import os
import io

import boto3
from botocore.client import Config
import pymupdf
import chromadb
from fastembed import TextEmbedding
from django.conf import settings

from contributeur.models import DocumentVente


NOM_MODELE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHEMIN_INDEX = os.path.join(settings.BASE_DIR, "chroma_index")

_modele = None
_client_chroma = None
_collection = None
_client_s3 = None


def _get_modele():
    global _modele

    if _modele is None:
        _modele = TextEmbedding(
            model_name=NOM_MODELE
        )

    return _modele


def _get_collection():
    global _client_chroma, _collection

    if _collection is None:
        _client_chroma = chromadb.PersistentClient(
            path=CHEMIN_INDEX
        )

        _collection = _client_chroma.get_or_create_collection(
            "contenu_documents",
            metadata={
                "hnsw:space": "cosine"
            }
        )

    return _collection


def _get_client_s3():
    global _client_s3

    if _client_s3 is None:
        _client_s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY"],
            aws_secret_access_key=os.environ["R2_SECRET_KEY"],
            region_name=os.environ.get(
                "R2_REGION",
                "auto"
            ),
            config=Config(
                signature_version="s3v4",
                s3={
                    "addressing_style": "path"
                },
            ),
        )

    return _client_s3


def telecharger_pdf_depuis_s3(chemin_fichier):
    s3 = _get_client_s3()

    bucket = os.environ.get(
        "R2_BUCKET",
        "storage"
    )

    prefixe = os.environ.get(
        "R2_LOCATION",
        "media"
    )

    cle = f"{prefixe}/{chemin_fichier}"

    reponse = s3.get_object(
        Bucket=bucket,
        Key=cle
    )

    return reponse["Body"].read()


def telecharger_pdf_depuis_local(chemin_fichier):
    chemin_complet = os.path.join(
        settings.MEDIA_ROOT,
        chemin_fichier
    )

    with open(chemin_complet, "rb") as f:
        return f.read()


def extraire_texte_pdf(contenu_binaire):
    doc = pymupdf.open(
        stream=io.BytesIO(contenu_binaire),
        filetype="pdf"
    )

    texte = "\n".join(
        page.get_text()
        for page in doc
    )

    doc.close()

    return texte


def decouper_en_blocs(texte, taille_mots=150):
    mots = texte.split()

    blocs = []

    for i in range(
        0,
        len(mots),
        taille_mots
    ):
        bloc = " ".join(
            mots[i:i + taille_mots]
        )

        if bloc.strip():
            blocs.append(bloc)

    return blocs


def indexer_ressource(ressource):
    """
    Indexe une ressource uniquement si elle possède
    un DocumentVente dont le statut est 'Publié'.

    Condition unique :

        DocumentVente.statut == "Publié"
    """

    # ============================================================
    # 1. VÉRIFICATION DU DOCUMENT DE VENTE
    # ============================================================

    document_vente_publie = DocumentVente.objects.filter(
        ressource=ressource,
        statut="Publié"
    ).exists()

    # Si aucun DocumentVente publié n'existe
    # pour cette ressource, on ne l'indexe pas.
    if not document_vente_publie:
        print(
            f"Ressource '{ressource.nom}' "
            f"(ID {ressource.id}) non indexée : "
            f"DocumentVente non publié."
        )

        return 0

    # ============================================================
    # 2. TÉLÉCHARGEMENT DU PDF
    # ============================================================

    try:
        chemin_fichier = ressource.fichier.name

        contenu = (
            telecharger_pdf_depuis_s3(
                chemin_fichier
            )
            if settings.PRODUCTION
            else telecharger_pdf_depuis_local(
                chemin_fichier
            )
        )

        texte = extraire_texte_pdf(
            contenu
        )

    except Exception as e:
        print(
            f"Échec téléchargement/extraction "
            f"pour {ressource.nom} : {e}"
        )

        return 0

    # ============================================================
    # 3. PRÉPARATION DU TEXTE
    # ============================================================

    texte_complet = (
        f"{ressource.nom}\n"
        f"{ressource.description}\n"
        f"{texte}"
    )

    blocs = decouper_en_blocs(
        texte_complet
    )

    if not blocs:
        return 0

    # ============================================================
    # 4. GÉNÉRATION DES EMBEDDINGS
    # ============================================================

    modele = _get_modele()
    collection = _get_collection()

    embeddings = list(
        modele.embed(blocs)
    )

    # ============================================================
    # 5. SUPPRESSION DE L'ANCIEN INDEX
    # ============================================================

    collection.delete(
        where={
            "ressource_id": ressource.id
        }
    )

    # ============================================================
    # 6. AJOUT DU NOUVEL INDEX
    # ============================================================

    collection.add(
        ids=[
            f"res{ressource.id}_bloc{i}"
            for i in range(len(blocs))
        ],

        embeddings=[
            embedding.tolist()
            for embedding in embeddings
        ],

        documents=blocs,

        metadatas=[
            {
                "ressource_id": ressource.id
            }
            for _ in blocs
        ],
    )

    print(
        f"Ressource '{ressource.nom}' "
        f"indexée avec succès : "
        f"{len(blocs)} blocs."
    )

    return len(blocs)


def rechercher_par_contenu(
    question,
    top_k=3,
    score_min=0.35
):
    """
    Recherche sémantique dans les ressources.

    Une ressource peut être retournée uniquement si elle
    possède un DocumentVente avec :

        statut == "Publié"
    """

    collection = _get_collection()

    # ============================================================
    # 1. VÉRIFIER QUE L'INDEX CONTIENT DES DOCUMENTS
    # ============================================================

    if collection.count() == 0:
        return []

    # ============================================================
    # 2. RÉCUPÉRER LES RESSOURCES PUBLIÉES
    # ============================================================

    ids_ressources_publiees = set(
        DocumentVente.objects
        .filter(
            statut="Publié",
            ressource__isnull=False
        )
        .values_list(
            "ressource_id",
            flat=True
        )
    )

    # Aucun document publié
    if not ids_ressources_publiees:
        return []

    # ============================================================
    # 3. TRANSFORMER LA QUESTION EN VECTEUR
    # ============================================================

    modele = _get_modele()

    vecteur_question = list(
        modele.embed([question])
    )[0].tolist()

    # ============================================================
    # 4. RECHERCHE DANS CHROMA
    # ============================================================

    resultats = collection.query(
        query_embeddings=[
            vecteur_question
        ],
        n_results=min(
            top_k * 10,
            collection.count()
        ),
    )

    # ============================================================
    # 5. FILTRER LES RESSOURCES NON PUBLIÉES
    # ============================================================

    meilleurs_scores = {}

    for distance, meta in zip(
        resultats["distances"][0],
        resultats["metadatas"][0]
    ):
        score = 1 - distance

        rid = meta["ressource_id"]

        # ========================================================
        # CONDITION UNIQUE :
        # DocumentVente.statut == "Publié"
        # ========================================================

        if rid not in ids_ressources_publiees:
            continue

        # Garder uniquement le meilleur score
        # pour chaque ressource.
        if (
            rid not in meilleurs_scores
            or score > meilleurs_scores[rid]
        ):
            meilleurs_scores[rid] = score

    # ============================================================
    # 6. TRI DES RÉSULTATS
    # ============================================================

    resultats_tries = sorted(
        meilleurs_scores.items(),
        key=lambda p: p[1],
        reverse=True
    )

    # ============================================================
    # 7. FILTRE PAR SCORE MINIMUM
    # ============================================================

    return [
        (rid, score)
        for rid, score in resultats_tries
        if score >= score_min
    ][:top_k]