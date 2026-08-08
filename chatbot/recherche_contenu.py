import os
import io
import boto3
from botocore.client import Config
import fitz  # pymupdf
import chromadb
from fastembed import TextEmbedding
from django.conf import settings

NOM_MODELE = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHEMIN_INDEX = os.path.join(settings.BASE_DIR, "chroma_index")

_modele = None
_client_chroma = None
_collection = None
_client_s3 = None


def _get_modele():
    global _modele
    if _modele is None:
        _modele = TextEmbedding(model_name=NOM_MODELE)
    return _modele


def _get_collection():
    global _client_chroma, _collection
    if _collection is None:
        _client_chroma = chromadb.PersistentClient(path=CHEMIN_INDEX)
        _collection = _client_chroma.get_or_create_collection("contenu_documents", metadata={"hnsw:space": "cosine"})
    return _collection


def _get_client_s3():
    global _client_s3
    if _client_s3 is None:
        _client_s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["CONTABO_ENDPOINT_URL"],
            aws_access_key_id=os.environ["CONTABO_ACCESS_KEY"],
            aws_secret_access_key=os.environ["CONTABO_SECRET_KEY"],
            region_name=os.environ.get("CONTABO_REGION", "eu2"),
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        )
    return _client_s3


def telecharger_pdf_depuis_s3(chemin_fichier):
    s3 = _get_client_s3()
    bucket = os.environ.get("CONTABO_BUCKET", "storage")
    prefixe = os.environ.get("CONTABO_LOCATION", "media")
    cle = f"{prefixe}/{chemin_fichier}"
    reponse = s3.get_object(Bucket=bucket, Key=cle)
    return reponse["Body"].read()


def extraire_texte_pdf(contenu_binaire):
    doc = fitz.open(stream=io.BytesIO(contenu_binaire), filetype="pdf")
    texte = "\n".join(page.get_text() for page in doc)
    doc.close()
    return texte


def decouper_en_blocs(texte, taille_mots=150):
    mots = texte.split()
    blocs = []
    for i in range(0, len(mots), taille_mots):
        bloc = " ".join(mots[i:i + taille_mots])
        if bloc.strip():
            blocs.append(bloc)
    return blocs


def indexer_ressource(ressource):
    try:
        contenu = telecharger_pdf_depuis_s3(ressource.fichier)
        texte = extraire_texte_pdf(contenu)
    except Exception as e:
        print(f"  Echec telechargement/extraction pour {ressource.nom} : {e}")
        return 0

    texte_complet = f"{ressource.nom}\n{ressource.description}\n{texte}"
    blocs = decouper_en_blocs(texte_complet)
    if not blocs:
        return 0

    modele = _get_modele()
    collection = _get_collection()
    embeddings = list(modele.embed(blocs))

    collection.delete(where={"ressource_id": ressource.id})
    collection.add(
        ids=[f"res{ressource.id}_bloc{i}" for i in range(len(blocs))],
        embeddings=[e.tolist() for e in embeddings],
        documents=blocs,
        metadatas=[{"ressource_id": ressource.id} for _ in blocs],
    )
    return len(blocs)


def rechercher_par_contenu(question, top_k=3, score_min=0.35):
    collection = _get_collection()
    if collection.count() == 0:
        return []

    modele = _get_modele()
    vecteur_question = list(modele.embed([question]))[0].tolist()

    resultats = collection.query(
        query_embeddings=[vecteur_question],
        n_results=min(top_k * 4, collection.count()),
    )

    meilleurs_scores = {}
    for distance, meta in zip(resultats["distances"][0], resultats["metadatas"][0]):
        score = 1 - distance
        rid = meta["ressource_id"]
        if rid not in meilleurs_scores or score > meilleurs_scores[rid]:
            meilleurs_scores[rid] = score

    resultats_tries = sorted(meilleurs_scores.items(), key=lambda p: p[1], reverse=True)
    return [(rid, score) for rid, score in resultats_tries if score >= score_min][:top_k]
