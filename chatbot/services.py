from django.db import connections

from .models import FAQEntry, Conversation
from .recherche_contenu import rechercher_par_contenu
from site_principal.models import Ressources
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import re
import unicodedata

# Il est important de noter que les mots vides sont des mots courants qui n'apportent pas beaucoup de valeur semantique a une phrase. En les excluant, on peut ameliorer la pertinence des resultats de recherche et la qualite des reponses generees par le chatbot.
MOTS_VIDES_FR = [
    "de", "la", "le", "les", "un", "une", "des", "du", "et", "est",
    "qui", "que", "quoi", "quel", "quelle", "quels", "quelles",
    "à", "au", "aux", "ce", "cette", "ces", "pour", "dans", "sur",
    "avec", "par", "en", "sont", "ai", "as", "a", "ont", "je", "tu",
    "cherche", "cherches", "recherche", "veux", "voudrais", "aimerais",
    "peux", "pourrais", "trouve", "trouver", "donne", "montre",
]
_stemmer_fr = SnowballStemmer("french")


def _normaliser_accents(texte):
    """Minuscules + suppression des accents (ete -> e, series -> series,
    Séries -> series). Utilise partout ou on doit comparer du texte issu
    de sources differentes (question tapee sans accent vs titre en base
    avec accent) : PostgreSQL ILIKE est sensible aux accents par defaut,
    donc une comparaison cote base echouerait silencieusement sans ca
    (teste et corrige - voir _candidats_par_mot_cle)."""
    return ''.join(
        c for c in unicodedata.normalize('NFKD', texte.lower())
        if not unicodedata.combining(c)
    )


def _analyseur_stemming(texte):
    texte = _normaliser_accents(texte)
    mots = re.findall(r"\b\w\w+\b", texte)
    return [_stemmer_fr.stem(mot) for mot in mots if mot not in MOTS_VIDES_FR]


SEUIL_PERTINENCE_FAQ = 0.28

# Seuil de similarite cosinus sur les EMBEDDINGS, applique au score final
# HYBRIDE (semantique + bonus lexical, voir plus bas), pas au score
# semantique brut.
SEUIL_PERTINENCE_DOCUMENT = 0.35

# A l'echelle des 182 documents reels, le score semantique seul ne suffit
# pas : des documents hors-sujet (ex: catalogue d'ecole, guide de these)
# scorent parfois plus haut (jusqu'a 0.55) que le bon document (0.41-0.44),
# car ce petit modele multilingue capte surtout un signal de "style
# academique francais" plutot que le sujet precis sur des questions
# courtes. On corrige avec un score hybride : score semantique + bonus si
# les mots-cles de la question apparaissent dans le titre/la description.
# Teste sur "mecanique du point" : les 2 bons documents remontent bien en
# tete (0.76-0.78) devant les faux positifs (0.49 max).
POIDS_LEXICAL = 0.35
CANDIDATS_SEMANTIQUES = 20
SEUIL_SEMANTIQUE_CANDIDAT = 0.30

# Seuil applique UNIQUEMENT quand aucun mot-cle de la question n'est
# retrouve dans le titre/la description (score_lex == 0) : le bruit
# semantique pur (documents hors-sujet) peut alors monter jusqu'a ~0.51
# sur des questions sans reponse reelle dans le corpus, donc 0.35 ne
# suffit pas a filtrer ce cas. Sans mot-cle pour confirmer, on exige un
# score brut nettement plus haut.
SEUIL_SEMANTIQUE_SANS_LEXICAL = 0.55

# Mots "porteurs de requete" - le type de chose demandee plutot que son
# sujet (ex: "je veux un DOCUMENT sur..."). Presents dans presque toutes
# les descriptions quel que soit le sujet, ils fausseraient le score
# lexical s'ils n'etaient pas exclus (teste : "document" faisait
# artificiellement remonter des documents sans rapport).
MOTS_GENERIQUES_DEMANDE = {
    "document", "documents", "fichier", "fichiers", "pdf", "cours",
    "sujet", "sujets", "support", "supports", "epreuve", "epreuves",
    "corrige", "corriges", "examen", "examens", "devoir", "devoirs",
}
_STEMS_GENERIQUES = {_stemmer_fr.stem(m) for m in MOTS_GENERIQUES_DEMANDE}


def _stemmes_utiles(texte):
    return set(_analyseur_stemming(texte)) - _STEMS_GENERIQUES


def _mots_cles_bruts(question):
    """Comme _stemmes_utiles, mais renvoie les mots ORIGINAUX (non
    stemmes, mais accents deja retires) : un stem (ex: "vector" pour
    "vectoriel") n'est pas forcement une sous-chaine cherchable telle
    quelle dans un titre, donc on a besoin des mots dans leur forme
    normale pour la recherche complementaire par mot-cle (voir
    _candidats_par_mot_cle)."""
    texte = _normaliser_accents(question)
    mots = re.findall(r"\b\w\w+\b", texte)
    resultat = []
    for mot in mots:
        if mot in MOTS_VIDES_FR:
            continue
        if _stemmer_fr.stem(mot) in _STEMS_GENERIQUES:
            continue
        resultat.append(mot)
    return resultat


def _candidats_par_mot_cle(question, limite=10):
    """Recherche complementaire DIRECTE dans les titres/descriptions,
    pour ne jamais rater un document dont le titre contient litteralement
    un mot-cle de la question, meme s'il n'est pas remonte dans le pool
    des CANDIDATS_SEMANTIQUES meilleurs candidats par embeddings.

    Teste et necessaire : pour la question "sujet d'examen sur les
    series", le document "Suites et Series" scorait 0.56 en semantique
    brut (largement pertinent) mais se classait 14e, juste hors du pool
    de 20 candidats retenus avant re-classement - il n'apparaissait donc
    jamais dans les resultats. Une reformulation minime le faisait
    remonter au rang 7 et le document ressortait alors correctement :
    preuve que la similarite d'embeddings seule est trop sensible a la
    formulation exacte pour qu'on lui fasse confiance a elle seule.

    Comparaison faite en PYTHON (pas via ILIKE cote base) : PostgreSQL
    ILIKE est sensible aux accents par defaut, donc chercher "series"
    (question tapee sans accent) ne matcherait pas "Séries" (titre avec
    accent) - teste et confirme, meme piege que pour le diagnostic
    precedent. Avec seulement 182 ressources au total, charger et
    comparer en Python est trivial en performance et evite ce probleme
    sans dependre d'une extension PostgreSQL (unaccent) potentiellement
    indisponible sur une base geree en lecture seule."""
    mots = _mots_cles_bruts(question)
    if not mots:
        return []

    candidats = []
    for ressource in Ressources.objects.select_related("ecole", "pays").all():
        texte = _normaliser_accents(f"{ressource.nom} {ressource.description}")
        if any(mot in texte for mot in mots):
            candidats.append(ressource)
            if len(candidats) >= limite:
                break
    return candidats


def _score_lexical(question, texte):
    """Fraction des mots-cles significatifs de la question retrouves dans
    le texte (titre + description d'une ressource). 0 si la question ne
    contient aucun mot-cle significatif (que des mots vides/generiques)."""
    mots_question = _stemmes_utiles(question)
    if not mots_question:
        return 0.0
    mots_texte = _stemmes_utiles(texte)
    return len(mots_question & mots_texte) / len(mots_question)


BASE_URL_SITE = "https://valideurlmd.com"


def _construire_liens(ids_ressources):
    """Construit les liens publics d'acces aux documents, en remontant
    Ressources -> documentvente -> pack (contributeur_packdocumentedition),
    via une requete SQL groupee (pas de modele Django pour ces tables,
    la base du site principal etant en lecture seule).

    Renvoie un dict {ressource_id: url}. Un document qui n'a pas de pack
    associe (ex: certains documents administratifs) n'apparait simplement
    pas dans le dict : on gere ce cas cote appelant en n'affichant pas de
    lien plutot que de planter.

    Note : le segment d'URL "pack_ressource_academique" a ete verifie
    pour des ressources academiques (types RessourcesAcademique). Il est
    possible que les documents de type concours/administratif/pro suivent
    un autre schema d'URL sur le site - a verifier si des liens errones
    apparaissent pour ces types-la."""
    if not ids_ressources:
        return {}

    placeholders = ",".join(["%s"] * len(ids_ressources))
    sql = f"""
        SELECT dv.ressource_id, pde.slug
        FROM contributeur_documentvente dv
        JOIN contributeur_packdocumentedition_pack link ON link.documentvente_id = dv.id
        JOIN contributeur_packdocumentedition pde ON pde.id = link.packdocumentedition_id
        WHERE dv.ressource_id IN ({placeholders})
    """
    liens = {}
    with connections["site_principal"].cursor() as cur:
        cur.execute(sql, ids_ressources)
        for ressource_id, slug in cur.fetchall():
            if ressource_id not in liens:  # on garde le premier pack trouve
                liens[ressource_id] = f"{BASE_URL_SITE}/details/pack_ressource_academique/{slug}/"
    return liens


def rechercher_documents(question, top_k=3):
    """Recherche semantique dans le CONTENU reel des documents (PDF sur
    Contabo), via l'index Chroma construit par indexer_documents,
    COMPLETEE par une recherche directe par mot-cle (_candidats_par_mot_cle,
    voir sa docstring pour le cas concret qui a motive cet ajout), puis
    RE-CLASSEMENT HYBRIDE : le score semantique brut ne suffit pas a bien
    ordonner les resultats a l'echelle des 182 documents reels (voir
    POIDS_LEXICAL ci-dessus), donc on reordonne en ajoutant un bonus si
    les mots-cles de la question apparaissent dans le titre/la
    description, avant de ne garder que les meilleurs top_k.

    Renvoie une liste de tuples (ressource, score, lien_ou_None)."""
    candidats = rechercher_par_contenu(
        question, top_k=CANDIDATS_SEMANTIQUES, score_min=SEUIL_SEMANTIQUE_CANDIDAT
    )
    scores_semantiques = dict(candidats)
    ressources_par_id = {}
    if scores_semantiques:
        ressources_par_id = {
            r.id: r for r in Ressources.objects.select_related("ecole", "pays")
            .filter(id__in=scores_semantiques)
        }

    # Complement mot-cle : un document trouve ainsi mais absent du pool
    # semantique recoit un score semantique "neutre" de 0 - c'est le bonus
    # lexical (POIDS_LEXICAL) qui portera son score final, pas une
    # similarite d'embeddings qu'on n'a pas calculee pour lui.
    for ressource in _candidats_par_mot_cle(question):
        if ressource.id not in scores_semantiques:
            scores_semantiques[ressource.id] = 0.0
            ressources_par_id[ressource.id] = ressource

    if not scores_semantiques:
        return []

    liens_par_id = _construire_liens(list(scores_semantiques))

    sortie = []
    for rid, score_sem in scores_semantiques.items():
        ressource = ressources_par_id.get(rid)
        if ressource is None:
            continue
        score_lex = _score_lexical(question, f"{ressource.nom} {ressource.description}")
        score_final = score_sem + POIDS_LEXICAL * score_lex

        if score_lex == 0:
            # Aucun mot-cle en commun avec le titre/la description : le
            # bonus lexical ne peut pas confirmer la pertinence, donc on
            # exige un score semantique brut nettement plus eleve. Teste
            # sur une question sans reponse reelle dans les 182 documents
            # ("sujet d'examen sur les series") : le bruit semantique pur
            # montait jusqu'a 0.51 sans qu'aucun document ne soit vraiment
            # pertinent - 0.35 est trop bas pour filtrer ce cas.
            if score_sem < SEUIL_SEMANTIQUE_SANS_LEXICAL:
                continue
        elif score_final < SEUIL_PERTINENCE_DOCUMENT:
            continue

        sortie.append((ressource, score_final, liens_par_id.get(rid)))

    sortie.sort(key=lambda triplet: triplet[1], reverse=True)
    return sortie[:top_k]


def _construire_corpus_faq():
    entrees = list(FAQEntry.objects.all())
    corpus = [entree.question for entree in entrees]
    return entrees, corpus


def rechercher_faq(question):
    entrees, corpus = _construire_corpus_faq()
    if not entrees:
        return None, 0.0
    vectorizer = TfidfVectorizer(analyzer=_analyseur_stemming)
    matrice = vectorizer.fit_transform(corpus + [question])
    vecteur_question = matrice[-1]
    vecteurs_entrees = matrice[:-1]
    scores = cosine_similarity(vecteur_question, vecteurs_entrees)[0]
    resultats = sorted(
        zip(entrees, scores),
        key=lambda paire: paire[1],
        reverse=True
    )
    meilleure_entree, meilleur_score = resultats[0]
    if meilleur_score >= SEUIL_PERTINENCE_FAQ:
        return meilleure_entree, meilleur_score
    else:
        return None, meilleur_score


# Prendre en compte les salutations dans la question de l'utilisateur. Si la question contient une salutation, le chatbot repondra avec un message de salutation approprie.
SALUTATIONS = ["salut", "bonjour", "bonsoir", "coucou", "hello", "hey", "bjr"]


def _est_salutation(question):
    mots = question.lower().strip().split()
    return any(mot.strip(" !?.,") in SALUTATIONS for mot in mots)


# Remerciements/cloture de conversation : sans ce cas special, "Merci" ou
# "ok" tombaient directement dans la recherche de documents (aucun mot-cle
# utile pour filtrer), et remontaient des documents au hasard.
#
# Contrairement a SALUTATIONS (verifie mot par mot), on compare ici le
# MESSAGE ENTIER (nettoye) a cette liste, pas mot par mot : des mots comme
# "bon", "oui" ou "ca va" sont trop courants dans de vraies questions
# ("quel est le BON niveau...") pour etre detectes en tant que simple mot
# isole sans faux positifs. Ne fonctionne donc que si l'utilisateur envoie
# UNIQUEMENT une formule de cloture, ce qui est le cas typique apres avoir
# recu une reponse utile.
MOTS_CLOTURE = {
    "merci", "mercii", "merci beaucoup", "merci bcp", "thanks", "thank you",
    "ok", "okay", "oki", "d'accord", "dacc", "daccord", "entendu",
    "compris", "recu", "reçu", "note", "noté", "noted",
    "c'est bon", "cest bon", "ca marche", "ça marche",
    "ca va", "ça va", "impeccable", "impec", "nickel",
    "cool", "top", "parfait", "super", "genial", "génial", "bien",
    "très bien", "tres bien", "au top",
}


def _est_cloture(question):
    return question.lower().strip().strip(" !?.,") in MOTS_CLOTURE


# La fonction `repondre` prend une question en entree et tente de trouver une reponse appropriee en recherchant d'abord dans la FAQ, puis dans les documents reels. Si aucune reponse pertinente n'est trouvee, elle renvoie un message indiquant qu'aucune reponse fiable n'a ete trouvee.
def repondre(question):
    if _est_salutation(question):
        reponse = "Bonjour ! Je suis l'assistant de Valideur LMD. Pose-moi une question sur la plateforme, ou dis-moi quel document tu cherches."
        besoin = "RF4"
        sources = []
    elif _est_cloture(question):
        reponse = "Avec plaisir ! N'hésite pas si tu as d'autres questions ou si tu cherches un autre document."
        besoin = "RF4"
        sources = []
    else:
        entree, score_faq = rechercher_faq(question)
        if entree is not None:
            besoin = "RF3" if entree.est_guide else "RF1"
            reponse = entree.reponse
            sources = []
        elif not _stemmes_utiles(question):
            # Aucun mot-cle significatif dans la question (que des mots
            # vides/generiques) : lancer la recherche donnerait un
            # resultat quasi aleatoire plutot qu'une vraie non-reponse.
            reponse = (
                "Je n'ai pas bien compris ta demande. Peux-tu préciser le sujet, "
                "la matière ou le type de document que tu cherches ?"
            )
            besoin = "RF5"
            sources = []
        else:
            resultats_pertinents = rechercher_documents(question, top_k=3)
            if resultats_pertinents:
                lignes = []
                for doc, score, lien in resultats_pertinents:
                    ligne = f"- {doc.nom} : {doc.description}"
                    if lien:
                        ligne += f"\n  Accès : {lien}"
                    lignes.append(ligne)
                reponse = (
                    "Voici les documents qui correspondent le mieux à ta recherche :\n"
                    + "\n".join(lignes)
                )
                besoin = "RF2"
                sources = [lien or doc.fichier for doc, score, lien in resultats_pertinents]
            else:
                reponse = (
                    "Je suis désolé, je n'ai pas trouvé de réponse fiable à ce sujet. "
                    "N'hésite pas à reformuler ta question, ou à me demander autre chose."
                )
                besoin = "RF5"
                sources = []
    Conversation.objects.create(
        question=question,
        reponse=reponse,
        besoin_fonctionnel=besoin,
    )
    return {
        "reponse": reponse,
        "besoin_fonctionnel": besoin,
        "sources": sources,
    }
