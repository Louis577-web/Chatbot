from django.db import connections

from .models import FAQEntry, Conversation
from .recherche_contenu import rechercher_par_contenu
from site_principal.models import (
    Ressources,
    RessourcesAcademique,
    RessourcesConcours,
    RessourcesAdministratif,
    RessourcesPro,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import difflib
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
# retrouve dans le titre (score_lex == 0) : le bruit semantique pur
# (documents hors-sujet) peut monter jusqu'a ~0.60 sur des questions avec
# un mot tres generique ("suites"), donc 0.55 ne suffisait pas non plus a
# filtrer ce cas ("GUIDE DE REDACTION DES THESES" a 0.603). Sans mot-cle
# pour confirmer, on exige un score brut nettement plus haut. Note : ce
# n'est pas une garantie absolue - le plafond du bruit semantique varie
# selon la question (0.51 puis 0.60 observes), rien ne prouve qu'il ne
# peut pas depasser 0.65 pour une autre formulation. C'est un compromis
# empirique, pas une solution definitive.
SEUIL_SEMANTIQUE_SANS_LEXICAL = 0.65

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
    normale pour la recherche complementaire par mot-cle et le score
    lexical (voir _candidats_par_mot_cle et _score_lexical)."""
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


def _mots_titre(ressource):
    """Mots du titre d'une ressource, accents retires, prets a etre
    compares (exactement ou approximativement) aux mots-cles bruts de
    la question."""
    return re.findall(r"\b\w\w+\b", _normaliser_accents(ressource.nom))


# Ratio difflib (0-1) au-dela duquel deux mots sont consideres comme "la
# meme faute de frappe pres". Calibre pour rattraper une faute typique
# (une lettre manquante/en trop/inversee) sans devenir trop permissif :
# teste sur "tologie" vs "topologie" -> ratio ~0.875 (rattrape), largement
# au-dessus du seuil.
SEUIL_SIMILARITE_FLOUE = 0.82

# En dessous de cette longueur, un mot est trop court pour qu'une
# comparaison approximative ait un sens (une seule lettre de difference
# represente deja une trop grande partie du mot, risque de faux positifs
# entre mots courts sans rapport) : on exige alors une correspondance
# exacte ou une sous-chaine.
LONGUEUR_MIN_COMPARAISON_FLOUE = 5


def _mots_correspondent(mot_question, mot_titre):
    """Un mot de la question "correspond" a un mot du titre si l'un est
    une sous-chaine de l'autre (gere pluriels, radicaux : "vectoriel"
    dans "vectoriels", "serie" dans "series" une fois les accents
    retires), OU si les deux sont suffisamment proches au sens de
    difflib (tolerance aux fautes de frappe, ex: "tologie" ~ "topologie",
    testee et necessaire - une simple faute de frappe empechait de
    trouver un document existant, la comparaison exacte ne suffisait
    pas)."""
    if mot_question in mot_titre or mot_titre in mot_question:
        return True
    if len(mot_question) < LONGUEUR_MIN_COMPARAISON_FLOUE or len(mot_titre) < LONGUEUR_MIN_COMPARAISON_FLOUE:
        return False
    return difflib.SequenceMatcher(None, mot_question, mot_titre).ratio() >= SEUIL_SIMILARITE_FLOUE


# Dictionnaire de synonymes de discipline. Necessaire car aucun titre de
# cours specifique ne contient litteralement le mot "math", "physique" ou
# "info" - les titres reels utilisent le nom precis du chapitre (Algebre
# Lineaire, Suites et Series, Topologie, Mecanique du point...). Teste sur
# "je veux un document de math" : le mot-cle "math" ne correspondait a
# rien de pertinent (aucun cours ne s'appelle "Math"), seuls les 2 tests
# de recrutement (titre contenant litteralement "Mathematiques") et le
# CATALOGUE DES FILIERES (bruit semantique pur au-dessus du seuil sans
# lexical) remontaient - aucun vrai cours de maths.
#
# Chaque valeur est un ensemble de mots-cles (accents deja retires)
# frequents dans les titres des cours de cette discipline, construit a
# partir de la liste reelle des matieres en base (contributeur_matieres).
# Ce n'est pas une taxonomie exhaustive/officielle, juste une liste
# pragmatique batie sur les intitules de cours reellement presents sur la
# plateforme - a completer si de nouveaux types de cours sont ajoutes.
DISCIPLINES = {
    frozenset({"math", "maths", "mathematique", "mathematiques"}): {
        "algebre", "analyse", "calcul", "suite", "suites", "serie", "series",
        "geometrie", "topologie", "probabilite", "probabilites",
        "statistique", "statistiques", "equation", "equations",
        "integration", "integrale", "integrales", "integral", "variable",
        "variables", "matriciel", "vectoriel", "vectoriels", "vectoriel",
        "espace", "logique", "numerique", "groupes", "anneaux",
        "bilineaire", "lineaire", "derivable", "derivables", "fonctions",
        "nombres",
    },
    frozenset({"info", "infos", "informatique", "informatiques"}): {
        "algorithme", "algorithmes", "programmation", "python", "django",
        "cloud", "iot", "logiciel", "reseau", "architecture", "ordinateurs",
        "machine", "learning", "donnees", "scilab", "big", "data", "web",
    },
    frozenset({"physique", "physiques"}): {
        "mecanique", "optique", "electricite", "electromagnetisme",
        "thermodynamique", "elasticite", "tensoriel", "electronique",
        "quantique", "mmc",
    },
}


def _mots_cles_discipline(mot_normalise):
    """Si mot_normalise est un synonyme de discipline connu (ex: "math"),
    renvoie l'ensemble des mots-cles de cette discipline. Sinon None."""
    for declencheurs, mots_cles in DISCIPLINES.items():
        if mot_normalise in declencheurs:
            return mots_cles
    return None


def _mot_correspond_a_ressource(mot_question, mots_titre):
    """Un mot de la question correspond a une ressource si le mot
    lui-meme matche un des mots du titre (voir _mots_correspondent :
    sous-chaine ou faute de frappe proche), OU si le mot est un
    synonyme de discipline (ex: "math") et qu'un des mots-cles de
    cette discipline apparait dans le titre. Ce deuxieme cas permet a
    une recherche par categorie large ("un document de math") de
    trouver les cours specifiques (Algebre Lineaire, Topologie...) sans
    que leur titre contienne le mot "math" lui-meme."""
    if any(_mots_correspondent(mot_question, mot_titre) for mot_titre in mots_titre):
        return True
    mots_cles_discipline = _mots_cles_discipline(mot_question)
    if mots_cles_discipline:
        return any(
            mot_cle in mot_titre or mot_titre in mot_cle
            for mot_cle in mots_cles_discipline
            for mot_titre in mots_titre
        )
    return False


def _candidats_par_mot_cle(question, limite=10):
    """Recherche complementaire DIRECTE dans les TITRES uniquement (pas
    les descriptions - voir _score_lexical pour la raison), pour ne
    jamais rater un document dont le titre contient (exactement ou avec
    une faute de frappe proche) un mot-cle de la question, meme s'il
    n'est pas remonte dans le pool des CANDIDATS_SEMANTIQUES meilleurs
    candidats par embeddings.

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
        mots_titre = _mots_titre(ressource)
        if any(_mot_correspond_a_ressource(mot, mots_titre) for mot in mots):
            candidats.append(ressource)
            if len(candidats) >= limite:
                break
    return candidats


def _score_lexical(question, ressource):
    """Fraction des mots-cles de la question retrouves (exactement ou
    approximativement - voir _mots_correspondent) dans le TITRE de la
    ressource (nom) UNIQUEMENT - pas la description.

    Restriction au titre teste et necessaire : pour la question "un
    document sur les suites", la ressource "TEST EXPRESSION" (un test de
    comprehension du francais, hors-sujet) contenait le mot "suite" dans
    sa description ("la SUITE probable dans le developpement d'un
    message") - mais au sens de "ce qui suit", pas de "suite
    mathematique". Le mot etait identique, le sens different, et le
    score lexical la faisait remonter en tete (0.78) devant le vrai
    document sur les suites. Un titre est court et cure pour refleter le
    sujet reel ; une description longue en texte libre est bien plus
    sujette a ce genre de collision de sens.

    Tolerance aux fautes de frappe testee et necessaire : sans elle, une
    question comme "un document de tologie" (faute de frappe pour
    "topologie") ne retrouvait jamais le document "Topologie + Corriges"
    qui existe pourtant bel et bien, faute de correspondance exacte."""
    mots_question = _mots_cles_bruts(question)
    if not mots_question:
        return 0.0
    mots_titre = _mots_titre(ressource)
    trouves = sum(
        1 for mot in mots_question
        if _mot_correspond_a_ressource(mot, mots_titre)
    )
    return trouves / len(mots_question)


def _vocabulaire_titres():
    """Ensemble de tous les mots (accents retires) apparaissant dans les
    titres des ressources reelles. Sert de reference pour distinguer un
    vrai mot utilise sur la plateforme d'un charabia sans rapport (ex:
    "ahyyy") - voir _mot_est_plausible.

    Recalcule a chaque appel plutot que mis en cache : avec seulement
    ~182 ressources, le cout est negligeable, et ca reste toujours a jour
    si de nouveaux documents sont ajoutes/indexes entre deux questions."""
    vocabulaire = set()
    for ressource in Ressources.objects.all():
        vocabulaire.update(_mots_titre(ressource))
    return vocabulaire


def _mot_est_plausible(mot, vocabulaire):
    """Un mot de la question est "plausible" s'il correspond (exactement
    ou approximativement) a un mot reellement present dans un titre de
    document, OU s'il declenche une des DISCIPLINES connues (ex: "math"
    lui-meme n'apparait dans aucun titre, mais reste un mot-cle valide).

    Sert de garde-fou dans rechercher_documents : sans ca, une saisie
    absurde ("ahyyy") peut tout de meme depasser SEUIL_SEMANTIQUE_SANS_LEXICAL
    par hasard - l'embedding d'un texte, meme denue de sens, a toujours
    une similarite cosinus non nulle avec les documents indexes, et rien
    ne garantit qu'elle reste sous le seuil. Teste et necessaire : "ahyyy"
    faisait remonter "SUJETS CONCOURS BACHELIERS ESATIC 2021", totalement
    sans rapport."""
    if _mots_cles_discipline(mot) is not None:
        return True
    return any(_mots_correspondent(mot, mot_vocab) for mot_vocab in vocabulaire)


BASE_URL_SITE = "https://valideurlmd.com"

# Segment d'URL propre a chaque type de ressource, verifie directement sur
# le site (exemples reels donnes par Louis) :
#   ressource_pro           -> https://valideurlmd.com/details/ressource_pro/<slug>/
#   ressource_administrative -> https://valideurlmd.com/details/ressource_administrative/<slug>/
#   ressource_concours      -> https://valideurlmd.com/details/ressource_concours/<slug>/
#   ressource_academique    -> https://valideurlmd.com/details/ressource_academique/<slug>/
# Cle = nom du type cote Python (attribut ajoute par _types_ressources),
# valeur = segment reel dans l'URL (attention : "administratif" en base
# devient "administrative" dans l'URL, accord au feminin avec "ressource").
SEGMENT_URL_PAR_TYPE = {
    "academique": "ressource_academique",
    "concours": "ressource_concours",
    "administratif": "ressource_administrative",
    "pro": "ressource_pro",
}


def _types_ressources(ids_ressources):
    """Determine le type (academique/concours/administratif/pro) de
    chaque ressource, en interrogeant les 4 tables de sous-type. Chaque
    ressource appartient a exactement un seul de ces 4 types (relation
    OneToOne obligatoire cote modele)."""
    if not ids_ressources:
        return {}
    types = {}
    for modele, nom_type in (
        (RessourcesAcademique, "academique"),
        (RessourcesConcours, "concours"),
        (RessourcesAdministratif, "administratif"),
        (RessourcesPro, "pro"),
    ):
        for rid in modele.objects.filter(ressources_id__in=ids_ressources).values_list("ressources_id", flat=True):
            types[rid] = nom_type
    return types


def _construire_liens(ids_ressources):
    """Construit les liens publics d'acces aux documents, a partir du
    slug de vente individuel (contributeur_documentvente.slug) et du
    type de la ressource (voir SEGMENT_URL_PAR_TYPE).

    Corrige un bug reel : la version precedente ne construisait un lien
    qu'en passant par un PACK (contributeur_packdocumentedition), ce qui
    ratait tout document vendu individuellement sans faire partie d'un
    pack - teste et confirme sur "Python pour la Data Science" et "Big
    Data avec MongoDB" (documentvente existant, mais aucun pack associe :
    ces documents n'affichaient donc jamais de lien). Chaque ressource a
    une entree documentvente (avec son propre slug), qu'elle soit ou non
    en plus regroupee dans un pack - c'est donc une base plus fiable et
    plus universelle pour construire le lien que de dependre d'un pack.

    Renvoie un dict {ressource_id: url}. Une ressource sans documentvente
    ou sans type reconnu n'apparait simplement pas dans le dict : gere
    cote appelant en n'affichant pas de lien plutot que de planter."""
    if not ids_ressources:
        return {}

    placeholders = ",".join(["%s"] * len(ids_ressources))
    sql = f"""
        SELECT ressource_id, slug
        FROM contributeur_documentvente
        WHERE ressource_id IN ({placeholders})
    """
    slugs = {}
    with connections["site_principal"].cursor() as cur:
        cur.execute(sql, ids_ressources)
        for ressource_id, slug in cur.fetchall():
            if ressource_id not in slugs:  # on garde la premiere vente trouvee
                slugs[ressource_id] = slug

    types = _types_ressources(list(slugs))

    liens = {}
    for rid, slug in slugs.items():
        segment = SEGMENT_URL_PAR_TYPE.get(types.get(rid))
        if segment:
            liens[rid] = f"{BASE_URL_SITE}/details/{segment}/{slug}/"
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

    # Calcules une seule fois (pas par document candidat) : reutilises
    # dans le garde-fou anti-charabia ci-dessous.
    mots_question = _mots_cles_bruts(question)
    vocabulaire = _vocabulaire_titres()

    sortie = []
    for rid, score_sem in scores_semantiques.items():
        ressource = ressources_par_id.get(rid)
        if ressource is None:
            continue
        score_lex = _score_lexical(question, ressource)
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
            # Garde-fou supplementaire : meme au-dessus du seuil, on
            # n'accepte un match "sans lexical" que si au moins un mot de
            # la question ressemble a un vrai terme utilise sur la
            # plateforme (ou une discipline connue). Sans ca, une saisie
            # absurde ("ahyyy") pouvait par hasard depasser le seuil
            # semantique et faire remonter un document totalement
            # sans rapport (teste : "SUJETS CONCOURS BACHELIERS ESATIC").
            if not any(_mot_est_plausible(mot, vocabulaire) for mot in mots_question):
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
