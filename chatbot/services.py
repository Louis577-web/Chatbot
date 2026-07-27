from .models import DocumentValide

#Il est important de noter que les mots vides sont des mots courants qui n'apportent pas beaucoup de valeur sémantique à une phrase. En les excluant, on peut améliorer la pertinence des résultats de recherche et la qualité des réponses générées par le chatbot.
MOTS_VIDES_FR = [
    "de", "la", "le", "les", "un", "une", "des", "du", "et", "est",
    "qui", "que", "quoi", "quel", "quelle", "quels", "quelles",
    "à", "au", "aux", "ce", "cette", "ces", "pour", "dans", "sur",
    "avec", "par", "en", "sont", "ai", "as", "a", "ont", "je", "tu",
]

#consteructeur de corpus de documents
def _construire_corpus_documents():
    documents = list(DocumentValide.objects.all())
    corpus = [
        f"{doc.titre} {doc.matiere} {doc.niveau} {doc.etablissement}"
        for doc in documents
    ]
    return documents, corpus

#calculer les scores de similarité entre la question et les documents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _calculer_scores(question):
    documents, corpus = _construire_corpus_documents()

    vectorizer = TfidfVectorizer()
    matrice = vectorizer.fit_transform(corpus + [question])

    vecteur_question = matrice[-1]
    vecteurs_documents = matrice[:-1]

    scores = cosine_similarity(vecteur_question, vecteurs_documents)[0]

    return documents, scores
#rechercher les documents les plus pertinents pour une question donnée
def rechercher_documents(question, top_k=3):
    documents, scores = _calculer_scores(question)

    resultats = sorted(
        zip(documents, scores),
        key=lambda paire: paire[1],
        reverse=True
    )

    return resultats[:top_k]

from chatbot.models import FAQEntry

FAQEntry.objects.create(
    question="Comment soumettre un document ?",
    reponse="Pour soumettre un document, connecte-toi à ton compte, va dans la section 'Déposer un document', puis suis les étapes indiquées.",
    categorie="soumission",
    est_guide=True
)

FAQEntry.objects.create(
    question="Combien de temps prend la validation d'un document ?",
    reponse="Le délai de validation peut aller jusqu'à un mois, selon la disponibilité des correcteurs.",
    categorie="delais",
    est_guide=False
)

FAQEntry.objects.create(
    question="Qui peut corriger un document ?",
    reponse="Seuls les membres valideurs de la plateforme, désignés par l'équipe, peuvent corriger un document soumis.",
    categorie="validation",
    est_guide=False
)

SEUIL_PERTINENCE_FAQ = 0.28


def _construire_corpus_faq():
    entrees = list(FAQEntry.objects.all())
    corpus = [entree.question for entree in entrees]
    return entrees, corpus


def rechercher_faq(question):
    entrees, corpus = _construire_corpus_faq()

    vectorizer = TfidfVectorizer(stop_words=MOTS_VIDES_FR)
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