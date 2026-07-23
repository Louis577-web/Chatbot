from .models import DocumentValide

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