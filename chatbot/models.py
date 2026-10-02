from django.db import models

#Base de données pour les questions fréquentes (FAQ)
class FAQEntry(models.Model):
    question = models.CharField(max_length=255)
    reponse = models.TextField()
    categorie = models.CharField(max_length=50, blank=True)
    est_guide = models.BooleanField(default=False)

    def __str__(self):
        return self.question


#Base de données pour les documents validés
class DocumentValide(models.Model):

    # Lien entre le document validé et la ressource originale.
    # null=True et blank=True sont TEMPORAIRES : ils permettent
    # d'ajouter ce nouveau champ sans casser les anciens documents
    # déjà présents dans la base de données.
    ressource = models.OneToOneField(
        "contributeur.Ressources",
        on_delete=models.CASCADE,
        related_name="document_valide",
        null=True,
        blank=True,
    )

    titre = models.CharField(max_length=255)
    matiere = models.CharField(max_length=100)
    niveau = models.CharField(max_length=50)
    etablissement = models.CharField(max_length=150)
    annee = models.PositiveIntegerField()
    lien = models.URLField()

    def __str__(self):
        return self.titre
    
#Base de données pour les conversations du chatbot
class Conversation(models.Model):
    BESOIN_CHOICES = [
        ("RF1", "Réponse FAQ"),
        ("RF2", "Recherche de document"),
        ("RF3", "Guidage utilisateur"),
        ("RF4", "Salutations"),
        ("RF5", "Hors périmètre"),
    ]

    question = models.TextField()
    reponse = models.TextField()
    besoin_fonctionnel = models.CharField(max_length=10, choices=BESOIN_CHOICES)
    date_creation = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.date_creation} — {self.besoin_fonctionnel}"