from django.db import models


class Pays(models.Model):
    id = models.BigAutoField(primary_key=True)
    nom = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = "administrateur_pays"
        app_label = "site_principal"

    def __str__(self):
        return self.nom


class Ecole(models.Model):
    id = models.BigAutoField(primary_key=True)
    nom = models.CharField(max_length=255)
    pays = models.ForeignKey(Pays, on_delete=models.DO_NOTHING, db_column="pays_id", related_name="+")

    class Meta:
        managed = False
        db_table = "administrateur_ecole"
        app_label = "site_principal"

    def __str__(self):
        return self.nom


class Matieres(models.Model):
    id = models.BigAutoField(primary_key=True)
    nom = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = "contributeur_matieres"
        app_label = "site_principal"

    def __str__(self):
        return self.nom


class Niveau(models.Model):
    id = models.BigAutoField(primary_key=True)
    nom = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = "contributeur_niveau"
        app_label = "site_principal"

    def __str__(self):
        return self.nom


class Filieres(models.Model):
    id = models.BigAutoField(primary_key=True)
    nom = models.CharField(max_length=255)
    ecole = models.ForeignKey(Ecole, on_delete=models.DO_NOTHING, db_column="ecole_id", related_name="+")

    class Meta:
        managed = False
        db_table = "contributeur_filieres"
        app_label = "site_principal"

    def __str__(self):
        return self.nom


class Ressources(models.Model):
    id = models.BigAutoField(primary_key=True)
    nom = models.CharField(max_length=255)
    description = models.TextField()
    fichier = models.CharField(max_length=500)
    annee_academique = models.CharField(max_length=50, null=True, blank=True)
    contributeur_id = models.BigIntegerField(null=True, blank=True)
    ecole = models.ForeignKey(Ecole, on_delete=models.DO_NOTHING, db_column="ecole_id", null=True, related_name="+")
    pays = models.ForeignKey(Pays, on_delete=models.DO_NOTHING, db_column="pays_id", null=True, related_name="+")
    date_creation = models.DateTimeField(null=True, blank=True)
    date_mise_a_jour = models.DateTimeField(null=True, blank=True)
    file_hash = models.CharField(max_length=255)

    class Meta:
        managed = False
        db_table = "contributeur_ressources"
        app_label = "site_principal"

    def __str__(self):
        return self.nom


class RessourcesAcademique(models.Model):
    ressources = models.OneToOneField(
        Ressources, primary_key=True, db_column="ressources_ptr_id",
        on_delete=models.DO_NOTHING, related_name="academique",
    )
    session = models.CharField(max_length=100, null=True, blank=True)
    motif_refus = models.TextField(null=True, blank=True)
    statut = models.CharField(max_length=50, null=True, blank=True)
    matiere = models.ForeignKey(Matieres, on_delete=models.DO_NOTHING, db_column="matiere_id", related_name="+")
    filiere = models.ForeignKey(Filieres, on_delete=models.DO_NOTHING, db_column="filiere_id", null=True, related_name="+")
    niveau = models.ForeignKey(Niveau, on_delete=models.DO_NOTHING, db_column="niveau_id", null=True, related_name="+")

    class Meta:
        managed = False
        db_table = "contributeur_ressourcesacademique"
        app_label = "site_principal"


class RessourcesConcours(models.Model):
    ressources = models.OneToOneField(
        Ressources, primary_key=True, db_column="ressources_ptr_id",
        on_delete=models.DO_NOTHING, related_name="concours",
    )
    statut = models.CharField(max_length=50, null=True, blank=True)
    type_document_id = models.BigIntegerField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "contributeur_ressourcesconcours"
        app_label = "site_principal"


class RessourcesAdministratif(models.Model):
    ressources = models.OneToOneField(
        Ressources, primary_key=True, db_column="ressources_ptr_id",
        on_delete=models.DO_NOTHING, related_name="administratif",
    )
    motif_refus = models.TextField(null=True, blank=True)
    statut = models.CharField(max_length=50, null=True, blank=True)
    type_document_id = models.BigIntegerField()

    class Meta:
        managed = False
        db_table = "contributeur_ressourcesadministratif"
        app_label = "site_principal"


class RessourcesPro(models.Model):
    ressources = models.OneToOneField(
        Ressources, primary_key=True, db_column="ressources_ptr_id",
        on_delete=models.DO_NOTHING, related_name="pro",
    )
    statut = models.CharField(max_length=50, null=True, blank=True)
    type_document_id = models.BigIntegerField(null=True, blank=True)

    class Meta:
        managed = False
        db_table = "contributeur_ressourcespro"
        app_label = "site_principal"
