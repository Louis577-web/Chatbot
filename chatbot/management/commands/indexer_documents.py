from django.core.management.base import BaseCommand
from site_principal.models import Ressources
from chatbot.recherche_contenu import indexer_ressource


class Command(BaseCommand):
    help = "Indexe le contenu des documents reels (PDF) pour la recherche semantique"

    def add_arguments(self, parser):
        parser.add_argument("--limite", type=int, default=None)

    def handle(self, *args, **options):
        queryset = Ressources.objects.all().order_by("id")
        if options["limite"]:
            queryset = queryset[:options["limite"]]

        total = queryset.count() if not options["limite"] else options["limite"]
        total_blocs = 0
        reussites = 0

        for i, ressource in enumerate(queryset, start=1):
            self.stdout.write(f"[{i}/{total}] {ressource.nom} ...")
            nb_blocs = indexer_ressource(ressource)
            if nb_blocs:
                reussites += 1
                total_blocs += nb_blocs
                self.stdout.write(f"    -> {nb_blocs} bloc(s) indexe(s)")

        self.stdout.write(self.style.SUCCESS(
            f"\nTermine : {reussites}/{total} document(s) indexe(s) avec succes, "
            f"{total_blocs} bloc(s) au total."
        ))
