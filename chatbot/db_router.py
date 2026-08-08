class SitePrincipalRouter:
    route_app_labels = {"site_principal"}

    def db_for_read(self, model, **hints):
        if model._meta.app_label in self.route_app_labels:
            return "site_principal"
        return None

    def db_for_write(self, model, **hints):
        if model._meta.app_label in self.route_app_labels:
            raise PermissionError(
                "Ecriture interdite sur la base du site principal (lecture seule)."
            )
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        if app_label in self.route_app_labels:
            return False
        return None
