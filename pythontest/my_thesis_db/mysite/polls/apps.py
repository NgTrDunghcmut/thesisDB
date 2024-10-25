from django.apps import AppConfig

# from .views import mqtt_connect


class PollsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "polls"

    # def ready(self):
    #     mqtt_connect()
