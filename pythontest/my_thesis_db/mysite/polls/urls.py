from django.urls import path

from polls import views

# from rest_framework_simplejwt.views import (
#     TokenObtainPairView,
#     TokenRefreshView,
# )

urlpatterns = [
    path("api/chartdata", views.chartdata, name="chartdata"),
    path("api/downloadcsv", views.csvexport, name="csvexport"),
    path("list", views.getitems),
    path("mlactivate", views.MLactive),
    path("connectmqtt", views.mqtt_connect),
    path("deviceon", views.mqtt_publish_request),
    path("disconnectmqtt", views.mqtt_disconnect),
    path("getvariables", views.resulttraining),
    path("training", views.train),
    path("newtopic", views.mqtt_subcribe),
    # path("api/", views.get_routes),
    # path("api/token/", views.MyTokenObtainPairView.as_view(), name="token_obtain_pair"),
    # path("api/token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    # path("profile/", views.get_profile),
]
