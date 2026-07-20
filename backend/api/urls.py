from django.urls import path

from .views import (
    HealthView,
    ModelMetadataView,
    PredictView,
    PredictionDetailView,
    PredictionFeedbackView,
    PredictionListView,
)


app_name = "api"

urlpatterns = [
    path("health/", HealthView.as_view(), name="health"),
    path("model/metadata/", ModelMetadataView.as_view(), name="model-metadata"),
    path("predict/", PredictView.as_view(), name="predict"),
    path("predictions/", PredictionListView.as_view(), name="prediction-list"),
    path(
        "predictions/<uuid:prediction_id>/",
        PredictionDetailView.as_view(),
        name="prediction-detail",
    ),
    path(
        "predictions/<uuid:prediction_id>/feedback/",
        PredictionFeedbackView.as_view(),
        name="prediction-feedback",
    ),
]
