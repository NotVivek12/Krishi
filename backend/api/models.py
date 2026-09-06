import uuid

from django.conf import settings
from django.db import models


class Prediction(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="crop_predictions",
    )
    input_data = models.JSONField()
    predicted_crop = models.CharField(max_length=120)
    confidence = models.FloatField()
    recommendations = models.JSONField(default=list)
    model_version = models.CharField(max_length=80)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(
                fields=["user", "-created_at"],
                name="api_pred_user_created_idx",
            ),
        ]

    def __str__(self):
        return f"{self.predicted_crop} ({self.confidence:.1%})"


class PredictionFeedback(models.Model):
    prediction = models.OneToOneField(
        Prediction,
        on_delete=models.CASCADE,
        related_name="feedback",
    )
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="prediction_feedback",
    )
    helpful = models.BooleanField()
    actual_crop = models.CharField(max_length=120, blank=True)
    notes = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Feedback for {self.prediction_id}"
