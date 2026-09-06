from django.contrib import admin

from .models import Prediction, PredictionFeedback


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "predicted_crop",
        "confidence",
        "user",
        "model_version",
        "created_at",
    )
    list_filter = ("predicted_crop", "model_version", "created_at")
    search_fields = ("id", "predicted_crop", "user__username", "user__email")
    readonly_fields = ("id", "created_at")


@admin.register(PredictionFeedback)
class PredictionFeedbackAdmin(admin.ModelAdmin):
    list_display = ("prediction", "user", "helpful", "updated_at")
    list_filter = ("helpful", "updated_at")
    search_fields = ("prediction__id", "user__username", "actual_crop")
