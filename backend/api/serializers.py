from rest_framework import serializers

from .exceptions import ModelUnavailable
from .models import Prediction, PredictionFeedback
from .services import ModelArtifactError, get_model_service


FIELD_ALIASES = {
    "soil": "soil",
    "SOIL": "soil",
    "season": "season",
    "SEASON": "season",
    "sown": "sown",
    "SOWN": "sown",
    "water_source": "water_source",
    "WATER_SOURCE": "water_source",
    "soil_ph": "soil_ph",
    "SOIL_PH": "soil_ph",
    "crop_duration": "crop_duration",
    "cropduration": "crop_duration",
    "CROPDURATION": "crop_duration",
    "temperature": "temperature",
    "temp": "temperature",
    "TEMP": "temperature",
    "water_required": "water_required",
    "WATERREQUIRED": "water_required",
    "relative_humidity": "relative_humidity",
    "RELATIVE_HUMIDITY": "relative_humidity",
    "nitrogen": "nitrogen",
    "N": "nitrogen",
    "phosphorus": "phosphorus",
    "P": "phosphorus",
    "potassium": "potassium",
    "K": "potassium",
    "top_k": "top_k",
    "TOP_K": "top_k",
}

MODEL_FIELD_MAP = {
    "soil": "SOIL",
    "season": "SEASON",
    "sown": "SOWN",
    "water_source": "WATER_SOURCE",
    "soil_ph": "SOIL_PH",
    "crop_duration": "CROPDURATION",
    "temperature": "TEMP",
    "water_required": "WATERREQUIRED",
    "relative_humidity": "RELATIVE_HUMIDITY",
    "nitrogen": "N",
    "phosphorus": "P",
    "potassium": "K",
}


class PredictionInputSerializer(serializers.Serializer):
    soil = serializers.CharField(max_length=120)
    season = serializers.CharField(max_length=30)
    sown = serializers.CharField(max_length=30)
    water_source = serializers.CharField(max_length=30)
    soil_ph = serializers.FloatField(min_value=5.0, max_value=9.0)
    crop_duration = serializers.FloatField(min_value=21.0, max_value=330.0)
    temperature = serializers.FloatField(min_value=5.0, max_value=47.0)
    water_required = serializers.FloatField(min_value=330.0, max_value=2499.8)
    relative_humidity = serializers.FloatField(min_value=15.0, max_value=100.0)
    nitrogen = serializers.FloatField(min_value=20.0, max_value=199.9)
    phosphorus = serializers.FloatField(min_value=20.0, max_value=100.0)
    potassium = serializers.FloatField(min_value=20.0, max_value=149.9)
    top_k = serializers.IntegerField(
        min_value=1,
        max_value=10,
        default=3,
        required=False,
    )

    def to_internal_value(self, data):
        payload = data.dict() if hasattr(data, "dict") else dict(data)
        unknown = sorted(set(payload) - set(FIELD_ALIASES))
        if unknown:
            raise serializers.ValidationError(
                {"non_field_errors": [f"Unknown fields: {', '.join(unknown)}."]}
            )

        normalized = {}
        for supplied_name, value in payload.items():
            canonical_name = FIELD_ALIASES[supplied_name]
            if (
                canonical_name in normalized
                and normalized[canonical_name] != value
            ):
                raise serializers.ValidationError(
                    {
                        canonical_name: [
                            "Conflicting values were supplied for this field."
                        ]
                    }
                )
            normalized[canonical_name] = value
        return super().to_internal_value(normalized)

    def validate(self, attrs):
        try:
            service = get_model_service()
            for api_name in ("soil", "season", "sown", "water_source"):
                model_name = MODEL_FIELD_MAP[api_name]
                attrs[api_name] = service.normalize_category(
                    model_name, attrs[api_name]
                )
        except ValueError as exc:
            raise serializers.ValidationError({api_name: [str(exc)]}) from exc
        except ModelArtifactError as exc:
            raise ModelUnavailable() from exc
        return attrs

    def model_payload(self):
        if not hasattr(self, "validated_data"):
            raise AssertionError("Call is_valid() before model_payload().")
        return {
            model_name: self.validated_data[api_name]
            for api_name, model_name in MODEL_FIELD_MAP.items()
        }


class FeedbackSerializer(serializers.ModelSerializer):
    class Meta:
        model = PredictionFeedback
        fields = (
            "helpful",
            "actual_crop",
            "notes",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("created_at", "updated_at")


class PredictionSerializer(serializers.ModelSerializer):
    feedback = FeedbackSerializer(read_only=True)

    class Meta:
        model = Prediction
        fields = (
            "id",
            "input_data",
            "predicted_crop",
            "confidence",
            "recommendations",
            "model_version",
            "feedback",
            "created_at",
        )
