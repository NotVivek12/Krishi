import hashlib
import re
import threading
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from django.conf import settings


FEATURE_SCHEMA = {
    "SOIL": {"kind": "category"},
    "SEASON": {"kind": "category"},
    "SOWN": {"kind": "category"},
    "WATER_SOURCE": {"kind": "category"},
    "SOIL_PH": {"kind": "number", "minimum": 5.0, "maximum": 9.0},
    "CROPDURATION": {"kind": "number", "minimum": 21.0, "maximum": 330.0},
    "TEMP": {"kind": "number", "minimum": 5.0, "maximum": 47.0},
    "WATERREQUIRED": {
        "kind": "number",
        "minimum": 330.0,
        "maximum": 2499.8,
    },
    "RELATIVE_HUMIDITY": {
        "kind": "number",
        "minimum": 15.0,
        "maximum": 100.0,
    },
    "N": {"kind": "number", "minimum": 20.0, "maximum": 199.9},
    "P": {"kind": "number", "minimum": 20.0, "maximum": 100.0},
    "K": {"kind": "number", "minimum": 20.0, "maximum": 149.9},
}

FEATURE_ORDER = list(FEATURE_SCHEMA)
CATEGORICAL_FEATURES = [
    name for name, definition in FEATURE_SCHEMA.items()
    if definition["kind"] == "category"
]


class ModelArtifactError(RuntimeError):
    pass


def _normalized_text(value):
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _display_text(value):
    return re.sub(r"\s+", " ", str(value)).strip()


class ModelService:
    def __init__(self, model_path, encoders_path, scaler_path=None):
        self.model_path = Path(model_path)
        self.encoders_path = Path(encoders_path)
        self.scaler_path = Path(scaler_path) if scaler_path else None
        self._model = None
        self._encoders = None
        self._scaler = None
        self._category_maps = None
        self._model_version = None
        self._load_lock = threading.RLock()
        self._prediction_lock = threading.Lock()

    @property
    def is_loaded(self):
        return self._model is not None and self._encoders is not None and self._scaler is not None

    def _ensure_loaded(self):
        if self.is_loaded:
            return

        with self._load_lock:
            if self.is_loaded:
                return

            missing = [
                str(path)
                for path in (self.model_path, self.encoders_path, self.scaler_path)
                if path and not path.is_file()
            ]
            if missing:
                raise ModelArtifactError(
                    f"Required model artifact not found: {', '.join(missing)}"
                )

            try:
                if self.model_path.suffix == ".keras":
                    import keras
                    model = keras.models.load_model(self.model_path)
                else:
                    model = joblib.load(self.model_path)
                encoders = joblib.load(self.encoders_path)
                scaler = joblib.load(self.scaler_path) if self.scaler_path else None
            except Exception as exc:
                raise ModelArtifactError(
                    "The crop recommendation model artifacts could not be loaded."
                ) from exc

            actual_features = list(
                getattr(model, "feature_names_in_", FEATURE_ORDER)
            )
            if actual_features != FEATURE_ORDER:
                raise ModelArtifactError(
                    "Model feature order does not match the serving schema."
                )

            required_encoders = set(CATEGORICAL_FEATURES + ["CROPS"])
            if not isinstance(encoders, dict) or not required_encoders.issubset(
                encoders
            ):
                raise ModelArtifactError(
                    "Encoder artifact is missing required feature encoders."
                )

            category_maps = {}
            for feature in CATEGORICAL_FEATURES:
                mapping = {}
                for item in encoders[feature].classes_:
                    mapping.setdefault(_normalized_text(item), str(item))
                category_maps[feature] = mapping

            self._model = model
            self._encoders = encoders
            self._scaler = scaler
            self._category_maps = category_maps
            try:
                self._model_version = self._artifact_version()
            except OSError as exc:
                self._model = None
                self._encoders = None
                self._scaler = None
                self._category_maps = None
                raise ModelArtifactError(
                    "The model artifact version could not be calculated."
                ) from exc

    def _artifact_version(self):
        digest = hashlib.sha256()
        for path in (self.model_path, self.encoders_path, self.scaler_path):
            if not path:
                continue
            with path.open("rb") as artifact:
                for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                    digest.update(chunk)
        prefix = "fl-keras" if self.model_path.suffix == ".keras" else "rf"
        return f"{prefix}-{digest.hexdigest()[:12]}"

    def normalize_category(self, feature, value):
        self._ensure_loaded()
        if feature not in self._category_maps:
            raise ValueError(f"{feature} is not a categorical feature.")
        canonical = self._category_maps[feature].get(_normalized_text(value))
        if canonical is None:
            choices = self.categories(feature)
            raise ValueError(
                f"Unsupported value. Choose one of: {', '.join(choices)}."
            )
        return canonical

    def categories(self, feature):
        self._ensure_loaded()
        values = {
            _display_text(value)
            for value in self._category_maps[feature].values()
        }
        return sorted(values, key=str.casefold)

    def predict(self, payload, top_k=3):
        self._ensure_loaded()
        encoded = {}

        for feature in FEATURE_ORDER:
            value = payload[feature]
            if feature in CATEGORICAL_FEATURES:
                canonical = self.normalize_category(feature, value)
                encoded[feature] = int(
                    self._encoders[feature].transform([canonical])[0]
                )
            else:
                encoded[feature] = float(value)

        frame = pd.DataFrame([encoded], columns=FEATURE_ORDER)
        if self._scaler is not None:
            # Important: apply scaler before prediction for keras model
            frame = self._scaler.transform(frame)
            
        with self._prediction_lock:
            if self.model_path.suffix == ".keras":
                probabilities = np.asarray(self._model.predict(frame, verbose=0)[0])
            else:
                probabilities = np.asarray(self._model.predict_proba(frame)[0])

        limit = min(max(int(top_k), 1), len(probabilities))
        ranked_indices = np.argsort(probabilities)[::-1][:limit]
        recommendations = []

        for rank, probability_index in enumerate(ranked_indices, start=1):
            if self.model_path.suffix == ".keras":
                class_id = int(probability_index)
            else:
                class_id = int(self._model.classes_[probability_index])
            crop = str(
                self._encoders["CROPS"].inverse_transform([class_id])[0]
            )
            recommendations.append(
                {
                    "rank": rank,
                    "crop": crop,
                    "confidence": round(float(probabilities[probability_index]), 6),
                }
            )

        return {
            "predicted_crop": recommendations[0]["crop"],
            "confidence": recommendations[0]["confidence"],
            "recommendations": recommendations,
            "model_version": self._model_version,
        }

    def metadata(self):
        self._ensure_loaded()
        features = []
        api_names = {
            "SOIL": "soil",
            "SEASON": "season",
            "SOWN": "sown",
            "WATER_SOURCE": "water_source",
            "SOIL_PH": "soil_ph",
            "CROPDURATION": "crop_duration",
            "TEMP": "temperature",
            "WATERREQUIRED": "water_required",
            "RELATIVE_HUMIDITY": "relative_humidity",
            "N": "nitrogen",
            "P": "phosphorus",
            "K": "potassium",
        }

        for name, definition in FEATURE_SCHEMA.items():
            item = {
                "name": api_names[name],
                "model_name": name,
                **definition,
            }
            if definition["kind"] == "category":
                item["choices"] = self.categories(name)
            features.append(item)

        return {
            "name": "AgriFL Federated DNN",
            "version": self._model_version,
            "status": "ready",
            "output_classes": len(self._encoders["CROPS"].classes_),
            "default_top_k": 3,
            "maximum_top_k": 10,
            "features": features,
            "limitations": (
                "The model was trained on controlled synthetic data and should "
                "support, not replace, local agronomic advice."
            ),
        }

    def healthcheck(self):
        try:
            self._ensure_loaded()
        except ModelArtifactError as exc:
            return {"status": "unavailable", "detail": str(exc)}
        return {
            "status": "ready",
            "model_version": self._model_version,
        }


@lru_cache(maxsize=1)
def get_model_service():
    return ModelService(
        settings.MODEL_PATH, 
        settings.ENCODERS_PATH, 
        getattr(settings, "SCALER_PATH", None)
    )
