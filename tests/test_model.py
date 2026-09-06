import pytest

from api.services import get_model_service


pytestmark = pytest.mark.django_db


def test_model_metadata_matches_training_contract():
    metadata = get_model_service().metadata()

    assert metadata["status"] == "ready"
    assert metadata["name"] == "RandomForestClassifier"
    assert metadata["output_classes"] == 57
    assert [feature["name"] for feature in metadata["features"]] == [
        "soil",
        "season",
        "sown",
        "water_source",
        "soil_ph",
        "crop_duration",
        "temperature",
        "water_required",
        "relative_humidity",
        "nitrogen",
        "phosphorus",
        "potassium",
    ]


def test_model_predicts_ranked_probabilities(valid_prediction_payload):
    service = get_model_service()
    payload = {
        "SOIL": valid_prediction_payload["soil"],
        "SEASON": valid_prediction_payload["season"],
        "SOWN": valid_prediction_payload["sown"],
        "WATER_SOURCE": valid_prediction_payload["water_source"],
        "SOIL_PH": valid_prediction_payload["soil_ph"],
        "CROPDURATION": valid_prediction_payload["crop_duration"],
        "TEMP": valid_prediction_payload["temperature"],
        "WATERREQUIRED": valid_prediction_payload["water_required"],
        "RELATIVE_HUMIDITY": valid_prediction_payload["relative_humidity"],
        "N": valid_prediction_payload["nitrogen"],
        "P": valid_prediction_payload["phosphorus"],
        "K": valid_prediction_payload["potassium"],
    }

    result = service.predict(payload, top_k=5)

    assert result["predicted_crop"]
    assert result["confidence"] == result["recommendations"][0]["confidence"]
    assert len(result["recommendations"]) == 5
    assert [item["rank"] for item in result["recommendations"]] == [1, 2, 3, 4, 5]
    assert all(
        left["confidence"] >= right["confidence"]
        for left, right in zip(
            result["recommendations"],
            result["recommendations"][1:],
        )
    )


def test_category_matching_tolerates_case_and_whitespace():
    service = get_model_service()

    assert service.normalize_category("SEASON", "  KHARIF ") == "kharif"
    assert service.normalize_category("SOIL", "alluvial   SOIL") == "Alluvial soil"


def test_unknown_category_is_rejected():
    service = get_model_service()

    with pytest.raises(ValueError, match="Unsupported value"):
        service.normalize_category("SEASON", "monsoonish")
