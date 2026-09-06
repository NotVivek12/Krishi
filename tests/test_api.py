import pytest

from api.models import Prediction


pytestmark = pytest.mark.django_db


def test_health_endpoint_checks_database_and_model(api_client):
    response = api_client.get("/health/")

    assert response.status_code == 200
    assert response.data["status"] == "healthy"
    assert response.data["checks"]["database"]["status"] == "ready"
    assert response.data["checks"]["model"]["status"] == "ready"
    assert response.headers["X-Request-ID"]


def test_model_metadata_endpoint_is_public(api_client):
    response = api_client.get("/api/model/metadata/")

    assert response.status_code == 200
    assert response.data["status"] == "ready"
    assert response.data["output_classes"] == 57
    assert len(response.data["features"]) == 12


def test_valid_prediction_is_persisted(api_client, valid_prediction_payload):
    response = api_client.post(
        "/api/predict/",
        valid_prediction_payload,
        format="json",
    )

    assert response.status_code == 200
    assert response.data["predicted_crop"]
    assert 0 <= response.data["confidence"] <= 1
    assert len(response.data["recommendations"]) == 3
    assert Prediction.objects.filter(id=response.data["prediction_id"]).exists()


def test_training_field_aliases_are_accepted(api_client, valid_prediction_payload):
    aliases = {
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
        "TOP_K": 2,
    }

    response = api_client.post("/api/predict/", aliases, format="json")

    assert response.status_code == 200
    assert len(response.data["recommendations"]) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("season", "winter"),
        ("soil_ph", 15),
        ("top_k", 0),
    ],
)
def test_invalid_prediction_returns_field_error(
    api_client,
    valid_prediction_payload,
    field,
    value,
):
    valid_prediction_payload[field] = value

    response = api_client.post(
        "/api/predict/",
        valid_prediction_payload,
        format="json",
    )

    assert response.status_code == 400
    assert field in response.data
    assert response.data["request_id"]


def test_unknown_prediction_fields_are_rejected(
    api_client,
    valid_prediction_payload,
):
    valid_prediction_payload["humidity"] = 80

    response = api_client.post(
        "/api/predict/",
        valid_prediction_payload,
        format="json",
    )

    assert response.status_code == 400
    assert "Unknown fields: humidity" in str(response.data["non_field_errors"])


def test_history_requires_authentication(api_client):
    response = api_client.get("/api/predictions/")

    assert response.status_code == 401


def test_authenticated_prediction_appears_in_history(
    api_client,
    registered_user,
    valid_prediction_payload,
):
    prediction = api_client.post(
        "/api/predict/",
        valid_prediction_payload,
        format="json",
    )

    history = api_client.get("/api/predictions/")

    assert prediction.status_code == 200
    assert history.status_code == 200
    assert history.data["count"] == 1
    assert history.data["results"][0]["id"] == str(
        prediction.data["prediction_id"]
    )
