import pytest
from rest_framework.test import APIClient


@pytest.fixture
def api_client():
    return APIClient()


@pytest.fixture
def valid_prediction_payload():
    return {
        "soil": "Alluvial soil",
        "season": "kharif",
        "sown": "Jun",
        "water_source": "irrigated",
        "soil_ph": 7.6,
        "crop_duration": 116.9,
        "temperature": 26.9,
        "water_required": 2462.3,
        "relative_humidity": 73.8,
        "nitrogen": 82.4,
        "phosphorus": 40.7,
        "potassium": 42.2,
        "top_k": 3,
    }


@pytest.fixture
def registered_user(api_client, django_user_model):
    user = django_user_model.objects.create_user(
        username="farmer",
        email="farmer@example.com",
        password="Strong-Passphrase-482!",
    )
    response = api_client.post(
        "/api/auth/token/",
        {
            "username": user.username,
            "password": "Strong-Passphrase-482!",
        },
        format="json",
    )
    api_client.credentials(HTTP_AUTHORIZATION=f"Bearer {response.data['access']}")
    return user
