import pytest


pytestmark = pytest.mark.django_db


def test_register_predict_review_and_logout_flow(
    api_client,
    valid_prediction_payload,
):
    registration = api_client.post(
        "/api/auth/register/",
        {
            "username": "grower",
            "email": "grower@example.com",
            "password": "A-Strong-Farm-Passphrase-928!",
            "first_name": "Asha",
        },
        format="json",
    )
    assert registration.status_code == 201
    assert registration.data["user"]["email"] == "grower@example.com"

    api_client.credentials(
        HTTP_AUTHORIZATION=f"Bearer {registration.data['access']}"
    )
    prediction = api_client.post(
        "/api/predict/",
        valid_prediction_payload,
        format="json",
    )
    assert prediction.status_code == 200

    feedback = api_client.post(
        f"/api/predictions/{prediction.data['prediction_id']}/feedback/",
        {
            "helpful": True,
            "actual_crop": prediction.data["predicted_crop"],
            "notes": "Recommendation matched the field plan.",
        },
        format="json",
    )
    assert feedback.status_code == 201
    assert feedback.data["helpful"] is True

    detail = api_client.get(
        f"/api/predictions/{prediction.data['prediction_id']}/"
    )
    assert detail.status_code == 200
    assert detail.data["feedback"]["helpful"] is True

    logout = api_client.post(
        "/api/auth/logout/",
        {"refresh": registration.data["refresh"]},
        format="json",
    )
    assert logout.status_code == 204


def test_users_cannot_read_or_review_each_others_predictions(
    api_client,
    django_user_model,
    valid_prediction_payload,
):
    first_user = django_user_model.objects.create_user(
        username="first",
        password="First-Secure-Passphrase-219!",
    )
    second_user = django_user_model.objects.create_user(
        username="second",
        password="Second-Secure-Passphrase-731!",
    )

    api_client.force_authenticate(first_user)
    prediction = api_client.post(
        "/api/predict/",
        valid_prediction_payload,
        format="json",
    )

    api_client.force_authenticate(second_user)
    detail = api_client.get(
        f"/api/predictions/{prediction.data['prediction_id']}/"
    )
    feedback = api_client.post(
        f"/api/predictions/{prediction.data['prediction_id']}/feedback/",
        {"helpful": False},
        format="json",
    )

    assert detail.status_code == 404
    assert feedback.status_code == 404
