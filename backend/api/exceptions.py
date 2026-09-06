from rest_framework.exceptions import APIException
from rest_framework.views import exception_handler


class ModelUnavailable(APIException):
    status_code = 503
    default_detail = "Prediction model is unavailable."
    default_code = "model_unavailable"


def api_exception_handler(exc, context):
    response = exception_handler(exc, context)
    if response is None:
        return None

    request = context.get("request")
    request_id = getattr(request, "request_id", None)
    if request_id and isinstance(response.data, dict):
        response.data["request_id"] = request_id
    return response
