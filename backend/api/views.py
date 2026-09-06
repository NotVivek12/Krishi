import logging

from django.db import connection
from django.shortcuts import get_object_or_404
from rest_framework import generics, status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Prediction, PredictionFeedback
from .permissions import IsPredictionOwner
from .serializers import (
    FeedbackSerializer,
    PredictionInputSerializer,
    PredictionSerializer,
)
from .services import ModelArtifactError, get_model_service


logger = logging.getLogger(__name__)


class HealthView(APIView):
    authentication_classes = []
    permission_classes = [AllowAny]
    throttle_classes = []

    def get(self, request):
        checks = {}

        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            checks["database"] = {"status": "ready"}
        except Exception:
            logger.exception("Database health check failed")
            checks["database"] = {"status": "unavailable"}

        checks["model"] = get_model_service().healthcheck()
        healthy = all(check["status"] == "ready" for check in checks.values())

        return Response(
            {
                "status": "healthy" if healthy else "unhealthy",
                "checks": checks,
            },
            status=status.HTTP_200_OK
            if healthy
            else status.HTTP_503_SERVICE_UNAVAILABLE,
        )


class ModelMetadataView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        try:
            metadata = get_model_service().metadata()
        except ModelArtifactError:
            logger.exception("Model metadata could not be loaded")
            return Response(
                {"detail": "Prediction model is unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        return Response(metadata)


class PredictView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        service = get_model_service()

        try:
            result = service.predict(
                serializer.model_payload(),
                top_k=serializer.validated_data["top_k"],
            )
        except ModelArtifactError:
            logger.exception("Prediction model is unavailable")
            return Response(
                {"detail": "Prediction model is unavailable."},
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        user = request.user if request.user.is_authenticated else None
        prediction = Prediction.objects.create(
            user=user,
            input_data={
                key: value
                for key, value in serializer.validated_data.items()
                if key != "top_k"
            },
            predicted_crop=result["predicted_crop"],
            confidence=result["confidence"],
            recommendations=result["recommendations"],
            model_version=result["model_version"],
        )

        return Response(
            {
                "prediction_id": prediction.id,
                **result,
                "created_at": prediction.created_at,
            },
            status=status.HTTP_200_OK,
        )


class PredictionListView(generics.ListAPIView):
    serializer_class = PredictionSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        if self.request.user.is_authenticated:
            return Prediction.objects.filter(user=self.request.user).select_related("feedback").order_by('-created_at')
        return Prediction.objects.all().select_related("feedback").order_by('-created_at')[:50]


class PredictionDetailView(generics.RetrieveAPIView):
    serializer_class = PredictionSerializer
    permission_classes = [AllowAny]
    lookup_url_kwarg = "prediction_id"

    def get_queryset(self):
        return Prediction.objects.all().select_related("feedback")


class PredictionFeedbackView(APIView):
    permission_classes = [AllowAny]

    def post(self, request, prediction_id):
        prediction = get_object_or_404(
            Prediction,
            id=prediction_id,
        )
        serializer = FeedbackSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        feedback, created = PredictionFeedback.objects.update_or_create(
            prediction=prediction,
            defaults={
                "user": request.user if request.user.is_authenticated else None,
                **serializer.validated_data,
            },
        )
        return Response(
            FeedbackSerializer(feedback).data,
            status=status.HTTP_201_CREATED
            if created
            else status.HTTP_200_OK,
        )
