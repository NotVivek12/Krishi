# Krishi Backend API

Base URL for local development: `http://127.0.0.1:8000`

All request and response bodies use JSON. Authenticated endpoints require:

```text
Authorization: Bearer <access-token>
```

Every response includes an `X-Request-ID` header for tracing.

## Endpoints

| Method | Path | Authentication | Purpose |
| --- | --- | --- | --- |
| GET | `/health/` | Public | Check database and model readiness |
| GET | `/api/health/` | Public | Namespaced health-check alias |
| GET | `/api/model/metadata/` | Public | Get feature ranges and categories |
| POST | `/api/predict/` | Public or optional JWT | Recommend crops and persist the result |
| GET | `/api/predictions/` | JWT | List the current user's predictions |
| GET | `/api/predictions/{id}/` | JWT | Get one owned prediction |
| POST | `/api/predictions/{id}/feedback/` | JWT | Create or update prediction feedback |
| POST | `/api/auth/register/` | Public | Register and receive JWT tokens |
| POST | `/api/auth/token/` | Public | Log in and receive JWT tokens |
| POST | `/api/auth/token/refresh/` | Public | Rotate an access token |
| GET/PATCH | `/api/auth/me/` | JWT | Read or update the current profile |
| POST | `/api/auth/logout/` | JWT | Blacklist a refresh token |

## Prediction

`POST /api/predict/`

```json
{
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
  "top_k": 3
}
```

The metadata endpoint is the canonical source for allowed category values and
numeric ranges. Category matching is case-insensitive and normalizes repeated
whitespace. The original training names (`SOIL`, `TEMP`, `N`, and so on) are
also accepted for compatibility.

Successful response:

```json
{
  "prediction_id": "5be4a7e0-b6ca-45bc-96f9-693185be634c",
  "predicted_crop": "Rice",
  "confidence": 0.93,
  "recommendations": [
    {"rank": 1, "crop": "Rice", "confidence": 0.93},
    {"rank": 2, "crop": "Maize", "confidence": 0.04},
    {"rank": 3, "crop": "Cotton", "confidence": 0.01}
  ],
  "model_version": "rf-12-character-hash",
  "created_at": "2026-07-21T00:00:00Z"
}
```

Invalid categories, out-of-range numbers, missing fields, conflicting aliases,
and unknown fields return HTTP 400 with field-level errors. A missing or corrupt
model returns HTTP 503.

## Authentication

Register with a unique username and email:

```json
{
  "username": "farmer",
  "email": "farmer@example.com",
  "password": "a-strong-password",
  "first_name": "Asha",
  "last_name": "Patel"
}
```

Registration and login return `access` and `refresh` tokens. Access tokens
expire after 30 minutes by default. Refresh tokens expire after seven days and
rotate when used. Prediction history and feedback are isolated by user; another
user receives HTTP 404 for an unowned prediction.

## Pagination and throttling

Prediction history is paginated with 20 records per page. Default throttles are
30 requests per minute for anonymous clients and 120 per minute for
authenticated users. Both values are configurable through environment
variables.
