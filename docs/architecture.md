# Backend Architecture

The backend is a Django REST application with four main boundaries:

```text
HTTP request
  -> validation and authentication
  -> model service
  -> Random Forest model + encoders
  -> prediction persistence and response
```

`config/` owns environment-driven Django settings, routing, ASGI, and WSGI.
`authentication/` owns registration, JWT login/rotation/logout, and profile
access. `api/` owns request validation, prediction history, feedback, health,
and model metadata. `api/services/model_service.py` is the only component that
loads or calls the machine-learning artifacts.

The model service loads artifacts lazily on the first health, metadata, or
prediction request. It verifies the model's 12-feature order and required label
encoders before declaring readiness. Category input is normalized and mapped
back to the exact training vocabulary. Predictions are serialized through a
lock because the persisted scikit-learn estimator is shared across request
threads.

Anonymous and authenticated callers can request predictions. Authenticated
predictions are attached to the caller and become visible in that user's
history. Object queries are user-scoped, preventing prediction IDs from being
used to access another account's data.

SQLite is the development and single-instance deployment database. A
multi-instance deployment should use PostgreSQL and shared migrations. The
model artifact is immutable application data and should be released together
with its matching encoders.
