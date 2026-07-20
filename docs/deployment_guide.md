# Backend Deployment Guide

## Local development

Requirements:

- Python 3.11
- The committed model files under `ml/models/`

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/manage.py migrate
python backend/manage.py createsuperuser
python backend/manage.py runserver
```

Run `pytest` before deploying. `python backend/manage.py check --deploy` can be
used with production environment variables to inspect Django's security
configuration.

## Environment variables

| Variable | Required | Default |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | Production | Development-only value |
| `DJANGO_DEBUG` | No | `1` |
| `DJANGO_ALLOWED_HOSTS` | Production | Local hosts |
| `CORS_ALLOWED_ORIGINS` | No | Local ports 3000 and 5173 |
| `DATABASE_PATH` | No | `backend/db.sqlite3` |
| `KRISHI_MODEL_PATH` | No | Committed Random Forest artifact |
| `KRISHI_ENCODERS_PATH` | No | Committed encoder artifact |
| `DJANGO_TIME_ZONE` | No | `Asia/Kolkata` |
| `API_ANON_RATE` | No | `30/min` |
| `API_USER_RATE` | No | `120/min` |
| `JWT_ACCESS_MINUTES` | No | `30` |
| `JWT_REFRESH_DAYS` | No | `7` |

When `DJANGO_DEBUG=0`, a secret key is mandatory and secure cookies, HTTPS
redirection, and HSTS are enabled. Deploy behind a TLS-terminating reverse proxy
that sets `X-Forwarded-Proto`.

## Docker Compose

Create `.env` from `.env.example`, replace `DJANGO_SECRET_KEY`, then run:

```powershell
docker compose up --build -d
docker compose ps
docker compose logs -f api
```

Compose runs migrations before Gunicorn starts, persists SQLite data in a named
volume, and checks `/health/`. The first health check can take several seconds
while the model loads.

For multi-instance production deployments, replace SQLite with PostgreSQL
before scaling the API horizontally. Keep model artifacts immutable and deploy
the model and encoder files as one versioned pair.

## Release checklist

1. Run the full test suite.
2. Run migrations against a backup-tested database.
3. Set a unique secret and exact allowed hosts/CORS origins.
4. Verify `/health/` reports both database and model as ready.
5. Send a known prediction payload and confirm the model version.
6. Monitor application logs using the response `X-Request-ID`.
