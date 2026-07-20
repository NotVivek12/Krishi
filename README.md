# Krishi

Krishi is a crop recommendation research system backed by centralized and
federated machine-learning experiments. The production-facing backend serves
the committed Random Forest model through a validated Django REST API.

The model was trained on controlled synthetic data. Its recommendations should
support, not replace, advice from local agricultural experts.

## Backend quick start

Create and activate a Python 3.11 virtual environment, then run:

```powershell
pip install -r backend/requirements.txt
python backend/manage.py migrate
python backend/manage.py runserver
```

The API is available at `http://127.0.0.1:8000`. Verify both the database and
model with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/
```

Run the complete backend test suite from the repository root:

```powershell
pytest
```

For container deployment, copy `.env.example` to `.env`, replace the secret,
and run:

```powershell
docker compose up --build
```

See `docs/api_documentation.md` for endpoint contracts and
`docs/deployment_guide.md` for production configuration.
