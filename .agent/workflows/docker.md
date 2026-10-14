---
description: Person B - Dockerfile and docker-compose for API and UI
---
1. Create Dockerfile.api (python:3.11-slim, install requirements, copy api/, optimizer/, preprocessing/, models/, configs/; run uvicorn on 0.0.0.0:8000).
2. Create Dockerfile.ui (copy frontend/, install streamlit deps; run streamlit on 8501; read API_URL from environment).
3. Create docker-compose.yml with services api and ui, ui depends_on api, API_URL=http://api:8000, ports 8000 and 8501, a healthcheck on GET /health.
4. Create .dockerignore (data/, mlruns/, mlflow.db, .venv/, __pycache__).
5. Check that every file predict_minutes needs at runtime (model, zone KMeans, feature list, weather table or fallback medians) is inside the image. List anything missing and tell me.
6. Run docker compose build and docker compose up from a clean state; show me logs and a successful /health and /optimize call.
