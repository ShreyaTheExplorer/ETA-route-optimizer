---
trigger: glob
globs: api/**,Dockerfile*,docker-compose.yml
---
Block 3 (Person B). Follow docs/CONTRACT.md section 4 exactly for request/response JSON.
Load the ETA model ONCE at startup (FastAPI lifespan), not per request. Build the matrix with ONE vectorised predict_minutes call for all n*n pairs.
Call the optimizer through optimizer/registry.py only. Validate input with pydantic and return 422 on bad input; never return a 500 for user mistakes.
Docker: slim Python image, copy only code + models/artifacts, no raw data, no mlruns. API URL for the UI comes from an environment variable.
