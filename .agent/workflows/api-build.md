---
description: Person B - build matrix builder and FastAPI /optimize endpoint
---
1. Read AGENTS.md and docs/CONTRACT.md.
2. Create api/schemas.py: pydantic models for the request and response in CONTRACT section 4 (NYC bounding box check on lat/lon, at least 1 stop, unique stop ids).
3. Create api/matrix_builder.py: build_time_matrix(points, departure) -> (n, n) numpy array. Build all n*n (i, j) pairs as arrays, call models.predict.predict_minutes ONCE, reshape, set the diagonal to 0.
   If models.predict is not available yet, use a stub (haversine / 30 km/h) behind the same function name so work is not blocked.
4. Create api/main.py with POST /optimize and GET /health. Load the model at startup. Call the optimizer via optimizer.registry.SOLVERS[algorithm].
   Compute: route, per-stop leg_minutes, cumulative eta_minutes, arrival_time = departure + eta, total_minutes, and a seeded random baseline route for before/after.
5. Write tests/test_api.py using FastAPI TestClient: normal request, 1 stop, 2 stops, repeated coordinates, duplicate ids (422), out-of-NYC (422), bad time (422), unknown algorithm (422).
6. Run uvicorn api.main:app --reload, send a sample request with curl, and show me the JSON and test results.
