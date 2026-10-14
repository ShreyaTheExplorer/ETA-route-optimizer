---
description: Person C validates Block 3 (B's API and Docker)
---
Do NOT modify B's code. Report in docs/validation_block3.md with PASS/FAIL per check.
1. Run tests/test_api.py. Start the API and send requests with curl or httpx.
2. Edge cases: 1 stop, 2 stops, 30 stops, repeated coordinates, duplicate ids, empty stops, lat/lon swapped, outside NYC, bad time format, unknown algorithm.
3. Consistency on a normal response: total_minutes equals the sum of leg_minutes; eta_minutes is the running sum; arrival_time = departure_time + eta_minutes; route contains every stop exactly once; improvement_pct = (baseline - total) / baseline * 100.
4. Matrix: confirm the API builds the time matrix with ONE vectorised predict_minutes call (not a loop of n*n calls) and that the model is loaded once at startup. Time a 20-stop request.
5. Docker: from a clean state (docker compose down -v; docker compose build --no-cache; docker compose up) confirm /health, /optimize and the UI all work.
6. Confirm every runtime artifact (model, KMeans, feature list, weather table or fallback) is inside the image.
