---
trigger: glob
globs: frontend/**,docs/report/**
---
Block 4 (Person D). The UI talks to the backend ONLY through HTTP (requests to API_URL, default http://localhost:8000). Never import models/ or optimizer/.
Show exactly the numbers the API returns; do not recompute or round differently. The map order must equal the returned route order.
Before/after view: baseline_route vs route, with total_minutes and improvement_pct from the API response.
Save every chart/table for the report into docs/report/ as PNG/CSV with a clear file name.
