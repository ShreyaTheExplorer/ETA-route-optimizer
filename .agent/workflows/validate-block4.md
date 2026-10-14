---
description: Person A validates Block 4 (D's Streamlit UI)
---
Do NOT modify D's code. Report in docs/validation_block4.md with PASS/FAIL per check.
1. Run the API and UI. For one fixed request, save the raw JSON from the API and compare it field by field with what the UI displays: total minutes, per-stop ETA, order, improvement percentage.
2. Map: the numbered markers and route line follow exactly the returned route order; before-map uses baseline_route and after-map uses route.
3. Errors: an out-of-NYC stop, an empty stop list and a stopped API each show a friendly message and no crash.
4. Confirm the UI never imports models/ or optimizer/ and reads API_URL from the environment.
5. Check every chart in docs/report/ has axis labels, units (minutes) and a title, and that numbers in the report tables match the CSV files.
