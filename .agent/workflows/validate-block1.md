---
description: Person D validates Block 1 (A's data, features, ETA models)
---
Do NOT modify A's code. Report findings in docs/validation_block1.md with PASS/FAIL per check.
1. Run pytest for tests/test_clean.py, test_features.py, test_predict.py.
2. Re-run the pipeline from the documented commands in docs/RUN.md on a clean checkout.
3. Cleaning: confirm the cleaning report row counts add up (before - dropped = after) and every rule in configs/config.yaml is actually applied in code.
4. Leakage: confirm the train/validation split happens BEFORE KMeans zones are fitted and that zones are fitted on training rows only.
5. Features: hand-check one row. Haversine of Times Square to JFK ~ 21 km; a Tuesday 08:00 timestamp has is_rush_hour = 1 and a Sunday 03:00 timestamp has 0.
6. Metrics: confirm docs/model_comparison.md matches the MLflow runs (same MAE/RMSE), and the best saved model is the one with the lowest validation MAE.
7. Overfitting: list train MAE vs validation MAE for every model and flag gaps over 1 minute.
8. Sanity: with the same pair, predicted time at 08:00 weekday > 03:00 weekday; a 5 km trip > a 1 km trip; heavy rain >= no rain (if weather is used).
9. Batch speed: predict_minutes on 10,000 pairs must be vectorised; report the time.
