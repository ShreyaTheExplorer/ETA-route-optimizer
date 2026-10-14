---
description: Person D - collect results, tables and charts for the report
---
1. Create frontend/report_assets.py (or notebooks/report_assets.ipynb) that collects into docs/report/: model comparison table (from MLflow or docs/model_comparison.md), the local-optima histogram and table (from C), and a before/after table over 20 sampled problems (call the API with fixed seeds): n stops, baseline minutes, optimized minutes, improvement_pct, mean and median improvement.
2. Save every figure at 200 dpi with a descriptive file name and a CSV beside it.
3. Draft a short results paragraph for each table in plain English with the actual numbers.
