# Running Block 1: ETA Predictor Pipeline

This document describes how to set up the environment, run the Block 1 pipeline, and execute tests.

## 1. Setup Environment
```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

## 2. Execute Data Cleaning
Drops invalid bounding box coordinates, zero-distance trips, trips > 2 hours, passenger anomalies, and impossible speeds (> 100 km/h).
```bash
python -m preprocessing.clean
```
Outputs: `data/processed/trips_clean.parquet` and `docs/cleaning_report.md`.

## 3. Ingest Weather & Build Features
Fetches hourly NYC weather from Open-Meteo and builds spatial, temporal, zone, and weather features.
```bash
python -m preprocessing.weather
python -m preprocessing.features
```
Outputs:
- `data/processed/features.parquet`
- `models/artifacts/zone_kmeans.joblib`
- `models/artifacts/feature_columns.json`
- `models/artifacts/weather_hourly.parquet`
- `models/artifacts/weather_medians.json`

## 4. Train Models & Track with MLflow
Trains baselines, Random Forest models, and XGBoost regressors, logging all metrics, artifacts, and parameters to SQLite MLflow.
```bash
python -m models.train
```
Outputs:
- `models/artifacts/best_eta_model.joblib`
- `docs/model_comparison.md`
- MLflow tracking database in `sqlite:///mlflow.db`

## 5. Run Validation & Unit Tests
```bash
pytest tests/
```
Runs 39 unit tests covering data cleaning rules, feature transformations, prediction latency benchmark (10,000 pairs), and the `models.predict.predict_minutes` interface contract.
