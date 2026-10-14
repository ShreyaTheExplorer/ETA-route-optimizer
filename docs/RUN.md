# Running Block 1: ETA Predictor Pipeline

This document contains the exact commands, in order, to set up the environment, run the entire data and model pipeline from scratch, execute tests, and launch the MLflow UI.

---

## 1. Environment Setup & Requirements Installation

Create and activate a virtual environment, then install all project dependencies from `requirements.txt`:

### On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 2. Download Data

### A. NYC Taxi Trips Data
Download `train.csv` from Kaggle's [NYC Taxi Trip Duration Competition](https://www.kaggle.com/c/nyc-taxi-trip-duration/data) and place it at `data/raw/train.csv`:
```bash
# Using Kaggle CLI:
kaggle competitions download -c nyc-taxi-trip-duration -f train.zip -p data/raw/
tar -xf data/raw/train.zip -C data/raw/
```
*(Verify that `data/raw/train.csv` exists before proceeding to cleaning).*

### B. Hourly NYC Weather Data
Downloads hourly NYC weather (temperature, precipitation, wind speed) from the Open-Meteo Archive API and stores it in `data/raw/weather_nyc_2016.csv` and `models/artifacts/`:
```bash
python -m preprocessing.weather
```

---

## 3. Clean the Trips Data

Applies the bounding box, duration filtering, speed plausibility checks, and passenger count constraints configured in `configs/config.yaml`:
```bash
python -m preprocessing.clean
```
- **Outputs**:
  - `data/processed/trips_clean.parquet` (cleaned dataset)
  - `docs/cleaning_report.md` (audit report detailing dropped rows per rule)

---

## 4. Feature Engineering

Computes haversine distance, Manhattan distance, bearing, temporal cyclical encodings (`hour_sin`, `hour_cos`), rush-hour flags, spatial KMeans coordinate clusters (zones), and joins weather data:
```bash
python -m preprocessing.features
```
- **Outputs**:
  - `data/processed/features.parquet` (full engineered feature table)
  - `models/artifacts/zone_kmeans.joblib` (spatial clustering model)
  - `models/artifacts/feature_columns.json` (canonical feature order)
  - Prints summary statistics (`describe()`) and correlation ranking with `duration_min`.

---

## 5. Model Training & MLflow Tracking

Trains baselines (median duration, distance-only linear regression), 3 Random Forest Regressor configurations, and 3 XGBoost Regressor configurations. Logs metrics, hyperparameters, feature importance plots, and models to MLflow:
```bash
python -m models.train
```
- **Outputs**:
  - `models/artifacts/best_eta_model.joblib` (best model by validation MAE)
  - `docs/model_comparison.md` (markdown comparison table and overfitting gap analysis)
  - `mlflow.db` (SQLite tracking database)

---

## 6. Run Tests

Executes the complete test suite (39 tests) covering data cleaning invariants, feature transformations, prediction latency (<50 ms for 10,000 pairs), and interface contract compliance:
```bash
pytest tests/
```

To run with verbose output and benchmark timing:
```bash
pytest tests/ -v -s
```

---

## 7. Start MLflow UI

Launch the MLflow tracking dashboard to view runs, parameters, metrics, and feature importance artifacts:

```bash
# With .venv activated:
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Or directly using the virtualenv executable:
# Windows:
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db
# Linux / macOS:
./.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000) to inspect experiment runs under `eta-prediction`.
