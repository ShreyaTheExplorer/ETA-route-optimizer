# Project: Delivery ETA Predictor + Route Optimizer (AI/ML Lab mini project)

Team of 4. Person A owns **Block 1**: data cleaning, feature engineering, ETA models, MLflow.
Validated by Person D. Do NOT edit optimizer/, api/, frontend/ (owned by B, C, D).

## Goal of Block 1
Train Random Forest and XGBoost regressors that predict trip duration (minutes) between two
lat/lon points at a given time. Save the best model so Person B can load it and predict an
all-pairs leg-time matrix for the API.

## Hard rules
1. Build in small steps. After each step, run it and show me the output before moving on.
2. Read all paths and parameters from `configs/config.yaml`. No hardcoded paths.
3. Training and inference MUST share ONE feature function: `preprocessing/features.py::build_features(df)`.
   Never duplicate feature logic in a notebook or the API (prevents train/serve skew).
4. Every training run is logged in MLflow: params, MAE, RMSE, train-vs-validation error, model.
5. Use a time-aware or random split with a fixed seed; never fit scalers/encoders/KMeans on the test set.
6. Keep code simple and commented so I can explain every line in the viva.
7. Use only packages in requirements.txt. Ask before adding any.
8. Write a small pytest for each module in tests/.
9. Stay on branch `feature/block1-eta` and commit after every working step.

## Interface contract (Person B depends on this, do not change without asking)
- `models/predict.py`:
  `predict_minutes(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, departure: datetime) -> np.ndarray`
  accepts scalars or arrays, returns predicted duration in MINUTES.
- Saved artifacts live in `models/artifacts/` (best model, zone KMeans, feature column list).
- Weather at inference: look up hourly weather if the date is in the dataset range, otherwise
  fall back to hour-of-day/month median (see `preprocessing/weather.py`).

## Data notes
- Trips: Kaggle NYC Taxi Trip Duration (train.csv, Jan-Jun 2016): id, vendor_id, pickup_datetime,
  dropoff_datetime, passenger_count, pickup/dropoff longitude/latitude, trip_duration (SECONDS).
- Target = duration in MINUTES = trip_duration / 60.
- Weather: hourly temperature, precipitation, wind, for NYC, same period, joined on the pickup hour.
