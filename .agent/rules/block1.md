---
trigger: glob
globs: preprocessing/**,models/**,evaluation/**,configs/**
---
Block 1 (Person A). Read all paths and parameters from configs/config.yaml.
Training and inference share ONE feature function: preprocessing/features.py::build_features(df). Never duplicate feature logic.
Log every training run in MLflow (params, MAE, RMSE, train-vs-validation error, model).
Fit KMeans zones and any encoders on training rows only.
Keep the contract in docs/CONTRACT.md section 1: predict_minutes(...) returns minutes and is vectorised.
