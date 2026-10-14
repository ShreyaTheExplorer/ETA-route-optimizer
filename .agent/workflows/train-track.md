---
description: Step 3 - train RF and XGBoost, log in MLflow, pick the best model
---
1. Create models/train.py. Load features, sample cfg.training.sample_rows, split with the fixed seed.
2. Train a baseline first (predict the median duration, plus a linear regression on haversine_km) so we can show the models add value.
3. Train RandomForestRegressor and XGBRegressor (tree_method=hist). Try 3 small hyperparameter settings each.
4. For every run log to MLflow: params, train_MAE, val_MAE, val_RMSE, val_R2, training time, the model, and a feature-importance plot.
5. Query MLflow for the best run by val_MAE, save that model to cfg.paths.best_model, and write docs/model_comparison.md (table of all runs).
6. Create models/predict.py implementing the contract in AGENTS.md (reuses build_features and the saved zone model).
7. Create tests/test_predict.py: Times Square to JFK on a weekday at 8am should predict roughly 40-70 minutes; a 1 km hop should be under 15 minutes.
8. Show me the comparison table and overfitting gap (train vs validation).
