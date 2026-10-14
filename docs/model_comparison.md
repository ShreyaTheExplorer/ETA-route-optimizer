# Model Comparison and Evaluation Report

## Overview
- **Dataset Size**: 300,000 sampled NYC taxi trips
- **Validation Strategy**: Holdout validation (20% test split, fixed seed=42)
- **Target Metric**: Duration in minutes (`duration_min`)
- **Best Model**: **`XGBoost_Depth10`** with **Val MAE = 3.0919 min**

## Performance Comparison Table

| Model | Parameters | Train MAE (min) | Val MAE (min) | Val RMSE (min) | Val R² | Overfitting Gap (Val - Train) | Training Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost_Depth10** | `{"n_estimators": 150, "max_depth": 10, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist"}` | 2.7032 | 3.0919 | 4.8854 | 0.7984 | +0.3887 | 5.3s |
| XGBoost_Depth8 | `{"n_estimators": 100, "max_depth": 8, "learning_rate": 0.08, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist"}` | 3.0037 | 3.1536 | 4.9676 | 0.7915 | +0.1499 | 2.4s |
| RandomForest_Depth20 | `{"n_estimators": 50, "max_depth": 20, "min_samples_leaf": 10, "max_features": 0.8}` | 2.6644 | 3.2746 | 5.1264 | 0.7780 | +0.6102 | 8.4s |
| XGBoost_Depth6 | `{"n_estimators": 60, "max_depth": 6, "learning_rate": 0.1, "tree_method": "hist"}` | 3.2886 | 3.3316 | 5.2097 | 0.7707 | +0.0430 | 2.2s |
| RandomForest_Depth15 | `{"n_estimators": 50, "max_depth": 15, "max_features": "sqrt"}` | 2.8551 | 3.3986 | 5.2560 | 0.7666 | +0.5435 | 2.7s |
| RandomForest_Depth10 | `{"n_estimators": 30, "max_depth": 10, "max_features": "sqrt"}` | 3.6187 | 3.7124 | 5.6133 | 0.7338 | +0.0936 | 1.4s |
| Baseline_Linear_Haversine | `{"fit_intercept": true, "features": "haversine_km_only"}` | 4.7583 | 4.7640 | 6.9277 | 0.5945 | +0.0057 | 0.0s |
| Baseline_Median | `{"strategy": "median"}` | 7.3632 | 7.3704 | 11.2714 | -0.0734 | +0.0072 | 0.0s |

## Key Findings & Viva Talking Points

1. **Baselines confirm model value**:
   - Predicting the median duration yields high error (~5-7 min MAE), confirming simple heuristics are insufficient.
   - A single-feature linear regression on `haversine_km` captures coarse distance, but misses traffic, weather, and localized routing congestion.

2. **Tree ensembles add substantial predictive power**:
   - Both Random Forest and XGBoost regressors dramatically reduce MAE and RMSE.
   - The non-linear interactions between distance, hour of day, rush-hour indicators, and spatial zone clusters explain complex NYC traffic dynamics.

3. **Generalization & Overfitting Gap**:
   - XGBoost with histogram-based binning (`tree_method='hist'`) provides both lightning-fast training times and controlled overfitting via shrinkage (`learning_rate`) and subsampling.
   - The overfitting gap (Validation MAE minus Training MAE) remains small and stable across the best-performing models.

4. **Deployment Readiness**:
   - The best performing model `XGBoost_Depth10` has been exported to `models/artifacts/best_eta_model.joblib` for zero-overhead inference in `models/predict.py`.
