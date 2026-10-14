"""
models/predict.py
=================
Interface contract for Block 1 (consumed by Person B in optimizer/ and api/).

Exposes:
    predict_minutes(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, departure) -> np.ndarray

Guarantees:
- Accepts scalars or numpy arrays / sequences.
- Returns predicted duration in MINUTES as a 1D numpy array.
- Uses the EXACT same feature pipeline (preprocessing/features.py::build_features)
  as training to prevent train/serve skew (AGENTS.md rule 3).
- Uses saved artifacts from models/artifacts/ (best model, zone KMeans, feature list).
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime
from typing import Sequence, Union

import joblib
import numpy as np
import pandas as pd
import yaml

from preprocessing.features import build_features

# ---------------------------------------------------------------------------
# Artifact cache for high-throughput inference
# ---------------------------------------------------------------------------

_cfg_cache: dict | None = None
_model_cache: object | None = None
_zone_model_cache: object | None = None
_feature_cols_cache: list[str] | None = None


def _get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent.parent


def _load_artifacts() -> tuple[dict, object, object, list[str]]:
    global _cfg_cache, _model_cache, _zone_model_cache, _feature_cols_cache

    if _model_cache is None:
        project_root = _get_project_root()
        config_path = project_root / "configs" / "config.yaml"

        with open(config_path) as fh:
            _cfg_cache = yaml.safe_load(fh)

        best_model_path = project_root / _cfg_cache["paths"]["best_model"]
        zone_model_path = project_root / _cfg_cache["paths"]["zone_model"]
        feat_list_path = project_root / _cfg_cache["paths"]["feature_list"]

        if not best_model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {best_model_path}. Run models/train.py first."
            )
        if not zone_model_path.exists():
            raise FileNotFoundError(
                f"Zone KMeans model not found at {zone_model_path}. Run preprocessing/features.py first."
            )
        if not feat_list_path.exists():
            raise FileNotFoundError(
                f"Feature columns list not found at {feat_list_path}. Run preprocessing/features.py first."
            )

        _model_cache = joblib.load(best_model_path)
        _zone_model_cache = joblib.load(zone_model_path)
        with open(feat_list_path) as fh:
            _feature_cols_cache = json.load(fh)

    return _cfg_cache, _model_cache, _zone_model_cache, _feature_cols_cache


# ---------------------------------------------------------------------------
# Public prediction API
# ---------------------------------------------------------------------------

def predict_minutes(
    pickup_lat: Union[float, Sequence[float], np.ndarray],
    pickup_lon: Union[float, Sequence[float], np.ndarray],
    dropoff_lat: Union[float, Sequence[float], np.ndarray],
    dropoff_lon: Union[float, Sequence[float], np.ndarray],
    departure: Union[datetime, str, Sequence[Union[datetime, str]], np.ndarray],
) -> np.ndarray:
    """
    Predict trip duration in MINUTES between pickup and dropoff points at departure time.

    Parameters
    ----------
    pickup_lat  : float or 1D array of pickup latitudes
    pickup_lon  : float or 1D array of pickup longitudes
    dropoff_lat : float or 1D array of dropoff latitudes
    dropoff_lon : float or 1D array of dropoff longitudes
    departure   : datetime, ISO-formatted string, or sequence/array of them

    Returns
    -------
    np.ndarray of predicted duration in MINUTES (shape: (N,))
    """
    cfg, model, zone_model, feature_cols = _load_artifacts()

    # Normalize scalars to 1D arrays
    p_lat = np.atleast_1d(pickup_lat).astype(float)
    p_lon = np.atleast_1d(pickup_lon).astype(float)
    d_lat = np.atleast_1d(dropoff_lat).astype(float)
    d_lon = np.atleast_1d(dropoff_lon).astype(float)

    if isinstance(departure, (datetime, str, pd.Timestamp)):
        dep_arr = np.array([departure] * len(p_lat))
    else:
        dep_arr = np.atleast_1d(departure)
        if len(dep_arr) == 1 and len(p_lat) > 1:
            dep_arr = np.repeat(dep_arr, len(p_lat))

    if not (len(p_lat) == len(p_lon) == len(d_lat) == len(d_lon) == len(dep_arr)):
        raise ValueError(
            f"Coordinate and departure array lengths must match. "
            f"Got shapes: pickup_lat={len(p_lat)}, pickup_lon={len(p_lon)}, "
            f"dropoff_lat={len(d_lat)}, dropoff_lon={len(d_lon)}, departure={len(dep_arr)}"
        )

    # Construct input dataframe
    input_df = pd.DataFrame({
        "pickup_latitude": p_lat,
        "pickup_longitude": p_lon,
        "dropoff_latitude": d_lat,
        "dropoff_longitude": d_lon,
        "pickup_datetime": pd.to_datetime(dep_arr),
    })

    # Build features using shared single source of truth
    feat_df, _ = build_features(
        input_df,
        zone_model=zone_model,
        fit_zones=False,
        cfg=cfg,
    )

    X = feat_df[feature_cols]

    # Model inference
    preds = model.predict(X)

    # Durations cannot be negative
    preds = np.maximum(preds, 0.5)

    return np.asarray(preds, dtype=float)
