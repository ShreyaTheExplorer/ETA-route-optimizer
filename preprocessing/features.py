"""
preprocessing/features.py
=========================
THE single source of feature logic for both training and inference.
All feature engineering funnels through:

    build_features(df, zone_model=None, fit_zones=False)

AGENTS.md rule 3: never duplicate this logic in notebooks or the API.

Features produced
-----------------
Geometry
  haversine_km      - great-circle distance (km)
  manhattan_km      - sum of |Δlat| + |Δlon| converted to km
  bearing           - compass bearing pickup → dropoff (degrees)

Time
  hour              - hour of day (0–23)
  weekday           - day of week (0=Mon … 6=Sun)
  month             - month (1–6)
  is_weekend        - 1 if Sat/Sun
  is_rush_hour      - 1 if weekday AND hour in cfg rush_hours
  hour_sin          - sin(2π·hour/24)  cyclic encoding
  hour_cos          - cos(2π·hour/24)  cyclic encoding

Zones (KMeans over pickup+dropoff coordinates)
  pickup_zone       - cluster id (0 … n_zones-1)
  dropoff_zone      - cluster id (0 … n_zones-1)

Weather (joined on floor(pickup_datetime, hour))
  temperature_2m    - °C
  precipitation     - mm
  wind_speed_10m    - km/h
"""

from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans

from preprocessing.weather import get_weather


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: pd.Series, lon1: pd.Series,
                  lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Vectorised haversine distance in km."""
    R = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _manhattan_km(lat1: pd.Series, lon1: pd.Series,
                  lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Approximate Manhattan distance in km.
    Converts degree differences to km using local scale factors:
      1° latitude  ≈ 111.0 km
      1° longitude ≈ 111.0 * cos(mean_lat) km
    """
    mean_lat = np.radians((lat1 + lat2) / 2)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(mean_lat)
    return np.abs(lat2 - lat1) * km_per_deg_lat + np.abs(lon2 - lon1) * km_per_deg_lon


def _bearing(lat1: pd.Series, lon1: pd.Series,
             lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Compass bearing from pickup to dropoff (0–360°)."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dlambda = np.radians(lon2 - lon1)
    x = np.sin(dlambda) * np.cos(phi2)
    y = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlambda)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# Config loader (cached so callers need not pass cfg every time)
# ---------------------------------------------------------------------------

_cfg_cache: dict | None = None

def _load_cfg() -> dict:
    global _cfg_cache
    if _cfg_cache is None:
        project_root = pathlib.Path(__file__).parent.parent
        with open(project_root / "configs" / "config.yaml") as fh:
            _cfg_cache = yaml.safe_load(fh)
    return _cfg_cache


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    zone_model: KMeans | None = None,
    fit_zones: bool = False,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, KMeans]:
    """
    Build all features from a cleaned trips DataFrame.

    Parameters
    ----------
    df         : cleaned DataFrame (output of clean_trips or trips_clean.parquet)
    zone_model : pre-fitted KMeans for zone assignment; required unless fit_zones=True
    fit_zones  : if True, fit a fresh KMeans on this df's coordinates
    cfg        : parsed config.yaml; loaded automatically if None

    Returns
    -------
    (feature_df, zone_model)
        feature_df  — original columns + all engineered features
        zone_model  — the (possibly newly fitted) KMeans
    """
    if cfg is None:
        cfg = _load_cfg()

    feat_cfg = cfg["features"]
    rush_hours = set(feat_cfg["rush_hours"])
    n_zones = feat_cfg["n_zones"]

    df = df.copy()

    # Ensure pickup_datetime is datetime64
    if not pd.api.types.is_datetime64_any_dtype(df["pickup_datetime"]):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    # ------------------------------------------------------------------
    # 1. Geometry features
    # ------------------------------------------------------------------
    df["haversine_km"] = _haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["manhattan_km"] = _manhattan_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    df["bearing"] = _bearing(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )

    # ------------------------------------------------------------------
    # 2. Time features
    # ------------------------------------------------------------------
    dt = df["pickup_datetime"]
    df["hour"]       = dt.dt.hour.astype(np.int8)
    df["weekday"]    = dt.dt.dayofweek.astype(np.int8)   # 0=Mon, 6=Sun
    df["month"]      = dt.dt.month.astype(np.int8)
    df["is_weekend"] = (df["weekday"] >= 5).astype(np.int8)
    df["is_rush_hour"] = (
        (df["weekday"] < 5) & (df["hour"].isin(rush_hours))
    ).astype(np.int8)

    # Cyclic hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # ------------------------------------------------------------------
    # 3. Zone assignment (KMeans over pickup + dropoff coordinates)
    # ------------------------------------------------------------------
    coords = df[["pickup_latitude", "pickup_longitude",
                 "dropoff_latitude", "dropoff_longitude"]].values

    if fit_zones:
        zone_model = KMeans(n_clusters=n_zones, random_state=42, n_init="auto")
        zone_model.fit(coords)
    elif zone_model is None:
        raise ValueError("Pass a fitted zone_model or set fit_zones=True.")

    # Assign zones separately for pickup and dropoff
    pickup_coords  = df[["pickup_latitude",  "pickup_longitude"]].values
    dropoff_coords = df[["dropoff_latitude", "dropoff_longitude"]].values

    # Use a 2-column sub-model: predict cluster using only first 2 dimensions
    # by building a 2-centroid mapping from the full 4-D model
    # Actually, simpler: refit two 2D KMeans or just predict from full-model centroids.
    # Per AGENTS.md simplicity rule: fit ONE 4-D model on (pu_lat,pu_lon,do_lat,do_lon)
    # then assign zone by finding the nearest centroid for each 2D point independently.
    pu_centroids = zone_model.cluster_centers_[:, :2]  # (n_zones, 2)
    do_centroids = zone_model.cluster_centers_[:, 2:]  # (n_zones, 2)

    def _assign_zone(points: np.ndarray, centroids: np.ndarray, chunk_size: int = 100_000) -> np.ndarray:
        """Nearest centroid (L2) for each row in points using chunked cdist."""
        from scipy.spatial.distance import cdist
        labels = np.empty(len(points), dtype=np.int32)
        for i in range(0, len(points), chunk_size):
            chunk = points[i : i + chunk_size]
            labels[i : i + chunk_size] = cdist(chunk, centroids).argmin(axis=1)
        return labels

    df["pickup_zone"]  = _assign_zone(pickup_coords,  pu_centroids)
    df["dropoff_zone"] = _assign_zone(dropoff_coords, do_centroids)

    # ------------------------------------------------------------------
    # 4. Weather features
    # ------------------------------------------------------------------
    weather = get_weather(df["pickup_datetime"], cfg)
    for col in ["temperature_2m", "precipitation", "wind_speed_10m"]:
        df[col] = weather[col].values

    return df, zone_model


# ---------------------------------------------------------------------------
# Feature column list (must match training order for inference)
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "haversine_km",
    "manhattan_km",
    "bearing",
    "hour",
    "weekday",
    "month",
    "is_weekend",
    "is_rush_hour",
    "hour_sin",
    "hour_cos",
    "pickup_zone",
    "dropoff_zone",
    "temperature_2m",
    "precipitation",
    "wind_speed_10m",
]


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pathlib
    import yaml

    project_root = pathlib.Path(__file__).parent.parent
    with open(project_root / "configs" / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    clean_path   = project_root / cfg["paths"]["clean_trips"]
    feat_path    = project_root / cfg["paths"]["features"]
    zone_path    = project_root / cfg["paths"]["zone_model"]
    feat_list_path = project_root / cfg["paths"]["feature_list"]

    print(f"[features] Loading cleaned trips from {clean_path} ...")
    df = pd.read_parquet(clean_path)
    print(f"[features] {len(df):,} rows loaded.")

    print("[features] Engineering features (fit_zones=True) ...")
    df_feat, zone_model = build_features(df, fit_zones=True, cfg=cfg)

    # Save zone KMeans
    zone_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(zone_model, zone_path)
    print(f"[features] Zone KMeans saved -> {zone_path}")

    # Save feature column list
    feat_list_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feat_list_path, "w") as fh:
        json.dump(FEATURE_COLUMNS, fh, indent=2)
    print(f"[features] Feature column list saved -> {feat_list_path}")

    # Save features parquet
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_parquet(feat_path, index=False)
    print(f"[features] Feature table saved -> {feat_path}")

    # ------------------------------------------------------------------
    # Print describe + correlation report
    # ------------------------------------------------------------------
    target = cfg["training"]["target"]
    all_cols = FEATURE_COLUMNS + [target]
    available = [c for c in all_cols if c in df_feat.columns]

    print("\n" + "=" * 65)
    print("  FEATURE DESCRIBE")
    print("=" * 65)
    print(df_feat[FEATURE_COLUMNS].describe().round(3).to_string())

    print("\n" + "=" * 65)
    print(f"  CORRELATION WITH {target}")
    print("=" * 65)
    corr = df_feat[available].corr()[target].drop(target).sort_values(
        key=abs, ascending=False
    )
    for feat, r in corr.items():
        bar = "#" * int(abs(r) * 40)
        sign = "+" if r >= 0 else "-"
        print(f"  {feat:<22} {sign}{abs(r):.4f}  {bar}")
    print("=" * 65)
    print("\n[features] Done.")

