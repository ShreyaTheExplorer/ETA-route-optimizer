"""
tests/test_features.py
======================
Unit tests for preprocessing/features.py.

Tests cover:
  - haversine distance for a known coordinate pair
  - manhattan distance for a known coordinate pair
  - bearing for a known coordinate pair
  - is_rush_hour flag logic (weekday peak, weekday off-peak, weekend)
  - is_weekend flag
  - hour_sin / hour_cos cyclic encoding
  - zone assignment returns valid cluster ids
  - FEATURE_COLUMNS list completeness
"""

import sys
import pathlib
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# We need to import without actually calling get_weather during import
# so we patch it before importing features
import preprocessing.features as feat_module
from preprocessing.features import (
    _haversine_km,
    _manhattan_km,
    _bearing,
    build_features,
    FEATURE_COLUMNS,
)


# ---------------------------------------------------------------------------
# Minimal config matching configs/config.yaml cleaning section
# ---------------------------------------------------------------------------
CFG = {
    "paths": {
        "raw_trips":   "data/raw/train.csv",
        "raw_weather": "data/raw/weather_nyc_2016.csv",
        "clean_trips": "data/processed/trips_clean.parquet",
        "features":    "data/processed/features.parquet",
        "zone_model":  "models/artifacts/zone_kmeans.joblib",
        "feature_list":"models/artifacts/feature_columns.json",
    },
    "cleaning": {
        "lat_min": 40.55, "lat_max": 40.92,
        "lon_min": -74.27, "lon_max": -73.68,
        "min_duration_min": 1, "max_duration_min": 120,
        "min_distance_km": 0.1, "max_speed_kmh": 100,
        "max_passengers": 6,
    },
    "features": {
        "n_zones":    5,
        "rush_hours": [7, 8, 9, 16, 17, 18, 19],
    },
    "training": {
        "sample_rows": 1000, "test_size": 0.2,
        "random_state": 42, "target": "duration_min",
    },
}

RUSH_HOURS = set(CFG["features"]["rush_hours"])


# ---------------------------------------------------------------------------
# Helper to build a small valid DataFrame
# ---------------------------------------------------------------------------

def _make_trip_df(
    pickup_lat=40.7580, pickup_lon=-73.9855,
    dropoff_lat=40.7527, dropoff_lon=-73.9772,
    pickup_datetime="2016-03-01 08:00:00",   # Tuesday 08:00 = rush hour
    duration_min=20.0,
    n=5,
) -> pd.DataFrame:
    """Return a tiny DataFrame with n identical-ish rows."""
    rows = []
    for i in range(n):
        rows.append({
            "id": f"t{i}",
            "vendor_id": 1,
            "pickup_datetime": pd.Timestamp(pickup_datetime),
            "dropoff_datetime": pd.Timestamp(pickup_datetime),
            "passenger_count": 1,
            "pickup_latitude":  pickup_lat,
            "pickup_longitude": pickup_lon,
            "dropoff_latitude": dropoff_lat,
            "dropoff_longitude": dropoff_lon,
            "trip_duration": int(duration_min * 60),
            "duration_min": duration_min,
            "distance_km": 1.0,
            "implied_speed_kmh": 30.0,
            "store_and_fwd_flag": "N",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fake weather so tests don't hit the network
# ---------------------------------------------------------------------------

def _fake_get_weather(datetimes, cfg):
    """Return constant weather values for any datetime series."""
    return pd.DataFrame({
        "temperature_2m":  [10.0] * len(datetimes),
        "precipitation":   [0.0]  * len(datetimes),
        "wind_speed_10m":  [5.0]  * len(datetimes),
    }, index=datetimes.index)


# ---------------------------------------------------------------------------
# Tests: geometry functions
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_same_point_is_zero(self):
        lat = pd.Series([40.758])
        lon = pd.Series([-73.985])
        dist = _haversine_km(lat, lon, lat, lon)
        assert float(dist.iloc[0]) == pytest.approx(0.0, abs=1e-9)

    def test_known_nyc_pair(self):
        """
        Times Square (40.7580, -73.9855) to Grand Central (40.7527, -73.9772).
        Known ground-truth ≈ 0.85–0.95 km.
        """
        lat1 = pd.Series([40.7580])
        lon1 = pd.Series([-73.9855])
        lat2 = pd.Series([40.7527])
        lon2 = pd.Series([-73.9772])
        dist = _haversine_km(lat1, lon1, lat2, lon2)
        assert float(dist.iloc[0]) == pytest.approx(0.89, abs=0.05)

    def test_symmetry(self):
        """Distance A→B == distance B→A."""
        lat1, lon1 = pd.Series([40.758]), pd.Series([-73.985])
        lat2, lon2 = pd.Series([40.753]), pd.Series([-73.977])
        d1 = _haversine_km(lat1, lon1, lat2, lon2)
        d2 = _haversine_km(lat2, lon2, lat1, lon1)
        assert float(d1.iloc[0]) == pytest.approx(float(d2.iloc[0]), rel=1e-9)

    def test_always_non_negative(self):
        lat1 = pd.Series([40.7, 40.8])
        lon1 = pd.Series([-73.9, -74.0])
        lat2 = pd.Series([40.75, 40.85])
        lon2 = pd.Series([-73.95, -74.05])
        assert (_haversine_km(lat1, lon1, lat2, lon2) >= 0).all()


class TestManhattan:
    def test_same_point_is_zero(self):
        lat = pd.Series([40.758])
        lon = pd.Series([-73.985])
        dist = _manhattan_km(lat, lon, lat, lon)
        assert float(dist.iloc[0]) == pytest.approx(0.0, abs=1e-9)

    def test_always_non_negative(self):
        lat1 = pd.Series([40.7])
        lon1 = pd.Series([-73.9])
        lat2 = pd.Series([40.75])
        lon2 = pd.Series([-73.95])
        assert float(_manhattan_km(lat1, lon1, lat2, lon2).iloc[0]) >= 0

    def test_manhattan_geq_haversine(self):
        """Manhattan distance is always >= haversine for the same pair."""
        lat1 = pd.Series([40.7580])
        lon1 = pd.Series([-73.9855])
        lat2 = pd.Series([40.7527])
        lon2 = pd.Series([-73.9772])
        h = float(_haversine_km(lat1, lon1, lat2, lon2).iloc[0])
        m = float(_manhattan_km(lat1, lon1, lat2, lon2).iloc[0])
        assert m >= h - 1e-6   # Manhattan >= haversine (within float tolerance)


class TestBearing:
    def test_north_bearing(self):
        """Moving due north should give bearing ≈ 0°."""
        lat1, lon1 = pd.Series([40.0]), pd.Series([-74.0])
        lat2, lon2 = pd.Series([41.0]), pd.Series([-74.0])
        b = float(_bearing(lat1, lon1, lat2, lon2).iloc[0])
        assert b == pytest.approx(0.0, abs=1.0)

    def test_south_bearing(self):
        """Moving due south should give bearing ≈ 180°."""
        lat1, lon1 = pd.Series([41.0]), pd.Series([-74.0])
        lat2, lon2 = pd.Series([40.0]), pd.Series([-74.0])
        b = float(_bearing(lat1, lon1, lat2, lon2).iloc[0])
        assert b == pytest.approx(180.0, abs=1.0)

    def test_range_0_to_360(self):
        lat1 = pd.Series([40.7, 40.8])
        lon1 = pd.Series([-73.9, -74.0])
        lat2 = pd.Series([40.75, 40.85])
        lon2 = pd.Series([-73.95, -74.05])
        b = _bearing(lat1, lon1, lat2, lon2)
        assert (b >= 0).all() and (b <= 360).all()


# ---------------------------------------------------------------------------
# Tests: rush-hour and time flags
# ---------------------------------------------------------------------------

class TestRushHourFlag:
    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_rush_hour_weekday_peak(self, mock_weather):
        """Tuesday 08:00 is a rush hour weekday → is_rush_hour == 1."""
        df = _make_trip_df(pickup_datetime="2016-03-01 08:00:00")  # Tuesday
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        assert (out["is_rush_hour"] == 1).all()

    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_no_rush_hour_weekday_off_peak(self, mock_weather):
        """Tuesday 13:00 is NOT a rush hour → is_rush_hour == 0."""
        df = _make_trip_df(pickup_datetime="2016-03-01 13:00:00")  # Tuesday
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        assert (out["is_rush_hour"] == 0).all()

    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_no_rush_hour_on_weekend(self, mock_weather):
        """Saturday 08:00 should NOT be rush hour even if hour matches."""
        df = _make_trip_df(pickup_datetime="2016-03-05 08:00:00")  # Saturday
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        assert (out["is_rush_hour"] == 0).all()

    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_is_weekend_flag(self, mock_weather):
        """Saturday → is_weekend == 1."""
        df = _make_trip_df(pickup_datetime="2016-03-05 10:00:00")  # Saturday
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        assert (out["is_weekend"] == 1).all()

    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_not_weekend_on_tuesday(self, mock_weather):
        """Tuesday → is_weekend == 0."""
        df = _make_trip_df(pickup_datetime="2016-03-01 10:00:00")  # Tuesday
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        assert (out["is_weekend"] == 0).all()


# ---------------------------------------------------------------------------
# Tests: cyclic hour encoding
# ---------------------------------------------------------------------------

class TestCyclicHour:
    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_midnight_sin_zero(self, mock_weather):
        """hour=0 → sin(0) = 0, cos(0) = 1."""
        df = _make_trip_df(pickup_datetime="2016-03-01 00:00:00")
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        assert float(out["hour_sin"].iloc[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(out["hour_cos"].iloc[0]) == pytest.approx(1.0, abs=1e-6)

    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_noon_sin_zero(self, mock_weather):
        """hour=12 → sin(π) ≈ 0, cos(π) = -1."""
        df = _make_trip_df(pickup_datetime="2016-03-01 12:00:00")
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        assert float(out["hour_sin"].iloc[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(out["hour_cos"].iloc[0]) == pytest.approx(-1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: zones
# ---------------------------------------------------------------------------

class TestZones:
    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_zone_ids_in_valid_range(self, mock_weather):
        """pickup_zone and dropoff_zone must be in [0, n_zones)."""
        df = _make_trip_df(n=20)
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        n_zones = CFG["features"]["n_zones"]
        assert out["pickup_zone"].between(0, n_zones - 1).all()
        assert out["dropoff_zone"].between(0, n_zones - 1).all()

    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_prefit_zone_model_is_reused(self, mock_weather):
        """Passing a pre-fitted zone_model should not raise and should produce zones."""
        from sklearn.cluster import KMeans
        df = _make_trip_df(n=10)
        coords = df[["pickup_latitude", "pickup_longitude",
                     "dropoff_latitude", "dropoff_longitude"]].values
        zm = KMeans(n_clusters=3, random_state=0, n_init="auto").fit(coords)
        out, returned_model = build_features(df, zone_model=zm, cfg=CFG)
        assert returned_model is zm
        assert "pickup_zone" in out.columns
        assert "dropoff_zone" in out.columns

    def test_missing_zone_model_raises(self):
        """Calling without zone_model and fit_zones=False must raise ValueError."""
        df = _make_trip_df(n=5)
        with pytest.raises(ValueError, match="zone_model"):
            build_features(df, cfg=CFG)   # no zone_model, fit_zones=False


# ---------------------------------------------------------------------------
# Tests: FEATURE_COLUMNS list
# ---------------------------------------------------------------------------

class TestFeatureColumns:
    def test_all_expected_features_present(self):
        expected = {
            "haversine_km", "manhattan_km", "bearing",
            "hour", "weekday", "month",
            "is_weekend", "is_rush_hour",
            "hour_sin", "hour_cos",
            "pickup_zone", "dropoff_zone",
            "temperature_2m", "precipitation", "wind_speed_10m",
        }
        assert expected.issubset(set(FEATURE_COLUMNS))

    @patch("preprocessing.features.get_weather", side_effect=_fake_get_weather)
    def test_feature_columns_all_in_output(self, mock_weather):
        """Every column in FEATURE_COLUMNS must exist in build_features output."""
        df = _make_trip_df(n=5)
        out, _ = build_features(df, fit_zones=True, cfg=CFG)
        missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
        assert missing == [], f"Missing columns: {missing}"
