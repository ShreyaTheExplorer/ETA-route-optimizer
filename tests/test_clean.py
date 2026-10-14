"""
tests/test_clean.py
===================
Unit tests for preprocessing/clean.py.

Each test injects exactly ONE bad row into a tiny hand-crafted DataFrame and
checks that clean_trips() drops that row and only that row.
"""

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is on the path when running from anywhere
PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.clean import clean_trips  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal config that mirrors configs/config.yaml cleaning section
# ---------------------------------------------------------------------------
CFG = {
    "paths": {
        "raw_trips": "data/raw/train.csv",
        "clean_trips": "data/processed/trips_clean.parquet",
    },
    "cleaning": {
        "lat_min": 40.55,
        "lat_max": 40.92,
        "lon_min": -74.27,
        "lon_max": -73.68,
        "min_duration_min": 1,
        "max_duration_min": 120,
        "min_distance_km": 0.1,
        "max_speed_kmh": 100,
        "max_passengers": 6,
    },
}


# ---------------------------------------------------------------------------
# Helper: build a valid base row that passes every rule
# ---------------------------------------------------------------------------

def _good_row() -> dict:
    """Return a dict representing a single clean NYC taxi trip."""
    return {
        "id": "good_0",
        "vendor_id": 1,
        "pickup_datetime": "2016-03-01 08:00:00",
        "dropoff_datetime": "2016-03-01 08:20:00",
        "passenger_count": 1,
        # Manhattan: pickup near Times Square, dropoff near Grand Central
        "pickup_latitude": 40.7580,
        "pickup_longitude": -73.9855,
        "dropoff_latitude": 40.7527,
        "dropoff_longitude": -73.9772,
        "trip_duration": 1200,  # 20 minutes in seconds
        "store_and_fwd_flag": "N",
    }


def _make_df(*extra_rows: dict) -> pd.DataFrame:
    """Return a DataFrame with one guaranteed-good row plus any extras."""
    rows = [_good_row()] + list(extra_rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanTrips:

    def test_good_row_is_kept(self):
        """A perfectly valid row must survive all filters."""
        df = _make_df()
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["total_dropped"] == 0

    def test_null_row_is_dropped(self):
        """A row with a null value in any column must be dropped."""
        bad = _good_row()
        bad["passenger_count"] = np.nan
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["nulls"] == 1

    def test_duplicate_row_is_dropped(self):
        """An exact duplicate of a good row must be dropped."""
        good = _good_row()
        df = pd.DataFrame([good, good])  # two identical rows
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["duplicates"] == 1

    def test_zero_distance_trip_is_dropped(self):
        """A trip with pickup == dropoff (0 km) must be dropped."""
        bad = _good_row()
        bad["dropoff_latitude"] = bad["pickup_latitude"]
        bad["dropoff_longitude"] = bad["pickup_longitude"]
        bad["id"] = "zero_dist"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["zero_distance"] == 1

    def test_outside_nyc_pickup_is_dropped(self):
        """A trip whose pickup is outside the NYC bounding box must be dropped."""
        bad = _good_row()
        bad["pickup_latitude"] = 41.5    # north of NYC bounding box
        bad["pickup_longitude"] = -73.9
        bad["id"] = "outside_pickup"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["outside_nyc"] == 1

    def test_outside_nyc_dropoff_is_dropped(self):
        """A trip whose dropoff is outside the NYC bounding box must be dropped."""
        bad = _good_row()
        bad["dropoff_latitude"] = 40.0   # south of NYC bounding box
        bad["dropoff_longitude"] = -74.0
        bad["id"] = "outside_dropoff"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["outside_nyc"] == 1

    def test_trip_over_2_hours_is_dropped(self):
        """A trip longer than max_duration_min (120 min) must be dropped."""
        bad = _good_row()
        bad["trip_duration"] = 8000  # ~133 minutes
        bad["id"] = "long_trip"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["bad_duration"] == 1

    def test_trip_under_min_duration_is_dropped(self):
        """A trip shorter than min_duration_min (1 min) must be dropped."""
        bad = _good_row()
        bad["trip_duration"] = 30  # 0.5 minutes
        bad["id"] = "short_trip"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["bad_duration"] == 1

    def test_impossible_speed_is_dropped(self):
        """A trip implying speed > 100 km/h must be dropped."""
        bad = _good_row()
        # Move dropoff ~3 km north of pickup (still inside NYC bounding box).
        # With trip_duration = 60 s (1 min), implied speed = 3 km / (1/60 h) = 180 km/h.
        # Duration passes the [1, 120] min range check, so only the speed filter fires.
        bad["dropoff_latitude"] = bad["pickup_latitude"] + 0.027   # ~3 km north
        bad["dropoff_longitude"] = bad["pickup_longitude"]
        bad["trip_duration"] = 60   # exactly 1 minute in seconds
        bad["id"] = "fast_trip"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["impossible_speed"] == 1

    def test_zero_passenger_count_is_dropped(self):
        """A trip with passenger_count == 0 must be dropped."""
        bad = _good_row()
        bad["passenger_count"] = 0
        bad["id"] = "zero_pax"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["bad_passenger_count"] == 1

    def test_excessive_passenger_count_is_dropped(self):
        """A trip with passenger_count > max_passengers (6) must be dropped."""
        bad = _good_row()
        bad["passenger_count"] = 9
        bad["id"] = "too_many_pax"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["bad_passenger_count"] == 1

    def test_report_row_counts_are_consistent(self):
        """rows_before == rows_after + total_dropped."""
        # Mix: 1 good + 1 null + 1 zero-distance
        bad_null = _good_row()
        bad_null["id"] = "bad_null"
        bad_null["passenger_count"] = np.nan

        bad_dist = _good_row()
        bad_dist["id"] = "bad_dist"
        bad_dist["dropoff_latitude"] = bad_dist["pickup_latitude"]
        bad_dist["dropoff_longitude"] = bad_dist["pickup_longitude"]

        df = _make_df(bad_null, bad_dist)
        clean, report = clean_trips(df, CFG)
        assert report["rows_before"] == report["rows_after"] + report["total_dropped"]
        assert report["rows_after"] == len(clean)
