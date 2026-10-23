"""
tests/test_clean.py
===================
Unit tests for preprocessing/clean.py - Food Delivery dataset.

Each test injects exactly ONE bad row into a tiny hand-crafted DataFrame and
checks that clean_trips() handles it correctly (drops, clamps, or fills).
All config values are read from a minimal CFG dict that mirrors config.yaml.
"""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.clean import clean_trips, _haversine_km  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal config mirroring configs/config.yaml cleaning section
# ---------------------------------------------------------------------------

CFG = {
    "seed": 42,
    "paths": {
        "raw_trips":  "data/raw/train.csv",
        "clean_trips": "data/processed/cleaned.csv",
    },
    "cleaning": {
        "lat_min": 6.0,
        "lat_max": 37.0,
        "lon_min": 68.0,
        "lon_max": 98.0,
        "min_distance_km": 0.1,
        "max_distance_km": 25.0,
        "max_speed_kmh": 80.0,
        "min_time_min": 10,
        "max_time_min": 54,
        "age_min": 18,
        "age_max": 70,
        "rating_min": 1.0,
        "rating_max": 5.0,
        "nan_strings": ["NaN", "nan", ""],
        "fill_unknown_cols": ["multiple_deliveries", "Festival"],
        "required_cols": ["Time_Orderd", "Road_traffic_density", "Weatherconditions"],
    },
}


# ---------------------------------------------------------------------------
# Helper: a valid base row that passes every cleaning rule
# ---------------------------------------------------------------------------

def _good_row(suffix: str = "0") -> dict:
    """Return a dict representing a single clean food-delivery trip."""
    return {
        "ID":                          f"0x{suffix:0>4}",
        "Delivery_person_ID":          f"BANGRES13DEL0{suffix}",
        "Delivery_person_Age":         "30",
        "Delivery_person_Ratings":     "4.5",
        "Restaurant_latitude":         12.914264,
        "Restaurant_longitude":        77.678400,
        "Delivery_location_latitude":  12.974264,   # ~6.7 km north
        "Delivery_location_longitude": 77.678400,
        "Order_Date":                  "19-03-2022",
        "Time_Orderd":                 "08:30:00",
        "Time_Order_picked":           "08:45:00",
        "Weatherconditions":           "conditions Sunny",
        "Road_traffic_density":        "Medium ",
        "Vehicle_condition":           "2",
        "Type_of_order":               "Snack",
        "Type_of_vehicle":             "motorcycle ",
        "multiple_deliveries":         "0",
        "Festival":                    "No ",
        "City":                        "Urban ",
        "Time_taken(min)":             "(min) 26",
    }


def _make_df(*extra_rows: dict) -> pd.DataFrame:
    """Return a DataFrame with one guaranteed-good row plus any extras."""
    rows = [_good_row()] + list(extra_rows)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCleanFoodDelivery:

    # --- basic pass-through ---------------------------------------------------

    def test_good_row_is_kept(self):
        """A perfectly valid row must survive all filters."""
        df = _make_df()
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["total_dropped"] == 0

    # --- Step 1: text normalisation ------------------------------------------

    def test_conditions_prefix_stripped(self):
        """'conditions Sunny' in Weatherconditions must be normalised to 'Sunny'."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert clean.loc[0, "Weatherconditions"] == "Sunny"

    def test_min_prefix_stripped_from_target(self):
        """'(min) 26' in Time_taken(min) must yield Time_taken_min == 26."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert clean.loc[0, "Time_taken_min"] == 26

    def test_trailing_whitespace_stripped_in_traffic(self):
        """'Medium ' must be cleaned to 'Medium' in Road_traffic_density."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert clean.loc[0, "Road_traffic_density"] == "Medium"

    def test_nan_string_becomes_real_nan(self):
        """'NaN' string in Weatherconditions must become a real NaN (row dropped)."""
        bad = _good_row("b1")
        bad["Weatherconditions"] = "conditions NaN"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        # NaN in required col -> row is dropped
        assert len(clean) == 1
        assert report["missing_required"] >= 1

    # --- Step 2: duplicates ---------------------------------------------------

    def test_exact_duplicate_row_dropped(self):
        """An exact duplicate row must be dropped."""
        good = _good_row()
        df = pd.DataFrame([good, good])
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["dup_rows"] == 1

    def test_duplicate_id_dropped(self):
        """Two rows sharing an ID keep only the first."""
        row1 = _good_row("dup")
        row2 = _good_row("dup")      # same ID, different lat
        row2["Restaurant_latitude"] = 13.0
        df = pd.DataFrame([row1, row2])
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["dup_ids"] == 1

    # --- Step 3: type conversions ---------------------------------------------

    def test_ordered_at_is_datetime(self):
        """ordered_at must be a datetime column."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert pd.api.types.is_datetime64_any_dtype(clean["ordered_at"])

    def test_ordered_at_value(self):
        """ordered_at must parse 19-03-2022 08:30:00 correctly."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        expected = pd.Timestamp("2022-03-19 08:30:00")
        assert clean.loc[0, "ordered_at"] == expected

    def test_time_taken_min_is_numeric(self):
        """Time_taken_min must be a numeric column."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert pd.api.types.is_numeric_dtype(clean["Time_taken_min"])

    def test_city_tier_column_dropped(self):
        """The original City (tier) column must be removed."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert "City" not in clean.columns

    def test_city_extracted_from_id(self):
        """city must be extracted from the Delivery_person_ID prefix."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert clean.loc[0, "city"] == "BANGRES"

    # --- Step 4: clamp Age / Ratings -----------------------------------------

    def test_age_below_min_set_to_nan(self):
        """Age < 18 must be set to NaN, row kept."""
        bad = _good_row("a1")
        bad["Delivery_person_Age"] = "15"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 2           # row kept
        assert report["age_clamped_to_nan"] == 1
        row = clean[clean["ID"] == "0x00a1"].iloc[0]
        assert pd.isna(row["Delivery_person_Age"])

    def test_age_above_max_set_to_nan(self):
        """Age > 70 must be set to NaN, row kept."""
        bad = _good_row("a2")
        bad["Delivery_person_Age"] = "80"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 2
        assert report["age_clamped_to_nan"] == 1

    def test_ratings_below_min_set_to_nan(self):
        """Ratings < 1.0 must be set to NaN, row kept."""
        bad = _good_row("r1")
        bad["Delivery_person_Ratings"] = "0.5"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 2
        assert report["ratings_clamped_to_nan"] == 1

    def test_ratings_above_max_set_to_nan(self):
        """Ratings > 5.0 must be set to NaN, row kept."""
        bad = _good_row("r2")
        bad["Delivery_person_Ratings"] = "5.5"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 2
        assert report["ratings_clamped_to_nan"] == 1

    # --- Step 5: required cols, fill unknowns --------------------------------

    def test_missing_time_orderd_row_dropped(self):
        """Row with NaN Time_Orderd must be dropped."""
        bad = _good_row("t1")
        bad["Time_Orderd"] = "NaN"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["missing_required"] >= 1

    def test_missing_traffic_row_dropped(self):
        """Row with NaN Road_traffic_density must be dropped."""
        bad = _good_row("t2")
        bad["Road_traffic_density"] = "NaN "
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["missing_required"] >= 1

    def test_missing_weather_row_dropped(self):
        """Row with NaN Weatherconditions must be dropped."""
        bad = _good_row("t3")
        bad["Weatherconditions"] = "conditions NaN"
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1

    def test_nan_multiple_deliveries_filled_unknown(self):
        """NaN multiple_deliveries must be filled with 'unknown', row kept."""
        bad = _good_row("m1")
        bad["multiple_deliveries"] = "NaN "
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 2
        # ID is '0xm1' (suffix 'm1' formatted with 0>4 padding -> '00m1'... but
        # '0>4' only pads digits; use direct filter on the full column instead)
        row = clean[clean["ID"].str.endswith("m1")].iloc[0]
        assert row["multiple_deliveries"] == "unknown"

    def test_nan_festival_filled_unknown(self):
        """NaN Festival must be filled with 'unknown', row kept."""
        bad = _good_row("f1")
        bad["Festival"] = "NaN "
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 2
        row = clean[clean["ID"] == "0x00f1"].iloc[0]
        assert row["Festival"] == "unknown"

    def test_time_order_picked_dropped(self):
        """Time_Order_picked must NOT appear in cleaned output."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert "Time_Order_picked" not in clean.columns

    # --- Step 6: coordinates / distance / speed ------------------------------

    def test_zero_lat_row_dropped(self):
        """Row with latitude = 0 must be dropped (outside India bbox)."""
        bad = _good_row("z1")
        bad["Restaurant_latitude"] = 0.01
        bad["Restaurant_longitude"] = 0.01
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["outside_bbox"] >= 1

    def test_negative_lat_row_dropped(self):
        """Row with negative Restaurant_latitude dropped (outside bbox)."""
        bad = _good_row("z2")
        bad["Restaurant_latitude"] = -12.5
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["outside_bbox"] >= 1

    def test_zero_distance_row_dropped(self):
        """Row where restaurant == delivery location (0 km) must be dropped."""
        bad = _good_row("d1")
        bad["Delivery_location_latitude"]  = bad["Restaurant_latitude"]
        bad["Delivery_location_longitude"] = bad["Restaurant_longitude"]
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["bad_distance"] >= 1

    def test_too_far_row_dropped(self):
        """Row with distance > 25 km must be dropped."""
        bad = _good_row("d2")
        bad["Delivery_location_latitude"] = 13.5   # ~65 km from 12.91
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["bad_distance"] >= 1

    def test_high_speed_row_dropped(self):
        """Row with implied speed > 80 km/h must be dropped."""
        # 10 km in 10 minutes = 60 km/h  (OK - kept as control to verify logic)
        # 10 km in 10 minutes but let's use ~10 km distance with Time_taken_min=10
        # -> speed = 10 / (10/60) = 60 km/h  (fine, not dropped)
        # Instead: ~15 km distance, Time_taken_min = 10 -> 90 km/h > 80 -> DROPPED
        # lat diff of 0.135 deg ~ 15 km
        bad = _good_row("s1")
        bad["Delivery_location_latitude"] = 12.914264 + 0.135  # ~15 km north
        bad["Delivery_location_longitude"] = 77.678400
        bad["Time_taken(min)"] = "(min) 10"   # 15 km / (10/60 h) = 90 km/h > 80
        df = _make_df(bad)
        clean, report = clean_trips(df, CFG)
        assert len(clean) == 1
        assert report["high_speed_dropped"] >= 1

    # --- Step 8: rare vehicle category ---------------------------------------

    def test_bicycle_kept(self):
        """Type_of_vehicle = 'bicycle' must be kept (rare but valid)."""
        bad = _good_row("v1")
        bad["Type_of_vehicle"] = "bicycle "
        df = _make_df(bad)
        clean, _ = clean_trips(df, CFG)
        assert "bicycle" in clean["Type_of_vehicle"].values

    # --- Step 9: assertions / report consistency -----------------------------

    def test_no_nan_except_allowed(self):
        """After cleaning, only Age and Ratings may have NaN."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        allowed = {"Delivery_person_Age", "Delivery_person_Ratings"}
        for col in clean.columns:
            if col in allowed:
                continue
            assert clean[col].isna().sum() == 0, f"Unexpected NaN in {col}"

    def test_ids_unique(self):
        """After cleaning, IDs must be unique."""
        df = _make_df()
        clean, _ = clean_trips(df, CFG)
        assert clean["ID"].is_unique

    def test_report_row_counts_consistent(self):
        """rows_before == rows_after + total_dropped."""
        bad_traffic = _good_row("c2")
        bad_traffic["Road_traffic_density"] = "NaN "
        df = _make_df(bad_traffic)
        clean, report = clean_trips(df, CFG)
        assert report["rows_before"] == report["rows_after"] + report["total_dropped"]
        assert report["rows_after"] == len(clean)

    # --- Haversine helper ----------------------------------------------------

    def test_haversine_zero_distance(self):
        """Same point must return 0 km."""
        s = pd.Series([12.9])
        dist = _haversine_km(s, s, s, s)
        assert float(dist.iloc[0]) == pytest.approx(0.0, abs=1e-9)

    def test_haversine_known_distance(self):
        """Approx 6.7 km between Bangalore test lat pair."""
        lat1 = pd.Series([12.914264])
        lon1 = pd.Series([77.678400])
        lat2 = pd.Series([12.974264])
        lon2 = pd.Series([77.678400])
        dist = _haversine_km(lat1, lon1, lat2, lon2)
        assert float(dist.iloc[0]) == pytest.approx(6.67, abs=0.1)
