"""
preprocessing/clean.py
======================
Cleans the raw NYC Taxi Trip Duration dataset.

All parameters are read from configs/config.yaml so no values are hardcoded.
Run as a script:
    python -m preprocessing.clean
or directly:
    python preprocessing/clean.py
"""

from __future__ import annotations

import pathlib
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Haversine distance (vectorised, returns kilometres)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: pd.Series, lon1: pd.Series,
                  lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Return great-circle distance in km between pairs of lat/lon points."""
    R = 6_371.0  # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_trips(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Apply all cleaning rules defined in cfg['cleaning'] to the raw trips DataFrame.

    Parameters
    ----------
    df  : Raw DataFrame loaded from data/raw/train.csv
    cfg : Parsed config.yaml dict (full document)

    Returns
    -------
    (cleaned_df, report_dict) where report_dict contains per-rule drop counts.
    """
    c = cfg["cleaning"]
    report: dict[str, int] = {}
    rows_before = len(df)

    # --- 1. Parse datetimes ------------------------------------------------
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])

    # --- 2. Convert trip_duration (seconds) to duration_min (minutes) ------
    df["duration_min"] = df["trip_duration"] / 60.0

    # --- 3. Drop nulls and exact duplicates --------------------------------
    before = len(df)
    df = df.dropna()
    report["nulls"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    report["duplicates"] = before - len(df)

    # --- 4. Drop zero-distance trips (haversine < min_distance_km) ---------
    df["distance_km"] = _haversine_km(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"],
    )
    before = len(df)
    df = df[df["distance_km"] >= c["min_distance_km"]]
    report["zero_distance"] = before - len(df)

    # --- 5. Drop trips outside the NYC bounding box (pickup OR dropoff) ----
    before = len(df)
    in_nyc = (
        df["pickup_latitude"].between(c["lat_min"], c["lat_max"]) &
        df["pickup_longitude"].between(c["lon_min"], c["lon_max"]) &
        df["dropoff_latitude"].between(c["lat_min"], c["lat_max"]) &
        df["dropoff_longitude"].between(c["lon_min"], c["lon_max"])
    )
    df = df[in_nyc]
    report["outside_nyc"] = before - len(df)

    # --- 6. Drop trips outside allowed duration range ----------------------
    before = len(df)
    df = df[df["duration_min"].between(c["min_duration_min"], c["max_duration_min"])]
    report["bad_duration"] = before - len(df)

    # --- 7. Drop trips with impossible speeds (> max_speed_kmh) ------------
    # implied_speed = distance_km / (duration_min / 60)  [km/h]
    df["implied_speed_kmh"] = df["distance_km"] / (df["duration_min"] / 60.0)
    before = len(df)
    df = df[df["implied_speed_kmh"] <= c["max_speed_kmh"]]
    report["impossible_speed"] = before - len(df)

    # --- 8. Drop invalid passenger counts ----------------------------------
    before = len(df)
    df = df[(df["passenger_count"] >= 1) & (df["passenger_count"] <= c["max_passengers"])]
    report["bad_passenger_count"] = before - len(df)

    rows_after = len(df)
    report["rows_before"] = rows_before
    report["rows_after"] = rows_after
    report["total_dropped"] = rows_before - rows_after

    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Cleaning report generator
# ---------------------------------------------------------------------------

def build_cleaning_report(report: dict, cfg: dict) -> str:
    """
    Format the cleaning statistics as a Markdown document.

    Parameters
    ----------
    report : dict produced by clean_trips()
    cfg    : full parsed config.yaml

    Returns
    -------
    Markdown string ready to write to docs/cleaning_report.md
    """
    c = cfg["cleaning"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def pct(n):
        return f"{n:,}  ({n / report['rows_before'] * 100:.2f} %)"

    md = textwrap.dedent(f"""\
        # Data Cleaning Report

        **Generated:** {now}
        **Source:** `{cfg['paths']['raw_trips']}`
        **Output:** `{cfg['paths']['clean_trips']}`

        ---

        ## Summary

        | Metric | Count |
        |--------|-------|
        | Rows before cleaning | {report['rows_before']:,} |
        | Rows after cleaning  | {report['rows_after']:,} |
        | **Total dropped**    | **{report['total_dropped']:,}** |
        | Retention rate       | {report['rows_after'] / report['rows_before'] * 100:.2f} % |

        ---

        ## Drop Breakdown (per rule)

        | Rule | Rows Dropped | % of Raw |
        |------|-------------|---------|
        | Nulls | {pct(report['nulls'])} |
        | Exact duplicates | {pct(report['duplicates'])} |
        | Zero / near-zero distance (< {c['min_distance_km']} km) | {pct(report['zero_distance'])} |
        | Outside NYC bounding box | {pct(report['outside_nyc'])} |
        | Trip duration outside [{c['min_duration_min']}, {c['max_duration_min']}] min | {pct(report['bad_duration'])} |
        | Impossible speed (> {c['max_speed_kmh']} km/h) | {pct(report['impossible_speed'])} |
        | Invalid passenger count (0 or > {c['max_passengers']}) | {pct(report['bad_passenger_count'])} |

        ---

        ## Cleaning Rules Applied

        | Parameter | Value |
        |-----------|-------|
        | NYC lat range | [{c['lat_min']}, {c['lat_max']}] |
        | NYC lon range | [{c['lon_min']}, {c['lon_max']}] |
        | Min duration (min) | {c['min_duration_min']} |
        | Max duration (min) | {c['max_duration_min']} |
        | Min distance (km)  | {c['min_distance_km']} |
        | Max speed (km/h)   | {c['max_speed_kmh']} |
        | Max passengers     | {c['max_passengers']} |

        ---

        ## Notes

        - **Zero-distance trips:** Trips with haversine distance < {c['min_distance_km']} km are
          treated as GPS glitches or test rides and dropped.
        - **Long trips:** Trips exceeding {c['max_duration_min']} minutes (2 hours) are likely
          data errors or very rare outliers; excluded to keep the model focused on realistic
          urban trips.
        - **Outside NYC:** Both pickup AND dropoff must be within the bounding box
          lat in [{c['lat_min']}, {c['lat_max']}], lon in [{c['lon_min']}, {c['lon_max']}].
        - **Impossible speeds:** Implied speed = distance_km / (duration_min / 60).
          Values > {c['max_speed_kmh']} km/h are physically impossible for NYC traffic and
          indicate GPS or timestamp errors.
    """)
    return md


# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Locate config.yaml relative to this file (project root / configs/)
    project_root = pathlib.Path(__file__).parent.parent
    config_path = project_root / "configs" / "config.yaml"

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    raw_path = project_root / cfg["paths"]["raw_trips"]
    clean_path = project_root / cfg["paths"]["clean_trips"]
    report_path = project_root / "docs" / "cleaning_report.md"

    print(f"[clean] Loading raw trips from: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    print(f"[clean] Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

    print("[clean] Applying cleaning rules ...")
    df_clean, report = clean_trips(df_raw, cfg)

    # Print live summary to console
    print("\n" + "=" * 55)
    print("  CLEANING REPORT")
    print("=" * 55)
    print(f"  Rows before   : {report['rows_before']:>10,}")
    print(f"  Rows after    : {report['rows_after']:>10,}")
    print(f"  Total dropped : {report['total_dropped']:>10,}")
    print("-" * 55)
    rules = [
        ("Nulls",                 report["nulls"]),
        ("Duplicates",            report["duplicates"]),
        ("Zero distance",         report["zero_distance"]),
        ("Outside NYC",           report["outside_nyc"]),
        ("Bad duration",          report["bad_duration"]),
        ("Impossible speed",      report["impossible_speed"]),
        ("Bad passenger count",   report["bad_passenger_count"]),
    ]
    for label, n in rules:
        pct = n / report["rows_before"] * 100
        print(f"  {label:<25}: {n:>8,}  ({pct:.2f} %)")
    print("=" * 55)

    # Save cleaned parquet
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(clean_path, index=False)
    print(f"\n[clean] Saved cleaned data -> {clean_path}")

    # Save markdown report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = build_cleaning_report(report, cfg)
    report_path.write_text(md, encoding="utf-8")
    print(f"[clean] Saved cleaning report -> {report_path}")
    print("\n[clean] Done.")
