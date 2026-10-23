"""
preprocessing/clean.py
======================
Cleans the raw Food Delivery dataset (data/raw/train.csv).

All thresholds are read from configs/config.yaml so nothing is hardcoded here.
Run as a script:
    python -m preprocessing.clean
or:
    python preprocessing/clean.py

Steps (match the 10 requirements in the task specification):
 1. Strip whitespace from column names & all object cells; strip "conditions " /
    "(min) " prefixes; replace nan-string sentinels with real NaN.
 2. Drop exact duplicate rows and duplicate ID rows.
 3. Type conversions: Time_taken_min (numeric), Age, Ratings (numeric),
    multiple_deliveries (Int8 nullable int), ordered_at (datetime from
    Order_Date + Time_Orderd).  Raises ValueError if datetime parse fails.
 4. Clamp Age / Ratings to valid range -> NaN if outside; report counts first.
 5. Drop rows missing required columns (Time_Orderd, traffic, weather);
    fill multiple_deliveries / Festival unknowns with "unknown".
    Extract city from Delivery_person_ID prefix; drop City tier column.
 6. Drop rows with coordinates outside India bbox; compute haversine;
    drop distance < min / > max; flag implied speed > max_speed_kmh, show, drop.
 7. Keep Time_taken_min as-is (range 10-54); do not use Time_Order_picked.
 8. Type_of_vehicle kept as-is (bicycle is a valid rare category).
 9. Final assertions: no NaN except Age/Ratings, ranges valid, IDs unique.
10. Save data/processed/cleaned.csv and docs/cleaning_report.md.
"""

from __future__ import annotations

import pathlib
import re
import sys
import textwrap
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Haversine (vectorised, returns km)
# ---------------------------------------------------------------------------

def _haversine_km(lat1: pd.Series, lon1: pd.Series,
                  lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Return great-circle distance in km between pairs of lat/lon points."""
    R = 6_371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi    = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = (np.sin(dphi / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Step 1 helpers: text normalisation
# ---------------------------------------------------------------------------

def _strip_text(df: pd.DataFrame, nan_strings: list[str]) -> pd.DataFrame:
    """
    Strip whitespace from column names and all string cells.
    Remove 'conditions ' and '(min) ' prefixes.
    Replace nan-string sentinels (NaN / nan / '') with real NaN.
    """
    # Column names
    df.columns = [c.strip() for c in df.columns]

    # Object columns: strip, remove prefixes, sentinel->NaN
    nan_set = set(nan_strings)
    prefix_re = re.compile(r'^(?:conditions\s+|\(min\)\s+)', re.IGNORECASE)

    # include both legacy object and new StringDtype columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].str.strip()
        s = s.str.replace(prefix_re, "", regex=True)
        s = s.where(~s.isin(nan_set), other=np.nan)
        df[col] = s

    return df


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------

def clean_trips(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    Apply all cleaning rules to the raw food-delivery DataFrame.

    Parameters
    ----------
    df  : Raw DataFrame loaded from data/raw/train.csv
    cfg : Parsed config.yaml dict (full document)

    Returns
    -------
    (cleaned_df, report_dict) where report_dict contains per-rule counts.
    """
    c      = cfg["cleaning"]
    report: dict[str, object] = {}
    df     = df.copy()

    rows_before = len(df)
    report["rows_before"] = rows_before

    # ==================================================================
    # Step 1 – strip / normalise text
    # ==================================================================
    df = _strip_text(df, nan_strings=c["nan_strings"])

    # Missingness BEFORE any dropping (for report)
    report["missing_before"] = df.isnull().sum().to_dict()

    # ==================================================================
    # Step 2 – drop exact duplicate rows, then duplicate IDs
    # ==================================================================
    before = len(df)
    df = df.drop_duplicates()
    report["dup_rows"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["ID"], keep="first")
    report["dup_ids"] = before - len(df)

    # ==================================================================
    # Step 3 – type conversions
    # ==================================================================

    # 3a. Target: Time_taken_min  (was "(min) 24" -> after strip-prefix -> "24")
    df["Time_taken_min"] = pd.to_numeric(df["Time_taken(min)"], errors="coerce")
    n_target_fail = df["Time_taken_min"].isna().sum()
    if n_target_fail > 0:
        raise ValueError(
            f"Step 3: {n_target_fail} rows have unparseable Time_taken(min) values. "
            f"Sample: {df.loc[df['Time_taken_min'].isna(), 'Time_taken(min)'].head(5).tolist()}"
        )
    df.drop(columns=["Time_taken(min)"], inplace=True)

    # 3b. Delivery_person_Age -> numeric
    df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")

    # 3c. Delivery_person_Ratings -> numeric
    df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"],
                                                   errors="coerce")

    # 3d. multiple_deliveries -> nullable integer (pandas Int8)
    df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"],
                                               errors="coerce").astype("Int8")

    # 3e. ordered_at datetime = Order_Date (dd-mm-yyyy) + Time_Orderd (HH:MM:SS)
    #     Fail loudly if parsing silently breaks.
    raw_dt = df["Order_Date"].str.strip() + " " + df["Time_Orderd"].str.strip()
    ordered_at = pd.to_datetime(raw_dt, format="%d-%m-%Y %H:%M:%S", errors="coerce")
    n_dt_fail = ordered_at.isna().sum() - df["Time_Orderd"].isna().sum()  # subtract pre-NaN
    if n_dt_fail > 0:
        bad_sample = raw_dt[ordered_at.isna() & df["Time_Orderd"].notna()].head(5).tolist()
        raise ValueError(
            f"Step 3: {n_dt_fail} rows could not be parsed as datetime. "
            f"Sample raw strings: {bad_sample}"
        )
    df["ordered_at"] = ordered_at
    # Keep Order_Date for reference; Time_Orderd kept for required-col check below
    report["type_conversions"] = "ok"

    # ==================================================================
    # Step 4 – clamp Age / Ratings to valid range -> NaN if outside
    # ==================================================================
    age_lo, age_hi = c["age_min"], c["age_max"]
    rat_lo, rat_hi = c["rating_min"], c["rating_max"]

    bad_age  = (~df["Delivery_person_Age"].isna() &
                ~df["Delivery_person_Age"].between(age_lo, age_hi))
    bad_rat  = (~df["Delivery_person_Ratings"].isna() &
                ~df["Delivery_person_Ratings"].between(rat_lo, rat_hi))

    report["age_clamped_to_nan"]     = int(bad_age.sum())
    report["ratings_clamped_to_nan"] = int(bad_rat.sum())
    print(f"[clean] Step 4: Age outside [{age_lo},{age_hi}]    -> set NaN: {report['age_clamped_to_nan']}")
    print(f"[clean] Step 4: Ratings outside [{rat_lo},{rat_hi}] -> set NaN: {report['ratings_clamped_to_nan']}")

    df.loc[bad_age, "Delivery_person_Age"]      = np.nan
    df.loc[bad_rat, "Delivery_person_Ratings"]  = np.nan

    # ==================================================================
    # Step 5 – drop rows missing required cols; fill unknowns; extract city
    # ==================================================================
    required = c["required_cols"]   # ["Time_Orderd", "Road_traffic_density", "Weatherconditions"]
    before = len(df)
    df = df.dropna(subset=required)
    report["missing_required"] = before - len(df)

    # Fill multiple_deliveries and Festival NaN -> "unknown"
    for col in c["fill_unknown_cols"]:
        if col == "multiple_deliveries":
            # It's Int8 nullable; convert to string for "unknown" fills
            df[col] = df[col].astype(object)
        df[col] = df[col].fillna("unknown")

    # Extract city from Delivery_person_ID prefix (e.g. "BANGRES18DEL02" -> "BANG")
    # Pattern: leading alpha characters before the first digit
    df["city"] = (df["Delivery_person_ID"]
                  .str.extract(r"^([A-Z]+)", expand=False)
                  .str.upper()
                  .str.strip())

    # Drop the original City tier column (ignored per spec)
    if "City" in df.columns:
        df.drop(columns=["City"], inplace=True)

    # ==================================================================
    # Step 6 – coordinates: bbox, haversine, speed flag
    # ==================================================================
    lat_min, lat_max = c["lat_min"], c["lat_max"]
    lon_min, lon_max = c["lon_min"], c["lon_max"]

    # 6a. Drop rows outside India bbox (catches 0/negative values)
    in_bbox = (
        df["Restaurant_latitude"].between(lat_min, lat_max) &
        df["Restaurant_longitude"].between(lon_min, lon_max) &
        df["Delivery_location_latitude"].between(lat_min, lat_max) &
        df["Delivery_location_longitude"].between(lon_min, lon_max)
    )
    before = len(df)
    df = df[in_bbox].copy()
    report["outside_bbox"] = before - len(df)

    # 6b. Haversine distance
    df["distance_km"] = _haversine_km(
        df["Restaurant_latitude"],   df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"],
    )

    # 6c. Distance gates
    min_d, max_d = c["min_distance_km"], c["max_distance_km"]
    before = len(df)
    df = df[df["distance_km"].between(min_d, max_d)].copy()
    report["bad_distance"] = before - len(df)

    # 6d. Flag implied speed > max_speed_kmh, show, then drop
    df["implied_speed_kmh"] = df["distance_km"] / (df["Time_taken_min"] / 60.0)
    speed_flag = df["implied_speed_kmh"] > c["max_speed_kmh"]
    report["high_speed_flagged"] = int(speed_flag.sum())
    if speed_flag.any():
        print(f"\n[clean] Step 6: {speed_flag.sum()} rows with implied speed "
              f"> {c['max_speed_kmh']} km/h (flagged before dropping):")
        print(df.loc[speed_flag,
                     ["ID", "distance_km", "Time_taken_min", "implied_speed_kmh"]
                     ].to_string(index=False))
        print()
    before = len(df)
    df = df[~speed_flag].copy()
    report["high_speed_dropped"] = before - len(df)

    # ==================================================================
    # Step 7 – drop Time_Order_picked (not used per spec)
    # ==================================================================
    if "Time_Order_picked" in df.columns:
        df.drop(columns=["Time_Order_picked"], inplace=True)

    # ==================================================================
    # Step 8 – vehicle category: kept as-is; note 'bicycle' is rare but valid.
    # The OrdinalEncoder/OneHotEncoder in features.py must use handle_unknown.
    # ==================================================================
    report["vehicle_counts"] = df["Type_of_vehicle"].value_counts().to_dict()

    # ==================================================================
    # Step 9 – final assertions
    # ==================================================================
    allowed_nan_cols = {"Delivery_person_Age", "Delivery_person_Ratings"}
    for col in df.columns:
        if col in allowed_nan_cols:
            continue
        n_null = df[col].isna().sum()
        assert n_null == 0, (
            f"Assertion failed: column '{col}' has {n_null} NaN values after cleaning."
        )

    assert df["Time_taken_min"].between(c["min_time_min"], c["max_time_min"]).all(), \
        "Assertion failed: Time_taken_min out of range."

    assert df["distance_km"].between(c["min_distance_km"], c["max_distance_km"]).all(), \
        "Assertion failed: distance_km out of range."

    assert df["implied_speed_kmh"].le(c["max_speed_kmh"]).all(), \
        "Assertion failed: implied_speed_kmh > max."

    assert df["ID"].is_unique, "Assertion failed: IDs are not unique."

    # ==================================================================
    # Wrap up report
    # ==================================================================
    rows_after = len(df)
    report["rows_after"]   = rows_after
    report["total_dropped"] = rows_before - rows_after
    report["missing_after"] = df.isnull().sum().to_dict()
    report["final_shape"]   = df.shape
    report["columns"]       = list(df.columns)

    return df.reset_index(drop=True), report


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def build_cleaning_report(report: dict, cfg: dict) -> str:
    """Format cleaning statistics as a Markdown document."""
    c   = cfg["cleaning"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    rb  = report["rows_before"]

    def pct(n: int) -> str:
        return f"{n:,}  ({n / rb * 100:.2f} %)"

    # Missingness table before
    missing_before_rows = "\n".join(
        f"| `{col}` | {cnt:,} |"
        for col, cnt in report["missing_before"].items()
        if cnt > 0
    ) or "| *(none)* | – |"

    # Missingness table after
    missing_after_rows = "\n".join(
        f"| `{col}` | {cnt:,} |"
        for col, cnt in report["missing_after"].items()
        if cnt > 0
    ) or "| *(none)* | – |"

    # Vehicle counts
    veh_rows = "\n".join(
        f"| `{v}` | {cnt:,} |"
        for v, cnt in sorted(report["vehicle_counts"].items(), key=lambda x: -x[1])
    )

    md = textwrap.dedent(f"""\
        # Data Cleaning Report – Food Delivery Dataset

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
        | Retention rate       | {report['rows_after'] / rb * 100:.2f} % |
        | Final shape          | {report['final_shape'][0]:,} rows × {report['final_shape'][1]} cols |

        ---

        ## Drop Breakdown (per rule)

        | Rule | Rows dropped | % of raw |
        |------|-------------|---------|
        | Exact duplicate rows          | {pct(report['dup_rows'])} |
        | Duplicate IDs                 | {pct(report['dup_ids'])} |
        | Missing required cols (Time_Orderd / traffic / weather) | {pct(report['missing_required'])} |
        | Outside India bbox (lat {c['lat_min']}–{c['lat_max']}, lon {c['lon_min']}–{c['lon_max']}) | {pct(report['outside_bbox'])} |
        | Distance < {c['min_distance_km']} km or > {c['max_distance_km']} km | {pct(report['bad_distance'])} |
        | Implied speed > {c['max_speed_kmh']} km/h | {pct(report['high_speed_dropped'])} |

        ---

        ## Value Clamping (set to NaN, rows kept)

        | Column | Count set to NaN |
        |--------|-----------------|
        | Age outside [{c['age_min']}, {c['age_max']}] | {report['age_clamped_to_nan']:,} |
        | Ratings outside [{c['rating_min']}, {c['rating_max']}] | {report['ratings_clamped_to_nan']:,} |

        ---

        ## Missingness Before Cleaning

        | Column | Null count |
        |--------|-----------|
        {missing_before_rows}

        ---

        ## Missingness After Cleaning

        | Column | Null count |
        |--------|-----------|
        {missing_after_rows}

        *(Only Age and Ratings are permitted to have NaN after cleaning.)*

        ---

        ## Vehicle Category Counts (after cleaning)

        | Vehicle | Count |
        |---------|-------|
        {veh_rows}

        > **Note:** `bicycle` is a rare but valid category (≈68 rows in raw data).
        > Feature encoders in `preprocessing/features.py` use `handle_unknown='use_encoded_value'`
        > (scikit-learn OrdinalEncoder) to handle unseen categories at inference time.

        ---

        ## Cleaning Thresholds Applied

        | Parameter | Value |
        |-----------|-------|
        | India lat range | [{c['lat_min']}, {c['lat_max']}] |
        | India lon range | [{c['lon_min']}, {c['lon_max']}] |
        | Min distance (km) | {c['min_distance_km']} |
        | Max distance (km) | {c['max_distance_km']} |
        | Max implied speed (km/h) | {c['max_speed_kmh']} |
        | Time_taken_min range | [{c['min_time_min']}, {c['max_time_min']}] |
        | Age valid range | [{c['age_min']}, {c['age_max']}] |
        | Ratings valid range | [{c['rating_min']}, {c['rating_max']}] |

        ---

        ## Final Columns

        ```
        {chr(10).join('  ' + c for c in report['columns'])}
        ```

        ---

        ## Limitations

        - The dataset contains delivery *legs*, not full multi-stop routes.
          Stops used in the optimizer are sampled points; legs are predicted by the ETA model.
        - NYC Taxi distance/speed norms do not apply here; India bounding box and
          80 km/h road-speed cap are used instead.
        - Age and Ratings are nullable (set to NaN if out of range); the ETA model
          must handle missingness in these columns (median imputation recommended).
    """)
    return md


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parent.parent
    config_path  = project_root / "configs" / "config.yaml"

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    raw_path    = project_root / cfg["paths"]["raw_trips"]
    clean_path  = project_root / cfg["paths"]["clean_trips"]
    report_path = project_root / "docs" / "cleaning_report.md"

    print(f"[clean] Loading raw data from: {raw_path}")
    df_raw = pd.read_csv(raw_path)
    print(f"[clean] Loaded {len(df_raw):,} rows, {df_raw.shape[1]} columns.")

    print("[clean] Applying cleaning rules ...")
    df_clean, report = clean_trips(df_raw, cfg)

    # --- console summary ---
    print("\n" + "=" * 60)
    print("  FOOD DELIVERY CLEANING REPORT")
    print("=" * 60)
    print(f"  Rows before   : {report['rows_before']:>10,}")
    print(f"  Rows after    : {report['rows_after']:>10,}")
    print(f"  Total dropped : {report['total_dropped']:>10,}")
    print("-" * 60)
    rules = [
        ("Dup rows",               report["dup_rows"]),
        ("Dup IDs",                report["dup_ids"]),
        ("Missing required cols",  report["missing_required"]),
        ("Outside India bbox",     report["outside_bbox"]),
        ("Bad distance",           report["bad_distance"]),
        ("High speed (dropped)",   report["high_speed_dropped"]),
    ]
    for label, n in rules:
        pct = n / report["rows_before"] * 100
        print(f"  {label:<30}: {n:>7,}  ({pct:.2f} %)")
    print("-" * 60)
    print(f"  Age clamped to NaN      : {report['age_clamped_to_nan']:>7,}")
    print(f"  Ratings clamped to NaN  : {report['ratings_clamped_to_nan']:>7,}")
    print("=" * 60)
    print(f"  Final shape: {report['final_shape']}")
    print("=" * 60)

    # --- save ---
    clean_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(clean_path, index=False)
    print(f"\n[clean] Saved cleaned data  -> {clean_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    md = build_cleaning_report(report, cfg)
    report_path.write_text(md, encoding="utf-8")
    print(f"[clean] Saved cleaning report -> {report_path}")
    print("\n[clean] Done.")
    sys.exit(0)
