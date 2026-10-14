"""
preprocessing/weather.py
========================
Downloads hourly NYC weather for Jan–Jun 2016 from Open-Meteo archive API
and saves it to cfg.paths.raw_weather.

Also exposes get_weather(datetimes, cfg) for joining weather at inference time,
with a median fallback when the requested datetime is outside the dataset range.

Run as a script:
    python -m preprocessing.weather
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import yaml


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

NYC_LAT = 40.71
NYC_LON = -74.01
START_DATE = "2016-01-01"
END_DATE   = "2016-06-30"
WEATHER_VARS = ["temperature_2m", "precipitation", "wind_speed_10m"]


def download_weather(save_path: pathlib.Path) -> pd.DataFrame:
    """
    Fetch hourly weather from Open-Meteo archive API and save as CSV.
    Raises RuntimeError if the request fails (no internet / API down).
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": NYC_LAT,
        "longitude": NYC_LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "America/New_York",
    }

    print(f"[weather] Requesting Open-Meteo archive: {START_DATE} to {END_DATE} ...")
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"[weather] Could not reach Open-Meteo API: {exc}\n"
            "Check your internet connection and try again."
        ) from exc

    data = resp.json()
    hourly = data["hourly"]
    df = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for var in WEATHER_VARS:
        df[var] = hourly[var]

    # Round to the start of the hour for easy joining
    df["pickup_hour"] = df["time"].dt.floor("h")
    df = df.drop(columns=["time"])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"[weather] Saved {len(df):,} rows -> {save_path}")
    return df


_weather_cache: pd.DataFrame | None = None
_medians_cache: tuple[pd.DataFrame, pd.Series] | None = None


def load_weather(cfg: dict | None = None) -> pd.DataFrame:
    """
    Load weather from models/artifacts/ (runtime) or data/raw/ (training/fallback).
    Results are cached in memory for zero-overhead inference.
    Returns DataFrame with columns: pickup_hour, temperature_2m, precipitation, wind_speed_10m.
    """
    global _weather_cache
    if _weather_cache is not None:
        return _weather_cache

    project_root = pathlib.Path(__file__).parent.parent

    # Check models/artifacts first so runtime/Docker has no dependency on data/
    candidates = []
    if cfg and "paths" in cfg:
        if "weather_artifact" in cfg["paths"]:
            candidates.append(project_root / cfg["paths"]["weather_artifact"])
        if "raw_weather" in cfg["paths"]:
            candidates.append(project_root / cfg["paths"]["raw_weather"])

    candidates.extend([
        project_root / "models" / "artifacts" / "weather_hourly.parquet",
        project_root / "models" / "artifacts" / "weather_nyc_2016.csv",
        project_root / "data" / "raw" / "weather_nyc_2016.csv",
    ])

    for path in candidates:
        if path.exists():
            if path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, parse_dates=["pickup_hour"])
            df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
            df = df.drop_duplicates(subset=["pickup_hour"]).reset_index(drop=True)
            _weather_cache = df
            return _weather_cache

    # If neither exists and config specifies raw_weather, download it
    raw_path = project_root / (cfg["paths"]["raw_weather"] if cfg else "data/raw/weather_nyc_2016.csv")
    print(f"[weather] Weather data not found. Downloading to {raw_path} ...")
    df = download_weather(raw_path)
    _weather_cache = df
    return _weather_cache


# ---------------------------------------------------------------------------
# Inference helper: join weather for a series of datetimes
# ---------------------------------------------------------------------------

def get_weather(datetimes: pd.Series, cfg: dict | None = None) -> pd.DataFrame:
    """
    Return weather features aligned to the given datetimes.
    Purely vectorised with no Python loops.
    """
    global _medians_cache
    weather_df = load_weather(cfg)

    if _medians_cache is None:
        w_calc = weather_df.copy()
        w_calc["hour"]  = w_calc["pickup_hour"].dt.hour
        w_calc["month"] = w_calc["pickup_hour"].dt.month
        medians = w_calc.groupby(["month", "hour"])[WEATHER_VARS].median()
        global_medians = weather_df[WEATHER_VARS].median()
        _medians_cache = (medians, global_medians)
    else:
        medians, global_medians = _medians_cache

    pickup_hours = datetimes.dt.floor("h")

    # Fast path: all datetimes share the same hour (common in route matrix inference)
    if len(pickup_hours) > 0 and (pickup_hours.iloc[0] == pickup_hours).all():
        single_h = pickup_hours.iloc[0]
        match = weather_df[weather_df["pickup_hour"] == single_h]
        if not match.empty:
            vals = match[WEATHER_VARS].values[0]
        else:
            key = (single_h.month, single_h.hour)
            if key in medians.index:
                vals = medians.loc[key].values
            else:
                vals = global_medians.values

        # Broadcast vectorised across all rows
        broadcasted = np.tile(vals, (len(datetimes), 1))
        return pd.DataFrame(broadcasted, columns=WEATHER_VARS, index=datetimes.index)

    # General path: multiple distinct datetimes (vectorised merge, no Python loops)
    query_df = pd.DataFrame({
        "pickup_hour": pickup_hours,
        "month": datetimes.dt.month,
        "hour": datetimes.dt.hour,
    }, index=datetimes.index)

    merged = query_df.merge(
        weather_df[["pickup_hour"] + WEATHER_VARS],
        on="pickup_hour",
        how="left"
    )
    merged.index = query_df.index

    if merged[WEATHER_VARS].isna().any().any():
        fallback = query_df[["month", "hour"]].merge(
            medians.reset_index(),
            on=["month", "hour"],
            how="left"
        )
        fallback.index = query_df.index
        merged[WEATHER_VARS] = merged[WEATHER_VARS].fillna(fallback[WEATHER_VARS]).fillna(global_medians)

    return merged[WEATHER_VARS]



# ---------------------------------------------------------------------------
# __main__ block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    project_root = pathlib.Path(__file__).parent.parent
    config_path = project_root / "configs" / "config.yaml"

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    weather_path = project_root / cfg["paths"]["raw_weather"]

    try:
        df = download_weather(weather_path)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(df.head())
    print(f"\n[weather] Done. Shape: {df.shape}")

