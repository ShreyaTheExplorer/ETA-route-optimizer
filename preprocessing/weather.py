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


def load_weather(cfg: dict) -> pd.DataFrame:
    """
    Load weather from disk (downloading first if not present).
    Returns DataFrame with columns: pickup_hour, temperature_2m,
    precipitation, wind_speed_10m.
    """
    project_root = pathlib.Path(__file__).parent.parent
    weather_path = project_root / cfg["paths"]["raw_weather"]

    if not weather_path.exists():
        print(f"[weather] File not found at {weather_path}. Downloading ...")
        download_weather(weather_path)

    df = pd.read_csv(weather_path, parse_dates=["pickup_hour"])
    return df


# ---------------------------------------------------------------------------
# Inference helper: join weather for a series of datetimes
# ---------------------------------------------------------------------------

def get_weather(datetimes: pd.Series, cfg: dict) -> pd.DataFrame:
    """
    Return weather features aligned to the given datetimes.

    Strategy
    --------
    - Round each datetime down to the hour.
    - Look up in the downloaded hourly table (dataset range).
    - For hours outside the dataset, fall back to the (hour-of-day, month)
      median computed from the training data.

    Parameters
    ----------
    datetimes : pd.Series of datetime64 values (pickup_datetime)
    cfg       : full config.yaml dict

    Returns
    -------
    DataFrame with columns temperature_2m, precipitation, wind_speed_10m,
    same length and index as datetimes.
    """
    weather_df = load_weather(cfg)
    weather_df["pickup_hour"] = pd.to_datetime(weather_df["pickup_hour"])
    weather_df = weather_df.drop_duplicates(subset=["pickup_hour"])

    # Compute (hour, month) medians as fallback
    w_calc = weather_df.copy()
    w_calc["hour"]  = w_calc["pickup_hour"].dt.hour
    w_calc["month"] = w_calc["pickup_hour"].dt.month
    medians = (
        w_calc.groupby(["month", "hour"])[WEATHER_VARS]
        .median()
    )
    global_medians = weather_df[WEATHER_VARS].median()

    # Create lookup frame preserving original index
    query_df = pd.DataFrame({
        "pickup_hour": datetimes.dt.floor("h"),
        "month": datetimes.dt.month,
        "hour": datetimes.dt.hour,
    }, index=datetimes.index)

    merged = query_df.merge(
        weather_df[["pickup_hour"] + WEATHER_VARS],
        on="pickup_hour",
        how="left"
    )
    merged.index = query_df.index

    # Check for missing values to apply fallback
    if merged[WEATHER_VARS].isna().any().any():
        fallback = query_df[["month", "hour"]].merge(
            medians.reset_index(),
            on=["month", "hour"],
            how="left"
        )
        fallback.index = query_df.index
        for var in WEATHER_VARS:
            merged[var] = merged[var].fillna(fallback[var]).fillna(global_medians[var])

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

