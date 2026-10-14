---
description: Step 2 - weather download/join and feature engineering
---
1. Create preprocessing/weather.py: download hourly NYC 2016-01-01..2016-06-30 weather (temperature_2m, precipitation, wind_speed_10m)
   from the Open-Meteo archive API (lat 40.71, lon -74.01) and save to cfg.paths.raw_weather. Add get_weather(datetimes) with the
   median fallback described in AGENTS.md. If there is no internet, tell me and stop.
2. Create preprocessing/features.py with build_features(df, zone_model=None, fit_zones=False):
   haversine_km, manhattan_km, hour, weekday, is_weekend, month, is_rush_hour (cfg rush_hours, weekdays only),
   hour sin/cos, pickup_zone and dropoff_zone (KMeans on coordinates, n_zones from config; fit only on training data),
   bearing, weather columns joined on the pickup hour.
3. Save zone KMeans with joblib and the final column order to feature_columns.json.
4. Save engineered table to cfg.paths.features. Print df.describe() and the correlation of every feature with duration_min.
5. Add tests/test_features.py (haversine of a known pair, rush flag on a known timestamp).
