---
description: Step 1 - load, clean and save the NYC taxi trips
---
1. Read configs/config.yaml and AGENTS.md.
2. Create preprocessing/clean.py with a function clean_trips(df, cfg) that: parses datetimes, converts trip_duration seconds to duration_min,
   drops rows outside the NYC bounding box, duration outside [min,max], haversine distance < min_distance_km,
   implied speed > max_speed_kmh, passenger_count == 0 or > max_passengers, duplicates and nulls.
3. Add a __main__ block that reads cfg.paths.raw_trips, cleans, saves cfg.paths.clean_trips (parquet).
4. Print a cleaning report: rows before, rows dropped per rule, rows after. Save it to docs/cleaning_report.md.
5. Write tests/test_clean.py using a tiny hand-made DataFrame containing one bad row per rule.
6. Run it and show me the report.
