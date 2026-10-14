# Data Cleaning Report

**Generated:** 2026-10-14 23:25 UTC
**Source:** `data/raw/train.csv`
**Output:** `data/processed/trips_clean.parquet`

---

## Summary

| Metric | Count |
|--------|-------|
| Rows before cleaning | 1,458,644 |
| Rows after cleaning  | 1,439,246 |
| **Total dropped**    | **19,398** |
| Retention rate       | 98.67 % |

---

## Drop Breakdown (per rule)

| Rule | Rows Dropped | % of Raw |
|------|-------------|---------|
| Nulls | 0  (0.00 %) |
| Exact duplicates | 0  (0.00 %) |
| Zero / near-zero distance (< 0.1 km) | 13,318  (0.91 %) |
| Outside NYC bounding box | 999  (0.07 %) |
| Trip duration outside [1, 120] min | 5,034  (0.35 %) |
| Impossible speed (> 100 km/h) | 34  (0.00 %) |
| Invalid passenger count (0 or > 6) | 13  (0.00 %) |

---

## Cleaning Rules Applied

| Parameter | Value |
|-----------|-------|
| NYC lat range | [40.55, 40.92] |
| NYC lon range | [-74.27, -73.68] |
| Min duration (min) | 1 |
| Max duration (min) | 120 |
| Min distance (km)  | 0.1 |
| Max speed (km/h)   | 100 |
| Max passengers     | 6 |

---

## Notes

- **Zero-distance trips:** Trips with haversine distance < 0.1 km are
  treated as GPS glitches or test rides and dropped.
- **Long trips:** Trips exceeding 120 minutes (2 hours) are likely
  data errors or very rare outliers; excluded to keep the model focused on realistic
  urban trips.
- **Outside NYC:** Both pickup AND dropoff must be within the bounding box
  lat in [40.55, 40.92], lon in [-74.27, -73.68].
- **Impossible speeds:** Implied speed = distance_km / (duration_min / 60).
  Values > 100 km/h are physically impossible for NYC traffic and
  indicate GPS or timestamp errors.
