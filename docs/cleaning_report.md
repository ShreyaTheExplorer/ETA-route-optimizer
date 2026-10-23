        # Data Cleaning Report – Food Delivery Dataset

        **Generated:** 2026-10-23 16:41 UTC
        **Source:** `data/raw/train.csv`
        **Output:** `data/processed/cleaned.csv`

        ---

        ## Summary

        | Metric | Count |
        |--------|-------|
        | Rows before cleaning | 45,593 |
        | Rows after cleaning  | 39,909 |
        | **Total dropped**    | **5,684** |
        | Retention rate       | 87.53 % |
        | Final shape          | 39,909 rows × 22 cols |

        ---

        ## Drop Breakdown (per rule)

        | Rule | Rows dropped | % of raw |
        |------|-------------|---------|
        | Exact duplicate rows          | 0  (0.00 %) |
        | Duplicate IDs                 | 0  (0.00 %) |
        | Missing required cols (Time_Orderd / traffic / weather) | 1,731  (3.80 %) |
        | Outside India bbox (lat 6.0–37.0, lon 68.0–98.0) | 3,665  (8.04 %) |
        | Distance < 0.1 km or > 25.0 km | 0  (0.00 %) |
        | Implied speed > 80.0 km/h | 288  (0.63 %) |

        ---

        ## Value Clamping (set to NaN, rows kept)

        | Column | Count set to NaN |
        |--------|-----------------|
        | Age outside [18, 70] | 38 |
        | Ratings outside [1.0, 5.0] | 53 |

        ---

        ## Missingness Before Cleaning

        | Column | Null count |
        |--------|-----------|
        | `Delivery_person_Age` | 1,854 |
| `Delivery_person_Ratings` | 1,908 |
| `Time_Orderd` | 1,731 |
| `Weatherconditions` | 616 |
| `Road_traffic_density` | 601 |
| `multiple_deliveries` | 993 |
| `Festival` | 228 |
| `City` | 1,200 |

        ---

        ## Missingness After Cleaning

        | Column | Null count |
        |--------|-----------|
        | `Delivery_person_Age` | 200 |
| `Delivery_person_Ratings` | 239 |

        *(Only Age and Ratings are permitted to have NaN after cleaning.)*

        ---

        ## Vehicle Category Counts (after cleaning)

        | Vehicle | Count |
        |---------|-------|
        | `motorcycle` | 23,445 |
| `scooter` | 13,311 |
| `electric_scooter` | 3,153 |

        > **Note:** `bicycle` is a rare but valid category (≈68 rows in raw data).
        > Feature encoders in `preprocessing/features.py` use `handle_unknown='use_encoded_value'`
        > (scikit-learn OrdinalEncoder) to handle unseen categories at inference time.

        ---

        ## Cleaning Thresholds Applied

        | Parameter | Value |
        |-----------|-------|
        | India lat range | [6.0, 37.0] |
        | India lon range | [68.0, 98.0] |
        | Min distance (km) | 0.1 |
        | Max distance (km) | 25.0 |
        | Max implied speed (km/h) | 80.0 |
        | Time_taken_min range | [10, 54] |
        | Age valid range | [18, 70] |
        | Ratings valid range | [1.0, 5.0] |

        ---

        ## Final Columns

        ```
          ID
  Delivery_person_ID
  Delivery_person_Age
  Delivery_person_Ratings
  Restaurant_latitude
  Restaurant_longitude
  Delivery_location_latitude
  Delivery_location_longitude
  Order_Date
  Time_Orderd
  Weatherconditions
  Road_traffic_density
  Vehicle_condition
  Type_of_order
  Type_of_vehicle
  multiple_deliveries
  Festival
  Time_taken_min
  ordered_at
  city
  distance_km
  implied_speed_kmh
        ```

        ---

        ## Limitations

        - The dataset contains delivery *legs*, not full multi-stop routes.
          Stops used in the optimizer are sampled points; legs are predicted by the ETA model.
        - NYC Taxi distance/speed norms do not apply here; India bounding box and
          80 km/h road-speed cap are used instead.
        - Age and Ratings are nullable (set to NaN if out of range); the ETA model
          must handle missingness in these columns (median imputation recommended).
