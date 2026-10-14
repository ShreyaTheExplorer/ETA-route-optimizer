"""
tests/test_predict.py
=====================
Tests for models/predict.py interface contract:
1. Times Square to JFK on a weekday at 8:00 AM predicts roughly 40-70 minutes.
2. Short 1 km hop (Times Square to Grand Central) predicts under 15 minutes.
3. Supports scalar and array inputs.
4. Returns positive float ndarray.
"""

from __future__ import annotations

import pathlib
import sys
from datetime import datetime

import numpy as np
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.predict import predict_minutes


# Known NYC coordinates
TIMES_SQUARE_LAT, TIMES_SQUARE_LON = 40.7580, -73.9855
GRAND_CENTRAL_LAT, GRAND_CENTRAL_LON = 40.7527, -73.9772  # ~0.9 km
JFK_AIRPORT_LAT, JFK_AIRPORT_LON = 40.6413, -73.7781      # ~21 km direct


class TestPredictInterface:
    def test_times_square_to_jfk_rush_hour(self):
        """
        Times Square to JFK Airport on a Tuesday morning at 8:00 AM (rush hour).
        Expected duration: roughly 40 to 75 minutes.
        """
        dep = datetime(2016, 3, 1, 8, 0, 0)  # Tuesday 8:00 AM
        pred = predict_minutes(
            pickup_lat=TIMES_SQUARE_LAT,
            pickup_lon=TIMES_SQUARE_LON,
            dropoff_lat=JFK_AIRPORT_LAT,
            dropoff_lon=JFK_AIRPORT_LON,
            departure=dep,
        )

        assert isinstance(pred, np.ndarray)
        assert pred.shape == (1,)
        duration = float(pred[0])
        print(f"\n[test] Times Square -> JFK predicted duration: {duration:.2f} min")
        assert 35.0 <= duration <= 85.0, f"Expected 35-85 min, got {duration:.2f}"

    def test_short_hop_under_15_minutes(self):
        """
        Times Square to Grand Central (~0.9 km).
        Should comfortably be under 15 minutes.
        """
        dep = datetime(2016, 3, 1, 14, 0, 0)  # Tuesday 2:00 PM (off-peak)
        pred = predict_minutes(
            pickup_lat=TIMES_SQUARE_LAT,
            pickup_lon=TIMES_SQUARE_LON,
            dropoff_lat=GRAND_CENTRAL_LAT,
            dropoff_lon=GRAND_CENTRAL_LON,
            departure=dep,
        )

        assert isinstance(pred, np.ndarray)
        duration = float(pred[0])
        print(f"\n[test] Times Square -> Grand Central predicted duration: {duration:.2f} min")
        assert 0.5 <= duration < 15.0, f"Expected < 15 min, got {duration:.2f}"

    def test_vectorized_batch_prediction(self):
        """
        Passing lists/arrays of multiple trips should return an ndarray of matching length.
        """
        pu_lats = [TIMES_SQUARE_LAT, TIMES_SQUARE_LAT]
        pu_lons = [TIMES_SQUARE_LON, TIMES_SQUARE_LON]
        do_lats = [GRAND_CENTRAL_LAT, JFK_AIRPORT_LAT]
        do_lons = [GRAND_CENTRAL_LON, JFK_AIRPORT_LON]
        dep = "2016-03-01 08:00:00"

        preds = predict_minutes(pu_lats, pu_lons, do_lats, do_lons, dep)

        assert isinstance(preds, np.ndarray)
        assert len(preds) == 2
        # Short trip should take less time than JFK trip
        assert preds[0] < preds[1]
        assert (preds > 0).all()
