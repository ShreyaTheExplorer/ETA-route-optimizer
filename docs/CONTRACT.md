# Interface contracts (agreed in Stage 0; change only by team agreement)

## 1. ETA function (owner: A, file models/predict.py)
predict_minutes(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, departure: datetime) -> np.ndarray
- Accepts scalars or equal-length arrays. Returns predicted duration in MINUTES. Must be vectorised.

## 2. Time matrix (builder owner: B in api/matrix_builder.py; plain version: C in optimizer/graph.py)
- numpy float array, shape (n, n), minutes. Index 0 is the depot, indexes 1..n-1 are stops in request order.
- T[i][j] = time to go from i to j. Diagonal is 0. NOT assumed symmetric.
- Plain version for early work: haversine km / 30 km/h * 60.

## 3. Optimizer interface (owner: C, folder optimizer/)
solve(time_matrix: np.ndarray, start: int = 0) -> list[int]
- Returns a visiting order of ALL node indexes, beginning with `start`, each index exactly once. Open route (no return to depot).
- route_cost(time_matrix, route) -> float  (sum of consecutive legs) lives in optimizer/base.py, used by everyone.
- optimizer/registry.py: SOLVERS = {"hill_climbing": solve_fn, ...}. Adding Tabu/Beam = one new file + one registry line.

## 4. API (owner: B, api/main.py)
POST /optimize
Request:
{
  "depot": {"lat": 40.7580, "lon": -73.9855},
  "stops": [{"id": "s1", "lat": 40.7484, "lon": -73.9857}, ...],
  "departure_time": "2026-10-20T08:00:00",
  "algorithm": "hill_climbing"        // optional, default hill_climbing
}
Response:
{
  "algorithm": "hill_climbing",
  "route": ["depot", "s3", "s1", "s2"],
  "stops": [{"id": "s3", "lat": ..., "lon": ..., "leg_minutes": 6.2, "eta_minutes": 6.2, "arrival_time": "2026-10-20T08:06:12"}, ...],
  "total_minutes": 21.4,
  "baseline_route": ["depot", "s1", "s2", "s3"],
  "baseline_total_minutes": 30.9,
  "improvement_pct": 30.7
}
GET /health -> {"status": "ok"}
Errors: 422 for fewer than 1 stop, coordinates outside NYC box, malformed time. Duplicate stop ids are rejected; duplicate coordinates are allowed.
