---
trigger: glob
globs: optimizer/**
---
Block 2 (Person C). Optimizers only see a numpy time matrix, never lat/lon and never the ETA model.
Every solver has the signature solve(time_matrix, start=0) -> list[int] and is registered in optimizer/registry.py.
Node `start` stays fixed in first position. Use a seeded random.Random/np.random.default_rng for reproducibility.
Recompute full route cost with route_cost() for every candidate (matrix may be asymmetric, so reversing a segment changes its cost).
Never add new dependencies for the optimizer; numpy and the standard library are enough.
