---
description: Person C - graph, plain matrix, hill climbing with swap and 2-opt
---
1. Read AGENTS.md and docs/CONTRACT.md.
2. Create optimizer/base.py: route_cost(T, route), random_route(n, start, rng).
3. Create optimizer/graph.py: plain_time_matrix(coords) (haversine km / 30 km/h * 60), matrix_to_adjacency_list(T), and a short docstring explaining why a matrix gives O(1) lookups.
4. Create optimizer/hill_climbing.py: solve(time_matrix, start=0, seed=None, max_iters=10000) -> list[int].
   Start from a random order. Neighbors = every pairwise swap and every 2-opt segment reversal (start node fixed). Move to the BEST improving neighbor; stop when none improves.
   Also provide hill_climbing_trace(...) returning the cost after each iteration for plotting.
5. Create optimizer/registry.py with SOLVERS = {"hill_climbing": solve}.
6. Tests: tests/test_optimizer.py. (a) result is a permutation starting with start, (b) cost never increases between iterations, (c) n=1 and n=2 work, (d) 5 nodes: compare against brute force over all 4! orders and report the gap.
7. Run and show me the results.
