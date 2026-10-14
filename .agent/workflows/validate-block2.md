---
description: Person B validates Block 2 (C's optimizer)
---
Do NOT modify C's code. Report in docs/validation_block2.md with PASS/FAIL per check.
1. Run tests/test_optimizer.py.
2. Brute force: for 5 nodes (depot + 4 stops) enumerate all orders with itertools.permutations, find the true optimum, and compare with hill climbing over 50 random starts. Report how many starts hit the true optimum.
3. Hand calculation: print a 4-node matrix and one returned route, compute its cost by hand leg by leg, and confirm route_cost matches.
4. Properties: route is a permutation, starts at the depot, has no repeats; cost is never worse than the random starting route; n=1 and n=2 do not crash; same seed gives the same route.
5. Asymmetric matrix: with T[i][j] != T[j][i] check that the reported cost still equals the direction-correct sum.
6. Confirm the optimizer does not import models/ or call predict_minutes (the ETA model must not be inside the search loop). Time one solve on a 30-node matrix.
