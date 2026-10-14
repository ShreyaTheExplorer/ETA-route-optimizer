---
description: Person C - run from many random starts and show the local-optima weakness
---
1. Create evaluation/local_optima_experiment.py (inside optimizer/ if you may not edit evaluation/; ask me).
2. For n = 8, 12, 20 sampled stops (fixed seed, plain matrix), run hill climbing from 100 random starts each.
3. For n=8 also compute the brute-force optimum. For larger n record the best cost found.
4. Save: a CSV of final costs per run; a histogram of final costs per n with the best cost marked; a table with min, mean, max, std, number of distinct local optima, % of runs that reached the best cost, mean runtime.
5. Write a 5-line plain-English conclusion that motivates Tabu and Beam Search. Save outputs to docs/report/.
