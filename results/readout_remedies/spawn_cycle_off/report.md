=== spawn_cycle_off: spawn_cycle_off ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.274** | 0.166 | 411/1500 | [0.252, 0.297] | 1.95 | 22.87 | 118.75 | 1.731 | 0.0205 |
| CTE | **0.336** | 0.243 | 504/1500 | [0.313, 0.360] | 2.41 | 24.33 | 122.54 | 1.679 | 0.0263 |
| DTE | **0.343** | 0.179 | 514/1500 | [0.319, 0.367] | 1.50 | 28.08 | 123.38 | 1.465 | 0.0161 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.040 | 0.187 | 0.407 | 0.280 | 0.253 | 0.600 | 0.167 | 0.160 | 0.200 | 0.447 | 3/10 |
| CTE | 0.873 | 0.327 | 0.160 | 0.413 | 0.100 | 0.107 | 0.560 | 0.407 | 0.140 | 0.273 | 5/10 |
| DTE | 0.067 | 0.213 | 0.400 | 0.293 | 0.627 | 0.287 | 0.253 | 0.407 | 0.633 | 0.247 | 2/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | -0.062 | 0.376 | -0.52 | 0.614 | 5/10 — not significant |
| CTDE vs DTE | -0.069 | 0.233 | -0.93 | 0.376 | 3/10 — not significant |
| CTE vs DTE | -0.007 | 0.392 | -0.05 | 0.958 | 5/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | -0.062 | 0.376 | -0.165 | ~388 |
| CTDE vs DTE | -0.069 | 0.233 | -0.295 | ~124 |
| CTE vs DTE | -0.007 | 0.392 | -0.017 | ~36146 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.667, df = 2, p = 0.717. Mean rank (1 = best): CTDE 2.20, CTE 1.95, DTE 1.85

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | -0.062 | 0.639 | not significant |
| CTDE vs DTE | -0.069 | 0.377 | not significant |
| CTE vs DTE | -0.007 | 0.969 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 1.95 [1.40, 2.61] | 22.9 [16.1, 31.1] | 118.7 [89.2, 149.7] | 0.0205 [0.0129, 0.0287] |
| CTE | 2.41 [1.71, 3.15] | 24.3 [17.7, 33.3] | 122.5 [82.1, 174.3] | 0.0263 [0.0160, 0.0379] |
| DTE | 1.50 [1.02, 2.12] | 28.1 [19.7, 36.8] | 123.4 [92.3, 153.5] | 0.0161 [0.0092, 0.0243] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 411/1500 vs 504/1500 | 0.75 | 0.0003 | **significant** |
| CTDE vs DTE | 411/1500 vs 514/1500 | 0.72 | 0.0001 | **significant** |
| CTE vs DTE | 504/1500 vs 514/1500 | 0.97 | 0.7286 | not significant |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.282** | 0.120 | 423/1500 | [0.260, 0.305] | 1.88 | 20.78 | 104.93 | 1.683 | 0.0253 |
| CTDE-Comm | **0.301** | 0.138 | 451/1500 | [0.278, 0.324] | 1.85 | 22.84 | 110.96 | 1.619 | 0.0226 |
| CTDE-VD | **0.379** | 0.202 | 569/1500 | [0.355, 0.404] | 2.38 | 21.29 | 122.70 | 1.921 | 0.0247 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.107 | 0.520 | 0.187 | 0.207 | 0.373 | 0.260 | 0.247 | 0.353 | 0.360 | 0.207 | 2/10 |
| CTDE-Comm | 0.120 | 0.160 | 0.520 | 0.267 | 0.340 | 0.433 | 0.240 | 0.380 | 0.413 | 0.133 | 4/10 |
| CTDE-VD | 0.753 | 0.307 | 0.287 | 0.527 | 0.267 | 0.140 | 0.433 | 0.240 | 0.200 | 0.640 | 4/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.019 | 0.177 | -0.33 | 0.746 | 4/10 — not significant |
| CTDE-CAC vs CTDE-VD | -0.097 | 0.292 | -1.05 | 0.320 | 5/10 — not significant |
| CTDE-Comm vs CTDE-VD | -0.079 | 0.322 | -0.77 | 0.460 | 5/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.019 | 0.177 | -0.106 | ~943 |
| CTDE-CAC vs CTDE-VD | -0.097 | 0.292 | -0.333 | ~97 |
| CTDE-Comm vs CTDE-VD | -0.079 | 0.322 | -0.244 | ~178 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.200, df = 2, p = 0.905. Mean rank (1 = best): CTDE-CAC 2.10, CTDE-Comm 1.90, CTDE-VD 2.00

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.019 | 0.740 | not significant |
| CTDE-CAC vs CTDE-VD | -0.097 | 0.324 | not significant |
| CTDE-Comm vs CTDE-VD | -0.079 | 0.455 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 1.88 [1.30, 2.52] | 20.8 [15.2, 27.2] | 104.9 [69.0, 148.7] | 0.0253 [0.0152, 0.0356] |
| CTDE-Comm | 1.85 [1.32, 2.46] | 22.8 [15.5, 31.3] | 111.0 [77.5, 149.2] | 0.0226 [0.0138, 0.0322] |
| CTDE-VD | 2.38 [1.66, 3.15] | 21.3 [16.6, 26.6] | 122.7 [94.4, 155.6] | 0.0247 [0.0147, 0.0356] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 423/1500 vs 451/1500 | 0.91 | 0.2780 | not significant |
| CTDE-CAC vs CTDE-VD | 423/1500 vs 569/1500 | 0.64 | 0.0000 | **significant** |
| CTDE-Comm vs CTDE-VD | 451/1500 vs 569/1500 | 0.70 | 0.0000 | **significant** |

### spawn_cycle_off vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.274 | -0.123 | -1.78 | 0.109 | 7/10 | 19.1 -> 22.9s | 2.43 -> 1.95 | 88 -> 119 | 1.54 -> 1.73 |
| exp1 | CTE | 0.265 | 0.336 | +0.071 | +0.86 | 0.414 | 4/10 | 16.1 -> 24.3s | 2.15 -> 2.41 | 71 -> 123 | 1.47 -> 1.68 |
| exp1 | DTE | 0.321 | 0.343 | +0.022 | +0.30 | 0.772 | 5/10 | 17.0 -> 28.1s | 1.91 -> 1.50 | 70 -> 123 | 1.36 -> 1.46 |
| exp2 | CTDE-CAC | 0.234 | 0.282 | +0.048 | +0.94 | 0.372 | 5/10 | 19.1 -> 20.8s | 1.82 -> 1.88 | 61 -> 105 | 1.07 -> 1.68 |
| exp2 | CTDE-Comm | 0.474 | 0.301 | -0.173 | -2.58 | 0.030 | 9/10 | 20.9 -> 22.8s | 2.60 -> 1.85 | 89 -> 111 | 1.42 -> 1.62 |
| exp2 | CTDE-VD | 0.261 | 0.379 | +0.118 | +1.83 | 0.100 | 2/10 | 17.9 -> 21.3s | 1.85 -> 2.38 | 71 -> 123 | 1.33 -> 1.92 |

wrote D:\rl-multi_marl-competition\results\readout_remedies\spawn_cycle_off\coupled_tests.json
