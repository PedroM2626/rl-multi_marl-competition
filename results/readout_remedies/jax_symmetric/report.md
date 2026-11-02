=== jax_symmetric: jax_symmetric ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.387** | 0.115 | 580/1500 | [0.362, 0.412] | 2.30 | 18.93 | 103.09 | 1.815 | 0.0245 |
| CTE | **0.215** | 0.089 | 322/1500 | [0.195, 0.236] | 1.92 | 15.07 | 83.15 | 1.839 | 0.0256 |
| DTE | **0.362** | 0.094 | 543/1500 | [0.338, 0.387] | 2.26 | 19.62 | 111.53 | 1.895 | 0.0224 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.300 | 0.593 | 0.360 | 0.267 | 0.540 | 0.407 | 0.293 | 0.480 | 0.333 | 0.293 | 4/10 |
| CTE | 0.200 | 0.147 | 0.380 | 0.153 | 0.187 | 0.267 | 0.280 | 0.067 | 0.180 | 0.287 | 1/10 |
| DTE | 0.453 | 0.247 | 0.240 | 0.507 | 0.267 | 0.300 | 0.380 | 0.420 | 0.440 | 0.367 | 5/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.172 | 0.172 | +3.16 | 0.011 | 9/10 — **significant** |
| CTDE vs DTE | +0.025 | 0.190 | +0.41 | 0.691 | 5/10 — not significant |
| CTE vs DTE | -0.147 | 0.156 | -2.99 | 0.015 | 1/10 — **significant** |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.172 | 0.172 | +1.001 | ~14 |
| CTDE vs DTE | +0.025 | 0.190 | +0.130 | ~622 |
| CTE vs DTE | -0.147 | 0.156 | -0.947 | ~15 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 9.600, df = 2, p = 0.008. Mean rank (1 = best): CTDE 1.60, CTE 2.80, DTE 1.60

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.172 | 0.010 | **significant** |
| CTDE vs DTE | +0.025 | 0.695 | not significant |
| CTE vs DTE | -0.147 | 0.021 | nominal only |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.30 [2.05, 2.54] | 18.9 [16.5, 21.9] | 103.1 [88.3, 115.1] | 0.0245 [0.0188, 0.0324] |
| CTE | 1.92 [1.69, 2.13] | 15.1 [12.5, 17.6] | 83.2 [65.6, 101.2] | 0.0256 [0.0205, 0.0310] |
| DTE | 2.26 [2.09, 2.42] | 19.6 [15.9, 23.4] | 111.5 [92.2, 127.6] | 0.0224 [0.0179, 0.0284] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 580/1500 vs 322/1500 | 2.31 | 0.0000 | **significant** |
| CTDE vs DTE | 580/1500 vs 543/1500 | 1.11 | 0.1744 | not significant |
| CTE vs DTE | 322/1500 vs 543/1500 | 0.48 | 0.0000 | **significant** |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.299** | 0.143 | 449/1500 | [0.277, 0.323] | 1.82 | 18.00 | 72.69 | 1.346 | 0.0315 |
| CTDE-Comm | **0.368** | 0.152 | 552/1500 | [0.344, 0.393] | 2.50 | 16.58 | 88.95 | 1.788 | 0.0377 |
| CTDE-VD | **0.298** | 0.078 | 447/1500 | [0.275, 0.322] | 2.03 | 20.96 | 71.11 | 1.131 | 0.0556 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.353 | 0.473 | 0.113 | 0.333 | 0.433 | 0.040 | 0.233 | 0.260 | 0.300 | 0.453 | 5/10 |
| CTDE-Comm | 0.340 | 0.273 | 0.727 | 0.280 | 0.247 | 0.500 | 0.407 | 0.387 | 0.300 | 0.220 | 4/10 |
| CTDE-VD | 0.280 | 0.220 | 0.160 | 0.327 | 0.307 | 0.453 | 0.327 | 0.287 | 0.353 | 0.267 | 1/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.069 | 0.283 | -0.77 | 0.462 | 5/10 — not significant |
| CTDE-CAC vs CTDE-VD | +0.001 | 0.184 | +0.02 | 0.983 | 5/10 — not significant |
| CTDE-Comm vs CTDE-VD | +0.070 | 0.185 | +1.20 | 0.262 | 6/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.069 | 0.283 | -0.243 | ~180 |
| CTDE-CAC vs CTDE-VD | +0.001 | 0.184 | +0.007 | ~210311 |
| CTDE-Comm vs CTDE-VD | +0.070 | 0.185 | +0.378 | ~76 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.154, df = 2, p = 0.926. Mean rank (1 = best): CTDE-CAC 1.95, CTDE-Comm 1.95, CTDE-VD 2.10

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.069 | 0.477 | not significant |
| CTDE-CAC vs CTDE-VD | +0.001 | 0.990 | not significant |
| CTDE-Comm vs CTDE-VD | +0.070 | 0.268 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 1.82 [1.32, 2.21] | 18.0 [14.0, 21.7] | 72.7 [43.9, 102.2] | 0.0315 [0.0182, 0.0494] |
| CTDE-Comm | 2.50 [2.08, 3.06] | 16.6 [13.0, 20.1] | 88.9 [68.1, 110.3] | 0.0377 [0.0218, 0.0599] |
| CTDE-VD | 2.03 [1.60, 2.47] | 21.0 [17.1, 24.7] | 71.1 [41.8, 101.6] | 0.0556 [0.0304, 0.0821] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 449/1500 vs 552/1500 | 0.73 | 0.0001 | **significant** |
| CTDE-CAC vs CTDE-VD | 449/1500 vs 447/1500 | 1.01 | 0.9682 | not significant |
| CTDE-Comm vs CTDE-VD | 552/1500 vs 447/1500 | 1.37 | 0.0001 | **significant** |

### jax_symmetric vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.387 | -0.010 | -0.13 | 0.898 | 6/10 | 19.1 -> 18.9s | 2.43 -> 2.30 | 88 -> 103 | 1.54 -> 1.82 |
| exp1 | CTE | 0.265 | 0.215 | -0.050 | -0.99 | 0.350 | 6/10 | 16.1 -> 15.1s | 2.15 -> 1.92 | 71 -> 83 | 1.47 -> 1.84 |
| exp1 | DTE | 0.321 | 0.362 | +0.041 | +0.58 | 0.579 | 4/10 | 17.0 -> 19.6s | 1.91 -> 2.26 | 70 -> 112 | 1.36 -> 1.90 |
| exp2 | CTDE-CAC | 0.234 | 0.299 | +0.065 | +1.32 | 0.220 | 3/10 | 19.1 -> 18.0s | 1.82 -> 1.82 | 61 -> 73 | 1.07 -> 1.35 |
| exp2 | CTDE-Comm | 0.474 | 0.368 | -0.106 | -1.61 | 0.143 | 7/10 | 20.9 -> 16.6s | 2.60 -> 2.50 | 89 -> 89 | 1.42 -> 1.79 |
| exp2 | CTDE-VD | 0.261 | 0.298 | +0.037 | +0.58 | 0.573 | 4/10 | 17.9 -> 21.0s | 1.85 -> 2.03 | 71 -> 71 | 1.33 -> 1.13 |

wrote D:\rl-multi_marl-competition\results\readout_remedies\jax_symmetric\coupled_tests.json
