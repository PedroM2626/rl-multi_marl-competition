=== JAX replication: study_jax ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.397** | 0.166 | 595/1500 | [0.372, 0.422] | 2.43 | 19.15 | 88.38 | 1.538 | 0.0361 |
| CTE | **0.265** | 0.165 | 397/1500 | [0.243, 0.288] | 2.15 | 16.07 | 70.90 | 1.471 | 0.0340 |
| DTE | **0.321** | 0.187 | 481/1500 | [0.298, 0.345] | 1.91 | 17.00 | 69.58 | 1.365 | 0.0310 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.480 | 0.287 | 0.533 | 0.313 | 0.193 | 0.587 | 0.547 | 0.213 | 0.593 | 0.220 | 5/10 |
| CTE | 0.247 | 0.093 | 0.260 | 0.240 | 0.280 | 0.047 | 0.453 | 0.253 | 0.160 | 0.613 | 1/10 |
| DTE | 0.253 | 0.593 | 0.207 | 0.400 | 0.520 | 0.340 | 0.000 | 0.513 | 0.227 | 0.153 | 4/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.132 | 0.269 | +1.55 | 0.155 | 7/10 — not significant |
| CTDE vs DTE | +0.076 | 0.316 | +0.76 | 0.466 | 6/10 — not significant |
| CTE vs DTE | -0.056 | 0.313 | -0.57 | 0.585 | 3/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.132 | 0.269 | +0.491 | ~47 |
| CTDE vs DTE | +0.076 | 0.316 | +0.241 | ~184 |
| CTE vs DTE | -0.056 | 0.313 | -0.179 | ~329 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 2.600, df = 2, p = 0.273. Mean rank (1 = best): CTDE 1.70, CTE 2.40, DTE 1.90

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.132 | 0.154 | not significant |
| CTDE vs DTE | +0.076 | 0.477 | not significant |
| CTE vs DTE | -0.056 | 0.570 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.43 [1.69, 3.22] | 19.1 [15.5, 23.1] | 88.4 [65.5, 111.2] | 0.0361 [0.0207, 0.0551] |
| CTE | 2.15 [1.53, 2.82] | 16.1 [13.9, 18.8] | 70.9 [56.0, 85.6] | 0.0340 [0.0222, 0.0481] |
| DTE | 1.91 [1.08, 2.84] | 17.0 [11.6, 23.3] | 69.6 [45.5, 91.7] | 0.0310 [0.0147, 0.0523] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 595/1500 vs 397/1500 | 1.83 | 0.0000 | **significant** |
| CTDE vs DTE | 595/1500 vs 481/1500 | 1.39 | 0.0000 | **significant** |
| CTE vs DTE | 397/1500 vs 481/1500 | 0.76 | 0.0009 | **significant** |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.234** | 0.111 | 351/1500 | [0.213, 0.256] | 1.82 | 19.09 | 61.19 | 1.068 | 0.0340 |
| CTDE-Comm | **0.474** | 0.205 | 711/1500 | [0.449, 0.499] | 2.60 | 20.89 | 88.95 | 1.419 | 0.0334 |
| CTDE-VD | **0.261** | 0.189 | 392/1500 | [0.240, 0.284] | 1.85 | 17.89 | 71.45 | 1.332 | 0.0269 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.273 | 0.140 | 0.280 | 0.180 | 0.380 | 0.080 | 0.120 | 0.427 | 0.213 | 0.247 | 2/10 |
| CTDE-Comm | 0.520 | 0.700 | 0.560 | 0.300 | 0.373 | 0.853 | 0.307 | 0.253 | 0.600 | 0.273 | 5/10 |
| CTDE-VD | 0.180 | 0.013 | 0.133 | 0.507 | 0.213 | 0.060 | 0.567 | 0.320 | 0.180 | 0.440 | 3/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.240 | 0.280 | -2.71 | 0.024 | 2/10 — nominal only |
| CTDE-CAC vs CTDE-VD | -0.027 | 0.217 | -0.40 | 0.699 | 7/10 — not significant |
| CTDE-Comm vs CTDE-VD | +0.213 | 0.379 | +1.78 | 0.110 | 6/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.240 | 0.280 | -0.858 | ~17 |
| CTDE-CAC vs CTDE-VD | -0.027 | 0.217 | -0.126 | ~662 |
| CTDE-Comm vs CTDE-VD | +0.213 | 0.379 | +0.562 | ~36 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 2.600, df = 2, p = 0.273. Mean rank (1 = best): CTDE-CAC 2.10, CTDE-Comm 1.60, CTDE-VD 2.30

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.240 | 0.020 | nominal only |
| CTDE-CAC vs CTDE-VD | -0.027 | 0.701 | not significant |
| CTDE-Comm vs CTDE-VD | +0.213 | 0.117 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 1.82 [1.05, 2.59] | 19.1 [13.0, 28.6] | 61.2 [36.8, 85.3] | 0.0340 [0.0194, 0.0499] |
| CTDE-Comm | 2.60 [1.80, 3.48] | 20.9 [14.8, 30.2] | 89.0 [69.3, 108.0] | 0.0334 [0.0212, 0.0472] |
| CTDE-VD | 1.85 [1.03, 2.70] | 17.9 [14.6, 22.6] | 71.4 [49.5, 91.2] | 0.0269 [0.0167, 0.0373] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 351/1500 vs 711/1500 | 0.34 | 0.0000 | **significant** |
| CTDE-CAC vs CTDE-VD | 351/1500 vs 392/1500 | 0.86 | 0.0906 | not significant |
| CTDE-Comm vs CTDE-VD | 711/1500 vs 392/1500 | 2.55 | 0.0000 | **significant** |

### study_jax vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.326 | 0.397 | +0.071 | +1.03 | 0.332 | 4/10 | 18.4 -> 19.1s | 1.97 -> 2.43 | 101 -> 88 | 1.84 -> 1.54 |
| exp1 | CTE | 0.252 | 0.265 | +0.013 | +0.21 | 0.840 | 4/10 | 22.7 -> 16.1s | 2.22 -> 2.15 | 122 -> 71 | 1.79 -> 1.47 |
| exp1 | DTE | 0.381 | 0.321 | -0.060 | -0.98 | 0.351 | 6/10 | 25.0 -> 17.0s | 1.95 -> 1.91 | 105 -> 70 | 1.40 -> 1.36 |
| exp2 | CTDE-CAC | 0.282 | 0.234 | -0.048 | -1.48 | 0.173 | 5/10 | 21.5 -> 19.1s | 1.98 -> 1.82 | 112 -> 61 | 1.74 -> 1.07 |
| exp2 | CTDE-Comm | 0.331 | 0.474 | +0.143 | +2.17 | 0.058 | 1/10 | 19.2 -> 20.9s | 2.18 -> 2.60 | 112 -> 89 | 1.94 -> 1.42 |
| exp2 | CTDE-VD | 0.367 | 0.261 | -0.106 | -1.79 | 0.107 | 8/10 | 23.1 -> 17.9s | 1.97 -> 1.85 | 93 -> 71 | 1.34 -> 1.33 |

wrote D:\rl-multi_marl-competition\results\study_jax\coupled_tests.json
