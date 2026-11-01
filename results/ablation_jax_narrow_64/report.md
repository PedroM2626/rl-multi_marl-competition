=== narrow_64 on the JAX runtime: ablation_jax_narrow_64 ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.403** | 0.232 | 604/1500 | [0.378, 0.428] | 2.27 | 16.43 | 86.23 | 1.750 | 0.0287 |
| CTE | **0.245** | 0.135 | 367/1500 | [0.224, 0.267] | 2.12 | 13.79 | 66.61 | 1.610 | 0.0326 |
| DTE | **0.338** | 0.222 | 507/1500 | [0.315, 0.362] | 2.14 | 15.87 | 74.52 | 1.565 | 0.0365 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.400 | 0.187 | 0.567 | 0.313 | 0.153 | 0.727 | 0.453 | 0.107 | 0.780 | 0.340 | 4/10 |
| CTE | 0.427 | 0.187 | 0.253 | 0.433 | 0.213 | 0.173 | 0.127 | 0.200 | 0.033 | 0.400 | 3/10 |
| DTE | 0.153 | 0.613 | 0.173 | 0.253 | 0.627 | 0.080 | 0.373 | 0.680 | 0.187 | 0.240 | 3/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.158 | 0.307 | +1.63 | 0.138 | 4/10 — not significant |
| CTDE vs DTE | +0.065 | 0.434 | +0.47 | 0.649 | 7/10 — not significant |
| CTE vs DTE | -0.093 | 0.284 | -1.04 | 0.327 | 5/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.158 | 0.307 | +0.514 | ~43 |
| CTDE vs DTE | +0.065 | 0.434 | +0.149 | ~475 |
| CTE vs DTE | -0.093 | 0.284 | -0.328 | ~100 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.667, df = 2, p = 0.717. Mean rank (1 = best): CTDE 1.85, CTE 1.95, DTE 2.20

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.158 | 0.141 | not significant |
| CTDE vs DTE | +0.065 | 0.643 | not significant |
| CTE vs DTE | -0.093 | 0.324 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.27 [1.42, 3.21] | 16.4 [14.1, 18.9] | 86.2 [69.5, 104.0] | 0.0287 [0.0172, 0.0416] |
| CTE | 2.12 [1.36, 2.88] | 13.8 [11.7, 15.8] | 66.6 [50.1, 80.8] | 0.0326 [0.0201, 0.0477] |
| DTE | 2.14 [1.27, 3.08] | 15.9 [13.2, 18.5] | 74.5 [47.6, 100.4] | 0.0365 [0.0238, 0.0506] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 604/1500 vs 367/1500 | 2.08 | 0.0000 | **significant** |
| CTDE vs DTE | 604/1500 vs 507/1500 | 1.32 | 0.0003 | **significant** |
| CTE vs DTE | 367/1500 vs 507/1500 | 0.63 | 0.0000 | **significant** |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.292** | 0.183 | 438/1500 | [0.270, 0.316] | 1.96 | 22.52 | 123.30 | 1.825 | 0.0196 |
| CTDE-Comm | **0.354** | 0.193 | 531/1500 | [0.330, 0.379] | 2.20 | 23.21 | 135.38 | 1.944 | 0.0229 |
| CTDE-VD | **0.305** | 0.146 | 457/1500 | [0.282, 0.328] | 1.81 | 23.74 | 90.20 | 1.266 | 0.0291 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.153 | 0.540 | 0.213 | 0.100 | 0.493 | 0.227 | 0.173 | 0.613 | 0.200 | 0.207 | 3/10 |
| CTDE-Comm | 0.440 | 0.140 | 0.640 | 0.287 | 0.220 | 0.620 | 0.293 | 0.147 | 0.547 | 0.207 | 5/10 |
| CTDE-VD | 0.373 | 0.293 | 0.133 | 0.273 | 0.267 | 0.140 | 0.513 | 0.220 | 0.253 | 0.580 | 2/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.062 | 0.334 | -0.59 | 0.571 | 3/10 — not significant |
| CTDE-CAC vs CTDE-VD | -0.013 | 0.261 | -0.15 | 0.882 | 5/10 — not significant |
| CTDE-Comm vs CTDE-VD | +0.049 | 0.293 | +0.53 | 0.607 | 5/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.062 | 0.334 | -0.186 | ~306 |
| CTDE-CAC vs CTDE-VD | -0.013 | 0.261 | -0.048 | ~4468 |
| CTDE-Comm vs CTDE-VD | +0.049 | 0.293 | +0.169 | ~372 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.462, df = 2, p = 0.794. Mean rank (1 = best): CTDE-CAC 2.15, CTDE-Comm 1.85, CTDE-VD 2.00

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.062 | 0.547 | not significant |
| CTDE-CAC vs CTDE-VD | -0.013 | 0.887 | not significant |
| CTDE-Comm vs CTDE-VD | +0.049 | 0.598 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 1.96 [1.24, 2.80] | 22.5 [15.6, 32.5] | 123.3 [85.7, 183.3] | 0.0196 [0.0128, 0.0267] |
| CTDE-Comm | 2.20 [1.38, 3.09] | 23.2 [16.5, 34.6] | 135.4 [92.9, 205.2] | 0.0229 [0.0125, 0.0344] |
| CTDE-VD | 1.81 [1.09, 2.61] | 23.7 [15.9, 36.7] | 90.2 [67.3, 109.1] | 0.0291 [0.0141, 0.0476] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 438/1500 vs 531/1500 | 0.75 | 0.0003 | **significant** |
| CTDE-CAC vs CTDE-VD | 438/1500 vs 457/1500 | 0.94 | 0.4726 | not significant |
| CTDE-Comm vs CTDE-VD | 531/1500 vs 457/1500 | 1.25 | 0.0046 | **significant** |

### ablation_jax_narrow_64 vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.403 | +0.006 | +0.17 | 0.866 | 5/10 | 19.1 -> 16.4s | 2.43 -> 2.27 | 88 -> 86 | 1.54 -> 1.75 |
| exp1 | CTE | 0.265 | 0.245 | -0.020 | -0.37 | 0.721 | 6/10 | 16.1 -> 13.8s | 2.15 -> 2.12 | 71 -> 67 | 1.47 -> 1.61 |
| exp1 | DTE | 0.321 | 0.338 | +0.017 | +0.31 | 0.765 | 5/10 | 17.0 -> 15.9s | 1.91 -> 2.14 | 70 -> 75 | 1.36 -> 1.56 |
| exp2 | CTDE-CAC | 0.234 | 0.292 | +0.058 | +1.16 | 0.276 | 5/10 | 19.1 -> 22.5s | 1.82 -> 1.96 | 61 -> 123 | 1.07 -> 1.82 |
| exp2 | CTDE-Comm | 0.474 | 0.354 | -0.120 | -2.15 | 0.060 | 9/10 | 20.9 -> 23.2s | 2.60 -> 2.20 | 89 -> 135 | 1.42 -> 1.94 |
| exp2 | CTDE-VD | 0.261 | 0.305 | +0.043 | +0.92 | 0.380 | 3/10 | 17.9 -> 23.7s | 1.85 -> 1.81 | 71 -> 90 | 1.33 -> 1.27 |
