=== jax_on: jax_on ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.413** | 0.084 | 620/1500 | [0.389, 0.438] | 2.25 | 24.39 | 105.01 | 1.435 | 0.0229 |
| CTE | **0.246** | 0.097 | 369/1500 | [0.225, 0.268] | 1.96 | 21.35 | 103.78 | 1.620 | 0.0230 |
| DTE | **0.297** | 0.109 | 445/1500 | [0.274, 0.320] | 1.80 | 22.90 | 104.09 | 1.515 | 0.0186 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.527 | 0.453 | 0.427 | 0.367 | 0.347 | 0.373 | 0.560 | 0.420 | 0.380 | 0.280 | 6/10 |
| CTE | 0.127 | 0.180 | 0.413 | 0.193 | 0.233 | 0.227 | 0.340 | 0.227 | 0.147 | 0.373 | 1/10 |
| DTE | 0.333 | 0.340 | 0.153 | 0.307 | 0.367 | 0.373 | 0.087 | 0.333 | 0.447 | 0.227 | 3/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.167 | 0.137 | +3.87 | 0.004 | 9/10 — **significant** |
| CTDE vs DTE | +0.117 | 0.160 | +2.30 | 0.047 | 7/10 — nominal only |
| CTE vs DTE | -0.051 | 0.197 | -0.81 | 0.437 | 3/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.167 | 0.137 | +1.224 | ~10 |
| CTDE vs DTE | +0.117 | 0.160 | +0.727 | ~23 |
| CTE vs DTE | -0.051 | 0.197 | -0.257 | ~161 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 8.051, df = 2, p = 0.018. Mean rank (1 = best): CTDE 1.35, CTE 2.60, DTE 2.05

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.167 | 0.006 | **significant** |
| CTDE vs DTE | +0.117 | 0.035 | nominal only |
| CTE vs DTE | -0.051 | 0.422 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.25 [1.88, 2.56] | 24.4 [19.6, 29.7] | 105.0 [86.7, 129.0] | 0.0229 [0.0182, 0.0278] |
| CTE | 1.96 [1.72, 2.27] | 21.4 [16.6, 26.7] | 103.8 [74.4, 138.9] | 0.0230 [0.0172, 0.0286] |
| DTE | 1.80 [1.43, 2.11] | 22.9 [18.2, 28.8] | 104.1 [80.4, 128.8] | 0.0186 [0.0150, 0.0225] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 620/1500 vs 369/1500 | 2.16 | 0.0000 | **significant** |
| CTDE vs DTE | 620/1500 vs 445/1500 | 1.67 | 0.0000 | **significant** |
| CTE vs DTE | 369/1500 vs 445/1500 | 0.77 | 0.0021 | **significant** |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.286** | 0.115 | 429/1500 | [0.264, 0.309] | 1.94 | 23.07 | 87.53 | 1.265 | 0.0240 |
| CTDE-Comm | **0.403** | 0.075 | 604/1500 | [0.378, 0.428] | 2.35 | 25.09 | 109.29 | 1.452 | 0.0220 |
| CTDE-VD | **0.261** | 0.100 | 391/1500 | [0.239, 0.283] | 1.65 | 24.15 | 100.35 | 1.385 | 0.0204 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.240 | 0.440 | 0.273 | 0.327 | 0.320 | 0.067 | 0.153 | 0.273 | 0.333 | 0.433 | 2/10 |
| CTDE-Comm | 0.400 | 0.427 | 0.493 | 0.373 | 0.333 | 0.540 | 0.413 | 0.400 | 0.373 | 0.273 | 7/10 |
| CTDE-VD | 0.213 | 0.040 | 0.220 | 0.273 | 0.327 | 0.273 | 0.433 | 0.307 | 0.240 | 0.280 | 1/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.117 | 0.175 | -2.11 | 0.064 | 2/10 — not significant |
| CTDE-CAC vs CTDE-VD | +0.025 | 0.187 | +0.43 | 0.678 | 6/10 — not significant |
| CTDE-Comm vs CTDE-VD | +0.142 | 0.136 | +3.31 | 0.009 | 8/10 — **significant** |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.117 | 0.175 | -0.667 | ~27 |
| CTDE-CAC vs CTDE-VD | +0.025 | 0.187 | +0.136 | ~573 |
| CTDE-Comm vs CTDE-VD | +0.142 | 0.136 | +1.047 | ~13 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 5.600, df = 2, p = 0.061. Mean rank (1 = best): CTDE-CAC 2.20, CTDE-Comm 1.40, CTDE-VD 2.40

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.117 | 0.055 | not significant |
| CTDE-CAC vs CTDE-VD | +0.025 | 0.691 | not significant |
| CTDE-Comm vs CTDE-VD | +0.142 | 0.014 | **significant** |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 1.94 [1.43, 2.32] | 23.1 [18.2, 28.3] | 87.5 [56.8, 123.0] | 0.0240 [0.0161, 0.0326] |
| CTDE-Comm | 2.35 [1.96, 2.67] | 25.1 [19.8, 31.2] | 109.3 [96.7, 125.1] | 0.0220 [0.0183, 0.0257] |
| CTDE-VD | 1.65 [1.28, 1.96] | 24.1 [19.4, 29.1] | 100.3 [66.1, 138.3] | 0.0204 [0.0159, 0.0247] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 429/1500 vs 604/1500 | 0.59 | 0.0000 | **significant** |
| CTDE-CAC vs CTDE-VD | 429/1500 vs 391/1500 | 1.14 | 0.1296 | not significant |
| CTDE-Comm vs CTDE-VD | 604/1500 vs 391/1500 | 1.91 | 0.0000 | **significant** |

### jax_on vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.413 | +0.017 | +0.35 | 0.733 | 3/10 | 19.1 -> 24.4s | 2.43 -> 2.25 | 88 -> 105 | 1.54 -> 1.43 |
| exp1 | CTE | 0.265 | 0.246 | -0.019 | -0.46 | 0.658 | 7/10 | 16.1 -> 21.4s | 2.15 -> 1.96 | 71 -> 104 | 1.47 -> 1.62 |
| exp1 | DTE | 0.321 | 0.297 | -0.024 | -0.52 | 0.618 | 5/10 | 17.0 -> 22.9s | 1.91 -> 1.80 | 70 -> 104 | 1.36 -> 1.52 |
| exp2 | CTDE-CAC | 0.234 | 0.286 | +0.052 | +1.22 | 0.253 | 5/10 | 19.1 -> 23.1s | 1.82 -> 1.94 | 61 -> 88 | 1.07 -> 1.26 |
| exp2 | CTDE-Comm | 0.474 | 0.403 | -0.071 | -1.40 | 0.194 | 6/10 | 20.9 -> 25.1s | 2.60 -> 2.35 | 89 -> 109 | 1.42 -> 1.45 |
| exp2 | CTDE-VD | 0.261 | 0.261 | -0.001 | -0.02 | 0.988 | 4/10 | 17.9 -> 24.1s | 1.85 -> 1.65 | 71 -> 100 | 1.33 -> 1.39 |

wrote D:\rl-multi_marl-competition\results\readout_remedies\jax_on\coupled_tests.json
