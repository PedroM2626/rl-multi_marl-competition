=== ARENA_LAYOUT=symmetric on the JAX runtime: ablation_jax_symmetric ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.374** | 0.099 | 561/1500 | [0.350, 0.399] | 2.11 | 20.44 | 105.40 | 1.719 | 0.0204 |
| CTE | **0.243** | 0.084 | 364/1500 | [0.222, 0.265] | 2.08 | 16.80 | 93.99 | 1.865 | 0.0249 |
| DTE | **0.345** | 0.074 | 517/1500 | [0.321, 0.369] | 2.20 | 20.69 | 116.05 | 1.869 | 0.0198 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.320 | 0.540 | 0.360 | 0.267 | 0.540 | 0.407 | 0.293 | 0.407 | 0.293 | 0.313 | 4/10 |
| CTE | 0.287 | 0.093 | 0.380 | 0.153 | 0.187 | 0.267 | 0.280 | 0.227 | 0.327 | 0.227 | 1/10 |
| DTE | 0.367 | 0.340 | 0.240 | 0.507 | 0.267 | 0.300 | 0.380 | 0.327 | 0.340 | 0.380 | 5/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.131 | 0.159 | +2.61 | 0.028 | 8/10 — nominal only |
| CTDE vs DTE | +0.029 | 0.154 | +0.60 | 0.561 | 5/10 — not significant |
| CTE vs DTE | -0.102 | 0.133 | -2.43 | 0.038 | 1/10 — nominal only |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.131 | 0.159 | +0.827 | ~18 |
| CTDE vs DTE | +0.029 | 0.154 | +0.191 | ~290 |
| CTE vs DTE | -0.102 | 0.133 | -0.768 | ~21 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 7.400, df = 2, p = 0.025. Mean rank (1 = best): CTDE 1.70, CTE 2.70, DTE 1.60

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.131 | 0.020 | nominal only |
| CTDE vs DTE | +0.029 | 0.568 | not significant |
| CTE vs DTE | -0.102 | 0.037 | nominal only |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.11 [1.76, 2.41] | 20.4 [17.5, 23.6] | 105.4 [91.9, 118.5] | 0.0204 [0.0169, 0.0243] |
| CTE | 2.08 [1.92, 2.25] | 16.8 [13.5, 20.0] | 94.0 [73.7, 114.9] | 0.0249 [0.0200, 0.0301] |
| DTE | 2.20 [2.02, 2.37] | 20.7 [17.6, 24.1] | 116.0 [102.3, 130.1] | 0.0198 [0.0169, 0.0231] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 561/1500 vs 364/1500 | 1.86 | 0.0000 | **significant** |
| CTDE vs DTE | 561/1500 vs 517/1500 | 1.14 | 0.1018 | not significant |
| CTE vs DTE | 364/1500 vs 517/1500 | 0.61 | 0.0000 | **significant** |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.333** | 0.064 | 499/1500 | [0.309, 0.357] | 2.38 | 18.72 | 97.67 | 1.739 | 0.0289 |
| CTDE-Comm | **0.356** | 0.110 | 534/1500 | [0.332, 0.381] | 2.27 | 17.95 | 101.41 | 1.883 | 0.0275 |
| CTDE-VD | **0.272** | 0.103 | 408/1500 | [0.250, 0.295] | 1.71 | 20.69 | 81.01 | 1.305 | 0.0334 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.353 | 0.400 | 0.327 | 0.333 | 0.307 | 0.360 | 0.233 | 0.260 | 0.300 | 0.453 | 3/10 |
| CTDE-Comm | 0.340 | 0.580 | 0.460 | 0.280 | 0.360 | 0.227 | 0.407 | 0.387 | 0.300 | 0.220 | 5/10 |
| CTDE-VD | 0.280 | 0.013 | 0.193 | 0.327 | 0.307 | 0.367 | 0.327 | 0.287 | 0.353 | 0.267 | 2/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.023 | 0.137 | -0.54 | 0.603 | 4/10 — not significant |
| CTDE-CAC vs CTDE-VD | +0.061 | 0.143 | +1.34 | 0.212 | 5/10 — not significant |
| CTDE-Comm vs CTDE-VD | +0.084 | 0.203 | +1.31 | 0.223 | 6/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.023 | 0.137 | -0.170 | ~363 |
| CTDE-CAC vs CTDE-VD | +0.061 | 0.143 | +0.425 | ~61 |
| CTDE-Comm vs CTDE-VD | +0.084 | 0.203 | +0.414 | ~64 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.474, df = 2, p = 0.789. Mean rank (1 = best): CTDE-CAC 2.00, CTDE-Comm 1.85, CTDE-VD 2.15

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.023 | 0.617 | not significant |
| CTDE-CAC vs CTDE-VD | +0.061 | 0.230 | not significant |
| CTDE-Comm vs CTDE-VD | +0.084 | 0.252 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 2.38 [2.06, 2.84] | 18.7 [15.7, 21.4] | 97.7 [77.2, 117.4] | 0.0289 [0.0200, 0.0424] |
| CTDE-Comm | 2.27 [2.05, 2.50] | 18.0 [14.4, 21.2] | 101.4 [79.4, 120.8] | 0.0275 [0.0189, 0.0406] |
| CTDE-VD | 1.71 [1.19, 2.16] | 20.7 [15.4, 25.4] | 81.0 [48.3, 109.6] | 0.0334 [0.0197, 0.0550] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 499/1500 vs 534/1500 | 0.90 | 0.1914 | not significant |
| CTDE-CAC vs CTDE-VD | 499/1500 vs 408/1500 | 1.33 | 0.0003 | **significant** |
| CTDE-Comm vs CTDE-VD | 534/1500 vs 408/1500 | 1.48 | 0.0000 | **significant** |

### ablation_jax_symmetric vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.374 | -0.023 | -0.31 | 0.761 | 6/10 | 19.1 -> 20.4s | 2.43 -> 2.11 | 88 -> 105 | 1.54 -> 1.72 |
| exp1 | CTE | 0.265 | 0.243 | -0.022 | -0.39 | 0.705 | 5/10 | 16.1 -> 16.8s | 2.15 -> 2.08 | 71 -> 94 | 1.47 -> 1.86 |
| exp1 | DTE | 0.321 | 0.345 | +0.024 | +0.36 | 0.725 | 4/10 | 17.0 -> 20.7s | 1.91 -> 2.20 | 70 -> 116 | 1.36 -> 1.87 |
| exp2 | CTDE-CAC | 0.234 | 0.333 | +0.099 | +2.23 | 0.053 | 2/10 | 19.1 -> 18.7s | 1.82 -> 2.38 | 61 -> 98 | 1.07 -> 1.74 |
| exp2 | CTDE-Comm | 0.474 | 0.356 | -0.118 | -1.70 | 0.123 | 8/10 | 20.9 -> 18.0s | 2.60 -> 2.27 | 89 -> 101 | 1.42 -> 1.88 |
| exp2 | CTDE-VD | 0.261 | 0.272 | +0.011 | +0.20 | 0.849 | 4/10 | 17.9 -> 20.7s | 1.85 -> 1.71 | 71 -> 81 | 1.33 -> 1.31 |

wrote D:\rl-multi_marl-competition\results\ablation_jax_symmetric\coupled_tests.json
