=== SPAWN_CYCLE_PER_MATCH=true on the JAX runtime: ablation_jax_spawn_cycle ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.346** | 0.030 | 519/1500 | [0.322, 0.370] | 2.11 | 24.22 | 130.95 | 1.802 | 0.0190 |
| CTE | **0.301** | 0.063 | 452/1500 | [0.279, 0.325] | 2.16 | 25.54 | 122.51 | 1.599 | 0.0201 |
| DTE | **0.321** | 0.086 | 482/1500 | [0.298, 0.345] | 1.63 | 26.40 | 117.44 | 1.483 | 0.0191 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.373 | 0.347 | 0.273 | 0.360 | 0.327 | 0.353 | 0.340 | 0.340 | 0.380 | 0.367 | 5/10 |
| CTE | 0.353 | 0.347 | 0.267 | 0.207 | 0.240 | 0.260 | 0.320 | 0.420 | 0.280 | 0.320 | 1/10 |
| DTE | 0.260 | 0.187 | 0.427 | 0.413 | 0.420 | 0.380 | 0.313 | 0.220 | 0.313 | 0.280 | 4/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.045 | 0.066 | +2.14 | 0.061 | 8/10 — not significant |
| CTDE vs DTE | +0.025 | 0.103 | +0.76 | 0.468 | 6/10 — not significant |
| CTE vs DTE | -0.020 | 0.145 | -0.44 | 0.672 | 5/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.045 | 0.066 | +0.677 | ~26 |
| CTDE vs DTE | +0.025 | 0.103 | +0.240 | ~185 |
| CTE vs DTE | -0.020 | 0.145 | -0.138 | ~551 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 3.436, df = 2, p = 0.179. Mean rank (1 = best): CTDE 1.55, CTE 2.35, DTE 2.10

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.045 | 0.059 | not significant |
| CTDE vs DTE | +0.025 | 0.469 | not significant |
| CTE vs DTE | -0.020 | 0.674 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.11 [1.93, 2.26] | 24.2 [18.7, 30.5] | 131.0 [105.7, 155.6] | 0.0190 [0.0139, 0.0259] |
| CTE | 2.16 [1.92, 2.38] | 25.5 [19.6, 32.2] | 122.5 [94.9, 156.8] | 0.0201 [0.0156, 0.0250] |
| DTE | 1.63 [1.25, 1.96] | 26.4 [21.4, 30.7] | 117.4 [93.8, 137.7] | 0.0191 [0.0107, 0.0321] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 519/1500 vs 452/1500 | 1.23 | 0.0100 | **significant** |
| CTDE vs DTE | 519/1500 vs 482/1500 | 1.12 | 0.1633 | not significant |
| CTE vs DTE | 452/1500 vs 482/1500 | 0.91 | 0.2528 | not significant |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.301** | 0.045 | 451/1500 | [0.278, 0.324] | 2.01 | 18.35 | 88.56 | 1.609 | 0.0270 |
| CTDE-Comm | **0.339** | 0.066 | 509/1500 | [0.316, 0.364] | 2.09 | 20.05 | 110.15 | 1.831 | 0.0206 |
| CTDE-VD | **0.335** | 0.084 | 502/1500 | [0.311, 0.359] | 2.16 | 19.44 | 108.77 | 1.865 | 0.0238 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.333 | 0.347 | 0.320 | 0.360 | 0.327 | 0.267 | 0.313 | 0.247 | 0.260 | 0.233 | 2/10 |
| CTDE-Comm | 0.273 | 0.433 | 0.380 | 0.333 | 0.307 | 0.347 | 0.380 | 0.227 | 0.293 | 0.420 | 5/10 |
| CTDE-VD | 0.380 | 0.200 | 0.293 | 0.293 | 0.313 | 0.333 | 0.273 | 0.500 | 0.427 | 0.333 | 3/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.039 | 0.073 | -1.67 | 0.129 | 4/10 — not significant |
| CTDE-CAC vs CTDE-VD | -0.034 | 0.118 | -0.91 | 0.387 | 5/10 — not significant |
| CTDE-Comm vs CTDE-VD | +0.005 | 0.144 | +0.10 | 0.920 | 6/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.039 | 0.073 | -0.529 | ~41 |
| CTDE-CAC vs CTDE-VD | -0.034 | 0.118 | -0.288 | ~130 |
| CTDE-Comm vs CTDE-VD | +0.005 | 0.144 | +0.033 | ~9894 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.600, df = 2, p = 0.741. Mean rank (1 = best): CTDE-CAC 2.10, CTDE-Comm 1.80, CTDE-VD 2.10

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.039 | 0.133 | not significant |
| CTDE-CAC vs CTDE-VD | -0.034 | 0.402 | not significant |
| CTDE-Comm vs CTDE-VD | +0.005 | 0.928 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 2.01 [1.80, 2.21] | 18.4 [15.5, 21.2] | 88.6 [67.1, 109.7] | 0.0270 [0.0200, 0.0349] |
| CTDE-Comm | 2.09 [1.90, 2.28] | 20.1 [16.6, 23.6] | 110.1 [92.8, 126.9] | 0.0207 [0.0164, 0.0252] |
| CTDE-VD | 2.16 [2.00, 2.30] | 19.4 [15.2, 23.9] | 108.8 [82.1, 136.8] | 0.0238 [0.0176, 0.0312] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 451/1500 vs 509/1500 | 0.84 | 0.0257 | nominal only |
| CTDE-CAC vs CTDE-VD | 451/1500 vs 502/1500 | 0.85 | 0.0499 | nominal only |
| CTDE-Comm vs CTDE-VD | 509/1500 vs 502/1500 | 1.02 | 0.8167 | not significant |

### ablation_jax_spawn_cycle vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.346 | -0.051 | -0.95 | 0.368 | 5/10 | 19.1 -> 24.2s | 2.43 -> 2.11 | 88 -> 131 | 1.54 -> 1.80 |
| exp1 | CTE | 0.265 | 0.301 | +0.037 | +0.69 | 0.509 | 4/10 | 16.1 -> 25.5s | 2.15 -> 2.16 | 71 -> 123 | 1.47 -> 1.60 |
| exp1 | DTE | 0.321 | 0.321 | +0.001 | +0.01 | 0.993 | 3/10 | 17.0 -> 26.4s | 1.91 -> 1.63 | 70 -> 117 | 1.36 -> 1.48 |
| exp2 | CTDE-CAC | 0.234 | 0.301 | +0.067 | +1.66 | 0.132 | 3/10 | 19.1 -> 18.4s | 1.82 -> 2.01 | 61 -> 89 | 1.07 -> 1.61 |
| exp2 | CTDE-Comm | 0.474 | 0.339 | -0.135 | -2.11 | 0.064 | 7/10 | 20.9 -> 20.1s | 2.60 -> 2.09 | 89 -> 110 | 1.42 -> 1.83 |
| exp2 | CTDE-VD | 0.261 | 0.335 | +0.073 | +1.15 | 0.281 | 3/10 | 17.9 -> 19.4s | 1.85 -> 2.16 | 71 -> 109 | 1.33 -> 1.86 |

wrote D:\rl-multi_marl-competition\results\ablation_jax_spawn_cycle\coupled_tests.json
