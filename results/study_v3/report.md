=== v3 (every docs/10 remedy): study_v3 ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.374** | 0.105 | 561/1500 | [0.350, 0.399] | 2.46 | 14.40 | 52.55 | 1.216 | 0.0521 |
| CTE | **0.201** | 0.074 | 302/1500 | [0.182, 0.222] | 2.27 | 13.02 | 42.47 | 1.087 | 0.0611 |
| DTE | **0.405** | 0.153 | 608/1500 | [0.381, 0.430] | 1.99 | 18.48 | 45.19 | 0.815 | 0.0525 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.293 | 0.360 | 0.280 | 0.393 | 0.453 | 0.233 | 0.460 | 0.567 | 0.427 | 0.273 | 4/10 |
| CTE | 0.120 | 0.120 | 0.193 | 0.147 | 0.160 | 0.153 | 0.307 | 0.253 | 0.313 | 0.247 | 0/10 |
| DTE | 0.567 | 0.507 | 0.520 | 0.447 | 0.367 | 0.580 | 0.220 | 0.160 | 0.233 | 0.453 | 6/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.173 | 0.098 | +5.60 | 0.000 | 10/10 — **significant** |
| CTDE vs DTE | -0.031 | 0.251 | -0.39 | 0.702 | 4/10 — not significant |
| CTE vs DTE | -0.204 | 0.216 | -2.99 | 0.015 | 3/10 — **significant** |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.173 | 0.098 | +1.771 | ~6 |
| CTDE vs DTE | -0.031 | 0.251 | -0.125 | ~674 |
| CTE vs DTE | -0.204 | 0.216 | -0.945 | ~15 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 7.400, df = 2, p = 0.025. Mean rank (1 = best): CTDE 1.60, CTE 2.70, DTE 1.70

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.173 | 0.002 | **significant** |
| CTDE vs DTE | -0.031 | 0.697 | not significant |
| CTE vs DTE | -0.204 | 0.020 | nominal only |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.46 [2.25, 2.69] | 14.4 [11.8, 17.9] | 52.6 [42.6, 63.3] | 0.0521 [0.0407, 0.0655] |
| CTE | 2.27 [2.07, 2.47] | 13.0 [11.1, 15.1] | 42.5 [32.2, 54.7] | 0.0611 [0.0493, 0.0745] |
| DTE | 1.99 [1.64, 2.29] | 18.5 [15.3, 22.1] | 45.2 [33.9, 56.9] | 0.0525 [0.0370, 0.0704] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 561/1500 vs 302/1500 | 2.37 | 0.0000 | **significant** |
| CTDE vs DTE | 561/1500 vs 608/1500 | 0.88 | 0.0850 | not significant |
| CTE vs DTE | 302/1500 vs 608/1500 | 0.37 | 0.0000 | **significant** |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.295** | 0.084 | 442/1500 | [0.272, 0.318] | 2.29 | 14.04 | 64.37 | 1.528 | 0.0452 |
| CTDE-Comm | **0.299** | 0.054 | 448/1500 | [0.276, 0.322] | 1.97 | 13.14 | 57.25 | 1.453 | 0.0438 |
| CTDE-VD | **0.380** | 0.088 | 570/1500 | [0.356, 0.405] | 2.43 | 15.81 | 73.60 | 1.552 | 0.0364 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.407 | 0.353 | 0.360 | 0.253 | 0.200 | 0.153 | 0.333 | 0.213 | 0.320 | 0.353 | 3/10 |
| CTDE-Comm | 0.213 | 0.367 | 0.300 | 0.260 | 0.293 | 0.380 | 0.240 | 0.273 | 0.313 | 0.347 | 1/10 |
| CTDE-VD | 0.353 | 0.267 | 0.313 | 0.447 | 0.487 | 0.447 | 0.387 | 0.507 | 0.320 | 0.273 | 6/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.004 | 0.112 | -0.11 | 0.913 | 5/10 — not significant |
| CTDE-CAC vs CTDE-VD | -0.085 | 0.164 | -1.65 | 0.133 | 4/10 — not significant |
| CTDE-Comm vs CTDE-VD | -0.081 | 0.116 | -2.21 | 0.054 | 2/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.004 | 0.112 | -0.036 | ~8214 |
| CTDE-CAC vs CTDE-VD | -0.085 | 0.164 | -0.522 | ~42 |
| CTDE-Comm vs CTDE-VD | -0.081 | 0.116 | -0.699 | ~25 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 2.205, df = 2, p = 0.332. Mean rank (1 = best): CTDE-CAC 2.05, CTDE-Comm 2.30, CTDE-VD 1.65

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.004 | 0.902 | not significant |
| CTDE-CAC vs CTDE-VD | -0.085 | 0.133 | not significant |
| CTDE-Comm vs CTDE-VD | -0.081 | 0.061 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 2.29 [2.04, 2.54] | 14.0 [10.9, 17.4] | 64.4 [45.7, 82.7] | 0.0452 [0.0320, 0.0607] |
| CTDE-Comm | 1.97 [1.70, 2.22] | 13.1 [11.1, 15.1] | 57.2 [41.7, 73.4] | 0.0438 [0.0301, 0.0581] |
| CTDE-VD | 2.43 [2.18, 2.68] | 15.8 [13.1, 19.6] | 73.6 [60.8, 86.4] | 0.0364 [0.0284, 0.0455] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 442/1500 vs 448/1500 | 0.98 | 0.8416 | not significant |
| CTDE-CAC vs CTDE-VD | 442/1500 vs 570/1500 | 0.68 | 0.0000 | **significant** |
| CTDE-Comm vs CTDE-VD | 448/1500 vs 570/1500 | 0.69 | 0.0000 | **significant** |

### study_v3 vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.374 | -0.023 | -0.32 | 0.759 | 5/10 | 19.1 -> 14.4s | 2.43 -> 2.46 | 88 -> 53 | 1.54 -> 1.22 |
| exp1 | CTE | 0.265 | 0.201 | -0.063 | -1.36 | 0.207 | 6/10 | 16.1 -> 13.0s | 2.15 -> 2.27 | 71 -> 42 | 1.47 -> 1.09 |
| exp1 | DTE | 0.321 | 0.405 | +0.085 | +1.16 | 0.275 | 3/10 | 17.0 -> 18.5s | 1.91 -> 1.99 | 70 -> 45 | 1.36 -> 0.82 |
| exp2 | CTDE-CAC | 0.234 | 0.295 | +0.061 | +1.32 | 0.219 | 2/10 | 19.1 -> 14.0s | 1.82 -> 2.29 | 61 -> 64 | 1.07 -> 1.53 |
| exp2 | CTDE-Comm | 0.474 | 0.299 | -0.175 | -3.09 | 0.013 | 8/10 | 20.9 -> 13.1s | 2.60 -> 1.97 | 89 -> 57 | 1.42 -> 1.45 |
| exp2 | CTDE-VD | 0.261 | 0.380 | +0.119 | +1.96 | 0.081 | 3/10 | 17.9 -> 15.8s | 1.85 -> 2.43 | 71 -> 74 | 1.33 -> 1.55 |

wrote D:\rl-multi_marl-competition\results\study_v3\coupled_tests.json
