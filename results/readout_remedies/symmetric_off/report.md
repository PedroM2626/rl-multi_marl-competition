=== symmetric_off: symmetric_off ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.378** | 0.187 | 567/1500 | [0.354, 0.403] | 2.24 | 21.25 | 87.22 | 1.368 | 0.0327 |
| CTE | **0.287** | 0.230 | 431/1500 | [0.265, 0.311] | 2.44 | 17.43 | 74.01 | 1.416 | 0.0376 |
| DTE | **0.319** | 0.180 | 479/1500 | [0.296, 0.343] | 1.70 | 18.97 | 78.77 | 1.384 | 0.0325 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.520 | 0.227 | 0.533 | 0.313 | 0.193 | 0.587 | 0.547 | 0.060 | 0.533 | 0.267 | 5/10 |
| CTE | 0.113 | 0.187 | 0.260 | 0.240 | 0.280 | 0.047 | 0.453 | 0.853 | 0.127 | 0.313 | 1/10 |
| DTE | 0.360 | 0.580 | 0.207 | 0.400 | 0.520 | 0.340 | 0.000 | 0.087 | 0.313 | 0.387 | 4/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.091 | 0.376 | +0.76 | 0.465 | 7/10 — not significant |
| CTDE vs DTE | +0.059 | 0.291 | +0.64 | 0.539 | 5/10 — not significant |
| CTE vs DTE | -0.032 | 0.367 | -0.28 | 0.789 | 3/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.091 | 0.376 | +0.241 | ~183 |
| CTDE vs DTE | +0.059 | 0.291 | +0.202 | ~260 |
| CTE vs DTE | -0.032 | 0.367 | -0.087 | ~1377 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 2.400, df = 2, p = 0.301. Mean rank (1 = best): CTDE 1.80, CTE 2.40, DTE 1.80

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.091 | 0.467 | not significant |
| CTDE vs DTE | +0.059 | 0.543 | not significant |
| CTE vs DTE | -0.032 | 0.797 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.24 [1.36, 3.13] | 21.2 [17.1, 25.7] | 87.2 [57.5, 120.6] | 0.0327 [0.0167, 0.0523] |
| CTE | 2.44 [1.76, 3.18] | 17.4 [14.9, 20.3] | 74.0 [55.3, 93.2] | 0.0376 [0.0259, 0.0511] |
| DTE | 1.70 [0.96, 2.56] | 19.0 [12.8, 25.7] | 78.8 [42.6, 114.3] | 0.0325 [0.0137, 0.0550] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 567/1500 vs 431/1500 | 1.51 | 0.0000 | **significant** |
| CTDE vs DTE | 567/1500 vs 479/1500 | 1.30 | 0.0009 | **significant** |
| CTE vs DTE | 431/1500 vs 479/1500 | 0.86 | 0.0619 | not significant |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.276** | 0.126 | 414/1500 | [0.254, 0.299] | 2.22 | 16.20 | 81.30 | 1.673 | 0.0278 |
| CTDE-Comm | **0.409** | 0.152 | 614/1500 | [0.385, 0.434] | 2.16 | 19.93 | 87.42 | 1.462 | 0.0309 |
| CTDE-VD | **0.299** | 0.151 | 448/1500 | [0.276, 0.322] | 2.05 | 17.37 | 76.04 | 1.459 | 0.0389 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.273 | 0.213 | 0.287 | 0.180 | 0.553 | 0.247 | 0.120 | 0.427 | 0.213 | 0.247 | 2/10 |
| CTDE-Comm | 0.520 | 0.573 | 0.507 | 0.300 | 0.213 | 0.547 | 0.307 | 0.253 | 0.600 | 0.273 | 5/10 |
| CTDE-VD | 0.180 | 0.193 | 0.193 | 0.507 | 0.220 | 0.187 | 0.567 | 0.320 | 0.180 | 0.440 | 3/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.133 | 0.235 | -1.80 | 0.106 | 2/10 — not significant |
| CTDE-CAC vs CTDE-VD | -0.023 | 0.232 | -0.31 | 0.764 | 7/10 — not significant |
| CTDE-Comm vs CTDE-VD | +0.111 | 0.276 | +1.27 | 0.236 | 5/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.133 | 0.235 | -0.568 | ~36 |
| CTDE-CAC vs CTDE-VD | -0.023 | 0.232 | -0.098 | ~1097 |
| CTDE-Comm vs CTDE-VD | +0.111 | 0.276 | +0.401 | ~68 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 1.400, df = 2, p = 0.497. Mean rank (1 = best): CTDE-CAC 2.10, CTDE-Comm 1.70, CTDE-VD 2.20

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.133 | 0.107 | not significant |
| CTDE-CAC vs CTDE-VD | -0.023 | 0.773 | not significant |
| CTDE-Comm vs CTDE-VD | +0.111 | 0.232 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 2.22 [1.56, 2.93] | 16.2 [13.5, 19.2] | 81.3 [65.8, 96.3] | 0.0278 [0.0210, 0.0351] |
| CTDE-Comm | 2.16 [1.40, 3.01] | 19.9 [14.1, 28.2] | 87.4 [67.0, 106.3] | 0.0309 [0.0176, 0.0460] |
| CTDE-VD | 2.05 [1.34, 2.80] | 17.4 [14.4, 20.9] | 76.0 [51.1, 97.6] | 0.0389 [0.0210, 0.0632] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 414/1500 vs 614/1500 | 0.55 | 0.0000 | **significant** |
| CTDE-CAC vs CTDE-VD | 414/1500 vs 448/1500 | 0.90 | 0.1830 | not significant |
| CTDE-Comm vs CTDE-VD | 614/1500 vs 448/1500 | 1.63 | 0.0000 | **significant** |

### symmetric_off vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.378 | -0.019 | -1.01 | 0.341 | 3/10 | 19.1 -> 21.2s | 2.43 -> 2.24 | 88 -> 87 | 1.54 -> 1.37 |
| exp1 | CTE | 0.265 | 0.287 | +0.023 | +0.31 | 0.762 | 3/10 | 16.1 -> 17.4s | 2.15 -> 2.44 | 71 -> 74 | 1.47 -> 1.42 |
| exp1 | DTE | 0.321 | 0.319 | -0.001 | -0.02 | 0.981 | 2/10 | 17.0 -> 19.0s | 1.91 -> 1.70 | 70 -> 79 | 1.36 -> 1.38 |
| exp2 | CTDE-CAC | 0.234 | 0.276 | +0.042 | +1.87 | 0.095 | 0/10 | 19.1 -> 16.2s | 1.82 -> 2.22 | 61 -> 81 | 1.07 -> 1.67 |
| exp2 | CTDE-Comm | 0.474 | 0.409 | -0.065 | -1.97 | 0.080 | 4/10 | 20.9 -> 19.9s | 2.60 -> 2.16 | 89 -> 87 | 1.42 -> 1.46 |
| exp2 | CTDE-VD | 0.261 | 0.299 | +0.037 | +1.81 | 0.103 | 0/10 | 17.9 -> 17.4s | 1.85 -> 2.05 | 71 -> 76 | 1.33 -> 1.46 |

wrote D:\rl-multi_marl-competition\results\readout_remedies\symmetric_off\coupled_tests.json
