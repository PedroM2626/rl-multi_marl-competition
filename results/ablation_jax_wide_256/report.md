=== wide_256 on the JAX runtime: ablation_jax_wide_256 ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.359** | 0.197 | 538/1500 | [0.335, 0.383] | 2.26 | 23.83 | 104.42 | 1.460 | 0.0424 |
| CTE | **0.262** | 0.191 | 393/1500 | [0.240, 0.285] | 2.27 | 21.83 | 107.08 | 1.635 | 0.0327 |
| DTE | **0.345** | 0.192 | 517/1500 | [0.321, 0.369] | 1.36 | 27.19 | 87.30 | 1.070 | 0.0175 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.380 | 0.240 | 0.600 | 0.373 | 0.073 | 0.193 | 0.180 | 0.360 | 0.707 | 0.480 | 3/10 |
| CTE | 0.387 | 0.173 | 0.187 | 0.513 | 0.253 | 0.327 | 0.593 | 0.047 | 0.093 | 0.047 | 3/10 |
| DTE | 0.233 | 0.373 | 0.207 | 0.100 | 0.660 | 0.467 | 0.220 | 0.593 | 0.147 | 0.447 | 4/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.097 | 0.331 | +0.92 | 0.380 | 5/10 — not significant |
| CTDE vs DTE | +0.014 | 0.343 | +0.13 | 0.900 | 5/10 — not significant |
| CTE vs DTE | -0.083 | 0.325 | -0.80 | 0.442 | 3/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.097 | 0.331 | +0.292 | ~126 |
| CTDE vs DTE | +0.014 | 0.343 | +0.041 | ~6279 |
| CTE vs DTE | -0.083 | 0.325 | -0.254 | ~165 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.800, df = 2, p = 0.670. Mean rank (1 = best): CTDE 2.00, CTE 2.20, DTE 1.80

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.097 | 0.379 | not significant |
| CTDE vs DTE | +0.014 | 0.906 | not significant |
| CTE vs DTE | -0.083 | 0.414 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.26 [1.67, 2.92] | 23.8 [14.8, 34.9] | 104.4 [57.2, 168.6] | 0.0423 [0.0206, 0.0707] |
| CTE | 2.27 [1.45, 3.11] | 21.8 [14.3, 30.8] | 107.1 [61.2, 165.3] | 0.0327 [0.0176, 0.0483] |
| DTE | 1.36 [0.48, 2.37] | 27.2 [18.5, 38.2] | 87.3 [61.8, 113.3] | 0.0175 [0.0059, 0.0337] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 538/1500 vs 393/1500 | 1.58 | 0.0000 | **significant** |
| CTDE vs DTE | 538/1500 vs 517/1500 | 1.06 | 0.4444 | not significant |
| CTE vs DTE | 393/1500 vs 517/1500 | 0.68 | 0.0000 | **significant** |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.378** | 0.286 | 567/1500 | [0.354, 0.403] | 2.25 | 18.40 | 98.98 | 1.794 | 0.0783 |
| CTDE-Comm | **0.258** | 0.194 | 387/1500 | [0.236, 0.281] | 1.78 | 20.51 | 101.70 | 1.653 | 0.0221 |
| CTDE-VD | **0.351** | 0.254 | 527/1500 | [0.328, 0.376] | 2.29 | 17.59 | 96.22 | 1.823 | 0.0289 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.147 | 0.467 | 0.733 | 0.853 | 0.020 | 0.173 | 0.233 | 0.673 | 0.287 | 0.193 | 4/10 |
| CTDE-Comm | 0.267 | 0.320 | 0.047 | 0.087 | 0.220 | 0.640 | 0.007 | 0.180 | 0.433 | 0.380 | 2/10 |
| CTDE-VD | 0.573 | 0.200 | 0.207 | 0.060 | 0.740 | 0.187 | 0.760 | 0.133 | 0.253 | 0.400 | 4/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | +0.120 | 0.416 | +0.91 | 0.386 | 5/10 — not significant |
| CTDE-CAC vs CTDE-VD | +0.027 | 0.504 | +0.17 | 0.871 | 5/10 — not significant |
| CTDE-Comm vs CTDE-VD | -0.093 | 0.354 | -0.83 | 0.426 | 5/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | +0.120 | 0.416 | +0.288 | ~129 |
| CTDE-CAC vs CTDE-VD | +0.027 | 0.504 | +0.053 | ~3740 |
| CTDE-Comm vs CTDE-VD | -0.093 | 0.354 | -0.264 | ~154 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 0.000, df = 2, p = 1.000. Mean rank (1 = best): CTDE-CAC 2.00, CTDE-Comm 2.00, CTDE-VD 2.00

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | +0.120 | 0.387 | not significant |
| CTDE-CAC vs CTDE-VD | +0.027 | 0.871 | not significant |
| CTDE-Comm vs CTDE-VD | -0.093 | 0.439 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 2.25 [1.42, 3.08] | 18.4 [13.0, 26.6] | 99.0 [59.2, 153.4] | 0.0783 [0.0205, 0.1800] |
| CTDE-Comm | 1.78 [1.05, 2.58] | 20.5 [14.1, 27.4] | 101.7 [64.1, 145.9] | 0.0221 [0.0136, 0.0322] |
| CTDE-VD | 2.29 [1.49, 3.13] | 17.6 [12.8, 24.1] | 96.2 [64.0, 138.7] | 0.0289 [0.0178, 0.0410] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 567/1500 vs 387/1500 | 1.75 | 0.0000 | **significant** |
| CTDE-CAC vs CTDE-VD | 567/1500 vs 527/1500 | 1.12 | 0.1390 | not significant |
| CTDE-Comm vs CTDE-VD | 387/1500 vs 527/1500 | 0.64 | 0.0000 | **significant** |

### ablation_jax_wide_256 vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.359 | -0.038 | -0.56 | 0.588 | 5/10 | 19.1 -> 23.8s | 2.43 -> 2.26 | 88 -> 104 | 1.54 -> 1.46 |
| exp1 | CTE | 0.265 | 0.262 | -0.003 | -0.03 | 0.974 | 5/10 | 16.1 -> 21.8s | 2.15 -> 2.27 | 71 -> 107 | 1.47 -> 1.64 |
| exp1 | DTE | 0.321 | 0.345 | +0.024 | +0.41 | 0.695 | 4/10 | 17.0 -> 27.2s | 1.91 -> 1.36 | 70 -> 87 | 1.36 -> 1.07 |
| exp2 | CTDE-CAC | 0.234 | 0.378 | +0.144 | +1.53 | 0.160 | 3/10 | 19.1 -> 18.4s | 1.82 -> 2.25 | 61 -> 99 | 1.07 -> 1.79 |
| exp2 | CTDE-Comm | 0.474 | 0.258 | -0.216 | -4.06 | 0.003 | 9/10 | 20.9 -> 20.5s | 2.60 -> 1.78 | 89 -> 102 | 1.42 -> 1.65 |
| exp2 | CTDE-VD | 0.261 | 0.351 | +0.090 | +1.03 | 0.330 | 3/10 | 17.9 -> 17.6s | 1.85 -> 2.29 | 71 -> 96 | 1.33 -> 1.82 |
