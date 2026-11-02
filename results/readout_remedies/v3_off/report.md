=== v3_off: v3_off ===

### exp1 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE | **0.376** | 0.178 | 564/1500 | [0.352, 0.401] | 2.32 | 12.21 | 41.73 | 1.139 | 0.0671 |
| CTE | **0.194** | 0.081 | 291/1500 | [0.175, 0.215] | 2.28 | 11.59 | 36.15 | 1.039 | 0.0705 |
| DTE | **0.413** | 0.195 | 620/1500 | [0.389, 0.438] | 2.24 | 15.45 | 44.63 | 0.963 | 0.0656 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE | 0.173 | 0.400 | 0.227 | 0.333 | 0.487 | 0.160 | 0.587 | 0.707 | 0.387 | 0.300 | 4/10 |
| CTE | 0.173 | 0.100 | 0.300 | 0.187 | 0.087 | 0.193 | 0.280 | 0.200 | 0.307 | 0.113 | 0/10 |
| DTE | 0.647 | 0.480 | 0.460 | 0.453 | 0.393 | 0.640 | 0.127 | 0.087 | 0.287 | 0.560 | 6/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.182 | 0.194 | +2.97 | 0.016 | 7/10 — **significant** |
| CTDE vs DTE | -0.037 | 0.365 | -0.32 | 0.754 | 4/10 — not significant |
| CTE vs DTE | -0.219 | 0.239 | -2.90 | 0.018 | 3/10 — nominal only |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE vs CTE | +0.182 | 0.194 | +0.938 | ~15 |
| CTDE vs DTE | -0.037 | 0.365 | -0.102 | ~1003 |
| CTE vs DTE | -0.219 | 0.239 | -0.918 | ~16 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 3.231, df = 2, p = 0.199. Mean rank (1 = best): CTDE 1.85, CTE 2.45, DTE 1.70

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE vs CTE | +0.182 | 0.020 | nominal only |
| CTDE vs DTE | -0.037 | 0.750 | not significant |
| CTE vs DTE | -0.219 | 0.025 | nominal only |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE | 2.32 [1.98, 2.64] | 12.2 [10.4, 15.1] | 41.7 [29.8, 53.5] | 0.0671 [0.0509, 0.0851] |
| CTE | 2.28 [2.07, 2.48] | 11.6 [8.8, 14.8] | 36.1 [29.2, 44.4] | 0.0705 [0.0555, 0.0860] |
| DTE | 2.24 [1.87, 2.58] | 15.5 [12.0, 19.3] | 44.6 [30.8, 60.1] | 0.0656 [0.0433, 0.0929] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE vs CTE | 564/1500 vs 291/1500 | 2.50 | 0.0000 | **significant** |
| CTDE vs DTE | 564/1500 vs 620/1500 | 0.86 | 0.0399 | nominal only |
| CTE vs DTE | 291/1500 vs 620/1500 | 0.34 | 0.0000 | **significant** |

### exp2 — 10 seeds x 150 held-out greedy matches

| Paradigm | Win rate (mean over seeds) | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Shots per alive-second | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| CTDE-CAC | **0.277** | 0.112 | 416/1500 | [0.255, 0.301] | 2.27 | 13.24 | 63.84 | 1.607 | 0.0656 |
| CTDE-Comm | **0.332** | 0.071 | 498/1500 | [0.309, 0.356] | 2.10 | 13.85 | 51.13 | 1.230 | 0.0560 |
| CTDE-VD | **0.365** | 0.152 | 548/1500 | [0.341, 0.390] | 2.39 | 15.14 | 67.49 | 1.486 | 0.0527 |

Per-seed held-out win rate (each column is an independent replicate):

| Paradigm | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | best seed? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-CAC | 0.473 | 0.393 | 0.307 | 0.093 | 0.280 | 0.153 | 0.327 | 0.193 | 0.300 | 0.253 | 2/10 |
| CTDE-Comm | 0.347 | 0.327 | 0.313 | 0.327 | 0.393 | 0.187 | 0.280 | 0.307 | 0.400 | 0.440 | 3/10 |
| CTDE-VD | 0.167 | 0.233 | 0.360 | 0.553 | 0.307 | 0.653 | 0.333 | 0.487 | 0.273 | 0.287 | 5/10 |

Pairwise tests at the **seed** level — n = 10 replicates, the independent unit:

| Comparison | Mean per-seed difference | SD | Paired t | p (df=9) | seeds favouring A |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.055 | 0.115 | -1.50 | 0.167 | 3/10 — not significant |
| CTDE-CAC vs CTDE-VD | -0.088 | 0.257 | -1.08 | 0.307 | 3/10 — not significant |
| CTDE-Comm vs CTDE-VD | -0.033 | 0.206 | -0.51 | 0.622 | 5/10 — not significant |

Forward power at two-sided alpha = 0.0167 (Bonferroni over three comparisons) and 80 % power — replicates needed to detect the **observed** difference at its observed spread:

| Comparison | Observed difference | SD of difference | Cohen d | Replicates needed |
|---|---:|---:|---:|---:|
| CTDE-CAC vs CTDE-Comm | -0.055 | 0.115 | -0.476 | ~49 |
| CTDE-CAC vs CTDE-VD | -0.088 | 0.257 | -0.342 | ~93 |
| CTDE-Comm vs CTDE-VD | -0.033 | 0.206 | -0.161 | ~405 |

Friedman test over 10 replicates (the three arms are coupled inside a match, so the replicate's ranking is the observation): chi2 = 2.400, df = 2, p = 0.301. Mean rank (1 = best): CTDE-CAC 2.40, CTDE-Comm 1.80, CTDE-VD 1.80

| Comparison | Mean per-seed difference | Exact sign-flip p | Bonferroni (alpha=0.0167) |
|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | -0.055 | 0.168 | not significant |
| CTDE-CAC vs CTDE-VD | -0.088 | 0.283 | not significant |
| CTDE-Comm vs CTDE-VD | -0.033 | 0.650 | not significant |

Secondary metrics, mean over replicates with a 95 % bootstrap interval (10,000 resamples of the replicates):

| Paradigm | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---|---|---|---|
| CTDE-CAC | 2.27 [1.91, 2.61] | 13.2 [10.3, 16.3] | 63.8 [40.2, 87.4] | 0.0656 [0.0331, 0.1075] |
| CTDE-Comm | 2.10 [1.83, 2.35] | 13.9 [10.2, 18.0] | 51.1 [33.9, 69.7] | 0.0560 [0.0382, 0.0762] |
| CTDE-VD | 2.39 [2.08, 2.68] | 15.1 [11.0, 20.4] | 67.5 [47.0, 89.7] | 0.0527 [0.0311, 0.0818] |

Pairwise Fisher exact tests on pooled held-out wins (**match-level, anti-conservative**):

| Comparison | Wins / matches | Odds | p (two-sided) | Bonferroni (alpha=0.0167) |
|---|---|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | 416/1500 vs 498/1500 | 0.77 | 0.0013 | **significant** |
| CTDE-CAC vs CTDE-VD | 416/1500 vs 548/1500 | 0.67 | 0.0000 | **significant** |
| CTDE-Comm vs CTDE-VD | 498/1500 vs 548/1500 | 0.86 | 0.0605 | not significant |

### v3_off vs control, paired on seed (10 shared seeds per arm)

| experiment | paradigm | control | arm | paired diff | t | p | seeds below control | survival (control -> arm) | elim./match | shots/match | shots per alive-second |
|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|---:|
| exp1 | CTDE | 0.397 | 0.376 | -0.021 | -0.22 | 0.827 | 4/10 | 19.1 -> 12.2s | 2.43 -> 2.32 | 88 -> 42 | 1.54 -> 1.14 |
| exp1 | CTE | 0.265 | 0.194 | -0.071 | -1.18 | 0.269 | 6/10 | 16.1 -> 11.6s | 2.15 -> 2.28 | 71 -> 36 | 1.47 -> 1.04 |
| exp1 | DTE | 0.321 | 0.413 | +0.093 | +1.12 | 0.294 | 3/10 | 17.0 -> 15.5s | 1.91 -> 2.24 | 70 -> 45 | 1.36 -> 0.96 |
| exp2 | CTDE-CAC | 0.234 | 0.277 | +0.043 | +0.89 | 0.397 | 3/10 | 19.1 -> 13.2s | 1.82 -> 2.27 | 61 -> 64 | 1.07 -> 1.61 |
| exp2 | CTDE-Comm | 0.474 | 0.332 | -0.142 | -1.82 | 0.102 | 6/10 | 20.9 -> 13.9s | 2.60 -> 2.10 | 89 -> 51 | 1.42 -> 1.23 |
| exp2 | CTDE-VD | 0.261 | 0.365 | +0.104 | +1.44 | 0.184 | 3/10 | 17.9 -> 15.1s | 1.85 -> 2.39 | 71 -> 67 | 1.33 -> 1.49 |

wrote D:\rl-multi_marl-competition\results\readout_remedies\v3_off\coupled_tests.json
