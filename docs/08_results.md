# 8. Results

All numbers in this chapter are recomputed from artefacts in the repository, by scripts rather than by
hand. The mapping from section to command is in
[§ Reproducing the numbers](README.md#reproducing-the-numbers); the ones this chapter depends on are
`scripts/report_study.py` (every study table, the paired *t* and Fisher tests, and the forward-power
table), `scripts/plot_study.py` (`results/study/learning_curves.png`), `scripts/random_baseline.py`
([§ 8.9](#random-baseline)), `scripts/run_cross_arena.py` + `scripts/report_cross_arena.py`
([§ 8.10](#cross-arena)), `scripts/run_absolute_eval.py` ([§ 8.11](#absolute-strength)),
`scripts/run_study.py` again for the arena-v2 and ablation campaigns ([§ 8.12](#arena-v2),
[§ 8.13](#ablations)) and `scripts/report_study.py --compare-to` for the paired arm tables. The
leave-one-out table in [§ 8.1](#leave-one-out) calls
`report_study.paired_t` on the same per-seed matrix with one replicate dropped, so it is not a second
statistics implementation.

The chapter has two halves, and they are **not** interchangeable:

* **[§ 8.1 Replicated study](#replicated-study)** — 10 independent replicates per experiment at 500,000
  steps, measured by held-out greedy evaluation of the final policy, produced under the corrected heading
  controller with the paradigm-to-slot assignment rotated across seeds. This is the primary evidence.
* **[§ 8.2 Historical single-seed runs](#historical-single-seed-runs)** — the two 100,000-step runs the
  repository originally shipped: one seed, measured during training, under the defective control law
  [A-1](11_code_audit.md#turn-control-defect) before it was fixed. Kept because those artefacts are what
  `data/` contains, and because the study **contradicts** their headline conclusion — which is the most
  important result in this document.

---

## 8.1 Replicated study <a name="replicated-study"></a>

### Design

| | |
|---|---|
| Replicates | 10 seeds × 2 experiments = 20 runs, all completed, no failures |
| Budget per replicate | 500,000 environment steps (5× the historical runs, half of the superseded 5-seed study) |
| Seeds | 1–10, controlling the `random`, NumPy and Torch streams |
| Slot rotation | `paradigm_rotation = seed % 3`, so each paradigm occupies each team slot 6 or 7 times across the 20 runs (observed: rotation 0 × 6, 1 × 8, 2 × 6) |
| Initialisation | fresh — `ARENA_DATA_DIR` isolates each run, so no checkpoint warm-start |
| Training-time metric records | every 100 matches; 87,863 training matches played in total |
| Held-out evaluation | 150 greedy matches on the final policy, plus 60 at each of 25/50/75/100 % |
| Evaluation conditions | `domain_randomization=False`, `set_rl_training(False)`, separate simulation instance |
| Trials per paradigm | 1,500 held-out matches (10 seeds × 150) |
| Wall clock | 117.5 min for all 20 replicates, run concurrently on 32 cores |
| Cost | 31.5 CPU-hours training + 4.7 h evaluation = **36.2 CPU-hours** |
| Throughput | 82.6–92.9 steps/s per process under 20-way load (368 steps/s alone) |

The budget was halved from the superseded 5-seed study in order to double the number of independent
replicates at a slightly lower total cost; with a between-seed SD near 0.2 and a per-seed standard error of
only ~0.04, replicate count is the binding constraint, not step count. That decision is revisited in
[§ 8.1 Power](#power-analysis) — at the effect sizes actually observed, neither lever is close to enough.

Held-out win rate is the response variable throughout. The training-time running average appears only in
[§ 8.2](#historical-single-seed-runs).

### Experiment 1 — CTE vs DTE vs CTDE

| Paradigm | Mean win rate over seeds | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| **DTE** | **0.381** | 0.191 | 571/1500 | [0.356, 0.406] | 1.95 | 24.98 | 105.2 | 2.71 % |
| CTDE | 0.326 | 0.187 | 489/1500 | [0.303, 0.350] | 1.97 | 18.37 | 101.2 | 2.65 % |
| CTE | 0.252 | 0.122 | 378/1500 | [0.231, 0.275] | 2.22 | 22.69 | 121.7 | 3.59 % |

Per-seed held-out win rate — each column is one independent replicate:

| Paradigm | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | best in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DTE | 0.153 | 0.527 | 0.280 | 0.187 | 0.533 | **0.760** | 0.180 | 0.400 | 0.407 | 0.380 | 6/10 |
| CTDE | **0.607** | 0.220 | **0.607** | 0.280 | 0.247 | **0.000** | 0.407 | 0.180 | 0.347 | 0.367 | 3/10 |
| CTE | 0.207 | 0.240 | 0.107 | **0.533** | 0.207 | 0.213 | 0.393 | 0.173 | 0.227 | 0.220 | 1/10 |

Seed-level paired tests (n = 10 replicates; the seed is the independent unit):

| Comparison | Mean per-seed difference | SD of difference | Paired *t* | *p* (df = 9) | Verdict |
|---|---:|---:|---:|---:|---|
| CTDE vs CTE | +0.074 | 0.236 | +0.99 | 0.348 | not significant |
| CTDE vs DTE | −0.055 | 0.357 | −0.48 | 0.640 | not significant |
| CTE vs DTE | −0.129 | 0.265 | −1.54 | 0.158 | not significant |

Match-level Fisher exact tests on the pooled 1,500 held-out matches per paradigm:

| Comparison | Wins / matches | Odds ratio | *p* (two-sided) |
|---|---|---:|---:|
| CTDE vs CTE | 489/1500 vs 378/1500 | 1.44 | 9 × 10⁻⁶ |
| CTDE vs DTE | 489/1500 vs 571/1500 | 0.79 | 0.0020 |
| CTE vs DTE | 378/1500 vs 571/1500 | 0.55 | < 10⁻⁴ |

> **The two test levels still disagree completely, and the seed-level one is the correct one.** Treating
> 1,500 matches as independent trials makes every exp-1 comparison significant; treating the 10 replicates
> as the independent unit makes none of them significant. Doubling the replicate count and adding slot
> rotation did not change that — it changed the *estimates*, not the *variance structure*. Matches within a
> replicate are not independent draws: they are played by one frozen policy against two others from the same
> run, sharing that run's initialisation, geometry stream and opponent dynamics. The match-level *p* values
> are anti-conservative and are shown only to make the size of that assumption visible.

### Experiment 2 — CTDE-VD vs CTDE-CAC vs CTDE-Comm

| Paradigm | Mean win rate over seeds | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival (s) | Shots/match | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| **CTDE-VD** | **0.367** | 0.171 | 551/1500 | [0.343, 0.392] | 1.97 | 23.08 | 92.8 | 3.01 % |
| CTDE-Comm | 0.331 | 0.218 | 496/1500 | [0.307, 0.355] | 2.18 | 19.23 | 111.6 | 2.40 % |
| CTDE-CAC | 0.282 | 0.182 | 423/1500 | [0.260, 0.305] | 1.98 | 21.47 | 112.3 | 2.35 % |

Per-seed held-out win rate:

| Paradigm | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 | best in |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-VD | 0.387 | 0.367 | 0.153 | **0.547** | 0.220 | 0.153 | 0.373 | 0.273 | **0.613** | **0.587** | 5/10 |
| CTDE-Comm | 0.253 | 0.253 | **0.553** | 0.173 | 0.153 | **0.767** | **0.580** | 0.193 | 0.187 | 0.193 | 3/10 |
| CTDE-CAC | 0.340 | 0.333 | 0.280 | 0.220 | **0.620** | 0.067 | 0.047 | **0.520** | 0.187 | 0.207 | 2/10 |

Seed-level paired tests:

| Comparison | Mean per-seed difference | SD | Paired *t* | *p* (df = 9) | Verdict |
|---|---:|---:|---:|---:|---|
| CTDE-CAC vs CTDE-Comm | −0.049 | 0.360 | −0.43 | 0.679 | not significant |
| CTDE-CAC vs CTDE-VD | −0.085 | 0.282 | −0.96 | 0.363 | not significant |
| CTDE-Comm vs CTDE-VD | −0.037 | 0.347 | −0.33 | 0.746 | not significant |

Match-level Fisher tests: CAC vs Comm *p* = 0.0043, CAC vs VD *p* < 10⁻⁴, Comm vs VD *p* = 0.0386 — two of
the three pass Bonferroni at the match level and none come close to it at the seed level.

### One replicate is enough to reverse the exp-1 ranking <a name="leave-one-out"></a>

The exp-1 per-seed matrix contains one extreme replicate: **seed 6's CTDE arm scores 0.000 under greedy
evaluation.** Over its 150 held-out matches it fired 3.8 shots per match against 325 and 372 for the other
two teams, recorded 0.01 eliminations per match, and won nothing — a zero on a 1/3 chance baseline. It is
the only such cell in 60 paradigm-seed evaluations (threshold: held-out eliminations below 0.25 per match),
and it is what drags CTDE's mean from 0.362 to 0.326.

**It is not a training collapse, and an earlier draft of this section said it was.** The same policy, read
through the stochastic action sampler it was actually trained with, was the *best* arm of that run:

| seed 6, CTDE | during training (4,929 matches, sampled actions) | held out (150 matches, greedy actions) |
|---|---:|---:|
| win rate | 0.315 | 0.000 |
| eliminations per match | **2.68** (highest of the three arms) | 0.01 |
| shot accuracy | **0.1041** (highest of the three arms) | 0.0018 |
| mean survival | 4.84 s | 9.3 s |

So the policy learned to fight — it led its own match in kills and accuracy — and only the argmax readout
turns it into something that never fires. What this replicate reveals is a **readout divergence**, not a
reward-landscape basin. The general form of it is in [§ 8.1 Readout](#readout-divergence) below, and it is
the larger finding.

Recomputing exp 1 with that one replicate removed, using the same shipped `paired_t`:

| Comparison | n = 10 | n = 9 (seed 6 removed) |
|---|---|---|
| CTDE mean win rate | 0.326 (2nd) | **0.362 (1st)** |
| DTE mean win rate | 0.381 (1st) | 0.339 (2nd) |
| CTDE vs CTE | +0.074, *t* = +0.99, *p* = 0.348 | +0.106, *t* = +1.40, *p* = 0.199 |
| CTDE vs DTE | −0.055, *t* = −0.48, *p* = 0.640 | +0.024, *t* = +0.26, *p* = 0.801 |
| CTE vs DTE | −0.129, *t* = −1.54, *p* = 0.158 | −0.082, *t* = −1.06, *p* = 0.322 |

**A single replicate out of thirty paradigm-slots flips which architecture is reported as the winner**, in
both the mean ordering and the sign of the paired difference, while nothing reaches significance either
way. The exp-1 ranking is therefore not a finding of this study; the divergence that produced the extreme
cell is, and it is a property of the measurement rather than of the optimiser.

### The readout diverges from the trained policy <a name="readout-divergence"></a>

The study's response variable is held-out **greedy** win rate, on the argument that a deployed policy should
be measured deterministically. That choice is defensible; treating the resulting numbers as a measurement of
*what training produced* is not. Across all 60 arm-replicate cells:

| | mean | SD | min | max |
|---|---:|---:|---:|---:|
| Win rate during training (sampled actions) | 0.331 | **0.072** | 0.176 | 0.464 |
| Held-out win rate (greedy actions) | 0.323 | **0.177** | 0.000 | 0.767 |

Two things follow. The greedy readout has **2.5× the spread** of the behaviour that was actually optimised,
so most of the variance this study had to fight is introduced at measurement time. And the orderings
disagree: across the 59 comparable within-replicate pairs, the greedy ranking matches the training ranking
in **21 — 35.6 %, worse than a coin flip.**

The largest single divergences are not marginal either:

| cell | training win rate | greedy win rate |
|---|---:|---:|
| exp-2 seed 7, CTDE-CAC | 0.390 | 0.047 |
| exp-2 seed 5, CTDE-CAC | 0.271 | 0.620 |
| exp-1 seed 4, CTE | 0.176 | 0.533 |
| exp-2 seed 6, CTDE-Comm | 0.271 | 0.767 |

This does not invalidate the study's null result — a measurement with more noise cannot detect an effect, and
the conclusion was that no effect is detectable. It does relocate the main finding: **the architecture
differences the repository spent two campaigns chasing are smaller than the difference between sampling and
argmax-ing the same network.** Any future campaign should fix the readout before it adds replicates: report
both, or evaluate under the stochastic policy at a fixed temperature, and treat a one-sided greedy number as
incomplete.

### Two regimes: wipeouts and timeouts <a name="two-regimes"></a>

Held-out median match duration is bimodal across replicates, not noisy around a mean:

| Regime | Replicates | Held-out median duration |
|---|---:|---|
| Matches end in a wipeout | 15 of 20 | 6.9 – 20.4 s |
| Matches run to the 90 s cap | 5 of 20 | 90.1 s (exp 1 seeds 6, 8; exp 2 seeds 2, 4, 9) |

Draws — every team alive with equal eliminations at the cap, under the [A-10](11_code_audit.md) tie-break —
occur in 62/1500 (4.1 %) of exp-1 held-out matches and 30/1500 (2.0 %) of exp-2 matches. Seed 6 is the same
replicate that produced the CTDE collapse: with one team not shooting, the other two spray (325 and 372
shots per match at 0.5–0.8 % accuracy), nobody dies, and matches end on the clock rather than on a kill.
Win rates measured in that regime are answering "who is most alive after 90 s", which is a different
question from "who wins a fight" — a further reason not to read the exp-1 ordering as a result.

### The historical conclusions do not survive <a name="reversal"></a>

| | Historical (1 seed, training-time, broken control law) | Study (10 seeds, held-out, fixed control law, rotated slots) |
|---|---|---|
| **Exp. 1 order** | CTDE 47.2 % > CTE 27.4 % > DTE 25.4 % | **DTE 38.1 % > CTDE 32.6 % > CTE 25.2 %** |
| **Exp. 2 order** | CAC 42.9 % > VD 40.7 % ≫ **Comm 16.4 %** | **VD 36.7 % > Comm 33.1 % > CAC 28.2 %** |
| The one defended finding | "Comm loses to both others" | **does not reproduce** — Comm is now 2nd of 3 |
| The exp-2 winner | CAC, by 2 points over VD | **CAC, by 8 points, last** |

Both experiments reverse, and the exp-2 reversal is more complete than the one reported by the superseded
5-seed study: CTDE-CAC was the historical winner and is now the last-place arm, while CTDE-Comm went from
last by 26 points to middle. CTDE-VD's per-seed results span 0.153 to 0.613 and CAC's span 0.047 to 0.620 —
each architecture produced one of the two best *and* one of the two worst replicates in its own experiment.

Two readings are possible and this data cannot separate them:

1. **The original findings were artefacts of the broken turn controller.** Under
   [A-1](11_code_audit.md#turn-control-defect) agents spun toward a fixed absolute heading rather than
   toward their waypoint. A communication channel built from relative-position features would be
   especially damaged by an actuator that cannot steer toward what it computes.
2. **They were single-seed flukes**, for which the per-seed matrices above show ample room: a spread from
   0.047 to 0.767 within one architecture makes any one-seed ranking fragile.

Both readings imply the same operational conclusion: **the historical rankings were not properties of the
architectures.**

### Learning curves

![Held-out learning curves](../results/study/learning_curves.png)

*Mean over 10 replicates (thick line) with every individual replicate shown (pale points). The dashed line
is the 1/3 three-way chance level. Curve points are held-out greedy evaluations of intermediate checkpoints,
so unlike the historical training-time metrics they are comparable across fractions.*

| Paradigm | 25 % | 50 % | 75 % | 100 % | Change 25 → 100 |
|---|---:|---:|---:|---:|---:|
| **Exp. 1** DTE | 0.465 | 0.130 | 0.243 | 0.351 | −0.114 |
| CTDE | 0.130 | 0.427 | 0.467 | 0.383 | +0.253 |
| CTE | 0.378 | 0.280 | 0.225 | 0.229 | −0.148 |
| **Exp. 2** CTDE-VD | 0.330 | 0.345 | 0.287 | 0.367 | +0.037 |
| CTDE-Comm | 0.190 | 0.335 | 0.258 | 0.331 | +0.141 |
| CTDE-CAC | 0.425 | 0.095 | 0.212 | 0.282 | −0.143 |

The 100 % column is the 60-match curve point, not the 150-match final evaluation, so it differs from the
headline means in the tables above by sampling noise (e.g. exp-1 CTDE: 0.383 here, 0.326 there). Each point
is 60 matches per replicate, giving a per-point binomial SE near 0.06 and an SE of the 10-replicate mean near
0.02 *if* replicates agreed; they do not, and the between-replicate spread dominates by roughly an order of
magnitude.

No curve is monotone. Three of six arms are best at the *earliest* checkpoint (DTE 0.465 at 25 %, CTE 0.378
at 25 %, CAC 0.425 at 25 %) and finish lower, which is what a degenerate-basin process looks like: early
policies are near-uniform and lose to each other in varied ways, then some runs settle into a fixed
attractor.

> **The project's central hypothesis about communication is still not supported, in a new way.** The claim —
> inherited from the original README — was that CTDE-Comm needs more steps than the others because messages
> must become informative before they help. In the 5-seed study Comm was *worst* at 25 % (0.650) and drifted
> down. With 10 replicates it is *worst* again at 25 % (0.190) but now drifts **up** (+0.141) — the sign of
> the trajectory flipped when the replicate count changed, which is itself the answer: the curve shape is
> not a property of the architecture. The one stable statement is the negative one: at 500,000 steps,
> Comm's advantage is not the monotone growth the hypothesis predicts, and VD's is barely any growth at all
> (+0.037).

### Component metrics moved far more than the ranking <a name="component-metrics-moved-far-more-than-the-ranking"></a>

Comparing the study to the historical run is confounded by the control-law fix, and the scale of the confound
is directly visible:

| Quantity | Historical runs | Replicated study |
|---|---:|---:|
| Shots per match | 21 – 27 | 92.8 – 121.7 |
| Shot accuracy | 8.1 – 12.8 % | 2.35 – 3.59 % |
| Mean survival | 7.5 – 10.2 s | 18.4 – 25.0 s |
| Eliminations per match | 1.7 – 3.4 | 1.95 – 2.22 |

Fixing the heading controller made agents drive at their waypoints, which put them in weapon range far more
often: they fire roughly four times as much and convert far less per shot, while surviving twice as long.
**The arena these architectures competed in is not the same arena as before**, which is why the historical
numbers are labelled as such rather than compared directly.

The identity of [§ 8.4](#metric-rank-deficiency) — eliminations = shots × accuracy — still holds, but only
**within** a replicate. It does not hold for the *mean* of per-seed ratios, which is what the tables above
report, because per-seed shot counts vary by more than 80× (3.8 to 372.4 shots per match).

### Power: how many replicates this would take <a name="power-analysis"></a>

Computed by `scripts/report_study.py` from the observed paired differences and their spread, for two-sided
α = 0.0167 (Bonferroni over three comparisons) at 80 % power, using the self-consistent
n = ((t<sub>crit</sub> + t<sub>β</sub>)·SD / mean)² solve. The solver was checked against a noncentral-*t*
power computation over the six observed pairs and agrees within one replicate everywhere.

| Comparison | Observed difference | SD of difference | Cohen *d* | Replicates needed |
|---|---:|---:|---:|---:|
| Exp. 1 CTDE vs CTE | +0.074 | 0.236 | +0.313 | ~110 |
| Exp. 1 CTDE vs DTE | −0.055 | 0.357 | −0.153 | ~450 |
| Exp. 1 CTE vs DTE | −0.129 | 0.265 | −0.486 | ~47 |
| Exp. 2 CAC vs Comm | −0.049 | 0.360 | −0.135 | ~576 |
| Exp. 2 CAC vs VD | −0.085 | 0.282 | −0.303 | ~117 |
| Exp. 2 Comm vs VD | −0.037 | 0.347 | −0.106 | ~939 |

These numbers are **five to sixty times larger** than those the 5-seed study reported (~13, ~49, ~91, ~272,
~421). That is not a mistake: doubling the replicates shrank the observed differences far faster than it
shrank their spread, so the effect sizes moved away from detectability as the estimates got better. This is
the characteristic signature of chasing a noise-dominated quantity.

At 1.8 CPU-hours per replicate, the cheapest comparison (CTE vs DTE, ~47 replicates) costs about **85
CPU-hours**, roughly 90 min of wall clock at 20-way concurrency — affordable. The exp-2 comparisons cost
100–1,700 CPU-hours. But the leave-one-out result above says a ranking that swings on a single degenerate
replicate is not a thing that more replicates would resolve: the next 100 replicates would add more collapses,
not less. The fix that would actually change this is to the *reward and termination design*, not to the
sample size.

---

## 8.2 Historical single-seed runs <a name="historical-single-seed-runs"></a>

Everything from here on describes the two original 100,000-step runs. It is retained because those
artefacts ship in `data/`, and because the analyses below are what justified re-running everything — but
their conclusions are superseded by [§ 8.1](#replicated-study).

### Experiment 1 — CTE vs DTE vs CTDE

Source: `data/metrics/summary.json` (463 matches played; snapshot taken at match 460).

| Team | Paradigm | Win rate | Elim./match | Mean survival (s) | Shot accuracy | Hits | Misses |
|---|---|---:|---:|---:|---:|---:|---:|
| Team 1 | CTE | 27.39 % | 1.72 | 9.63 | 8.10 % | 790 | 8,959 |
| Team 2 | DTE | 25.43 % | 1.84 | 8.59 | 8.13 % | 848 | 9,587 |
| Team 3 | **CTDE** | **47.17 %** | **3.42** | **9.93** | **12.84 %** | 1,571 | 10,663 |

### Experiment 2 — CTDE-VD vs CTDE-CAC vs CTDE-Comm

Source: [`ctde_arena/data/metrics/summary.json`](../ctde_arena/data/metrics/summary.json) (456 matches
played; snapshot at match 450).

| Team | Paradigm | Win rate | Elim./match | Mean survival (s) | Shot accuracy | Hits | Misses |
|---|---|---:|---:|---:|---:|---:|---:|
| Team 1 | CTDE-VD | 40.67 % | 2.50 | 10.20 | 11.29 % | 1,125 | 8,843 |
| Team 2 | **CTDE-CAC** | **42.89 %** | **2.63** | 9.83 | **12.33 %** | 1,184 | 8,417 |
| Team 3 | CTDE-Comm | 16.44 % | 1.96 | 7.47 | 12.14 % | 883 | 6,390 |

![Experiment 2 comparative dashboard](../ctde_arena/data/exports/comparative_dashboard.png)

*The four-panel dashboard regenerated from the versioned `team_match_metrics.csv`. The win-rate panel is a
**cumulative** ratio over recorded matches only, so the visible convergence near 0.41/0.43 is regression
to the mean of a 46-sample estimate, not a training effect.*

## 8.3 Analyses performed on the historical data <a name="historical-analyses"></a>

The following sections were the diagnostic work that motivated the study. Each is a correct analysis of the
historical artefacts, and each found something the study then overturned or recontextualised.

## 8.4 The metric set is rank-deficient <a name="metric-rank-deficiency"></a>

`eliminations_per_match` is not an independent measurement. Because every hit eliminates exactly one agent
and every elimination comes from a hit,

\[
\text{elim/match} \;=\; \text{shots/match} \times \text{shot accuracy}
\]

holds to the reported precision for all six historical teams (verified: CTE \(21.19 \times 0.0810 = 1.72\)
against a reported 1.72; CTDE-Comm \(16.16 \times 0.1214 = 1.96\) against 1.96). Four headline columns
therefore carry three degrees of freedom, and any interpretation treating "eliminations" and "accuracy" as
separate evidence double-counts. The identity also holds within each replicate of the study
([§ 8.1](#component-metrics-moved-far-more-than-the-ranking)).

## 8.5 Decomposing the elimination gap <a name="decomposing-the-elimination-gap"></a>

Expressing each historical team relative to a reference arm separates *how much a team shoots* from *how
well it converts*:

**Experiment 1**, relative to CTE:

| Team | Shot volume | Accuracy | Product |
|---|---:|---:|---:|
| DTE | ×1.070 | ×1.003 | ×1.073 |
| CTDE | ×1.255 | ×1.585 | **×1.989** |

CTDE's advantage was two-factor: it fired 26 % more often **and** converted 59 % better.

**Experiment 2**, relative to CTDE-VD:

| Team | Shot volume | Accuracy | Product |
|---|---:|---:|---:|
| CTDE-CAC | ×0.963 | ×1.093 | ×1.052 |
| CTDE-Comm | **×0.730** | **×1.076** | ×0.785 |

> CTDE-Comm was *not* a worse shot even in the historical run — its accuracy (12.14 %) was marginally
> higher than CTDE-VD's (11.29 %) and within 0.2 points of CTDE-CAC's. Its entire elimination deficit was a
> **firing-volume deficit**: it shot 27 % less often. At CTDE-VD's shot volume with its own accuracy it
> would have produced 2.69 eliminations per match, ahead of both other arms.

That analysis survives the replication in one respect: the mechanism it identified — a policy-level
decision to shoot rather than an aiming deficit — is architectural rather than seed-dependent. What did not
survive is the conclusion that this made Comm the weakest arm.

## 8.6 Statistical significance on 46 matches <a name="statistical-significance-historical"></a>

Computed on the 46 recorded matches per team, the most conservative estimate available from the historical
artefacts (Wilson score intervals, two-proportion *z*-tests).

**Experiment 1**

| Team | Wins / recorded | Win rate | 95 % CI |
|---|---:|---:|---|
| CTE | 14 / 46 | 0.304 | [0.191, 0.448] |
| DTE | 13 / 46 | 0.283 | [0.173, 0.425] |
| CTDE | 19 / 46 | 0.413 | [0.283, 0.557] |

| Comparison | *z* | *p* | Verdict |
|---|---:|---:|---|
| CTE vs DTE | +0.23 | 0.819 | not significant |
| CTE vs CTDE | −1.09 | 0.277 | not significant |
| DTE vs CTDE | −1.31 | 0.189 | not significant |

**Experiment 2**

| Team | Wins / recorded | Win rate | 95 % CI |
|---|---:|---:|---|
| VD | 19 / 46 | 0.413 | [0.283, 0.557] |
| CAC | 21 / 46 | 0.457 | [0.322, 0.598] |
| Comm | 6 / 46 | 0.130 | [0.061, 0.257] |

| Comparison | *z* | *p* | Verdict |
|---|---:|---:|---|
| VD vs CAC | −0.42 | 0.674 | not significant |
| VD vs Comm | +3.05 | 0.002 | significant |
| CAC vs Comm | +3.43 | 0.001 | significant |

### Sensitivity to the independence assumption <a name="independence-sensitivity"></a>

Treating the full cumulative denominators (460 / 450 matches) as independent Bernoulli trials instead makes
every Experiment 1 gap significant at *p* < 10⁻⁵ — a swing of ten orders of magnitude produced by an
assumption choice. That assumption is false: the matches are played by policies that change every match, and
exactly one of three teams wins each, so outcomes are negatively correlated within a match.

This was the clearest internal signal that the historical evidence could not support its conclusions, and it
is what the two-level reporting in [§ 8.1](#replicated-study) resolves properly.

## 8.7 Drift in the historical data: was the ranking a learning effect? <a name="drift"></a>

Splitting each team's 46 recorded matches at the midpoint separates "the architecture is better" from "the
architecture learned to be better".

**Experiment 1** — win rate by segment:

| Team | All 46 | First 23 | Last 23 | Last 10 | Change |
|---|---:|---:|---:|---:|---:|
| CTE | 0.304 | 0.304 | 0.304 | 0.200 | **+0.000** |
| DTE | 0.283 | 0.304 | 0.261 | 0.200 | −0.043 |
| CTDE | 0.413 | 0.391 | 0.435 | 0.600 | +0.043 |

> Experiment 1 showed essentially **no learning trend**: the ordering was already present in the first 23
> recorded matches, which begin at match 10. Either the policies converged within a few thousand steps, or
> the ordering was not a learning effect at all.

**Experiment 2** — win rate by segment:

| Team | All 46 | First 23 | Last 23 | Last 10 | Change |
|---|---:|---:|---:|---:|---:|
| VD | 0.413 | 0.478 | 0.348 | 0.200 | **−0.130** |
| CAC | 0.457 | 0.478 | 0.435 | 0.500 | −0.043 |
| Comm | 0.130 | 0.043 | 0.217 | 0.300 | **+0.174** |

The historical data appeared to support the project's hypothesis here: Comm started worst by a wide margin
and improved the most. The held-out curves in [§ 8.1](#learning-curves) show the opposite sign — Comm starts
best and declines — so this apparent improvement was an artefact of either the broken control law or of
measuring a training-time running average rather than a frozen policy.

## 8.8 What MLflow actually contains <a name="mlflow-content"></a>

The versioned export [`ctde_arena/data/mlflow_export/`](../ctde_arena/data/mlflow_export/) holds the 48
metric points that reached the tracker. The 100,000-step run contributed **one** point, at step 50,793:

| Team | Win rate @50,793 | Final cumulative | Δ |
|---|---:|---:|---:|
| CTDE-VD | 0.4039 | 0.4067 | +0.003 |
| CTDE-CAC | 0.4433 | 0.4289 | −0.014 |
| CTDE-Comm | 0.1527 | 0.1644 | +0.012 |

The mid-training snapshot reproduces the final historical ordering to within 1.4 points, corroborating
§ 8.7's finding that the ranking was established early. The 5,000-step smoke run (`457ce1e8`, logged every
1,000 steps) reaches 0.250 / 0.417 / 0.333 for VD / CAC / Comm by step 3,958 — under 4 % of the budget, a
configuration check rather than a result.

The replicated study does not use MLflow; it writes its own held-out curve to
`data/runs/*/evaluation/run_summary.json`.

## 8.9 Baseline behaviour from random initialisation <a name="random-baseline"></a>

The repository previously quoted a 60-match "random baseline" taken before the control-law fix. That has
been re-measured under the current code with `scripts/random_baseline.py`, 300 headless fixed-variant
matches, and it produced a methodological finding of its own: **there are two different things one can mean
by "untrained", and they disagree about almost everything.**

`set_rl_training(False)` makes `decide()` take `torch.argmax` of the actor's logits, which is the right
selection rule for a trained policy and a misleading one for an untrained network — the argmax of random
weights is close to a constant, so such an agent replays roughly one action instead of sampling across the
eight. The script therefore offers `--policy greedy` (what the harness actually runs) and `--policy uniform`
(actions drawn from the action space, weights ignored). 300 matches each.

**Experiment 1 — CTE / DTE / CTDE**

| | Uniform random | Greedy, untrained | Trained (study mean) |
|---|---:|---:|---:|
| Ended by wipeout | **97.0 %** | 35.7 % | mixed — [§ 8.1](#two-regimes) |
| Hit the 90 s cap / drawn | 2.0 % / 1.0 % | 43.0 % / 21.3 % | 4.1 % drawn |
| Median match duration | 12.6 s | 90.1 s | 6.9 – 90.1 s |
| Survival CTE / DTE / CTDE | 9.4 / 8.1 / 10.3 s | 40.7 / 48.8 / 60.4 s | 22.7 / 25.0 / 18.4 s |
| Shots/match | 43.5 / 37.5 / 43.4 | **4.6 / 6.8 / 82.1** | 121.7 / 105.2 / 101.2 |
| Shot accuracy | 5.1 / 5.9 / 6.2 % | 1.5 / 3.4 / 3.8 % | 3.6 / 2.7 / 2.7 % |
| Win rate | 0.257 / 0.270 / 0.463 | **0.013 / 0.023 / 0.750** | 0.252 / 0.381 / 0.326 |

**Experiment 2 — CTDE-VD / CTDE-CAC / CTDE-Comm**

| | Greedy, untrained | Trained (study mean) |
|---|---:|---:|
| Ended by wipeout | **2.0 %** | mixed — [§ 8.1](#two-regimes) |
| Hit the 90 s cap / drawn | 56.7 % / **41.3 %** | 2.0 % drawn |
| Median match duration | 90.1 s | 6.9 – 90.1 s |
| Survival VD / CAC / Comm | 83.3 / 63.6 / 84.9 s | 23.1 / 21.5 / 19.2 s |
| Shots/match | 358.2 / 88.2 / 397.9 | 92.8 / 112.3 / 111.6 |
| Shot accuracy | **0.10 / 0.10 / 0.20 %** | 3.0 / 2.4 / 2.4 % |
| Win rate | 0.167 / 0.037 / 0.383 | 0.367 / 0.282 / 0.331 |

The uniform column is tree-independent — it never queries a network, so experiment 2's uniform baseline is
the same 300 matches as experiment 1's relabelled, not a second measurement. The greedy column is not: it is
a function of each tree's random initialisations, and the two trees differ there in both degree and kind.

Four conclusions follow, and they are the most load-bearing in this chapter:

1. **Training does something.** Against true random play, trained exp-1 agents survive 2–3× longer (18–25 s
   against 8–10 s), fire 2.4–2.8× more often, and finish matches by elimination instead of running the
   clock. The earlier worry in this section — that trained survival of 16.5–25.6 s looked indistinguishable
   from untrained play — was an artefact of comparing against the *greedy* column, whose 40–85 s survival
   comes from agents that cannot aim, not from agents that fight well.
2. **Training massively improves conversion, but only against the greedy floor.** Exp-2's untrained policies
   shoot 88–398 times per match at 0.1–0.2 % accuracy; trained they shoot 93–112 times at 2.4–3.0 % — a
   15–30× gain in accuracy for a third of the volume. Against *uniform random aiming* (5–6 % accuracy) the
   picture inverts: trained policies convert **worse** while firing more. The learned behaviour is
   engagement and positioning, not marksmanship.
3. **Untrained greedy play already sits in the timeout regime.** 41 % of exp-2's baseline matches are
   three-way draws and 57 % hit the cap, against 2 % of the trained study's. So the timeout behaviour
   described in [§ 8.1](#two-regimes) is not something training created — training reduces it sharply — but
   training does not eliminate it.
4. **The paradigm label predicts firing rate before any learning happens.** Exp-1's untrained CTE fires 4.6
   times per match against CTDE's 82.1, and untrained CTDE wins 75 % of matches. That 18× spread comes only
   from which action each architecture's random argmax happens to favour. Any comparison of these arms'
   *engagement* is therefore partly a comparison of their initialisations, and win rate in this arena cannot
   be interpreted against a chance level of 1/3. Mechanism in
   [A-30](11_code_audit.md#greedy-evaluation-of-an-untrained-network-is-not-random-play).

Regenerate either table with:

```
python scripts/random_baseline.py --matches 300 --policy greedy
python scripts/random_baseline.py --matches 300 --policy uniform
```

## 8.10 Cross-experiment round-robin: all six architectures on one scale <a name="cross-arena"></a>

Everything above compares the three arms of one experiment against each other. The two experiments never
meet: experiment 1's CTE/DTE/CTDE only ever fight each other, experiment 2's VD/CAC/Comm only ever fight
each other, so placing a number from one table next to a number from the other has been an assumption
rather than a measurement.

`scripts/run_cross_arena.py` drops it. The trained, already-versioned policies are loaded — six
architectures, three independently trained checkpoints each — each driven by the controller class of the tree
that produced it, because re-implementing an actuator would silently invalidate the comparison — and every
three-architecture combination is played to completion: C(6,3) = 20 triples x 30 matches, with the slot
permutation cycled inside every triple and no training of any kind.

### Design: three sets of checkpoints x five match streams <a name="cross-arena-design"></a>

The first version of this section used one checkpoint per architecture over five harness seeds (3,000
matches). The caveat it recorded — see [below](#cross-arena-limits) — was that a single checkpoint makes any
statement a claim about *two particular networks that were measured many times*, not about two
architectures. That caveat is now closed at its cheap end: the study's other replicates have been versioned
under `data/training_replicates/` (18 checkpoints, three blocks) and the whole round-robin replayed over

* **three training blocks** — each block pairs one experiment-1 replicate with one experiment-2 replicate,
  so every architecture is represented by three independently trained networks:
  `exp1_seed1+exp2_seed7`, `exp1_seed10+exp2_seed5`, `exp1_seed9+exp2_seed2`;
* **five harness seeds** — five independent match streams over the same checkpoints.

**9,000 matches / 27,000 team-matches, 4,500 per architecture.**

The third block is the promoted checkpoint set. Its six `.pt` files are byte-identical (SHA-256) to
`data/checkpoints/` and `ctde_arena/data/checkpoints/`, which turns the re-run into a parity control rather
than a fresh instrument: block 3's per-architecture win rates reproduce the table this section published
before the replication, to four decimal places, on every one of the six arms.

`scripts/combine_cross_arena.py` stamps every row with its `training_block` and `harness_seed`;
`scripts/report_cross_arena.py` does the analysis. **Both columns have to be part of the match key.** The
first pooled analysis omitted `harness_seed`, which merged five match streams onto the same thirty keys: it
ran its paired tests on one fifth of the evidence and still printed a complete-looking table with plausible
p-values. Adding the training block one level up reproduced the same trap, and
`tests/test_cross_arena_report.py` now pins the seam — three runs of two matches must count six, and a key
that drops either column reports four or two. The McNemar implementation itself was checked against
`scipy.stats.binomtest` over 200 random discordant tables and agrees to 4.9e-14. Its tail is summed in log
space, which is what lets it report the 1.11e-15 below rather than underflowing to zero at the first
multiplication.

### Pooled result

| Rank | Architecture | Win rate | 95 % CI | vs 1/3 | Elim./match | Survival | Shots | Accuracy |
|---|---|---:|---|---:|---:|---:|---:|---:|
| 1 | **CTDE-Comm** | **0.373** | [0.359, 0.388] | +0.040 | 2.42 | 20.4 s | 116.3 | 2.08 % |
| 2 | **DTE** | **0.368** | [0.354, 0.382] | +0.034 | 1.98 | 24.0 s | 96.0 | 2.06 % |
| 3 | CTDE | 0.329 | [0.316, 0.343] | −0.004 | 2.15 | 20.4 s | 93.8 | 2.29 % |
| 4 | CTDE-VD | 0.297 | [0.284, 0.311] | −0.036 | 1.77 | 23.4 s | 87.1 | 2.03 % |
| 5 | CTDE-CAC | 0.284 | [0.271, 0.297] | −0.050 | 1.87 | 20.4 s | 78.7 | 2.37 % |
| 6 | **CTE** | **0.266** | [0.254, 0.280] | −0.067 | 2.07 | 19.9 s | 94.1 | 2.20 % |

Decided by wipeout 60.0 %, with the clock still running 35.9 %, drawn 4.1 %. The three rows of a match are
negatively coupled by construction, so overlapping intervals are the wrong test; the pairwise comparison is
McNemar's exact test on the discordant outcomes of the matches where the two architectures actually met —
**1,800 such matches per pair** — with Bonferroni correction over 15 pairs at alpha = 0.00333.

### Eight of the fifteen pairs separate

| Pair | Meetings | A won | B won | Discordant b/c | *p* | Verdict |
|---|---:|---:|---:|---|---:|---|
| CTDE-Comm vs CTE | 1800 | 694 | 447 | 694/447 | 2.65e-13 | **Comm** |
| CTE vs DTE | 1800 | 441 | 713 | 441/713 | 1.11e-15 | **DTE** |
| CTDE-CAC vs CTDE-Comm | 1800 | 503 | 688 | 503/688 | 9.19e-08 | **Comm** |
| CTDE-CAC vs DTE | 1800 | 484 | 676 | 484/676 | 1.91e-08 | **DTE** |
| CTDE-VD vs DTE | 1800 | 490 | 660 | 490/660 | 5.97e-07 | **DTE** |
| CTDE vs CTE | 1800 | 616 | 478 | 616/478 | 3.37e-05 | **CTDE** |
| CTDE-Comm vs CTDE-VD | 1800 | 677 | 525 | 677/525 | 1.30e-05 | **Comm** |
| CTDE vs CTDE-Comm | 1800 | 549 | 664 | 549/664 | 0.00106 | **Comm** |
| CTDE vs CTDE-CAC | 1800 | 601 | 526 | 601/526 | 0.02746 | nominal only |
| CTDE-VD vs CTE | 1800 | 580 | 518 | 580/518 | 0.06559 | not significant |
| CTDE vs CTDE-VD | 1800 | 598 | 538 | 598/538 | 0.07999 | not significant |
| CTDE vs DTE | 1800 | 600 | 633 | 600/633 | 0.36214 | not significant |
| CTDE-CAC vs CTDE-VD | 1800 | 510 | 541 | 510/541 | 0.35477 | not significant |
| CTDE-CAC vs CTE | 1800 | 531 | 514 | 531/514 | 0.62066 | not significant |
| CTDE-Comm vs DTE | 1800 | 637 | 628 | 637/628 | 0.82205 | not significant |

**Four times as many separations as the previous, single-block version of this analysis reported** — two
pairs at 120 meetings per pair, then two at 600, now eight at 1,800. Read as a graph, the eight significant
edges are acyclic and give a clean partial order:

* **Top band, mutually indistinguishable:** CTDE-Comm and DTE (*p* = 0.82). Each beats four of the other
  four arms; between them the arena cannot tell them apart.
* **Middle:** plain CTDE, which beats CTE (*p* = 3.37e-05) and is separated from Comm only nominally.
* **Bottom band:** CTDE-VD and CTDE-CAC, mutually indistinguishable (*p* = 0.35), each beaten by Comm or
  DTE, and neither able to separate from CTE.
* **CTE is last**, losing three corrected-significant pairs (to Comm, DTE and CTDE) and winning none. It is
  the most separated-against arm in the pool; CTDE-CAC also wins none, but loses only two.

### But only four of those eight survive leaving a block out

Pooling blocks raises power and can still hide that one set of checkpoints produced the whole signal, which
is the same unit-of-analysis mistake this chapter makes a point of elsewhere. Re-testing every
corrected-significant pair on the subset that excludes each training block in turn:

| Pair | Pooled *p* | Without block 1 | Without block 2 | Without block 3 | Per-block discordant (solo *p*) | Robust |
|---|---:|---:|---:|---:|---|---:|
| CTDE-Comm vs CTE | 2.65e-13 | 8.64e-08 | 1.96e-05 | 5.36e-17 | 235/137 (4.25e-07), 237/111 (1.25e-11), 222/199 (0.284) | **yes** |
| CTE vs DTE | 1.11e-15 | 1.26e-10 | 1.05e-05 | 9.04e-19 | 148/242 (2.23e-06), 106/253 (5.42e-15), 187/218 (0.136) | **yes** |
| CTDE-VD vs DTE | 5.97e-07 | 0.00023 | 2.12e-05 | 2.12e-05 | 164/232 (0.00074), 163/214 (0.0099), 163/214 (0.0099) | **yes** |
| CTDE-CAC vs DTE | 1.91e-08 | 0.00327 | 7.93e-11 | 1.21e-05 | 118/225 (7.96e-09), 199/213 (0.522), 167/238 (0.00049) | **yes** |
| CTDE vs CTE | 3.37e-05 | **0.07894** | **0.13431** | 4.43e-12 | 234/144 (4.26e-06), 220/124 (2.55e-07), 162/210 (0.0147) | no |
| CTDE-Comm vs CTDE-VD | 1.30e-05 | **0.01042** | 0.00014 | 2.04e-05 | 245/166 (0.00011), 228/184 (0.034), 204/175 (0.150) | no |
| CTDE-CAC vs CTDE-Comm | 9.19e-08 | **0.17607** | 8.58e-12 | 1.05e-06 | 128/274 (2.62e-13), 204/197 (0.765), 171/217 (0.0222) | no |
| CTDE vs CTDE-Comm | 0.00106 | 0.00010 | 0.00124 | **0.35213** | 210/215 (0.846), 197/220 (0.281), 142/229 (7.32e-06) | no |

The four robust edges are exactly the ones that point **from the top band at the bottom band**: DTE over
CAC, VD and CTE, and Comm over CTE. Remove any one set of checkpoints and they still clear the corrected
threshold. The four fragile ones all involve a *middle* architecture — plain CTDE, or CTDE-Comm's relation to
VD and CAC — and each is carried by a single block. The clearest case is the pair this section previously
advertised as its headline significant result: **CTDE-Comm over plain CTDE**. It holds pooled
(*p* = 0.00106), but block by block it is 210/215, then 197/220, then 142/229 — no signal in the first two
blocks, all of it in the third, which is the promoted checkpoint set. That is precisely the failure mode the
single-caveat version of this analysis warned about, now measured instead of predicted.

### Replication across match streams, and across checkpoints

Two dimensions, two very different amounts of noise.

Harness seeds (the same six networks, five independent match streams):

| Architecture | 1 | 2 | 3 | 4 | 5 | Mean | SD | Ranks seen |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CTDE-Comm | 0.384 | 0.349 | 0.361 | 0.374 | 0.398 | 0.373 | 0.017 | 1, 1, 1, 2, 2 |
| DTE | 0.389 | 0.383 | 0.357 | 0.374 | 0.336 | 0.368 | 0.020 | 1, 1, 1, 2, 3 |
| CTDE | 0.306 | 0.314 | 0.336 | 0.346 | 0.346 | 0.329 | 0.016 | 2, 3, 3, 3, 3 |
| CTDE-VD | 0.298 | 0.289 | 0.312 | 0.291 | 0.296 | 0.297 | 0.008 | 4, 4, 4, 4, 5 |
| CTDE-CAC | 0.274 | 0.293 | 0.293 | 0.271 | 0.287 | 0.284 | 0.009 | 4, 5, 5, 5, 5 |
| CTE | 0.260 | 0.287 | 0.264 | 0.263 | 0.258 | 0.266 | 0.010 | 6, 6, 6, 6, 6 |

Training blocks (three independently trained networks per architecture, all five streams pooled inside
each):

| Architecture | Block 1 | Block 2 | Block 3 (promoted) | Mean | SD | Ranks seen |
|---|---:|---:|---:|---:|---:|---|
| CTDE-Comm | 0.399 | 0.359 | 0.362 | 0.373 | 0.018 | 1, 2, 2 |
| DTE | 0.364 | 0.369 | 0.370 | 0.368 | 0.003 | 1, 1, 3 |
| CTDE | 0.370 | 0.351 | 0.267 | 0.329 | 0.045 | 2, 3, 6 |
| CTDE-VD | 0.296 | 0.307 | 0.289 | 0.297 | 0.007 | 4, 5, 5 |
| CTDE-CAC | 0.204 | 0.351 | 0.296 | 0.284 | 0.061 | 3, 4, 6 |
| CTE | 0.264 | 0.196 | 0.339 | 0.266 | 0.059 | 3, 5, 6 |

Block 1 = `exp1_seed1+exp2_seed7`, block 2 = `exp1_seed10+exp2_seed5`, block 3 = `exp1_seed9+exp2_seed2`.

The pattern is the one the study's own variance analysis predicted: **the evaluation is reproducible, the
training is not.** Across match streams the SD is 0.008–0.020 and no architecture changes band; across
training blocks it is 0.003–0.061, and two architectures cross the whole table. Comm and DTE are stable in
both dimensions (block SD 0.018 and 0.003). Plain CTDE swings 0.370 → 0.351 → 0.267 and CTE swings the other
way, 0.264 → 0.196 → 0.339.

**This reverses the headline of the previous version of this section.** With one checkpoint per architecture
the ranking was DTE > Comm > CTE > CAC > VD > CTDE, and the summary sentence was "plain CTDE is last in all
five seeds". That was a property of the promoted replicate. Over three replicates CTDE is **third**, not
last, and CTE — the architecture experiment 1 was built to improve on — is **last**, not third. The claim
that survived both ways is that the paradigm experiment 1 promoted is not at the top of the pool.

### The pooled order is the fight order, not the buzzer order

35.9 % of these matches ran to the 90 s clock, so a pooled win rate mixes "won the fight" with "was most
alive at the buzzer" — the concern item 8 of the audit's open-work list raises. Splitting the same 9,000
matches by how they ended (a column the report now prints, `scripts/report_cross_arena.py`):

| Architecture | Wipeout matches | Timeout matches | Drawn | Pooled | Wipeout − timeout |
|---|---:|---:|---:|---:|---:|
| CTDE-Comm | 0.427 (2,738) | 0.320 (1,594) | 0.000 (168) | 0.373 | **+0.107** |
| DTE | 0.394 (2,653) | 0.369 (1,654) | 0.000 (193) | 0.368 | +0.025 |
| CTDE | 0.349 (2,747) | 0.332 (1,574) | 0.000 (179) | 0.329 | +0.018 |
| CTDE-VD | 0.288 (2,557) | 0.345 (1,743) | 0.000 (200) | 0.297 | −0.057 |
| CTDE-CAC | 0.279 (2,782) | 0.330 (1,515) | 0.000 (203) | 0.284 | −0.051 |
| CTE | 0.261 (2,717) | 0.303 (1,616) | 0.000 (167) | 0.266 | −0.041 |

The wipeout column reproduces the pooled ranking exactly, so the result is not an artefact of survival: the
arm that finishes fights best also wins the most. What the split does show is a second axis, engagement
versus endurance. CTDE-Comm's advantage is 0.107 larger among decided fights than among clock-outs; VD, CAC
and CTE win a *higher* share of timeout matches than of wipeouts, which is what a survive-and-wait strategy
looks like. That ordering is stable enough to be worth its own measurement and is consistent with the
elimination and survival columns above (VD survives 23.4 s against Comm's 20.4 s while eliminating 0.65 fewer
agents per match). It is not enough to change any of the fifteen comparisons: the regimes are descriptive,
and every corrected significance in this section is computed on all matches together.

### The hardest limit of this comparison <a name="cross-arena-limits"></a>

**Closed:** each architecture is no longer one checkpoint. Three independently trained networks represent
each of them, the checkpoints are versioned, and a reader can rebuild `results/cross_arena/blocks.csv` from
them — see [§ 9.3](09_reproducibility.md#93-commands) for the fifteen-run loop. Re-merging the fifteen run
directories reproduces the versioned CSV's SHA-256 exactly, so the pooled table is derived data, not a
transcription.

**Still open, in order of what it costs:**

* **n = 3 blocks.** The between-block SD of 0.003–0.061 is enough to reverse a ranking, as it just did,
  which is the argument for why three is not enough to settle one. Six blocks would cost about twice what
  these three did, and all of it is evaluation time — the other replicates' checkpoints already exist under
  `data/runs/`, just not in version control.
* **Block pairings are not balanced across experiments.** A block is one experiment-1 seed times one
  experiment-2 seed, and the three exp-1 arms and three exp-2 arms each see the same three pairings. A
  cross-experiment comparison therefore cannot separate "this exp-1 replicate was strong" from "this
  exp-1 x exp-2 pairing interacted" with only three pairings.
* **It is still the same arena.** Every number here inherits the spawn bias below and the 90 s clock, and
  the round-robin says nothing about transfer to another game.

### The slot effect, measured at scale

Pooled over 9,000 team-matches per slot, with every architecture spending exactly one third of its matches
in each:

| Slot | Pooled win rate |
|---|---:|
| Team 1 | 0.216 |
| Team 2 | 0.276 |
| **Team 3** | **0.467** |

All six architectures do best in Team 3 regardless of what they are; within an architecture the gap between
its best and worst slot is 0.17–0.33. The arena's three spawn centres are two far corners and the middle of
the near edge, and the obstacle layout is not three-fold symmetric, so the third position is objectively
worth about twice a corner. Documented as [A-32](11_code_audit.md#the-arena-spawn-geometry-is-not-symmetric).

It also explains a scare during development: the first version of this harness passed the paradigms in
alphabetical order rather than the recorded order, which moved CTDE-Comm from 0.253 to 0.700. That looked
like a broken instrument and was in fact a correct instrument measuring a biased arena.

The rotation makes the pooled comparison fair in expectation, but the slot effect inflates the variance of
every number here and of every number in [§ 8.1](#replicated-study), and is a plausible contributor to the
0.12–0.22 between-seed spread that makes the study's own ranking unresolvable.

### What this adds to the conclusions

The cross-arena design removes the co-adaptation confound, balances slots, uses a deterministic instrument
and a paired test that respects the dependence between arms. Over 9,000 matches and three independently
trained networks per architecture, the six arms are **not** all indistinguishable: eight of fifteen pairs
clear Bonferroni-corrected McNemar tests, and **four of those still clear the threshold with any one set of
checkpoints removed**. Those four all separate the top band (CTDE-Comm, DTE) from the bottom band (CTE,
CTDE-CAC, CTDE-VD) — so the defensible statement is a partial order with two bands, not a ranking.

Two things this section said before the replication are now corrected rather than merely extended. Plain
CTDE is not last; it is third, and the checkpoint that put it in the bottom band was one of three. And the
pair that carried the previous headline — CTDE-Comm over plain CTDE — is real in the pooled data and absent
from two of its three blocks, which is why the leave-one-out column is now part of the instrument.

That is still a stronger claim than the study's own, because it is no longer conditional on each arm
fighting its own neighbours — and it was only reachable by replicating the evaluation twice over, once in
the match stream and once in the checkpoints.

---

## 8.11 Absolute strength against a fixed random pool <a name="absolute-strength"></a>

Every win rate above is **relative**: an arm beating two other arms that trained against it. That is why
nothing in [§ 8.1](#replicated-study) or [§ 8.10](#cross-arena) can be placed next to a number from the
literature. `scripts/run_absolute_eval.py` measures the other thing: one trained team, two teams whose
controllers draw uniformly from the action space, and a floor that does not depend on who else was in the run.

Three design points before the numbers.

* **The opponents are drawn, not argmax'd.** The greedy action of an untrained network is a near-constant
  policy ([A-30](11_code_audit.md#greedy-evaluation-of-an-untrained-network-is-not-random-play)), so a
  "random" opponent read off untrained weights would be a stationary target rather than a random one. The
  controller here is the real class with its action selection replaced by a uniform draw.
* **The null is 0.330, not 1/3.** Each arm spends nearly a third of its matches in each spawn slot, and the
  slots are not equal: [§ 8.9](#random-baseline) measured three *mutually random* teams splitting
  0.257 / 0.270 / 0.463 by slot. Averaged over the rotation that gives **0.3300**, which is close to 1/3 by
  construction rather than by symmetry — and it is the matched comparison, because a random focal team in
  that baseline run also faced two random opponents.
* **Three checkpoint sets, 450 matches per architecture.** `results/absolute_blocks/` plays each arm's three
  independently trained replicates (the same blocks as [§ 8.10](#cross-arena)) over three harness seeds at 50
  matches per architecture per block per seed. The rotation is not exactly balanced, because 50 is not a
  multiple of 3: the pooled slot counts are 918 / 918 / 864 rather than 900 each, and the largest effect on
  an arm's pooled rate is 0.0025. `results/absolute/` is the same instrument on the promoted checkpoints
  only, at 150 matches per architecture per seed; it is kept because it is the case study below.

### Every arm clears the floor, and the two scales agree

Pooled over the three checkpoint sets:

| Rank | Architecture | Win rate vs random | 95 % CI | Excess over the matched null | Draws | Losses | SD across blocks |
|---|---|---:|---|---:|---:|---:|---:|
| 1 | DTE | **0.753** | [0.712, 0.791] | +0.423 | 0.002 | 0.244 | 0.022 |
| 2 | CTDE-Comm | **0.740** | [0.698, 0.778] | +0.410 | 0.000 | 0.260 | 0.022 |
| 3 | CTDE | **0.673** | [0.629, 0.715] | +0.343 | 0.002 | 0.324 | 0.153 |
| 4 | CTDE-VD | **0.671** | [0.626, 0.713] | +0.341 | 0.002 | 0.327 | 0.079 |
| 5 | CTDE-CAC | **0.607** | [0.561, 0.651] | +0.277 | 0.004 | 0.389 | 0.129 |
| 6 | CTE | **0.558** | [0.512, 0.603] | +0.228 | 0.004 | 0.438 | 0.086 |

Matches are short and decided by elimination: median 11.0 s of the 90 s cap (10.6 s when the trained team
wins, 12.0 s when it loses), only 4.1 % reach the clock, and 0.0–0.4 % are drawn — against 35.9 % timeouts
and 4.1 % draws in the round-robin. Against random opponents the buzzer regime barely exists, which is part
of why this scale is cleaner than the peer-relative one.

**This is the repository's strongest positive result.** Ranking the same six architectures by how often they
beat *trained peers* and by how often they beat *uniform noise* — two opponent pools with nothing in common
except the arena — gives:

| Architecture | Rank vs random | Rank vs trained peers |
|---|---:|---:|
| DTE | 1 | 2 |
| CTDE-Comm | 2 | 1 |
| CTDE | 3 | 3 |
| CTDE-VD | 4 | 4 |
| CTDE-CAC | 5 | 5 |
| CTE | 6 | 6 |

Spearman **+0.94** (average ranks) and Pearson **+0.95** on the win rates, computed by
`run_absolute_eval.py --compare-pool results/cross_arena/blocks.json`. The only disagreement at the top is
which of DTE and CTDE-Comm is first — the two arms [§ 8.10](#cross-arena) could not separate from each other
(*p* = 0.82). The bottom of one scale is the bottom of the other: CTE last against random play, last against
trained opponents.

Per checkpoint set, rather than pooled, the agreement is **+0.77, +0.70 and +0.41** (blocks
`exp1_seed1+exp2_seed7`, `exp1_seed10+exp2_seed5`, `exp1_seed9+exp2_seed2`). It rises as replicates are
averaged, which is what noise-dominated measurement predicts: within one set of six networks the two
instruments mostly agree, and averaging three sets makes them agree almost completely.

### The correction that produced this result

An earlier version of this section, run on **one** checkpoint set, reported the opposite: that the two
scales were uncorrelated (Spearman −0.04), with CTE last against peers and second against random play.
That was wrong in two ways at once, and both are the same lesson [§ 8.10](#cross-arena) already teaches.

* **Scope mismatch.** The absolute numbers came from one checkpoint set while the peer ranks came from the
  pooled three. Compared like for like, that same block gives **+0.41**, not −0.04.
* **The block chosen was the atypical one.** It is the promoted replicate, the set that put plain CTDE last
  in the round-robin and CTE third. On it, CTDE reads 0.460 against random play and CTE 0.673 — the two
  arms exactly reversed relative to the three-block picture.

The promoted-only measurement is still reported in `results/absolute/` and in the artefact inventory, so the
reversal is checkable rather than hidden. The general point is the one worth keeping: **a single replicate
destroys the correlation between two independent measurements of the same six networks**, even when each
measurement uses 450 matches. Sample size inside a replicate is not a substitute for replication across
replicates.

### Slot sensitivity, and one cell that is not an architecture effect

Pooled over the three sets, the trained team wins 0.662 / 0.588 / 0.756 from Team 1 / 2 / 3 — the same
direction as the round-robin's 0.216 / 0.276 / 0.467, compressed because half the field is random noise.
Per architecture the best-to-worst slot gap is 0.14–0.26.

The sharpest cell in the whole instrument is plain CTDE from Team 2 in the promoted block: **0.118**, against
0.765 and 0.745 for the other two blocks' CTDE in that same slot, and 0.270 for a purely random team there.
A checkpoint of the CTDE architecture therefore lost to uniform noise from one corner far more often than
noise loses to noise, while two other checkpoints of the same architecture won three quarters of the time.
That is a property of one network, not of centralised training with decentralised execution, and the
three-block value (0.542) is unremarkable.

### What this section can and cannot support

**Can:** that all three trained replicates of every architecture beat uniform-random play by an interval
that excludes the matched null; that the six architectures rank nearly identically on two opponent pools
that share nothing but the engine (**+0.94** pooled, +0.41 to +0.77 within each set); and that a
single-replicate measurement, at any sample size, can invert that relationship.

**Cannot:** place any number here against a published result — uniform-random play in this arena is not any
paper's baseline, and there is still no scripted opponent, no second game and no shared benchmark; claim
that 0.558 for CTE means it is weak, since it is +0.228 over its own matched null; or treat the
three-checkpoint agreement as a substitute for significance — [§ 8.10](#cross-arena) still cannot separate
CTDE from CTDE-VD or CTDE-CAC, and agreement between two noisy scales is consistency, not proof.

---

## 8.12 Arena v2: what happens when surviving the clock is not a win <a name="arena-v2"></a>

Two of the four remedies chapter 10 names as the reason the study cannot resolve an architecture ranking
were about the objective rather than the statistics: the alive bonus dominates the discounted return
([§ 4.4.1](04_mdp_formalisation.md#441-reward-scale-versus-horizon)), and a third of matches are decided by
the 90 s clock, where winning means "most alive at the buzzer"
([§ 8.1](#two-regimes)). Both are now configuration, not source
([A-34](11_code_audit.md#the-termination-and-reward-knobs-are-configurable)), so both were tested.

`scripts/run_arena_v2_campaign.sh` re-runs the whole replicated study — 10 seeds x 2 experiments, 500,000
steps, slot rotation, 150 held-out greedy matches per replicate — changing exactly three keys:

| Key | Committed arena | Arena v2 |
|---|---:|---:|
| `REWARD_ALIVE_BONUS` | 0.015 | 0.0015 |
| `REWARD_ALIVE_DECAY_HALF_LIFE_S` | 0 (off) | 30 |
| `TERMINATION_CAP_AWARDS_WIN` | true | false |

Survival is worth a tenth as much and halves every 30 s of the match, and a match that runs to the clock is
a **draw** instead of a win for whichever team has more agents standing. Everything else is the same code,
the same seeds and the same budget; wall clock was 104.3 min for the 20 runs at 20-way concurrency (the
committed study took 117.5 min), and `results/study_v2/` holds the aggregates.

### The draw regime is larger than the timeout count suggested

Under the committed rule, held-out matches ended in a draw 4.1 % of the time in experiment 1 and 2.0 % in
experiment 2 — because a clock-out already had a winner. With the rule off, the same replicates fail to
produce a winner in **44.3 %** (experiment 1, range 20.0–87.3 % across seeds) and **39.0 %** (experiment 2,
22.0–70.7 %) of held-out matches.

That is the direct measurement of the confound the two-regimes analysis inferred from median match durations:
roughly four matches in ten of the study's headline response variable were awarded by the buzzer, not won by
elimination. One replicate (`exp1_seed2`) is an extreme case: 87 % of its held-out matches are draws under
v2, and its three teams read 0.073 / 0.040 / 0.013 wins (CTDE / CTE / DTE), against 0.220 / 0.240 / 0.527
for the same seed in the committed arena, where a clock-out still had a winner.

### Win rates fall by about the draw share, and nothing separates

| Experiment | Paradigm | Committed win rate (SD) | Arena v2 win rate (SD) | SD of the change |
|---|---|---:|---:|---:|
| 1 | CTDE | 0.326 (0.187) | **0.212** (0.155) | −0.114 |
| 1 | DTE | 0.381 (0.191) | **0.176** (0.148) | −0.205 |
| 1 | CTE | 0.252 (0.122) | **0.169** (0.134) | −0.083 |
| 2 | CTDE-VD | 0.367 (0.171) | **0.255** (0.208) | −0.112 |
| 2 | CTDE-Comm | 0.331 (0.218) | **0.166** (0.185) | −0.165 |
| 2 | CTDE-CAC | 0.282 (0.182) | **0.189** (0.099) | −0.093 |

No seed-level paired comparison reaches significance in either experiment under v2 — experiment 1 gives
*p* = 0.528 and 0.646 for CTDE against CTE and DTE, experiment 2 gives *p* = 0.750 and 0.439 — and the
forward-power column is as prohibitive as ever (~246 to ~10,662 replicates). The match-level Fisher test,
which [§ 8.1](#statistical-significance-historical) already shows to be anti-conservative on
these data, calls CTDE above both CTE (*p* = 0.0034) and DTE (*p* = 0.0144) in experiment 1; that is
reported as the instrument's opinion, not as a result.

**The ranking of experiment 1 flips** — DTE first under the committed rule, CTDE first under v2 — but the
difference is 0.036 with an SD of 0.240, so the flip is well inside the noise the study has been reporting
all along. It is the same lesson as [§ 8.1](08_results.md#leave-one-out), arrived at from the other side: an
ordering that survives neither a different replicate nor a different rule is not a measurement of the
architectures.

### What the change did *not* do

Three expectations from the diagnosis, and what the data says:

* **Variance was expected to shrink; it barely did.** Mean between-seed SD goes from 0.167 to 0.146 in
  experiment 1 and 0.190 to 0.164 in experiment 2 — a 10–13 % reduction against a spread that is still
  smaller than the differences between arms in the committed arena. The instability is in training, not in
  the buzzer.
* **Engagement was expected to rise once survival paid less; it is mixed.** CTDE fires 140.3 shots per
  match under v2 against 101.2 under the committed rule and eliminates more (2.27 vs 1.97), but CTE fires
  *less* (107.4 vs 121.7) and DTE eliminates *fewer* agents (1.72 vs 1.95). Making survival cheap does not
  reliably buy aggression at this reward scale, which weakens the "no cost for not engaging" reading that
  motivated the knob.
* **The two experiments did not converge on one story.** In experiment 2 the order of CTDE-Comm and CTDE-CAC
  swaps relative to the committed arena while VD stays first, so the v2 ranking is not a cleaned-up version
  of the v1 ranking in either experiment.

### Standing of this section

Arena v2 is **not** the committed configuration: every other number in this document, including
[§ 8.10](#cross-arena) and [§ 8.11](#absolute-strength), is the default arena, and the defaults are the old
constants ([A-34](11_code_audit.md#the-termination-and-reward-knobs-are-configurable) is about the knobs
existing, not about them being on). The v2 replicates were trained before `run_summary.json` recorded
`domain_randomization`; domain randomisation was on in both campaigns, so the comparison holds, and the
three ablations queued behind this one (`scripts/run_ablation_campaign.sh`) do record it.

What v2 does settle is the accounting: the repository can now state what fraction of its headline response
variable was decided by the clock rather than by fighting (about four matches in ten), and that removing
that route does not resolve the ranking — it lowers every win rate, leaves the spread almost as large, and
moves the top of experiment 1 inside noise.


## 8.13 Configuration ablations: one key at a time <a name="ablations"></a>

The study fixes every knob by convention and varies only the paradigm, which the protocol calls a control.
A control that is never varied cannot say whether the ranking came from the paradigm or from the
configuration. Three arms re-train experiment 1 **and** experiment 2 at 500,000 steps on seeds 1, 2 and 3 —
the same seed numbers as the committed study, so the paradigm-to-slot rotation matches the control seed by
seed and every comparison is paired — changing exactly one key each.

| Arm | Key | Control → arm |
|---|---|---|
| `no_dr` | `DOMAIN_RANDOMIZATION` | true → false |
| `narrow_64` | `RL_HIDDEN_DIM` | 128 → 64 |
| `wide_256` | `RL_HIDDEN_DIM` | 128 → 256 |

**The knobs were checked before their results were believed.** Actor parameter counts read back from the
saved checkpoints are 9,545 / 35,465 / 136,457 for the three widths, and every `no_dr` run records
`"domain_randomization": false` in its own `run_summary.json`. The second is only possible because of
[A-33](11_code_audit.md#the-study-runner-overrode-the-domain-randomisation-knob): before that fix the arm
would have re-trained the control and the difference would have been reported as an effect.

Reproduce with `python scripts/report_study.py --results-dir results/ablation_no_dr --compare-to results/study`
(same for the other two arms).

### A configuration key moves the outcome more than a paradigm does

Held-out greedy win rate, mean over seeds 1-3, paired against the control on the same seeds:

| Paradigm | Control | `no_dr` | Δ | *p* | below | `narrow_64` | Δ | *p* | `wide_256` | Δ | *p* |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **exp1** CTDE | 0.478 | 0.284 | −0.193 | 0.430 | 2/3 | 0.438 | −0.040 | 0.750 | 0.491 | +0.013 | 0.774 |
| **exp1** CTE | 0.184 | 0.287 | +0.102 | 0.518 | 1/3 | 0.120 | −0.064 | 0.255 | 0.124 | −0.060 | **0.016** |
| **exp1** DTE | 0.320 | 0.202 | −0.118 | 0.496 | 2/3 | 0.351 | +0.031 | 0.894 | 0.373 | +0.053 | 0.340 |
| **exp2** CAC | 0.318 | 0.302 | −0.016 | 0.935 | 2/3 | 0.302 | −0.016 | 0.916 | 0.320 | +0.002 | 0.965 |
| **exp2** Comm | 0.353 | 0.100 | **−0.253** | 0.238 | **3/3** | 0.333 | −0.020 | 0.737 | 0.280 | −0.073 | 0.076 |
| **exp2** VD | 0.302 | 0.349 | +0.047 | 0.892 | 2/3 | 0.331 | +0.029 | 0.703 | 0.382 | +0.080 | 0.276 |

"below" counts the arm's seeds that landed under the control's value on the same seed. With three seeds and
18 comparisons the corrected threshold is alpha = 0.0028 and **no cell in this table is significant** — two
reach nominal p < 0.05. What carries the finding is the size of the columns, not their p-values:

* **The largest single-key move is 0.253** (CTDE-Comm, `no_dr`) and the second is 0.193 (CTDE, `no_dr`).
  For scale, the entire between-paradigm spread in the ten-seed study is 0.129 in experiment 1 (DTE 0.381
  over CTE 0.252) and 0.085 in experiment 2 (VD 0.367 over CAC 0.282). **One configuration key moves these
  outcomes further than the thing the study was built to measure.**
* **The rankings invert.** In experiment 1 the control on these seeds reads CTDE > DTE > CTE; `no_dr` reads
  CTE ≈ CTDE > DTE, putting the control's last arm first. In experiment 2 the control reads Comm > CAC > VD
  and `no_dr` reads VD > CAC > Comm, with Comm at 0.100 — under the control's *worst* seed in every one of
  its own three.
* **Capacity is nearly free, which is the opposite of what the parameter counts suggest.** Halving the
  network costs at most 0.064 and changes no ranking. Doubling it is similarly mild, and in both
  directions the arms stay within the seed-to-seed noise the study already reports (between-seed SD
  0.122–0.218 in the ten-seed run). A reader who inferred "CTDE-Comm loses because its critic is too
  small" from the parameter table would be wrong in the direction of that inference.

### The arms play a different game, not only a different score

The behavioural columns show the configuration changes *what the policies do*, which matters for how much
weight their win rates can carry:

* Turning domain randomisation off roughly **triples mean survival** (13.9–21.8 s in the control to
  40.0–46.1 s) while eliminations fall (2.77 → 1.94 for CTDE, 2.29 → 0.24 for Comm). With a fixed arena the
  agents do not finish each other; they stall. Those win rates are therefore more contaminated by the
  buzzer than the control's, which [§ 8.12](#arena-v2) showed to be a large effect on its own — the two
  knobs interact through the termination rule, and neither arm's number can be read as pure fighting skill.
* `narrow_64` **doubles CTDE's firing rate** (95 → 203 shots per match) at lower accuracy (3.40 % → 1.82 %):
  a smaller network shoots more and hits less. The same pattern appears for CTE (60 → 154 shots).
* `wide_256` raises CTDE's accuracy to 4.64 % while cutting its volume to 59 shots — the opposite policy,
  same nominal win rate (0.491 against the control's 0.478). Two configurations reaching the same score by
  different means is exactly what makes the pooled win rate a weak summary.

### Status of this section

`scripts/run_ablation_campaign.sh` runs the three arms in series behind the arena-v2 campaign. Its header
comment said experiment 1 only, while `run_study.py` defaults to both experiments, so what actually ran is
**both** — 6 runs per arm and 18 in total. The campaign was left as launched rather than restarted, so the
results are the wider ones, and the script's comment has since been corrected to describe the command.

Two collection notes. `no_dr`'s first aggregate came out with 9 rows instead of 18 because of
[A-35](11_code_audit.md#the-study-collector-read-only-one-of-the-two-trees); `--collect-only` rebuilt it
from the run summaries already on disk, with no re-training. And on this mostly-idle machine six concurrent
replicates run at 243–350 steps/s in 24–34 minutes, against the 85–92 steps/s and 90–101 minutes the
committed study measured under its 20-way load ([§ 9.5](09_reproducibility.md#95-cost-of-a-run)).

**What this does not do:** it does not re-run the common-opponent pool or the random-pool instrument under
these configurations, so it cannot say whether the [§ 8.10](#cross-arena) band structure or the
[§ 8.11](#absolute-strength) agreement survives a different `RL_HIDDEN_DIM` or no domain randomisation —
each arm still measures teams that co-adapted inside their own run. It also does not close the power gap:
three seeds is the sample the study's own forward-power table puts an order of magnitude short of what
would resolve the differences it reports.

**What it establishes:** the repository's architecture comparisons are conditional on one configuration, and
that configuration is not a neutral backdrop — moving one key reorders both experiments. Any statement of
the form "paradigm X beats paradigm Y" in this document should be read as "under hidden width 128 and
domain randomisation on".


------

## 8.14 Summary of answers <a name="summary-of-answers"></a>

| Question | Answer from the replicated study (10 replicates per arm) |
|---|---|
| **RQ1** — does CTDE beat CTE and DTE? | **No, and the ordering is not ours to report.** CTDE finishes **second** at 0.326 behind DTE at 0.381 and ahead of CTE at 0.252. The paired per-seed differences are +0.074 vs CTE (*p* = 0.348) and −0.055 vs DTE (*p* = 0.640); detecting them would need ~110 and ~450 replicates. Removing the single degenerate replicate (seed 6) puts CTDE first at 0.362 and flips the sign of the CTDE–DTE difference — so the honest answer is that **this study measures no exp-1 ordering at all.** |
| **RQ2** — which CTDE variant wins? | **No detectable difference.** VD 0.367, Comm 0.331, CAC 0.282; every paired per-seed comparison has *p* ≥ 0.363, and the smallest gap costs ~117 replicates to confirm. The historical claim that Comm loses decisively **does not reproduce**, and the historical winner CAC is now last of three. |
| **RQ3** — is it stable? | **No — and that is the headline result.** Between-seed SD is 0.122–0.218 against a 1/3 chance baseline; single replicates range from **0.000 to 0.767** within one architecture; 1 of 30 paradigm-slots reads out as a never-fires policy under greedy action selection while the same network led its own match on kills during training ([§ Readout](#readout-divergence)); 5 of 20 replicates play a different game entirely, matches decided by the 90 s clock rather than by kills. The dominant finding is variance, not ranking, and most of that variance is introduced by the readout rather than by training. |
| **RQ4** — do the two experiments' numbers belong on one scale? | **Yes, as a two-band partial order — and replicating it twice changed the ranking.** The cross-experiment round-robin ([§ 8.10](#cross-arena)) puts all six arms on common opponents over **9,000 matches and three independently trained networks per architecture**: Comm 0.373, DTE 0.368, CTDE 0.329, VD 0.297, CAC 0.284, CTE 0.266. Eight of fifteen pairs clear Bonferroni-corrected McNemar, and four clear it with any one set of checkpoints removed; those four all separate {Comm, DTE} from {CTE, CAC, VD}. Two claims of the single-checkpoint version did not survive: plain CTDE is **third, not last** — it was last only for the promoted replicate — and the earlier headline, Comm > CTDE, is present in one block of three. The first 600-match run said no pair separates at all, so this answer is a product of replication. |

**What this repository can defensibly claim.**

1. Six MARL information-flow architectures have been run against each other in one engine with matched
   hyperparameters, replicated ten times each with slot rotation, and measured held out — and the ranking a
   single seed produced turned out to be wrong in *both* experiments.
2. On a common opponent pool, over 9,000 matches and three trained networks per architecture,
   **eight cross-experiment differences clear multiplicity correction and four of them survive leaving any
   one set of checkpoints out**: DTE over CTE, CAC and VD, and Comm over CTE. The other four — Comm over
   CTDE, VD and CAC, and CTDE over CTE — are significant pooled but lose the corrected threshold as soon as
   one particular block is removed ([§ 8.10](#cross-arena)). The robust set is a partial order with a top
   band and a bottom band, not a ranking. This is the only family of claims in the repository that clears a
   corrected significance threshold, and it appeared only after the evaluation was replicated in both
   dimensions.
3. A measurement finding that is probably the most consequential thing here: the greedy readout used as the
   headline response variable has 2.5× the spread of the sampled policy it summarises, and agrees with its
   rankings only 35.6 % of the time ([§ Readout](#readout-divergence)).
4. The arena itself is biased: one of the three spawn positions wins 47 % of matches against 22 % and 28 %
   for the other two, for every architecture
   ([A-32](11_code_audit.md#the-arena-spawn-geometry-is-not-symmetric)).
5. Every architecture — all three trained replicates of each — beats uniform-random play by an interval
   that excludes the slot-matched null (+0.228 to +0.423), and the two opponent pools rank the six
   architectures **nearly identically**: Spearman +0.94 and Pearson +0.95 between "beats trained peers" and
   "beats random play", with the same arm last on both scales. The one disagreement is which of DTE and
   CTDE-Comm is first, which [§ 8.10](#cross-arena) also cannot settle ([§ 8.11](#absolute-strength)). The
   same comparison on a single replicate gave −0.04, which is the sharpest available demonstration that
   checkpoint replication, not match count, is what this arena's measurements need.
6. Re-measuring the study under an arena that removes both objective-side confounds — an alive bonus an
   order of magnitude smaller and decaying with a 30 s half-life, and no win for surviving the clock — shows
   that **44 % and 39 % of held-out matches have no winner at all**, that no arm separates (paired *p* ≥
   0.439), and that experiment 1's top place moves from DTE to CTDE inside noise. The study's inability to
   rank the paradigms is not an artefact of the buzzer rule, and removing that route is not a path to a
   resolution ([§ 8.12](#arena-v2)).
7. The comparison is **conditional on one configuration**, and that configuration is not a neutral
   backdrop: changing one key — domain randomisation off — moves a paradigm's held-out win rate by up to
   **0.253** and inverts the ranking of both experiments, while the whole between-paradigm spread of the
   ten-seed study is 0.085–0.129. Every "X beats Y" statement here should be read as "under hidden width
   128 and domain randomisation on" ([§ 8.13](#ablations)).

**What it cannot claim:** that any architecture is best, or that the six admit a strict ordering — CTDE-Comm
and DTE cannot be told apart, and neither can CTDE-CAC, CTDE-VD and CTE; that the ordering is settled by
three checkpoints per architecture, since replacing one block of three reversed two positions in the pooled
table; that within-experiment differences exist at all, since every corrected significance is
*cross*-experiment; that win rate here measures fighting skill rather than, in a third of matches,
survival to the clock; that the ablated configurations generalise, since the three arms were measured only
inside their own co-adapted runs and never through the common-opponent or random pools; or that any of it
holds under a different setting of the keys [§ 8.13](#ablations) shows to matter.
