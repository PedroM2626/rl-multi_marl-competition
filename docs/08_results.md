# 8. Results

All numbers in this chapter are recomputed from artefacts in the repository, by scripts rather than by
hand. The mapping from section to command is in
[§ Reproducing the numbers](README.md#reproducing-the-numbers); the ones this chapter depends on are
`scripts/report_study.py` (every study table, the paired *t* and Fisher tests, and the forward-power
table), `scripts/plot_study.py` (`results/study/learning_curves.png`) and `scripts/random_baseline.py`
([§ 8.9](#random-baseline)). The leave-one-out table in [§ 8.1](#leave-one-out) calls
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

The exp-1 per-seed matrix contains a degenerate replicate: **seed 6's CTDE team never learned to shoot.**
Over its 150 held-out matches it fired 3.8 shots per match against 325 and 372 for the other two teams,
recorded 0.01 eliminations per match, and won **0.000** of its matches — a zero on a 1/3 chance baseline.
It is the only such collapse in 60 paradigm-seed cells (threshold: held-out eliminations below 0.25 per
match), and it is what drags CTDE's mean from 0.362 to 0.326.

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
way. The exp-1 ranking is therefore not a finding of this study; the collapse itself is. It is a property of
PPO in this reward landscape — a policy that stops firing forfeits nothing early, gets no negative signal
until matches run to the 90 s cap, and can sit in that basin for the whole budget.

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

## 8.10 Summary of answers <a name="summary-of-answers"></a>

| Question | Answer from the replicated study (10 replicates per arm) |
|---|---|
| **RQ1** — does CTDE beat CTE and DTE? | **No, and the ordering is not ours to report.** CTDE finishes **second** at 0.326 behind DTE at 0.381 and ahead of CTE at 0.252. The paired per-seed differences are +0.074 vs CTE (*p* = 0.348) and −0.055 vs DTE (*p* = 0.640); detecting them would need ~110 and ~450 replicates. Removing the single degenerate replicate (seed 6) puts CTDE first at 0.362 and flips the sign of the CTDE–DTE difference — so the honest answer is that **this study measures no exp-1 ordering at all.** |
| **RQ2** — which CTDE variant wins? | **No detectable difference.** VD 0.367, Comm 0.331, CAC 0.282; every paired per-seed comparison has *p* ≥ 0.363, and the smallest gap costs ~117 replicates to confirm. The historical claim that Comm loses decisively **does not reproduce**, and the historical winner CAC is now last of three. |
| **RQ3** — is it stable? | **No — and that is the headline result.** Between-seed SD is 0.122–0.218 against a 1/3 chance baseline; single replicates range from **0.000 to 0.767** within one architecture; 1 of 30 paradigm-slots collapses to a never-fires policy; 5 of 20 replicates play a different game entirely (matches decided by the 90 s clock, not by kills). The dominant finding is variance, not ranking, and the [leave-one-out analysis](#leave-one-out) shows a ranking that a single replicate can reverse. |

**What this repository can defensibly claim:** six MARL information-flow architectures have been run against
each other in one engine with matched hyperparameters, replicated ten times each with slot rotation, and
measured held out — and the resulting ranking is not statistically distinguishable, while the ranking a
single seed produced turned out to be wrong in *both* experiments. It additionally contributes a concrete
characterisation of a PPO degenerate-attractor failure mode in a sparse-elimination reward.

**What it cannot claim:** that any of the six architectures is better than any other; that the CTDE
advantage over CTE is anything more than suggestive; or that win rate here is measuring fighting skill
rather than, in a quarter of replicates, survival to the clock.
