# 7. Experimental protocol

## 7.1 Design

Both experiments use a **between-subjects competitive design**: three architectures occupy the three
team slots simultaneously in the same match, share the environment, the reward function, the
observation encoding, the optimiser hyperparameters and the step budget, and the response variable is
the competitive outcome between them.

There is no control arm that could isolate the architecture from the slot. Team identity is confounded
with:

* **spawn corner** — Team 1 at \((-m,1,-m)\), Team 2 at \((+m,1,-m)\), Team 3 at \((0,1,+m)\) in the
  fixed variant;
* **RNG seed** — controller seeds are `RANDOM_SEED + {11, 23, 37}`;
* **initial heading** — drawn per agent from the simulation RNG in team order;
* **parameter-initialisation order** — networks are constructed in team order from the same global
  Torch RNG state, so the three teams start from different random weights by construction.

Under domain randomisation the spawn geometry is re-sampled every match, which **partially** breaks the
first confound during training but not during evaluation, where the fixed variant is used. Nothing
breaks the seed or initialisation confounds: no seed replication was run. See
[§ Threats to validity](10_threats_to_validity.md#101-internal-validity).

## 7.2 Procedure

```
1.  ArenaSimulation(domain_randomization=True)
      → three RLTeamControllers, each loading data/checkpoints/<slug>_<paradigm>.pt if present
2.  set_rl_training(controllers, True)          # stochastic action sampling, rollout recording on
3.  loop until total_env_steps >= RL_TRAIN_TOTAL_STEPS:
      step(dt) until the match terminates
      finish_match()                             # assign win, update metrics, one PPO update per team
      every RL_METRICS_EVERY_MATCHES: record_match()  → CSV append + summary.json + dashboard PNG
      every RL_LOG_EVERY_STEPS:       append training_log entry (+ MLflow metrics in experiment 2)
      every RL_SAVE_EVERY_STEPS:      save checkpoints
      reset_match()                              # sample a new domain-randomised variant
4.  save checkpoints, write training_log.json
5.  evaluation / visualisation: python main.py
      → ArenaSimulation(domain_randomization=False), set_rl_training(..., False)
      → greedy argmax actions, fixed default variant, metrics still appended to the same CSVs
```

Two properties of this procedure are load-bearing:

* **The reported metrics are training-time metrics, not evaluation metrics.** `record_match` is called
  from inside the training loop, so every row in `team_match_metrics.csv` comes from a match played by
  a mid-training policy with stochastic action sampling. No held-out evaluation of the final policies
  was performed. `main.py` would produce evaluation data — it writes to the same `MetricsStore` — but
  it requires a display and no such run is versioned.
* **`summary.json` is a cumulative average over the whole non-stationary trajectory**, not a measurement
  of the final policy. A team that was strong at match 50 and weak at match 450 contributes equally to
  both. [§ Results](08_results.md#drift) quantifies how much this matters.

## 7.3 Runs actually performed

| Run | Experiment | Target steps | Matches | Evidence |
|---|---|---:|---:|---|
| Historical single-seed run 1 | 1 (CTE/DTE/CTDE) | 100,000 | 463 | `data/checkpoints/training_log.json`, `data/metrics/summary.json` |
| Historical single-seed run 2 | 2 (VD/CAC/Comm) | 100,000 | 456 | `ctde_arena/data/…` |
| MLflow smoke run `457ce1e8` | 2 | 5,000 | — | 3 logged points at steps 1,217 / 2,526 / 3,958; `rl_log_every_steps` was 1,000 |
| MLflow run `c762d435` | 2 | 100,000 | — | **1 logged point**, at step 50,793 |
| Replicated study | 1 and 2 | 1,000,000 | see § 7.3 | `results/study/per_seed_metrics.csv`, `analysis.json`; raw per-run output under `data/runs/` (not versioned) |

The two 100 k-step runs predate the A-1 heading-control fix and used a single seed, so they are labelled
*historical* throughout this documentation set. The replicated study is the primary evidence.

The last row is a real gap rather than a documentation slip. `train_rl.py` logs to MLflow only inside
the `RL_LOG_EVERY_STEPS` branch, which is evaluated at a match boundary; with a 50,000-step cadence and
a 100,000-step budget the branch fires once (at 50,793) and the loop then exits before a second fire.
The final policy's metrics therefore never reached MLflow, and the "learning curves in MLflow" claim
reduces to a single point. The exported evidence is
[`ctde_arena/data/mlflow_export/mlflow_logged_metrics.csv`](../ctde_arena/data/mlflow_export/mlflow_logged_metrics.csv)
(48 rows).

## 7.4 Metric definitions

All aggregate metrics come from `TeamMetrics.as_summary()` in `src/marl_arena/models.py`. Let \(M\) be
`matches_played`, and note the guard \(M' = \max(M, 1)\).

| Metric | Formula | Denominator subtlety |
|---|---|---|
| **Win rate** | \(W / M'\) | \(W\) counts wins over all recorded matches |
| **Eliminations per match** | \(E / M'\) | \(E\) is the running total of kills by the team |
| **Mean survival time** | \(S / (3M')\) | divides by *three agents* × matches, so it is the mean lifetime **per agent**, in seconds |
| **Shot accuracy** | \(H / \max(H + X, 1)\) | pooled over the whole run, not averaged per match |

where \(S = \sum_{\text{matches}} \sum_{\text{agents}} \text{survival\_time}\).

Additional definitions used in this documentation set:

* **Recorded match** — a match written to `team_match_metrics.csv`; one in every
  `RL_METRICS_EVERY_MATCHES = 10`. 46 per team in both experiments.
* **Cumulative match** — a match counted in `matches_played`, i.e. every match played. 463 / 456 total,
  but the `summary.json` snapshot was taken at match 460 / 450 — the last multiple of ten — so
  `win_rate` denominators are 460 and 450, not 463 and 456.
* **Per-match survival** — `mean_survival_time` in the CSV is the mean over the team's three agents
  **in that match**, without the \(M\) division.

## 7.5 Artefact schemas

### `data/metrics/team_match_metrics.csv` — one row per team per recorded match

| Column | Type | Meaning |
|---|---|---|
| `match_index` | int | 1-based match counter, increments on every `reset_match` |
| `variant_id` | int | domain-randomisation sample id |
| `team_name` | str | `Team 1` … `Team 3` |
| `paradigm` | str | arm label |
| `winner` | int | 1 for the single team awarded the win |
| `eliminations` | int | kills by this team **in this match** |
| `mean_survival_time` | float | seconds, mean over the team's three agents |
| `shots_hit` / `shots_missed` | int | this match |
| `shot_accuracy` | int ratio | \(H/(H+X)\) this match |
| `remaining_agents` | int | alive at the end |
| `match_duration_seconds` | float | elapsed at termination |

### `data/metrics/agent_match_metrics.csv` — one row per agent per recorded match

`match_index, agent_id, team_name, paradigm, kills, shots_hit, shots_missed, survival_time, alive_at_end`

### `data/metrics/trajectory_metrics.csv` — one row per team **per simulation step**

`match_index, time_seconds, team_name, paradigm, alive_agents, cumulative_eliminations,
cumulative_accuracy, cumulative_win_rate, obstacle_count`

This is the largest artefact (29,058 rows / 2.1 MB for experiment 2) and the only one containing
in-match dynamics. It is written for every step of every recorded match.

### `data/checkpoints/training_log.json`

```
{ target_steps, completed_steps, matches_played,
  entries: [ { env_steps, matches, winner, variant{...}, summary{ Team N: {...} } } ] }
```

`entries` is appended on the `RL_LOG_EVERY_STEPS` cadence. **Both versioned logs contain exactly one
entry**, at step 50,587 (experiment 1) and 50,793 (experiment 2) — the same single-fire cadence problem
described in § 7.3. The file is therefore not a training history; it is one snapshot from mid-training.

### Schema rotation

`MetricsStore._rotate_csv_if_schema_changed` renames an existing CSV to `<stem>.legacy[<n>]<suffix>`
and starts a fresh file when the column set changes, emitting a warning. This is why
`data/metrics/team_match_metrics.legacy.csv` (1,191 rows) and
`trajectory_metrics.legacy.csv` (480,021 rows) exist in the working tree for experiment 1: the CSV
schema changed at least once during development, and the pre-rotation history is orphaned. Those files
are not reflected in `summary.json`, so **experiment 1's local data directory contains three mutually
inconsistent histories**; `**/data/metrics/*.legacy*.csv` is now ignored explicitly, so they stay out of
the repository.

## 7.6 Statistical treatment used in this documentation

Because matches are neither independent (non-stationary policies) nor balanced (three teams in one
match, so exactly one win per match), the following conventions are used in
[§ Results](08_results.md):

1. **Point estimates** are quoted exactly as `summary.json` reports them, with the caveat of § 7.2.
2. **Uncertainty** is computed on the 46 recorded matches per team with Wilson score intervals, which
   is the most conservative defensible estimate available from the versioned data.
3. **Pairwise comparisons** use a two-proportion \(z\)-test on the same 46-match subsample, and are
   additionally reported on the full cumulative denominator so the sensitivity of the conclusion to
   the independence assumption is visible.
4. No correction for multiple comparisons is applied; with three pairwise tests per experiment the
   Bonferroni-adjusted threshold would be \(\alpha = 0.0167\), and this is noted where it changes a
   verdict.
