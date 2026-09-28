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


These two are precisely what the replicated study changes. Its procedure differs as follows:

```
scripts/run_experiment.py --seed S --steps 1000000 --out <run dir>
  1. ARENA_DATA_DIR=<run dir>, RANDOM_SEED=S, RL_TRAIN_TOTAL_STEPS=1000000
       -> one directory per replicate; nothing can warm-start from another run's checkpoints
  2. seed random.Random, NumPy's global RNG (which drives the PPO minibatch shuffle)
     and Torch (which drives weight initialisation)
  3. train exactly as above, recording training metrics every 100 matches
  4. at 25 / 50 / 75 / 100 % of the budget:
       save checkpoints -> construct a SEPARATE ArenaSimulation that loads them
       -> 40 greedy matches, domain_randomization=False, set_rl_training(False)
       -> discard that simulation; resume training the original
  5. at the end: 150 further greedy held-out matches -> evaluation/eval_match_metrics.csv
```

Three consequences worth stating explicitly:

* The evaluation numbers describe **the shipped policy**, sampled greedily on the fixed variant, which is
  what a "result" was always meant to be. The training-time running average is reported alongside it, not
  instead of it.
* The evaluation runs in a separate simulation instance, so it cannot contaminate training: with
  `training_enabled = False`, `_record_step` returns early, no rollout rows accumulate, and the
  `finish_match()` it calls performs no gradient step. Covered by
  `test_greedy_evaluation_is_deterministic_and_collects_nothing`.
* The curve points evaluate *intermediate* checkpoints of the same run, so they show learning within a
  replicate rather than a running average across ~10,000 different policies.

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

Two conventions are in force, and the difference between them is the reason the study reports two test
levels per comparison.

**Primary — the replicated study.** The independent unit is the **replicate**, not the match. Matches
inside one replicate are played by a single frozen policy against two opponents from the same run, sharing
that run's initialisation and geometry stream, so they are not independent draws. Accordingly:

1. **Effect estimates** are the mean of the per-replicate held-out win rates, reported with the
   between-replicate standard deviation.
2. **Pairwise tests** are two-sided paired *t*-tests over the per-replicate differences, with
   df = replicates − 1. The paired form is used because the three arms share a match, so their win rates
   are negatively coupled by construction.
3. **Match-level Fisher exact tests** on the pooled held-out matches are reported alongside, explicitly
   labelled anti-conservative, so the size of the unit-of-analysis error is visible rather than hidden.
4. **Multiplicity** is controlled at α = 0.0167 (Bonferroni over the three pairwise comparisons per
   experiment) and both the raw verdict and the Bonferroni verdict are shown.
5. **Power** is reported forward, not retrospectively: [§ 8.1](08_results.md#power-analysis) states how
   many replicates each observed effect size would need at 80 % power.

**Secondary — the historical single-seed runs.** Point estimates as `summary.json` reports them (a
cumulative average over a non-stationary trajectory), Wilson intervals on the 46 recorded matches per
team, and two-proportion *z*-tests on the same subsample, with the full cumulative denominator shown as
a sensitivity check. This treatment exists only to characterise artefacts that are already in `data/`;
no conclusion rests on it.

Draws are excluded from win-rate numerators but not from denominators, so the three arms' win rates sum
to less than 1 by exactly the draw rate ([§ 3.5](03_arena_system_model.md#35-episode-structure-and-termination)).

