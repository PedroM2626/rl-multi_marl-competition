# Experiment 2 — CTDE Variants Arena (`ctde_arena/`)

Self-contained sub-project of [RL Multi MARL Competition](../README.md). It holds the engine, reward
function and optimiser fixed at the **CTDE** paradigm and varies the architecture inside it, comparing
three ways of giving three decentralised actors a learning signal:

1. **CTDE-VD — Value Decomposition.** Team value is the sum of per-agent terms, in the spirit of VDN.
2. **CTDE-CAC — Centralised Actor-Critic.** Local actors, one critic over the full joint state, in the
   spirit of MAPPO.
3. **CTDE-Comm — Explicit Communication.** Actors exchange differentiable, mean-pooled messages at
   execution time, in the spirit of CommNet.

Each paradigm controls one team; three teams of three agents fight in the same arena.

> The parent repository's root is **Experiment 1**, which compares CTE × DTE × CTDE.
> Full technical detail for both lives in [`docs/`](../docs/README.md). This document covers what is
> specific to Experiment 2.

## Status of this experiment

| Aspect | State |
|---|---|
| Historical 100k-step run | 456 matches, artefacts committed under `data/` |
| Replicated study | 5 seeds × 1,000,000 steps, held-out greedy evaluation — [results](../docs/08_results.md#replicated-study) |
| MLflow curves | **1 logged point** for the 100k run — see [A-9](../docs/11_code_audit.md#mlflow-cadence) |
| Tests | 33 passing, **88 %** statement coverage |
| Docker image | never built or run in this session |

Read [Results](../docs/08_results.md) and [Threats to validity](../docs/10_threats_to_validity.md)
before quoting any number from here. The tables in this file are the **historical** single-seed run,
kept because those artefacts ship in `data/`; the study supersedes them.

## Architecture

| Team | Paradigm | Actor | Critic | Actor params | Critic params |
|---|---|---|---|---:|---:|
| Team 1 | CTDE-VD | `ActorNetwork` (local obs) | `ValueDecompositionCriticNetwork` | 35,337 | 17,281 |
| Team 2 | CTDE-CAC | `ActorNetwork` (local obs) | `CentralizedCriticNetwork` (joint state) | 35,337 | 37,889 |
| Team 3 | CTDE-Comm | `CommActorNetwork` (local obs + message) | `CentralizedCriticNetwork` (joint state) | 37,388 | 37,889 |

Layer-by-layer specifications: [§ Network architectures](../docs/05_network_architectures.md).

Two implementation facts that change how the comparison should be read:

* **`update_ctde_vd` is a one-line alias of `update_ctde`.** The only difference between the VD and CAC
  arms is the critic's `forward` — and its input capacity, since VD's critic is half the size.
  [§ 6.4](../docs/06_optimisation_procedure.md#64-the-three-update-paths).
* **VD does not decompose over local observations.** The documented formula is
  \(V_{\text{tot}}(s) = \sum_i V_i(o_i)\); the code computes \(\sum_i V_i(\text{slice}_i(s))\) from the
  *global* vector, so it is not a decentralisable factorisation.
  [§ 5.4](../docs/05_network_architectures.md#54-valuedecompositioncriticnetwork--additive-per-agent-critic-ctde-vd).
* **Comm uses one communication round**, and dead allies contribute a learned non-zero constant message
  because their observations are zero-filled rather than masked.
  [§ 5.5](../docs/05_network_architectures.md#55-commactornetwork--one-round-differentiable-communication-ctde-comm).

## Differences from the root experiment

Only these files differ from the root tree; the rest are byte-identical copies.

| File | Change |
|---|---|
| `src/marl_arena/rl/networks.py` | adds `ValueDecompositionCriticNetwork`, `CommActorNetwork` |
| `src/marl_arena/rl/ppo.py` | adds `update_ctde_vd`, `update_ctde_comm` |
| `src/marl_arena/controllers/rl_controller.py` | paradigm selection and the Comm decision path |
| `src/marl_arena/systems/match_variant.py` | `TEAM_META` labels; also drops `zip(..., strict=True)` |
| `src/marl_arena/systems/simulation.py` | status-text lines only |
| `scripts/train_rl.py` | MLflow run, params, metrics, artefacts, registered models |
| `main.py` | legend text |
| `requirements.txt` | adds `mlflow==2.17.2` |
| `Dockerfile` | training container (root has none) |

The full divergence list, including unintended drift, is
[§ Cross-tree divergence](../docs/11_code_audit.md#cross-tree-divergence).

## Installation

```bash
cd ctde_arena
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pandas                # only for scripts/plot_metrics.py
```

Pinned: `ursina==6.1.2`, `numpy==2.2.6`, `matplotlib==3.10.3`, `python-dotenv==1.0.1`, `torch==2.6.0`,
`pytest==8.3.5`, `mlflow==2.17.2`.

## Training and MLOps

```bash
python scripts/train_rl.py
```

The run registers, under experiment `CTDE_Comparison_Arena`, run `ppo_training_run`:

* **43 scalar parameters** — every simple field of `CONFIG`;
* **tags** `domain_randomization` and `device`;
* **per team, per `RL_LOG_EVERY_STEPS`:** `<team>_win_rate`, `<team>_shot_accuracy`,
  `<team>_mean_survival_time`, `<team>_eliminations_per_match`;
* **artefacts:** `plots/`, `checkpoints/`, `logs/training_log.json`;
* **Model Registry:** `CTDE_Arena_TEAM_1_ACTOR`, `..._TEAM_2_...`, `..._TEAM_3_...`.

```bash
mlflow ui --backend-store-uri file:./mlruns
```

> **What the tracker actually contains.** The metric call sits inside the `RL_LOG_EVERY_STEPS` branch,
> which is evaluated at a match boundary; with a 50,000-step cadence and a 100,000-step budget it fires
> **once**, at step 50,793, and the loop then exits. The 100k run therefore has a single logged point per
> metric and no learning curve. `mlruns/` is git-ignored; the 48 points that do exist are exported to
> [`data/mlflow_export/`](../docs/07_experimental_protocol.md#73-runs-actually-performed) so the claim is
> checkable. [A-9](../docs/11_code_audit.md#mlflow-cadence).

The mid-training snapshot at 50,793 steps already reproduces the final ordering:

| Team | Win rate @50,793 | Final cumulative |
|---|---:|---:|
| CTDE-VD | 0.4039 | 0.4067 |
| CTDE-CAC | 0.4433 | 0.4289 |
| CTDE-Comm | 0.1527 | 0.1644 |

## Visual simulation

```bash
python main.py
```

Loads `data/checkpoints/team_N_<paradigm>.pt` and runs the arena with greedy actions. The legend panel
maps Team 1 → red / CTDE-VD, Team 2 → blue / CTDE-CAC, Team 3 → green / CTDE-Comm, and the obstacle
colours. Orbit with `EditorCamera`; **Restart Match** resets.

## Docker

```bash
cd ctde_arena
docker build -t ctde-arena .
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena
```

`python:3.10-slim` plus OpenGL/X11 libraries; `CMD` is headless training. **Not built or tested in this
session** — treat as unverified.

## Results

Cumulative over 456 matches (the `summary.json` snapshot is taken at match 450).

| Team | Paradigm | Win rate | Elim./match | Mean survival | Shot accuracy | Hits | Misses |
|---|---|---:|---:|---:|---:|---:|---:|
| Team 1 | CTDE-VD | 40.67 % | 2.50 | 10.20 s | 11.29 % | 1,125 | 8,843 |
| Team 2 | CTDE-CAC | **42.89 %** | **2.63** | 9.83 s | **12.33 %** | 1,184 | 8,417 |
| Team 3 | CTDE-Comm | 16.44 % | 1.96 | 7.47 s | 12.14 % | 883 | 6,390 |

![Comparative dashboard](data/exports/comparative_dashboard.png)

### What the numbers support

**CTDE-Comm's deficit is a firing-volume deficit, not an aiming deficit.** Decomposing
eliminations = shots × accuracy relative to CTDE-VD:

| Team | Shot volume | Accuracy | Product |
|---|---:|---:|---:|
| CTDE-CAC | ×0.963 | ×1.093 | ×1.052 |
| CTDE-Comm | **×0.730** | **×1.076** | ×0.785 |

Comm's accuracy (12.14 %) is *higher* than VD's (11.29 %) and within 0.2 points of CAC's. At VD's firing
volume with its own accuracy, Comm would score **2.69 eliminations per match** — the best of the three.
Whatever limits this arm, it acts through the decision to shoot or through dying earlier, not through
missing. [§ 8.3](../docs/08_results.md#83-decomposing-the-elimination-gap).

**Comm improved the most over training.** Splitting each team's 46 recorded matches at the midpoint:

| Team | First 23 | Last 23 | Change |
|---|---:|---:|---:|
| CTDE-VD | 0.478 | 0.348 | −0.130 |
| CTDE-CAC | 0.478 | 0.435 | −0.043 |
| CTDE-Comm | 0.043 | 0.217 | **+0.174** |

This is consistent with the "learn to act and to communicate simultaneously" hypothesis — but it is also
consistent with ordinary co-adaptation among three concurrently trained teams, and with n = 1 seed it
cannot be resolved.

**VD vs CAC is a tie.** 2.2 points cumulative, 4.4 points on the recorded matches, p = 0.674.

## Configuration

Same variables as the root experiment; the full table is in
[§ Hyperparameters as trained](../docs/06_optimisation_procedure.md#67-hyperparameters-as-trained).
The values that matter most here:

| Variable | Committed value | Note |
|---|---|---|
| `RL_TRAIN_TOTAL_STEPS` | 100,000 | code default is 3,000,000 — the `.env` wins |
| `RL_LOG_EVERY_STEPS` | 50,000 | with a 100k budget this yields one MLflow point |
| `RL_METRICS_EVERY_MATCHES` | 10 | the CSVs hold one match in ten |
| `RL_HIDDEN_DIM` | 128 | two Tanh layers |
| `RL_DEVICE` | cpu | `cuda` / `mps` supported |
| `RANDOM_SEED` | 7 | single seed; minibatch shuffling is unseeded regardless |

## Tests

```bash
cd ctde_arena
python -m pytest tests/ -q        # 33 passed, 88% statement coverage
```

`test_components.py` checks tensor shapes for the two new networks; `test_ctde_training.py` terminates
real matches and asserts that all three variants' parameters move after a PPO update, that the VD critic
slices the right agents, and that the Comm arm stores 24-wide team observations; `test_control_law.py`
pins the heading-controller fix; `test_metrics_store.py` covers the artefact pipeline. The remaining gap
is `ui/dashboard.py`. [§ Coverage](../docs/11_code_audit.md#coverage).

## Open work for this experiment

1. Lower `RL_LOG_EVERY_STEPS` (or move the MLflow call outside the boundary check) so a run produces an
   actual curve — `run_experiment.py` already records its own curve, but MLflow still does not.
2. Rotate which team slot each variant occupies, so a slot effect cannot masquerade as an
   architecture effect.
3. Mask dead allies in `CommActorNetwork` instead of zero-filling their observations.
4. Match critic capacity across VD and CAC before interpreting that contrast.
5. Report shots per second-of-alive-life per agent — the only measurement that separates "Comm shoots
   less" from "Comm dies sooner".
