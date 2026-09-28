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
| Historical 100k-step run | 456 matches, artefacts committed under `data/historical_100k/` |
| Replicated study | 10 seeds × 500,000 steps, slot-rotated, held-out greedy evaluation — [results](../docs/08_results.md#replicated-study) |
| Promoted replicate in `data/` | seed 2, chosen as the one least distant from the study means (`data/PROVENANCE.json`) |
| Untrained baseline | 300 matches in both greedy and uniform modes (`results/baseline/`) |
| Bit-for-bit reproducible training | yes, verified with `--no-mlflow` |
| MLflow curves | **1 logged point** for the 100k run — see [A-9](../docs/11_code_audit.md#mlflow-cadence) |
| Tests | 47 passing, **87 %** statement coverage |
| Docker image | **never built or run** — unverified configuration |

Read [Results](../docs/08_results.md) and [Threats to validity](../docs/10_threats_to_validity.md)
before quoting any number from here. Everything in the "historical" tables is superseded by the replicated
study; it is kept because those artefacts still ship.

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
  [§ 5.4](../docs/05_network_architectures.md#valuedecompositioncriticnetwork).
* **Comm uses one communication round**, and dead allies contribute a learned non-zero constant message
  because their observations are zero-filled rather than masked.
  [§ 5.5](../docs/05_network_architectures.md#commactornetwork).

## Differences from the root experiment

**Five source files differ, and the engine is not one of them.** `simulation.py`, `main.py`, `config.py`,
`models.py`, `controllers/base.py`, `rl/actions.py`, `rl/buffer.py`, `systems/metrics.py`,
`systems/plotting.py`, `ui/dashboard.py`, `scripts/run_experiment.py`, `scripts/random_baseline.py` and
`scripts/plot_metrics.py` are byte-identical copies, and `tests/test_tree_parity.py` fails the suite if
they drift. Measured by `diff`:

| File | Changed lines | Why |
|---|---:|---|
| `src/marl_arena/controllers/rl_controller.py` | 116 | paradigm selection and the Comm decision path |
| `src/marl_arena/rl/networks.py` | 84 | adds `ValueDecompositionCriticNetwork`, `CommActorNetwork` |
| `src/marl_arena/rl/ppo.py` | 58 | adds `update_ctde_vd`, `update_ctde_comm` |
| `scripts/train_rl.py` | 135 | MLflow run, params, metrics, artefacts, registered models |
| `src/marl_arena/systems/match_variant.py` | **2** | `PARADIGM_CYCLE` only — a pinned, asserted difference |

Non-code differences: `requirements.txt` adds `mlflow==2.17.2`, and this tree has a `Dockerfile` the root
lacks. `RANDOM_SEED` also differs (2 here, 9 at the root) because each value is the one that makes
`python main.py` load that tree's promoted checkpoints.

**The physics, the reward, the termination rule, the observation encoder and the collision solver are the
same code in both experiments**, so every engine fix listed in
[§ Code audit](../docs/11_code_audit.md) — the control law, the memory leak, the tie-break, the observation
scaling, the slot rotation — applies to experiment 2 as well. They were applied twice, and the parity test
is what proves it rather than what hopes it.

The full divergence history, including the unintended drift that was removed, is
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

Loads `data/checkpoints/team_N_<paradigm>.pt` and runs the arena with greedy actions. Which paradigm is in
which slot is `RANDOM_SEED % 3`; at the shipped value of 2 the promoted replicate (seed 2, rotation 2) maps
Team 1 → red / CTDE-Comm, Team 2 → blue / CTDE-VD, Team 3 → green / CTDE-CAC, and the legend panel and
obstacle colours are built from the live variant rather than hard-coded. Orbit with `EditorCamera`;
**Restart Match** resets.

If you change `RANDOM_SEED` to a value with a different remainder, the checkpoint files for those slots do
not exist and each controller now **warns** instead of silently running random networks. To demo the shipped
policies, keep the seed's remainder at 2.

## Docker

```bash
cd ctde_arena
docker build -t ctde-arena .
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena
```

`python:3.10-slim` plus OpenGL/X11 libraries; `CMD` is headless training. **Not built or tested in this
session** — treat as unverified.

## Results

### Replicated study — primary evidence

10 independent replicates at 500,000 steps each, with the paradigm-to-slot assignment rotated by
`seed % 3`, each measured by 150 held-out greedy matches on the fixed evaluation variant. Full analysis:
[§ 8.1 Replicated study](../docs/08_results.md#replicated-study).

| Paradigm | Mean win rate | Between-seed SD | Pooled wins | Pooled 95 % CI | Elim./match | Survival | Shots/match | Accuracy |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| **CTDE-VD** | **0.367** | 0.171 | 551/1500 | [0.343, 0.392] | 1.97 | 23.1 s | 92.8 | 3.01 % |
| CTDE-Comm | 0.331 | 0.218 | 496/1500 | [0.307, 0.355] | 2.18 | 19.2 s | 111.6 | 2.40 % |
| CTDE-CAC | 0.282 | 0.182 | 423/1500 | [0.260, 0.305] | 1.98 | 21.5 s | 112.3 | 2.35 % |

Per-seed held-out win rate — every column is one independent replicate:

| Paradigm | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 | s10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CTDE-VD | 0.387 | 0.367 | 0.153 | 0.547 | 0.220 | 0.153 | 0.373 | 0.273 | 0.613 | 0.587 |
| CTDE-Comm | 0.253 | 0.253 | 0.553 | 0.173 | 0.153 | **0.767** | 0.580 | 0.193 | 0.187 | 0.193 |
| CTDE-CAC | 0.340 | 0.333 | 0.280 | 0.220 | 0.620 | 0.067 | **0.047** | 0.520 | 0.187 | 0.207 |

> **Nothing here is statistically distinguishable.** Paired per-seed *t*-tests give *p* = 0.679 (CAC vs
> Comm), 0.363 (CAC vs VD) and 0.746 (Comm vs VD). The forward power analysis in
> [§ 8.1](../docs/08_results.md#power-analysis) puts those comparisons at ~576, ~117 and ~939 replicates —
> 100 to 1,700 CPU-hours. The match-level Fisher tests reject two of the three, which is the anti-conservative
> treatment this design warns about.
>
> Three of these ten replicates end most of their matches on the 90 s clock rather than by elimination
> ([§ 8.1](../docs/08_results.md#two-regimes)), so part of what this table ranks is survival-to-buzzer, not
> fighting.

### Both historical conclusions reverse

| | Historical (1 seed, training-time, pre-A-1-fix) | Study (10 seeds, held-out, fixed, rotated) |
|---|---|---|
| Order | CAC 42.9 % > VD 40.7 % ≫ **Comm 16.4 %** | **VD 36.7 % > Comm 33.1 % > CAC 28.2 %** |
| Defended claim | "Comm loses to both others" | **does not reproduce** — Comm is now middle |
| Historical winner | CTDE-CAC | **CTDE-CAC finishes last** |

CTDE-Comm moves from last by 26 points to middle, and the arm that historically won finishes third. Either
the broken heading controller produced the original deficit or the single seed did; the data cannot separate
the two, and both readings imply the same operational conclusion — the old ranking was not a property of the
architectures.

The hypothesis this experiment was built around, that Comm needs more steps than the others because messages
must become informative before they help, is not supported either — and its *trajectory* is not stable enough
to test. At 25 % of the budget Comm was the **best** arm in the 5-seed study (0.650) and the **worst** here
(0.190); there it drifted down, here it drifts up (+0.141). When the sign of a learning curve flips on the
replicate count, the curve is measuring initialisation.

### Against untrained play

`scripts/random_baseline.py`, 300 headless matches. [§ 8.9](../docs/08_results.md#random-baseline) has both
experiments and the interpretation.

| | Uniform random actions | Greedy, untrained | Trained (study mean) |
|---|---:|---:|---:|
| Ended by wipeout | 97.0 % | **2.0 %** | mostly |
| Hit the cap / drawn | 2.0 % / 1.0 % | 56.7 % / **41.3 %** | 2.0 % drawn |
| Survival VD / CAC / Comm | 9.4 / 8.1 / 10.3 s | 83.3 / 63.6 / 84.9 s | 23.1 / 21.5 / 19.2 s |
| Shots/match | 43.5 / 37.5 / 43.4 | 358.2 / 88.2 / 397.9 | 92.8 / 112.3 / 111.6 |
| Accuracy | 5.1 / 5.9 / 6.2 % | **0.10 / 0.10 / 0.20 %** | 3.0 / 2.4 / 2.4 % |
| Win rate | 0.257 / 0.270 / 0.463 | 0.167 / 0.037 / 0.383 | 0.367 / 0.282 / 0.331 |

Training raises conversion by 15–30× over the greedy-untrained floor and largely removes the
timeout-and-draw regime, which is the strongest positive result in this repository. Against genuinely random
aiming, though, trained policies fire more and convert less — the learned skill is engagement, not
marksmanship. The uniform column is tree-independent (it never queries a network), so it is the same 300
matches as the root's, relabelled.

### Historical single-seed run

Cumulative over 456 matches, kept because these artefacts ship in `data/historical_100k/`.
[§ 8.2](../docs/08_results.md#historical-single-seed-runs).

| Team | Paradigm | Win rate | Elim./match | Mean survival | Shot accuracy | Hits | Misses |
|---|---|---:|---:|---:|---:|---:|---:|
| Team 1 | CTDE-VD | 40.67 % | 2.50 | 10.20 s | 11.29 % | 1,125 | 8,843 |
| Team 2 | **CTDE-CAC** | **42.89 %** | **2.63** | 9.83 s | **12.33 %** | 1,184 | 8,417 |
| Team 3 | CTDE-Comm | 16.44 % | 1.96 | 7.47 s | 12.14 % | 883 | 6,390 |

![Comparative dashboard](data/exports/comparative_dashboard.png)

### One historical analysis that survives

**CTDE-Comm's deficit was a firing-volume deficit, not an aiming deficit.** Decomposing
eliminations = shots × accuracy relative to CTDE-VD:

| Team | Shot volume | Accuracy | Product |
|---|---:|---:|---:|
| CTDE-CAC | ×0.963 | ×1.093 | ×1.052 |
| CTDE-Comm | **×0.730** | **×1.076** | ×0.785 |

Comm's accuracy (12.14 %) was *higher* than VD's (11.29 %). At VD's firing volume with its own accuracy it
would have led on eliminations. Whatever limited this arm acted through the decision to shoot, or through
dying sooner — not through missing. That mechanism is architectural and is the best hypothesis left to
test; what did not survive replication is the conclusion that it made Comm the weakest arm.
[§ 8.5](../docs/08_results.md#decomposing-the-elimination-gap).

## Configuration

Same variables as the root experiment; the full table is in
[§ Hyperparameters as trained](../docs/06_optimisation_procedure.md#67-hyperparameters-as-trained).
The values that matter most here:

| Variable | Committed value | Note |
|---|---|---|
| `RL_TRAIN_TOTAL_STEPS` | 1,000,000 | matches the code default; the study overrode it to 500,000 per replicate |
| `RL_LOG_EVERY_STEPS` | 50,000 | with the historical 100k budget this yields one MLflow point |
| `RL_METRICS_EVERY_MATCHES` | 10 | the CSVs hold one match in ten |
| `RL_HIDDEN_DIM` | 128 | two Tanh layers |
| `RL_DEVICE` | cpu | `cuda` / `mps` supported, and both are *slower* here — [A-26](../docs/11_code_audit.md#gpu-and-thread-scaling) |
| `RANDOM_SEED` | 2 | also the slot rotation: `2 % 3 = 2`, which is what the promoted checkpoints need |

## Tests

```bash
cd ctde_arena
python -m pytest tests/ -q        # 47 passed, 87% statement coverage
```

`test_components.py` checks tensor shapes for the two new networks; `test_ctde_training.py` terminates real
matches and asserts that all three variants' parameters move after a PPO update, that the VD critic slices
the right agents, and that the Comm arm stores 9-wide local observations; `test_control_law.py` pins the
heading-controller fix; `test_collision.py` verifies the swept collision solver against hand-computed entry
parameters and drives a full projectile step long enough to tunnel a barrier; `test_metrics_store.py` covers
the artefact pipeline. The remaining gap is `ui/dashboard.py`.
[§ Coverage](../docs/11_code_audit.md#coverage).

Run the two trees' suites separately — from the root, `pytest tests/ ctde_arena/tests/` fails at collection
because each tree prepends its own `src/`.

## Open work for this experiment

1. Lower `RL_LOG_EVERY_STEPS` (or move the MLflow call outside the boundary check) so a run produces an
   actual curve — `run_experiment.py` already records its own curve, but MLflow still does not.
2. ~~Rotate which team slot each variant occupies~~ **Done** — `paradigm_rotation = seed % 3`, recorded in
   every `run_summary.json`.
3. Mask dead allies in `CommActorNetwork` instead of zero-filling their observations.
4. Match critic capacity across VD and CAC before interpreting that contrast — VD's critic is 17,281
   parameters against CAC's 37,889, and the study's exp-2 arms now differ in critic size as well as in
   information flow.
5. Report shots per second-of-alive-life per agent — the only measurement that separates "Comm shoots less"
   from "Comm dies sooner". [A-30](../docs/11_code_audit.md#greedy-evaluation-of-an-untrained-network-is-not-random-play)
   makes this urgent: untrained firing volume already spans 88 to 398 shots per match.
6. Build and run the Docker image, or delete the Dockerfile. It is currently unverified configuration.
