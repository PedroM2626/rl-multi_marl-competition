# RL Multi MARL Competition

A 3D battle arena with **nine agents — three teams of three** — where each team is trained with PPO
(PyTorch) under a different multi-agent reinforcement learning architecture. The architectures compete
against each other inside one engine, so the competitive outcome is itself the measurement.

> **Full technical documentation lives in [`docs/`](docs/README.md)** — system model, MDP
> formalisation, architectures, optimisation, protocol, results with confidence intervals,
> reproducibility, limitations, and a 25-item code audit. This file is the quick start.

## Documentation

| If you want to… | Read |
|---|---|
| understand what is being tested and why | [Introduction and research questions](docs/01_introduction.md) |
| see how this relates to VDN / MAPPO / CommNet | [Related work](docs/02_related_work.md) |
| understand the physics and combat model | [Arena system model](docs/03_arena_system_model.md) |
| see the exact state, action and reward definitions | [MDP formalisation](docs/04_mdp_formalisation.md) |
| look up a network layer by layer | [Network architectures](docs/05_network_architectures.md) |
| understand what PPO is and is not doing here | [Optimisation procedure](docs/06_optimisation_procedure.md) |
| know what the CSV columns mean | [Experimental protocol](docs/07_experimental_protocol.md) |
| get the numbers, intervals and significance tests | [Results](docs/08_results.md) |
| reproduce a run, or estimate its cost | [Reproducibility](docs/09_reproducibility.md) |
| know what these results cannot support | [Threats to validity](docs/10_threats_to_validity.md) |
| find the bugs before trusting a ranking | [Code audit](docs/11_code_audit.md) |

## Repository status

Honest summary of what is finished and what is not:

| | Experiment 1 (root) | Experiment 2 (`ctde_arena/`) |
|---|---|---|
| Engine and training code | complete | complete |
| Historical 100k-step run | yes (463 matches) | yes (456 matches) |
| Results committed to Git | yes | yes |
| Checkpoints committed | yes | yes |
| MLflow tracking | n/a | wired, but only **1 logged point** for the 100k run |
| Tests | 36 passing, **93 %** statement coverage | 33 passing, **88 %** coverage |
| Multi-seed replication | 5 seeds × 1 M steps | 5 seeds × 1 M steps |
| Held-out greedy evaluation | yes, 150 matches per replicate | yes, 150 matches per replicate |

Three defects documented in the audit were **fixed** as part of the work below, and each changed what the
results mean:

* **A-1** — the low-level heading controller computed its bearing from the vertical axis, so turning was
  effectively saturated noise. Every policy trained before the fix learned in that world, which is why the
  original single-seed numbers are now labelled historical and the study was re-run.
  [Details](docs/11_code_audit.md#turn-control-defect).
* **A-2** — `BaseTeamController` retained every transition forever: 4.53 KB per env step, ~13.6 GB
  extrapolated to 3 M steps. That, not compute, is what made a replicated study impossible. Now 16 B per
  step. [Details](docs/11_code_audit.md#unbounded-transition-retention).
* **A-3** — root-anchored `.gitignore` patterns excluded experiment 1's data while committing experiment
  2's, so cloning the repo left you unable to recompute experiment 1's headline table. Both trees are now
  versioned. [Details](docs/11_code_audit.md#versioning).

Still open, and load-bearing for how the results may be read: the paradigm-to-slot assignment is not
rotated ([A-27](docs/11_code_audit.md#slot-confounding)) and the remaining open items are
statistical rather than code-level: the study needs ~13 replicates to settle CTDE vs CTE
([§ 8.1](docs/08_results.md#power-analysis)), `ui/dashboard.py` is still untested, and the
`ctde_arena/` Docker image has never been built.

## The two experiments

| | Location | Compares | Tracking |
|---|---|---|---|
| **1** | repository root | CTE × DTE × CTDE | JSON / CSV / PNG under `data/` |
| **2** | `ctde_arena/` | CTDE-VD × CTDE-CAC × CTDE-Comm | MLflow + `ctde_arena/data/` |

**Experiment 1 — training and execution paradigms.**

| Team | Paradigm | Actor | Critic |
|---|---|---|---|
| Team 1 | CTE | joint state + slot | joint state |
| Team 2 | DTE | local observation | own value head |
| Team 3 | CTDE | local observation | joint state |

**Experiment 2 — CTDE internals.**

| Team | Paradigm | Idea |
|---|---|---|
| Team 1 | CTDE-VD | additive per-agent value terms (VDN-style) |
| Team 2 | CTDE-CAC | centralised critic over the joint state (MAPPO-style) |
| Team 3 | CTDE-Comm | differentiable mean-pooled messages (CommNet-style) |

Each experiment is self-contained: its own `marl_arena` package, `.env`, `requirements.txt` and `data/`.
Run all commands **from the experiment's own directory**.

## Results at a glance

Held-out greedy win rate, 5 independent replicates per experiment at 1,000,000 steps, 150 evaluation
matches each. Full analysis: [§ Results](docs/08_results.md#replicated-study).

| Experiment | Paradigm | Mean win rate | Between-seed SD | Pooled 95 % CI | Best in |
|---|---|---:|---:|---|---:|
| 1 | **CTDE** | **0.495** | 0.244 | [0.459, 0.530] | 3/5 |
| 1 | DTE | 0.289 | 0.207 | [0.258, 0.323] | 2/5 |
| 1 | CTE | 0.216 | 0.100 | [0.188, 0.247] | 0/5 |
| 2 | **CTDE-Comm** | **0.365** | 0.305 | [0.332, 0.400] | 2/5 |
| 2 | CTDE-CAC | 0.360 | 0.211 | [0.326, 0.395] | 2/5 |
| 2 | CTDE-VD | 0.275 | 0.309 | [0.244, 0.308] | 1/5 |

**No pairwise comparison is significant at the seed level** — the independent unit is the replicate, not
the match. Paired per-seed *t*-tests give *p* = 0.115 to 0.979 across all six comparisons. The
match-level Fisher tests call several of them significant; [§ 8.1](docs/08_results.md#replicated-study)
shows why that treatment is anti-conservative here.

Two findings worth knowing before reading anything else:

* **The previous headline result does not reproduce.** With one seed, CTDE-Comm finished last by 26 points
  and that was the project's only statistically-defended claim. Replicated, it is tied for first.
* **Run-to-run variance exceeds the architecture effects.** Single replicates of one architecture range
  from 0.007 to 0.873 win rate. Power analysis in
  [§ 8.1](docs/08_results.md#power-analysis) puts the cheapest worthwhile comparison at ~13 replicates.

The two historical single-seed 100k-step runs are still reported, relabelled, in
[§ 8.2](docs/08_results.md#historical-single-seed-runs), because their artefacts ship in `data/`.

## Installation

Requires Python 3.10+. Repeat for each experiment directory.

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
                                  # source .venv/bin/activate on POSIX
pip install -r requirements.txt
pip install pandas                # only for scripts/plot_metrics.py; not in requirements.txt
```

`.env` is tracked and identical to `.env.example`; copy the latter if you delete it.

## Usage

| Command | Run from | Effect |
|---|---|---|
| `python scripts/train_rl.py` | either | Train for `RL_TRAIN_TOTAL_STEPS` env steps |
| `python main.py` | either | 3D arena with the versioned checkpoints, greedy actions |
| `python -m pytest tests/ -q` | either | Test suite (**run the two separately**) |
| `python scripts/plot_metrics.py` | either | `data/metrics/summary.json` → `exports/metrics/` |
| `mlflow ui --backend-store-uri file:./mlruns` | `ctde_arena/` | Tracking UI |

```bash
# experiment 1
python scripts/train_rl.py
python main.py

# experiment 2
cd ctde_arena
python scripts/train_rl.py
mlflow ui --backend-store-uri file:./mlruns
```

> **Running the test suites together fails at collection** (`pytest tests/ ctde_arena/tests/`), because
> each tree prepends its own `src/` to `sys.path`. See
> [§ Commands](docs/09_reproducibility.md#93-commands).

> **`train_rl.py` silently resumes from existing checkpoints** while resetting the step counter and
> optimizer state ([A-4](docs/11_code_audit.md#silent-warm-start)). Delete `data/checkpoints/*.pt` first
> if you want training from random initialisation.

**Cost.** Measured at 368 env steps/s on CPU with one torch thread per process (more threads and CUDA
both make it *slower* — see [A-26](docs/11_code_audit.md#gpu-and-thread-scaling)): a 100k-step run is
≈ 4.5 minutes, a 1 M-step run ≈ 45 minutes, and the whole 10-replicate study runs in about the time of
one replicate because the processes are independent.

## Controls in the 3D view

The window uses Ursina's `EditorCamera`: drag to orbit, scroll to zoom. The **Restart Match** button
resets the arena. The right-hand panel is the legend; the top-left overlay lists per-agent status.

## Configuration

Every tunable is read from `.env` by `src/marl_arena/config.py`. The full table with defaults, ranges
and units is in [§ Hyperparameters as trained](docs/06_optimisation_procedure.md#67-hyperparameters-as-trained)
and [§ Match variants](docs/03_arena_system_model.md#36-match-variants-and-domain-randomisation).

Note: `RL_TRAIN_TOTAL_STEPS` in the committed `.env` is **100,000** (what the shipped artefacts used);
the code default when the variable is absent is 3,000,000. `RESPAWN_ENABLED` has no effect
([A-16](docs/11_code_audit.md#dead-configuration)).

## Repository layout

```text
.
├── main.py                     # experiment 1: 3D arena
├── scripts/
│   ├── train_rl.py             # experiment 1: headless PPO training
│   └── plot_metrics.py         # summary.json -> CSV/PNG (needs pandas)
├── src/marl_arena/
│   ├── config.py               # .env -> ArenaConfig
│   ├── models.py               # snapshots, TeamMetrics.as_summary, MatchResult
│   ├── controllers/            # base.py (features, waypoints), rl_controller.py (CTE/DTE/CTDE)
│   ├── rl/                     # actions, buffer + GAE, networks, ppo
│   ├── systems/                # match_variant, simulation, metrics, plotting
│   └── ui/dashboard.py         # in-game overlay text
├── tests/                      # 36 tests
├── data/                       # experiment 1 artefacts — NOT versioned (see A-3)
├── exports/metrics/            # derived charts (versioned)
├── docs/                       # the full documentation set
└── ctde_arena/                 # experiment 2: same engine, CTDE variants + MLflow
    ├── main.py  Dockerfile  requirements.txt  README.md
    ├── scripts/train_rl.py     # PPO + MLflow logging
    ├── src/marl_arena/         # own copy; VD / CAC / Comm networks
    ├── tests/test_components.py
    └── data/                   # checkpoints, metrics, exports, mlflow_export — versioned
```

## Output files

| Path | Contents | Versioned |
|---|---|---|
| `data/checkpoints/team_N_<paradigm>.pt` | `{paradigm, actor, critic}` state dicts | exp. 2 only |
| `data/checkpoints/training_log.json` | one mid-training snapshot per run | exp. 2 only |
| `data/metrics/team_match_metrics.csv` | one row per team per recorded match | exp. 2 only |
| `data/metrics/agent_match_metrics.csv` | one row per agent per recorded match | exp. 2 only |
| `data/metrics/trajectory_metrics.csv` | one row per team per simulation step | exp. 2 only |
| `data/metrics/summary.json` | cumulative aggregate behind the tables above | exp. 2 only |
| `data/exports/comparative_dashboard.png` | four-panel cumulative comparison | exp. 2 only |
| `data/mlflow_export/` | the 48 metric points that reached MLflow | yes |

Schemas and the difference between "recorded" and "cumulative" matches:
[§ Artefact schemas](docs/07_experimental_protocol.md#75-artefact-schemas).

## Docker

```bash
cd ctde_arena
docker build -t ctde-arena .
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena
```

The image installs OpenGL/X11 libraries but defaults to headless training. It has **not been built or
run in this session** — see [§ Reproducibility](docs/09_reproducibility.md).

## Contact

Open an issue or a pull request in this repository to discuss the experiments, the audit findings, or
proposed fixes.
