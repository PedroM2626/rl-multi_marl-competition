# RL Multi MARL Competition

A 3D battle arena with **nine agents — three teams of three** — where each team is trained with PPO
(PyTorch) under a different multi-agent reinforcement learning architecture. The architectures compete
against each other inside one engine, so the competitive outcome is itself the measurement.

> **Full technical documentation lives in [`docs/`](docs/README.md)** — system model, MDP
> formalisation, architectures, optimisation, protocol, results with confidence intervals and power
> analysis, reproducibility, limitations, and a 31-item code audit. This file is the quick start.

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
| get the numbers, intervals, significance and power tests | [Results](docs/08_results.md) |
| reproduce a run, or estimate its cost | [Reproducibility](docs/09_reproducibility.md) |
| know what these results cannot support | [Threats to validity](docs/10_threats_to_validity.md) |
| find the bugs before trusting a ranking | [Code audit](docs/11_code_audit.md) |

## Repository status

Honest summary of what is finished and what is not:

| | Experiment 1 (root) | Experiment 2 (`ctde_arena/`) |
|---|---|---|
| Engine and training code | complete | complete |
| Engine shared with the other tree | byte-identical, enforced by `tests/test_tree_parity.py` | byte-identical, same guard |
| Historical 100k-step run | yes (463 matches) | yes (456 matches) |
| Replicated study | 10 seeds × 500k steps | 10 seeds × 500k steps |
| Held-out greedy evaluation | yes, 150 matches per replicate | yes, 150 matches per replicate |
| Slot rotation across replicates | yes (`paradigm_rotation = seed % 3`) | yes |
| Random-play baseline | yes, greedy and uniform modes | yes, greedy and uniform modes |
| Results committed to Git | yes | yes |
| Checkpoints committed | yes — newly versioned, and loadable by the current code | yes, re-promoted from the study |
| Bit-for-bit reproducible training | verified | verified with `--no-mlflow` |
| MLflow tracking | n/a | wired, but only **1 logged point** for the 100k run |
| Tests | 55 passing, **92 %** statement coverage | 47 passing, **87 %** coverage |
| Docker | n/a | **never built or run** |

Six defects documented in the audit were **fixed** during this work, and each changed what the results
mean:

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
* **A-10** — a three-way tie on survivors was resolved by dictionary order, so a slot name chose the winner.
  Ties are now reported as draws.
* **A-17** — observations mixed metres with radians and normalised nothing, so the same agent produced
  features three orders of magnitude apart. Local and global observations are now scaled by the arena
  half-size — which also means the old checkpoints no longer load, and are labelled as historical.
* **A-27** — paradigm-to-slot assignment was fixed, so a slot effect was indistinguishable from an
  architecture effect even across seeds. It is now rotated by `seed % 3`.

Still open, and load-bearing for how the results may be read: the study is **underpowered by a wide
margin** (~47 to ~939 replicates, [§ 8.1](docs/08_results.md#power-analysis)), a quarter of replicates
decide matches on the 90 s clock rather than by eliminations
([§ 8.1](docs/08_results.md#two-regimes)), one replicate in sixty collapses to a never-fires policy
([§ 8.1](docs/08_results.md#leave-one-out)), `ui/dashboard.py` is untested, and the `ctde_arena/` Docker
image has never been built.

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

Held-out greedy win rate, 10 independent replicates per experiment at 500,000 steps, 150 evaluation matches
each, with the paradigm-to-slot assignment rotated across seeds. Full analysis:
[§ Results](docs/08_results.md#replicated-study).

| Experiment | Paradigm | Mean win rate | Between-seed SD | Pooled 95 % CI | Best in |
|---|---|---:|---:|---|---:|
| 1 | **DTE** | **0.381** | 0.191 | [0.356, 0.406] | 6/10 |
| 1 | CTDE | 0.326 | 0.187 | [0.303, 0.350] | 3/10 |
| 1 | CTE | 0.252 | 0.122 | [0.231, 0.275] | 1/10 |
| 2 | **CTDE-VD** | **0.367** | 0.171 | [0.343, 0.392] | 5/10 |
| 2 | CTDE-Comm | 0.331 | 0.218 | [0.307, 0.355] | 3/10 |
| 2 | CTDE-CAC | 0.282 | 0.182 | [0.260, 0.305] | 2/10 |

**No pairwise comparison is significant at the seed level.** The independent unit is the replicate, not the
match; paired per-seed *t*-tests give *p* = 0.158 to 0.746 across all six comparisons. The match-level
Fisher tests call five of the six significant —
[§ 8.1](docs/08_results.md#replicated-study) shows why that treatment is anti-conservative here.

Four findings worth knowing before reading anything else:

* **Both historical headline results fail to reproduce.** With one seed, CTDE-Comm finished last by 26
  points and CTDE-CAC first; replicated, Comm is middle and CAC is last. In experiment 1 the CTDE arm is
  now second behind DTE.
* **Run-to-run variance exceeds the architecture effects.** Single replicates of one architecture range
  from 0.000 to 0.767 win rate, and forward power puts the cheapest comparison at ~47 replicates and the
  rest at 110–939 ([§ 8.1](docs/08_results.md#power-analysis)).
* **One replicate reverses the experiment 1 ranking.** Seed 6's CTDE arm learned to stop shooting — 3.8
  shots per match, 0.000 wins. Remove that single replicate and CTDE goes from second to first, and the
  sign of its difference against DTE flips ([§ 8.1](docs/08_results.md#leave-one-out)).
* **A quarter of replicates are not measuring fighting.** Five of twenty end most matches on the 90 s
  clock, where winning means "most alive at the buzzer"
  ([§ 8.1](docs/08_results.md#two-regimes)).

Against a genuine random-play baseline
([§ 8.9](docs/08_results.md#random-baseline)) training clearly does something — trained agents survive
2–3× longer and end matches by elimination instead of timing out — but it does **not** improve aim:
uniform random shots convert at 5–6 % while trained policies convert at 2.4–3.6 % while firing ~2.5× more.

The two historical single-seed 100k-step runs are still reported, relabelled, in
[§ 8.2](docs/08_results.md#historical-single-seed-runs); their artefacts now live in
`data/historical_100k/`.

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
| `python scripts/train_rl.py --seed 1` | either | Train; `--from-scratch` skips checkpoint loading |
| `python main.py` | either | 3D arena with the versioned checkpoints, greedy actions |
| `python -m pytest tests/ -q` | either | Test suite (**run the two separately**) |
| `python scripts/plot_metrics.py` | either | `data/metrics/summary.json` → `exports/metrics/` |
| `python scripts/run_experiment.py --seed 1 --steps 500000 --out data/runs/exp1_seed1` | either | One replicate: train, then held-out greedy evaluation |
| `python scripts/run_study.py --steps 500000 --seeds 1,2,3,4,5,6,7,8,9,10 --jobs 20` | root | Fan both experiments out and collect `results/study/` |
| `python scripts/report_study.py` | root | All study tables, paired tests, Fisher tests and forward power |
| `python scripts/plot_study.py` | root | `results/study/learning_curves.png` |
| `python scripts/random_baseline.py --matches 300 --policy uniform` | either | Untrained reference point |
| `python scripts/promote_run.py --experiment exp1 --seed median` | root | Copy one replicate into the versioned `data/` |
| `mlflow ui --backend-store-uri file:./mlruns` | `ctde_arena/` | Tracking UI |

```bash
# experiment 1: reproduce the whole study
python scripts/run_study.py --steps 500000 --seeds 1,2,3,4,5,6,7,8,9,10 --jobs 20
python scripts/report_study.py
python scripts/plot_study.py

# just look at it
python main.py
```

> **Running the test suites together fails at collection** (`pytest tests/ ctde_arena/tests/`), because
> each tree prepends its own `src/` to `sys.path`. See
> [§ Commands](docs/09_reproducibility.md#93-commands).

> **`train_rl.py` resumes from existing checkpoints** unless you pass `--from-scratch`, and it writes into
> `data/checkpoints/` — which is versioned evidence. Set `ARENA_DATA_DIR` to a scratch directory for
> exploratory runs so a throwaway training job cannot overwrite the shipped policies. This is how the
> original experiment 1 checkpoints were lost
> ([A-24](docs/11_code_audit.md#test-isolation)).

> **`RANDOM_SEED` decides which paradigm sits in which slot** (`seed % 3`), and checkpoint filenames carry
> the slot. The shipped `.env` is set so that `python main.py` loads the promoted policies; a different
> rotation warns rather than silently demoing random networks. See `data/PROVENANCE.json`.

**Cost.** Measured at 368 env steps/s on CPU with one torch thread per process (more threads and CUDA
both make it *slower* — see [A-26](docs/11_code_audit.md#gpu-and-thread-scaling)): a 100k-step run is
≈ 4.5 minutes, a 500k-step replicate is ≈ 25 minutes alone, and the full 20-replicate study cost
**36.2 CPU-hours but only 117.5 minutes of wall clock** at 20-way concurrency
([§ 9.5](docs/09_reproducibility.md#95-cost-of-a-run)).

## Controls in the 3D view

The window uses Ursina's `EditorCamera`: drag to orbit, scroll to zoom. The **Restart Match** button
resets the arena. The right-hand panel is the legend; the top-left overlay lists per-agent status.

## Configuration

Every tunable is read from `.env` by `src/marl_arena/config.py`. The full table with defaults, ranges
and units is in [§ Hyperparameters as trained](docs/06_optimisation_procedure.md#67-hyperparameters-as-trained)
and [§ Match variants](docs/03_arena_system_model.md#36-match-variants-and-domain-randomisation).

Note: `RL_TRAIN_TOTAL_STEPS` is **1,000,000** in both committed `.env` files, matching the code default;
`RANDOM_SEED` is deliberately different per tree (9 and 2) because it also selects the paradigm-to-slot
rotation that the shipped checkpoints need. `RESPAWN_ENABLED` was removed rather than left inert
([A-16](docs/11_code_audit.md#dead-configuration)).

## Repository layout

```text
.
├── main.py                     # experiment 1: 3D arena
├── scripts/
│   ├── train_rl.py             # headless PPO training, seeded by --seed
│   ├── run_experiment.py       # one replicate: train + held-out greedy evaluation (identical in both trees)
│   ├── run_study.py            # fan replicates out across processes, collect results/study/
│   ├── report_study.py         # tables, paired t, Fisher, forward power
│   ├── plot_study.py           # learning-curve figure
│   ├── random_baseline.py      # untrained reference point (greedy / uniform)
│   ├── promote_run.py          # copy one replicate into the versioned data/
│   └── plot_metrics.py         # summary.json -> CSV/PNG (needs pandas)
├── src/marl_arena/
│   ├── config.py               # .env -> ArenaConfig, plus seed_all()
│   ├── models.py               # snapshots, TeamMetrics.as_summary, MatchResult
│   ├── controllers/            # base.py (features, waypoints), rl_controller.py (CTE/DTE/CTDE)
│   ├── rl/                     # actions, buffer + GAE, networks, ppo
│   ├── systems/                # match_variant, simulation, metrics, plotting
│   └── ui/dashboard.py         # in-game overlay text
├── tests/                      # 55 tests, incl. tree-parity and collision geometry
├── data/                       # experiment 1: promoted replicate + historical_100k/ + PROVENANCE.json
├── results/
│   ├── study/                  # per_seed_metrics.csv, analysis.json, learning_curves.png
│   └── baseline/               # random_baseline_{greedy,uniform}.json
├── exports/metrics/            # derived charts
├── docs/                       # the full documentation set, 12 chapters
└── ctde_arena/                 # experiment 2: same engine, CTDE variants + MLflow
    ├── main.py  Dockerfile  requirements.txt  README.md
    ├── scripts/                # own train_rl.py (adds MLflow); the shared scripts are byte-identical
    ├── src/marl_arena/         # own copy; VD / CAC / Comm networks
    ├── tests/                  # 47 tests
    ├── results/baseline/       # experiment 2's untrained reference
    └── data/                   # promoted replicate, historical_100k/, mlflow_export/, PROVENANCE.json
```

## Output files

Both trees ship the same layout; `exp. 1` and `exp. 2` below mean "in each tree's own `data/`".

| Path | Contents | Versioned |
|---|---|---|
| `data/checkpoints/team_N_<paradigm>.pt` | `{paradigm, actor, critic}` state dicts, one promoted replicate | yes, both |
| `data/metrics/team_match_metrics.csv` | one row per team per recorded match | yes, both |
| `data/metrics/agent_match_metrics.csv` | one row per agent per recorded match | yes, both |
| `data/metrics/trajectory_metrics.csv` | one row per team per simulation step | yes, both |
| `data/metrics/summary.json` | cumulative aggregate behind the tables above | yes, both |
| `data/exports/comparative_dashboard.png` | four-panel cumulative comparison | yes, both |
| `data/PROVENANCE.json` | which replicate was promoted, how it was chosen, and its full run summary incl. the held-out evaluation | yes, both |
| `data/historical_100k/` | the superseded single-seed artefacts behind [§ 8.2](docs/08_results.md#historical-single-seed-runs), incl. its `training_log.json` | yes, both |
| `data/mlflow_export/` | the 48 metric points that reached MLflow | exp. 2 |
| `results/study/` | the 20-replicate study's aggregate CSV, JSON and figure | yes |
| `results/baseline/` | the untrained reference points, greedy and uniform | yes |
| `data/runs/` | per-replicate scratch (checkpoints, per-step CSVs, `evaluation/run_summary.json`) | **no** — see [§ 9.8](docs/09_reproducibility.md#the-versioning-asymmetry) |

A replicate produced by `run_experiment.py` writes its training record to `evaluation/run_summary.json`
rather than a `training_log.json`, so the promoted directory carries checkpoints only; the run summary is
copied into `data/PROVENANCE.json`, which is why that file is the authoritative description of what ships.

Schemas and the difference between "recorded" and "cumulative" matches:
[§ Artefact schemas](docs/07_experimental_protocol.md#75-artefact-schemas).

## Docker

```bash
cd ctde_arena
docker build -t ctde-arena .
docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena
```

The image installs OpenGL/X11 libraries but defaults to headless training. **It has never been built or
run** — this is a known, accepted gap, not an oversight; every number in the documentation was produced on
the host. Treat the Dockerfile as unverified configuration.

## Contact

Open an issue or a pull request in this repository to discuss the experiments, the audit findings, or
proposed fixes.
