# 9. Reproducibility

## 9.1 Environment

Requirements are pinned per experiment. Root: `ursina==6.1.2`, `numpy==2.2.6`, `matplotlib==3.10.3`,
`python-dotenv==1.0.1`, `torch==2.6.0`, `pytest==8.3.5`. `ctde_arena/` adds `mlflow==2.17.2`.

Two gaps in the declared environment:

* `scripts/plot_metrics.py` imports **pandas**, which appears in neither `requirements.txt`. The script
  exits with an install hint if the import fails, so this is a visible failure rather than a silent one.
* The `.venv/` directory in the working tree is **not usable**: its `pyvenv.cfg` points at
  `C:\Python313\python.exe` and records a creation command from `C:\Users\pedro\Downloads/…`. It was
  built on a different machine. It is git-ignored so it never reached the repository, but anyone
  copying the folder between machines will get a broken interpreter and a confusing error.

### Verification environment actually used

Every number in this documentation set was produced on:

| | |
|---|---|
| Python | 3.10.11 |
| NumPy | 1.26.4 (pinned requirement says 2.2.6) |
| PyTorch | 2.5.1+cu121 (pinned says 2.6.0) |
| Matplotlib | 3.10.9 (pinned says 3.10.3) |
| pytest | 9.1.1 (pinned says 8.3.5) |
| Device | CPU |
| Platform | Windows 10 / 11 x64 |

The minor-version drift did not affect any reported figure: all results were recomputed from committed
CSV/JSON artefacts, and the live simulation runs are used only for the qualitative and cost measurements
flagged as such.

## 9.2 Setup

```bash
# from the experiment directory (repository root for experiment 1, ctde_arena/ for experiment 2)
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate  on POSIX
pip install -r requirements.txt
pip install pandas                # only needed by scripts/plot_metrics.py
cp .env.example .env              # .env is tracked and identical to .env.example
```

## 9.3 Commands

| Command | Experiment | What it does |
|---|---|---|
| `python scripts/train_rl.py` | 1 | Headless PPO training, `RL_TRAIN_TOTAL_STEPS` env steps |
| `python main.py` | 1 | Ursina 3D window, greedy policies loaded from `data/checkpoints/` |
| `python scripts/plot_metrics.py` | 1 | Reads `data/metrics/summary.json` → `exports/metrics/` |
| `python -m pytest tests/ -q` | 1 | 7 tests |
| `cd ctde_arena && python scripts/train_rl.py` | 2 | Same, plus MLflow logging |
| `cd ctde_arena && mlflow ui --backend-store-uri file:./mlruns` | 2 | Tracking UI |
| `cd ctde_arena && python main.py` | 2 | 3D window for the CTDE variants |
| `cd ctde_arena && python -m pytest tests/ -q` | 2 | 2 tests |
| `cd ctde_arena && docker build -t ctde-arena . && docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena` | 2 | Containerised training |

**The two test suites must be run separately.** Invoking `pytest tests/ ctde_arena/tests/` from the root
fails at collection with `ImportError: cannot import name 'ValueDecompositionCriticNetwork'`, because
each tree injects its own `src/` at the front of `sys.path` and the root package shadows the
`ctde_arena` one. Verified:

```
$ python -m pytest tests/ ctde_arena/tests/ -q
ERROR ctde_arena/tests/test_components.py
ImportError: cannot import name 'ValueDecompositionCriticNetwork' from 'marl_arena.rl.networks'
            (D:\rl-multi_marl-competition\src\marl_arena\rl\networks.py)
```

## 9.4 Test results

```
$ python -m pytest tests/ -q            # experiment 1
7 passed in 2.38s
$ cd ctde_arena && python -m pytest tests/ -q
2 passed in 1.33s
```

What the nine tests do **not** cover is catalogued in
[§ Coverage](11_code_audit.md#coverage). In particular no test asserts that a policy improves, that the
collision solver returns the geometrically correct hit, or that any metric is computed correctly.

## 9.5 Cost of a run

Measured on this machine, CPU only, with domain randomisation enabled and all three teams training:

```
2000 env steps in 10.81 s  ->  185.1 steps/s
```

| Budget | Wall time | Notes |
|---|---:|---|
| 100,000 steps (versioned runs) | **≈ 9 minutes** | 463 / 456 matches |
| 1,000,000 steps | ≈ 1.5 h | |
| 3,000,000 steps (code default) | ≈ 4.5 h | feasible on CPU |
| 3 M × 5 seeds | ≈ 22.5 h | the replication study this project needs |

A 3 M-step run is therefore **not** computationally out of reach. It is, however, currently blocked by a
memory defect: `BaseTeamController.transitions` accumulates every `TransitionRecord` ever produced and
is never cleared. Measured with `tracemalloc`:

```
3,000 extra steps -> 11,731 transitions retained, 13.59 MB of marl_arena-attributed heap growth
                  -> 4.53 KB per env step
extrapolated:  100,000 steps ≈   453 MB
             3,000,000 steps ≈ 13.6 GB
```

So a 100 k-step run survives the leak; a 3 M-step run will exhaust a typical workstation. This is the
single change that most unblocks the project's stated next step — see
[§ Unbounded transition retention](11_code_audit.md#unbounded-transition-retention).

## 9.6 What is and is not reproducible

**Reproducible bit-for-bit from the repository:** every table in [§ Results](08_results.md), because all
of them are recomputed from committed CSV/JSON files with the commands quoted in each section.

**Not reproducible:** the training runs that produced those files. Four independent reasons:

1. **Minibatch shuffling is unseeded.** `np.random.shuffle` in all three `PPOTrainer.update_*` methods
   uses NumPy's global RNG, which nothing in the codebase seeds. `RANDOM_SEED=7` reaches `random.Random`
   and `np.random.default_rng` (the latter of which is created and then never used).
2. **No code version is recorded.** Checkpoints store `{paradigm, actor, critic}` and MLflow logs 43
   scalar config parameters — neither records a git commit, and the MLflow store itself is git-ignored.
3. **No environment lock.** Pinned versions exist but no hash lock file, and CUDA/CPU selection is a
   `.env` value.
4. **Torch determinism is not requested.** No `torch.manual_seed`, no `deterministic algorithms` flag.

A re-run with the same seed will produce a *statistically similar* run, not an identical one.

## 9.7 Artefact inventory

Versioned artefacts, with SHA-256 for integrity checking.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/data/checkpoints/team_1_ctde-vd.pt` | 216,346 | `c0ab135db4674649e16db19e4ab901e4895196e298d891e62487cb1353f4bb24` |
| `ctde_arena/data/checkpoints/team_2_ctde-cac.pt` | 299,324 | `6352a1f04d74165fcfc08e632d5b52ecc69f3346eaa250b35fab4fae3dee29ed` |
| `ctde_arena/data/checkpoints/team_3_ctde-comm.pt` | 308,194 | `a80855ba555d8579623deb07591983c1fe3cd8c9eef1b5fd346106ae89b461a4` |
| `ctde_arena/data/checkpoints/training_log.json` | 1,646 | `5d85eedbfda798b58099ad5841529a7161398075675c7036c55cf9c7f84ba228` |
| `ctde_arena/data/exports/comparative_dashboard.png` | 272,740 | `3d0248bf4c843540393bd36b0c476a0165522419ca08732c6bb88c3f00c4c7c7` |
| `ctde_arena/data/metrics/agent_match_metrics.csv` | 20,297 | `d1870d75d6a26024d651f90ad6bca1842297fb34df975011c489e77386964cc5` |
| `ctde_arena/data/metrics/summary.json` | 951 | `fc33dfb9fce1c361a915c28fdc80452b7d85e720c6fd93e2d4895a6c3733aa4f` |
| `ctde_arena/data/metrics/team_match_metrics.csv` | 11,722 | `01869e471d91411c7e13fc8b9c4944db2b8f0cd04daeb089f9e865597c34d230` |
| `ctde_arena/data/metrics/trajectory_metrics.csv` | 2,111,002 | `94a41c8c67861ba1a7d8c3578bae8f18f57adf58e58c4b211353f835678fbdb3` |
| `ctde_arena/data/mlflow_export/mlflow_logged_metrics.csv` | 4,494 | `a94164d2c0c44d84241c29e77c3eaf1102edd3f146c9fa7797172e141db1f471` |
| `ctde_arena/data/mlflow_export/mlflow_runs.json` | 2,017 | `c2e815aba57167c1aba1ac67bf1b0b28661944a54bb2f5fecccebbf3d38a4d12` |
| `exports/metrics/metrics_comparison.png` | 24,161 | `d1da22c5456b3cc7d666431c41f3a5f1b29a43eb632ce1836579ee48b8401145` |
| `exports/metrics/metrics_summary.csv` | 403 | `5e840a5ac26bca77de324d07d9cba77aca17307bb25d18e0b2e56f27d526f452` |

`exports/metrics/*` are derived from experiment 1's **untracked** `data/metrics/summary.json`, so they
are the only committed evidence of experiment 1's results — and they contain three rows and no
per-match detail.

## 9.8 The versioning asymmetry <a name="the-versioning-asymmetry"></a>

`.gitignore` contains:

```
data/exports/*.png
data/metrics/*.csv
data/metrics/*.json
data/checkpoints/*.pt
data/checkpoints/training_log.json
```

Because each pattern contains a `/`, Git anchors it at the repository root. **Experiment 1's data
directory is therefore excluded while experiment 2's — at `ctde_arena/data/…` — is not**, and was
committed. The result is inverted from what was presumably intended:

| | Experiment 1 (root) | Experiment 2 (`ctde_arena/`) |
|---|---|---|
| checkpoints | not in repo | in repo |
| per-match CSVs | not in repo | in repo |
| `summary.json` | not in repo | in repo |
| dashboard PNG | not in repo | in repo |
| MLflow store | git-ignored, present locally | never created |

**A reader who clones this repository cannot recompute a single number from Experiment 1.** Fixing it
means either anchoring the rules with `/**` so they apply to both trees and then force-adding
experiment 1's artefacts, or moving experiment 1's data under a path that is not matched. That is a
decision about repository content, so it is reported rather than made unilaterally
([§ Code audit](11_code_audit.md#versioning)).

## 9.9 Localisation rename <a name="localisation-rename"></a>

On 2026-09-27 the repository was converted from Portuguese to English. The rename touched identifiers
that appear in data, so it is documented here for provenance:

| Before | After | Where |
|---|---|---|
| `Equipe 1/2/3` | `Team 1/2/3` | `TEAM_META`, `TEAM_PARADIGMS`, CSV `team_name` column, JSON keys, MLflow metric names |
| `barreira_fixa` | `fixed_barrier` | obstacle type tag |
| `obstaculo_movel` | `moving_obstacle` | obstacle type tag |
| `passagem_restrita` | `restricted_passage` | obstacle type tag |
| `equipe_N_<paradigm>.pt` | `team_N_<paradigm>.pt` | checkpoint filenames, derived from the team slug |

**No numeric value was altered.** The rename was applied to the committed CSV/JSON artefacts as a
string substitution on the label fields only, and the two versioned dashboard PNGs were regenerated
from the relabelled rows — the curves are identical, only the legend text changed. The MLflow export in
`ctde_arena/data/mlflow_export/` normalises the metric keys to `Team N` as well, and records the
original run ids so the mapping back to the untracked `mlruns/` store is preserved.

Old clones will see the checkpoint files as renames (`git log --follow` resolves them). The local
`mlruns/` store still contains the original `equipe_*` metric keys because it is a historical record;
the export translates them.
