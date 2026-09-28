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
| `python scripts/train_rl.py --seed S` | 1 | Headless PPO training; bit-for-bit reproducible for a given seed |
| `python main.py` | 1 | Ursina 3D window, greedy policies loaded from `data/checkpoints/` |
| `python scripts/plot_metrics.py` | 1 | Reads `data/metrics/summary.json` → `exports/metrics/` |
| `python -m pytest tests/ -q` | 1 | 59 tests |
| `cd ctde_arena && python scripts/train_rl.py` | 2 | Same, plus MLflow logging |
| `cd ctde_arena && mlflow ui --backend-store-uri file:./mlruns` | 2 | Tracking UI |
| `cd ctde_arena && python main.py` | 2 | 3D window for the CTDE variants |
| `cd ctde_arena && python -m pytest tests/ -q` | 2 | 47 tests |
| `cd ctde_arena && docker build -t ctde-arena . && docker run --rm -v ${PWD}/data:/app/data -v ${PWD}/mlruns:/app/mlruns ctde-arena` | 2 | Containerised training |
| `python scripts/run_experiment.py --seed S --steps N --out DIR` | either | One seeded replicate: train, then greedy held-out evaluation |
| `python scripts/run_study.py --steps 500000 --seeds 1..10 --jobs 20` | both | Fan replicates out across processes and collect `results/study/` |
| `python scripts/report_study.py` | both | Render the study tables, paired tests, Fisher tests and forward-power table |
| `python scripts/plot_study.py` | both | Draw `results/study/learning_curves.png` |
| `python scripts/promote_run.py --experiment exp1 --list` | 1 or 2 | Show every replicate with its L1 distance to the study means |
| `python scripts/promote_run.py --experiment exp1 --seed median` | 1 or 2 | Copy one replicate into the versioned `data/` and write `PROVENANCE.json` |
| `python scripts/random_baseline.py --matches 300` | either | Untrained reference point: outcome mix, durations and per-arm metrics from random policies |
| `python scripts/report_inventory.py` | root | Regenerate the § 9.7 artefact table from `git ls-files` and live SHA-256 |

`run_experiment.py` and `random_baseline.py` are byte-identical in both trees; which engine they bind to is
decided by the working directory, exactly like `train_rl.py`. `run_study.py`, `report_study.py`,
`plot_study.py` and `promote_run.py` exist only at the root and drive both trees — `promote_run.py` writes
experiment 2's artefacts into `ctde_arena/data/`. `tests/test_tree_parity.py` enforces which files must stay
identical across the two trees and which are allowed to differ, so drift in the shared engine fails the
suite.

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
59 passed in 5.80s
$ cd ctde_arena && python -m pytest tests/ -q
47 passed in 5.93s
```

Statement coverage over `src/marl_arena`, measured with `coverage run --source=src/marl_arena -m pytest`
(`pytest-cov` is not installed, and the two suites must still be run separately):

| Tree | Statements | Missing | Coverage |
|---|---:|---:|---:|
| Experiment 1 | 1,527 | 111 | **93 %** |
| Experiment 2 (`ctde_arena/`) | 1,639 | 204–206 | **87–88 %** |

Experiment 2's total moves between two and three missing statements across repeated identical invocations,
so a single quoted figure would be spurious precision. Quoting the range is more honest than quoting
whichever run happened to be measured last. The cause was not tracked down; the effect is 0.1 %.

In both trees the bulk of the gap is `ui/dashboard.py` (8 statements, 0 %) plus the rendering-only paths,
and in experiment 2 there is a structural component explained in
[§ Coverage](11_code_audit.md#coverage).

What the 106 tests do **not** cover is catalogued in
[§ Coverage](11_code_audit.md#coverage). The collision solver is now geometrically verified against
hand-computed values, but no test checks a *metric* against an independent reference — see the note at the
end of that section on coverage not being correctness.

## 9.5 Cost of a run

Measured on this machine (32-core host, RTX 4070 Laptop GPU), CPU only, domain randomisation enabled,
all three teams training, one process:

```
torch.set_num_threads(1) -> 368 steps/s
torch.set_num_threads(2) -> 261 steps/s
torch.set_num_threads(4) -> 259 steps/s
CUDA, same loop          -> 150 steps/s
```

One thread is fastest and the GPU is the slowest option, because a step is nine batch-1 forwards through
a 35 k-parameter MLP: the cost is Python and kernel dispatch, not arithmetic. The full profile and the
implication for a JAX/vectorised-env port are in
[§ GPU and thread scaling](11_code_audit.md#gpu-and-thread-scaling).

| Budget | Wall time, one process | Notes |
|---|---:|---|
| 100,000 steps (historical runs) | ≈ 4.5 min | 463 / 456 matches |
| 500,000 steps | 90–101 min | one study replicate, **measured under 20-way concurrency** — ≈ 25 min alone |
| 1,000,000 steps | ≈ 45 min | the superseded 5-seed budget |
| 3,000,000 steps (code default) | ≈ 2.3 h | |
| 500 k × 10 seeds × 2 experiments | **117.5 min total** | 20 processes at once on 32 cores; 31.5 CPU-h training + 4.7 h evaluation |

The measured per-replicate throughput under full 20-way load is 82.6–92.9 steps/s against 368 steps/s for an
isolated process, so 20 concurrent replicates cost 36.2 CPU-hours and take about as long as four sequential
ones would. The load does not come from CPU contention — 32 cores for 20 single-threaded processes has
headroom — but from the shared memory bandwidth of the NumPy hot loop.

Parallelism across seeds, not throughput per step, is what makes replication affordable: the runs are
independent and each wants exactly one core.

The memory defect that previously made long runs impossible is fixed. Before, `BaseTeamController`
retained every `TransitionRecord` forever — 4.53 KB per env step, ~13.6 GB extrapolated to 3 M steps, so
ten concurrent replicates could never fit in RAM. After:

```
20,000 env steps -> 0.33 MB growth -> 16.3 B per env step -> ~0.05 GB at 3M steps
```

See [§ Unbounded transition retention](11_code_audit.md#unbounded-transition-retention).

## 9.6 What is and is not reproducible

**Reproducible bit-for-bit from the repository:** every table in [§ Results](08_results.md), because all
of them are recomputed from committed CSV/JSON files with the commands quoted in each section.

**Reproducible bit-for-bit by re-running:** `scripts/train_rl.py --seed S`, in both trees. This was
verified, not assumed — two 4,000-step runs at seed 42 and one at seed 43, each into its own
`ARENA_DATA_DIR`:

```
experiment 1 (root)
det1 log d76caaf23622cd5a | team_1_cte=2382baba8f team_2_dte=cb6520da69 team_3_ctde=38e49f9684
det2 log d76caaf23622cd5a | team_1_cte=2382baba8f team_2_dte=cb6520da69 team_3_ctde=38e49f9684
det3 log e734d423aab56c20 | team_1_dte=08565d0e15 team_2_ctde=be4fbd29aa team_3_cte=82e1530e18

experiment 2 (ctde_arena, --no-mlflow)
cdet1 log 229a52a77acb6fc1 | team_1_ctde-vd=27c37aa375 team_2_ctde-cac=004ad7e370 team_3_ctde-comm=a0b157caf9
cdet2 log 229a52a77acb6fc1 | team_1_ctde-vd=27c37aa375 team_2_ctde-cac=004ad7e370 team_3_ctde-comm=a0b157caf9
cdet3 log 594052f04bff8d43 | team_1_ctde-cac=c0272c7ac4 team_2_ctde-comm=84dfb63e82 team_3_ctde-vd=bf424449cd
```

Same seed → identical `training_log.json` **and identical checkpoint bytes** in both trees; different seed
→ different, and the slot rotation differs as it must (`42 % 3 = 0`, `43 % 3 = 1`, visible above as the
paradigm-to-slot mapping changing between the seed-42 and seed-43 rows). `seed_all` in
`src/marl_arena/config.py` seeds `random`, NumPy's global RNG and Torch, and `RolloutBuffer` now owns an
explicit `np.random.Generator` instead of calling `np.random.shuffle` on the global stream — which is what
makes the equality hold rather than merely being likely.

Experiment 2 is reproducible **only with `--no-mlflow`**: an MLflow run id and wall-clock timestamps are
written into the store, so the log is not byte-stable even when the checkpoints are.

`ARENA_DATA_DIR` isolates each run's artefacts, so a replicate cannot silently warm-start from another's
checkpoints, and `run_experiment.py` passes `load_checkpoints=False`.

**Not reproducible:**

1. **The two historical 100 k-step runs.** They were produced before the control-law fix, before the
   observation-encoding change, and before `seed_all` existed. Their artefacts are versioned, so every
   number in [§ 8.2](08_results.md#historical-single-seed-runs) can be *recomputed* from them, but the run
   itself cannot be reproduced.
2. **Any run made through MLflow.** The store records wall-clock timestamps and generates run ids; the
   training itself is still seed-reproducible with `--no-mlflow`.
3. **No code version is recorded inside the artefacts.** Checkpoints store `{paradigm, actor, critic}` and
   nothing else; `data/PROVENANCE.json` names the promoted replicate and the study, and Git supplies the
   commit, but a checkpoint file does not carry the hash that produced it.
4. **No environment lock.** Pinned versions exist in `requirements.txt` but no hash lock file, and
   CUDA/CPU selection is a `.env` value. Two machines with different resolved transitive dependencies can
   therefore diverge even though the seeding is correct.
5. **Torch determinism flags are not set.** Not needed at these shapes — the verification above is exact —
   but a kernel-level change in a future Torch release is not guarded against.

## 9.7 Artefact inventory

Every versioned experiment artefact (46 files), with SHA-256 for integrity checking. Regenerate this section with `python scripts/report_inventory.py`; it is built from `git ls-files`, so it cannot drift from what is actually committed.

**Promoted replicate.** The artefacts that match the current code. Chosen by `promote_run.py --seed median`, which is the replicate with the least L1 distance to the per-arm study means; `PROVENANCE.json` records the choice and the full run summary behind it.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/data/PROVENANCE.json` | 6,565 | `528a55f7bdf549eae03d170cc698a45acd87192935bcf09925ee2dcff2fa041b` |
| `ctde_arena/data/checkpoints/team_1_ctde-comm.pt` | 313,778 | `cb3fee053d68a7a3cc6e4cadbd3fdfc05a356e5a9542babad143a1aa2eb85ff1` |
| `ctde_arena/data/checkpoints/team_2_ctde-vd.pt` | 217,330 | `bf261e17c010ee0bb8f69b9d7e9ca1c4214cc9cfe0d8775038379dd71547d5f5` |
| `ctde_arena/data/checkpoints/team_3_ctde-cac.pt` | 304,400 | `451b14b4c8aae906c3e7b135f9e814426b268354e8eb222d420b42dcb981cccb` |
| `ctde_arena/data/exports/comparative_dashboard.png` | 269,089 | `474621e93c7817c93abedc1404cd4461b70d7258609706051e746e867438886e` |
| `ctde_arena/data/metrics/agent_match_metrics.csv` | 18,806 | `4e86f11187d317bf9ccee379b995e63485e22dc9d81d33a52d7526407ee4e405` |
| `ctde_arena/data/metrics/summary.json` | 975 | `d7403380fd69055c9087842818197fe8ae8875b9c4b3c2d79c14e1c66313915f` |
| `ctde_arena/data/metrics/team_match_metrics.csv` | 11,118 | `6fbebf4f161d9196d25bf93d976ec91bc3da6888651e4023582bda0ac6a945a3` |
| `ctde_arena/data/metrics/trajectory_metrics.csv` | 1,200,527 | `b975706a5eba42abb99b9db6ccff4567940cb011ea7d839d35638bdc9b9db6f8` |
| `data/PROVENANCE.json` | 6,491 | `95c9f868895b977641b28f00543e106064a851af0affacca3e1dda1352fba186` |
| `data/checkpoints/team_1_cte.pt` | 323,170 | `c978273409e60e45b9555d763d65c9ff13ba7774ffaf332ac4eccd11ffe59111` |
| `data/checkpoints/team_2_dte.pt` | 145,762 | `93fc387fbfbd1cb935a271672954675230e1326565ae2680333a4ded4e4e3897` |
| `data/checkpoints/team_3_ctde.pt` | 304,248 | `5a5273f3c73942e8fb0964bbb6a848734c58fee9cf0b52b4986f84cf8f2a4fae` |
| `data/exports/comparative_dashboard.png` | 272,710 | `d3daf6ac475005675a5411e39173c8ff53d5349ac39fb6c1655e9193753ab783` |
| `data/metrics/agent_match_metrics.csv` | 17,383 | `c408d9c3996f93c30e19cd602e70352fe285a5348ea30346ebdf21bdcf7bae05` |
| `data/metrics/summary.json` | 959 | `092c598f0136ea2adc3ce3bcbc8042bbafc13a4cbb62f8bd0859c7a0dad77e94` |
| `data/metrics/team_match_metrics.csv` | 10,945 | `2c03ab03f0cf2363ad2f039ab9a31b57e7070ccc6ca7ab8aacad4303f886afe7` |
| `data/metrics/trajectory_metrics.csv` | 1,363,141 | `94182ae194f1df7bbdf30475b621cb93671d9ca21aec78c25a05b4a64d90f498` |

**Study aggregates.** Recomputed by `report_study.py` and `plot_study.py` from the per-replicate summaries; every table in [§ 8.1](08_results.md#replicated-study) is derived from these three files.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/study/analysis.json` | 4,437 | `69a1756130e3a04663eef8b1ad131b088dd43df7435f1218acac9ad994fe302b` |
| `results/study/learning_curves.png` | 137,532 | `1915d48d117e8a96730b3659e0259e5ad6446c0fb7103d0642a6a5e7bd601cbe` |
| `results/study/per_seed_metrics.csv` | 5,287 | `a23d366199cbb78e805b52ac7b8569f4b3b72c4d3c05dd3e6977efadbcdac52f` |

**Untrained baselines.** `random_baseline.py` output, 300 fixed-variant headless matches per mode. The uniform files are byte-identical between the trees because that mode never queries a network.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/results/baseline/random_baseline_greedy.json` | 1,205 | `eccf400bf08621ea0ac8f5063742cfc3ae5bca93d78808942327508b14e6586e` |
| `ctde_arena/results/baseline/random_baseline_uniform.json` | 1,196 | `37d0c21947c0a56016df17f6862d61b98f81395fd5f7c76e03f5b73b4fd6106c` |
| `results/baseline/random_baseline_greedy.json` | 1,144 | `d453dddac636879222f7f727afd3fe0b762cd61d4eb50a6521da4db479d33c8c` |
| `results/baseline/random_baseline_uniform.json` | 1,182 | `8ce5317480f79bfc787b5176b35883473c2155192ab50db323c5ba7e3969dcc9` |

**Historical single-seed runs.** Superseded by the study but retained because [§ 8.2](08_results.md#historical-single-seed-runs) quotes them and their checkpoints no longer load into the current observation encoding.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/data/historical_100k/checkpoints/team_1_ctde-vd.pt` | 216,346 | `c0ab135db4674649e16db19e4ab901e4895196e298d891e62487cb1353f4bb24` |
| `ctde_arena/data/historical_100k/checkpoints/team_2_ctde-cac.pt` | 299,324 | `6352a1f04d74165fcfc08e632d5b52ecc69f3346eaa250b35fab4fae3dee29ed` |
| `ctde_arena/data/historical_100k/checkpoints/team_3_ctde-comm.pt` | 308,194 | `a80855ba555d8579623deb07591983c1fe3cd8c9eef1b5fd346106ae89b461a4` |
| `ctde_arena/data/historical_100k/checkpoints/training_log.json` | 1,646 | `5d85eedbfda798b58099ad5841529a7161398075675c7036c55cf9c7f84ba228` |
| `ctde_arena/data/historical_100k/exports/comparative_dashboard.png` | 272,740 | `3d0248bf4c843540393bd36b0c476a0165522419ca08732c6bb88c3f00c4c7c7` |
| `ctde_arena/data/historical_100k/metrics/agent_match_metrics.csv` | 20,297 | `d1870d75d6a26024d651f90ad6bca1842297fb34df975011c489e77386964cc5` |
| `ctde_arena/data/historical_100k/metrics/summary.json` | 951 | `fc33dfb9fce1c361a915c28fdc80452b7d85e720c6fd93e2d4895a6c3733aa4f` |
| `ctde_arena/data/historical_100k/metrics/team_match_metrics.csv` | 11,722 | `01869e471d91411c7e13fc8b9c4944db2b8f0cd04daeb089f9e865597c34d230` |
| `ctde_arena/data/historical_100k/metrics/trajectory_metrics.csv` | 2,111,002 | `94a41c8c67861ba1a7d8c3578bae8f18f57adf58e58c4b211353f835678fbdb3` |
| `data/historical_100k/checkpoints/training_log.json` | 1,629 | `38820ca4ee22bad1000aed3250370c716fde16e59fd8c949ea6f180facb4fd6a` |
| `data/historical_100k/exports/comparative_dashboard.png` | 285,844 | `0815af67bff72de2db06f0b7a42974c4e5bd1a5c2632ca766eb10e1ee4a0c321` |
| `data/historical_100k/metrics/agent_match_metrics.csv` | 181,660 | `4e8b527c751a079324a86663c5004ea66c6505b4f421e4a823cd80266ab85a79` |
| `data/historical_100k/metrics/summary.json` | 953 | `462070731a16d56162983ff7b3d08a6ed32e59025343c9be3d9fc827f3382a73` |
| `data/historical_100k/metrics/team_match_metrics.csv` | 11,075 | `338ffa7442e0386227611996a0592a8643b6223fed658775a7083dc4d2890eda` |
| `data/historical_100k/metrics/trajectory_metrics.csv` | 1,619,886 | `644342fc2f850c4a89fa35d0eb74b37d736339c2c9d51f3892f73d45f9edf406` |

**MLflow export and derived charts.** The 48 points that reached MLflow, plus the `plot_metrics.py` output for each tree.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/data/mlflow_export/mlflow_logged_metrics.csv` | 4,494 | `a94164d2c0c44d84241c29e77c3eaf1102edd3f146c9fa7797172e141db1f471` |
| `ctde_arena/data/mlflow_export/mlflow_runs.json` | 2,017 | `c2e815aba57167c1aba1ac67bf1b0b28661944a54bb2f5fecccebbf3d38a4d12` |
| `ctde_arena/exports/metrics/metrics_comparison.png` | 21,200 | `b5ae4c3368f3e4b665fc5f9f76982a69feebfee824da037ddfd36e43d30065a5` |
| `ctde_arena/exports/metrics/metrics_summary.csv` | 425 | `4536c72f92f2557c14a50f7728e253b44af39423877de21884c71e9ce27d04ad` |
| `exports/metrics/metrics_comparison.png` | 21,707 | `7e3a2eab959e1452b141c5563949f7064df3610b3afb47cd571ad04ff2c1368e` |
| `exports/metrics/metrics_summary.csv` | 409 | `d0bf34cc4dd622566ce468f5fe1af5c8f5bcfaa4e20d53bc8a2f0a42b633bdeb` |

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

**A reader who cloned the repository before 2026-09-27 could not recompute a single number from
Experiment 1.**

### Resolution

The rules now read `**/data/...` where they need to reach both trees, and experiment artefacts are
versioned by design. Experiment 1's five artefacts — `summary.json`, the three per-match CSVs and
`training_log.json` — plus its dashboard PNG are committed. What remains excluded is only what is not
evidence:

| Path | Why excluded |
|---|---|
| `mlruns/` | binary model artefacts and duplicated checkpoints; the logged points are exported to `ctde_arena/data/mlflow_export/` instead |
| `**/data/metrics/*.legacy*.csv` | orphaned pre-schema-rotation copies, one of them 29.9 MB |
| `**/data/runs/` | per-replicate study scratch (checkpoints and per-step trajectory CSVs); the aggregated analysis in `results/study/` is versioned instead |

One artefact needed care: experiment 1's `data/exports/comparative_dashboard.png` was still rendered with
the old Portuguese legends, because it was untracked and so the localisation pass never reached it. It was
regenerated from the relabelled CSV before being committed — same curves, English text.

The three untrained root checkpoints left in `data/checkpoints/` by the pre-fix test run were moved to
`data/checkpoints/_untrained_scratch/` rather than committed, so they cannot be mistaken for the
100 k-step policies.

## 9.9 Localisation rename <a name="localisation-rename"></a>

On 2026-09-27 the repository was converted from Portuguese to English. The rename touched identifiers
that appear in data, so it is documented here for provenance:

| Before | After | Where |
|---|---|---|
| `Equipe 1/2/3` | `Team 1/2/3` | `TEAM_NAMES`, `PARADIGM_CYCLE`, CSV `team_name` column, JSON keys, MLflow metric names |
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
