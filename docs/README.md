# Documentation index

This directory is the technical and academic record of the repository. It describes what was built,
how it was built, what was measured, what the measurements show, and — equally important — what has
**not** been established. Every numeric value in these documents was recomputed from the files that
ship with the repository or from a command executed against the current code; the derivation is
stated next to each value so it can be re-checked.

Read them in order if you want the full argument; jump directly if you only need one part.

| # | Document | Contents |
|---|----------|----------|
| 1 | [Introduction and research questions](01_introduction.md) | Problem statement, the two experiments, research questions, contributions, scope boundaries |
| 2 | [Related work](02_related_work.md) | Positioning against the MARL literature (CTDE, VDN, MAPPO, CommNet, QMIX, PPO, GAE) |
| 3 | [Arena system model](03_arena_system_model.md) | Entities, geometry, kinematics, combat resolution, swept collision detection, termination |
| 4 | [MDP formalisation](04_mdp_formalisation.md) | Observation and action spaces, feature engineering, reward function, episode structure |
| 5 | [Network architectures](05_network_architectures.md) | Layer-by-layer specification and parameter counts for all six paradigms |
| 6 | [Optimisation procedure](06_optimisation_procedure.md) | PPO objective, GAE, the three update paths, hyperparameters, what the implementation does and does not compute |
| 7 | [Experimental protocol](07_experimental_protocol.md) | Protocol, domain randomisation, metric definitions with formulas, artefact schemas |
| 8 | [Results](08_results.md) | The replicated study (primary), the historical single-seed runs, held-out learning curves, significance tests at both units of analysis, and a power analysis |
| 9 | [Reproducibility](09_reproducibility.md) | Environment setup, commands, measured runtime and cost, artefact inventory with checksums |
| 10 | [Threats to validity](10_threats_to_validity.md) | Internal, construct, external and statistical validity — the limitations section |
| 11 | [Code audit](11_code_audit.md) | Defects found by inspection and by execution, each with a reproduction command |
| 12 | [Glossary and references](12_glossary_and_references.md) | Terminology, symbol table, bibliography |

## Where the numbers come from

Four distinct data sources exist, and they are not interchangeable:

| Source | Contents | Versioned? |
|--------|----------|------------|
| `data/metrics/summary.json` (root) | Experiment 1, one promoted study replicate | Yes |
| `ctde_arena/data/` | Experiment 2 same | Yes |
| `ctde_arena/data/mlflow_export/` | The 48 metric points actually logged to MLflow across two runs | Yes (exported) |
| `results/study/` | 10 seeds × 500 k steps per experiment, slot-rotated, held-out greedy evaluation, paired *t*-tests, Fisher-exact tests and forward power | Yes |

The first two rows are **one promoted replicate each** — the run closest to its experiment's study means,
selected mechanically by `promote_run.py --seed median` and described by `data/PROVENANCE.json`. They are
what `python main.py` loads and what the dashboards show; they are **not** evidence about the ranking, and
a single replicate can reverse that ranking ([§ 8.1](08_results.md#leave-one-out)). The aggregated study in
`results/study/` is the primary evidence and is what [§ Results](08_results.md) leads with. The superseded
single-seed artefacts live on under `data/historical_100k/`, produced under the heading-control defect
[A-1](11_code_audit.md#turn-control-defect) before it was fixed.

Experiment 1's artefacts were themselves only committed on 2026-09-27 — before that, root-anchored
`.gitignore` patterns excluded them while committing experiment 2's
([A-3](11_code_audit.md#versioning)).

## Conventions

* **Experiment 1** is the repository root: CTE vs DTE vs CTDE.
* **Experiment 2** is `ctde_arena/`: CTDE-VD vs CTDE-CAC vs CTDE-Comm.
* Teams are addressed as `Team 1/2/3`; the paradigm is the experiment-relative label. Team identifiers
  were renamed from Portuguese (`Equipe N`) to English on 2026-09-27; the rename touched labels only,
  never metric values ([details](09_reproducibility.md#localisation-rename)).
* "Verified" means a claim was checked by running code in this session, and the command is quoted.
* "Asserted" means the claim appears in documentation or comments but was not confirmed by execution.

## Reproducing the numbers

| Artefact | Produced by |
|---|---|
| `results/study/per_seed_metrics.csv` | `python scripts/run_study.py --steps 500000 --seeds 1,2,3,4,5,6,7,8,9,10 --jobs 20 --eval-matches 150` |
| The tables in [§ Results](08_results.md#replicated-study) | `python scripts/report_study.py` |
| `results/study/learning_curves.png` | `python scripts/plot_study.py` |
| [§ 8.9](08_results.md#random-baseline) untrained baselines | `python scripts/random_baseline.py --matches 300 --policy greedy` and `--policy uniform`, in each tree |
| The leave-one-out table in [§ 8.1](08_results.md#leave-one-out) | `report_study.paired_t` over the per-seed matrix with one replicate dropped |
| The `--seed median` choice in [§ 9.7](09_reproducibility.md#97-artefact-inventory) | `python scripts/promote_run.py --experiment exp1 --list` |
| [§ 9.7](09_reproducibility.md#97-artefact-inventory) artefact table | `python scripts/report_inventory.py` |
| [§ 9.6](09_reproducibility.md#96-what-is-and-is-not-reproducible) reproducibility claim | two `train_rl.py --seed 42 --steps 4000` runs into separate `ARENA_DATA_DIR`s, then compare SHA-256 of the checkpoints and `training_log.json` |
| [§ 9.1](09_reproducibility.md#verification-environment-actually-used) NumPy-version comparison | the same seeded run under NumPy 1.26.4 and 2.2.6; variant parameters match to the last digit, saved weights do not ([A-31](11_code_audit.md#the-pinned-environment-does-not-reproduce-the-recorded-runs)) |
| Coverage figures in [§ Code audit](11_code_audit.md#coverage) | `coverage run --source=src/marl_arena -m pytest tests/ -q && coverage report` |
| Documentation cross-references | `python -m pytest tests/test_docs_links.py -q` |
