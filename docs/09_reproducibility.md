# 9. Reproducibility

> **Runtime update (2026-10-18).** The current training code uses JAX 0.4.38 and an optional Rust/PyO3
> extension for movement, obstacle correction and projectile collisions. `requirements.txt` now pins JAX;
> `native/arena_core/` builds the Rust extension with maturin. The `.pt` files from the historical PyTorch
> study are preserved and converted to non-pickle `.npz` checkpoints for the current runtime. The
> measurements and environment records below describe the prior PyTorch implementation; the current
> runtime has passed smoke/update checks but has not yet reproduced or replaced the published study.
> JAX documents native Windows x86_64 CPU support as experimental and Windows GPU support as unavailable;
> the verified device in this workspace is CPU. See the [JAX installation table](https://docs.jax.dev/en/latest/installation.html).

## 9.1 Environment

Current requirements are pinned per experiment. Root: `ursina==6.1.2`, `numpy==2.2.6`,
`matplotlib==3.10.3`, `pandas==2.3.3`, `python-dotenv==1.2.4`, `jax==0.4.38`, `jaxlib==0.4.38`, and
`pytest==9.2.0`. `ctde_arena/` adds `mlflow==3.17.2`. Building the Rust extension also requires Rust
stable and `maturin==1.15.0`.

The declared environment is complete: `pandas` is pinned in both `requirements.txt` files (it was once
only imported by `scripts/plot_metrics.py`, and the documentation used to call that a gap).

Two local directories are deliberately not part of the environment. `.venv/` is the repository runtime
the campaign scripts measure on: the JAX version pinned above plus the Rust extension built from
`native/arena_core`. `.venv-check/` was a CI-verification environment and has held a different JAX build
without that extension, so `scripts/_runtime.sh` refuses it and the campaign scripts no longer prefer it.
A stale `.venv.broken_old/` from an earlier host is git-ignored local state, not a supported runtime.

### Verification environment actually used

This is the historical PyTorch environment that produced the published measurements up to § 8.13. The
JAX/Rust runtime has since re-trained the study twice ([§ 8.14](08_results.md#v3): the JAX replication and
v3) and the capacity arms at ten seeds ([§ 8.13](08_results.md#capacity-ten)); those numbers were produced
on the pinned runtime of § 9.1, not on this table.

Every PyTorch-era number in this documentation set was produced on:

| | |
|---|---|
| Python | 3.10.11 |
| NumPy | 1.26.4 (pinned requirement says 2.2.6) |
| PyTorch | 2.5.1+cu121 (pinned then 2.6.0; no longer a dependency) |
| Matplotlib | 3.10.9 (pinned says 3.10.3) |
| pytest | 9.1.1 (pinned then 8.3.5) |
| Device | CPU |
| Platform | Windows 10 / 11 x64 |

**Every PyTorch-era number in this documentation set was produced on the versions above, not on the pinned
requirements** — NumPy 1.26 → 2.2 is a *major* version change. Rather than leave that as an untested
reassuring-sounding claim, it was measured, in a second environment holding everything else constant:

| Check | NumPy 1.26.4 vs 2.2.6 | Result |
|---|---|---|
| Experiment 1 test suite (59 tests then, 125 at this check) | both versions | **pass** |
| Experiment 2 test suite (47 tests then, 76 at this check) | both versions | **pass** |
| Collision geometry (14 hand-computed entry parameters and hit positions) | both versions | **identical to 1e-12** |
| Domain-randomisation stream (`arena_size`, speeds, cooldowns of match 41) | both versions | **identical, all digits** |
| `train_rl.py --seed 42 --steps 4000`, first PPO update | both versions | policy/value loss differ at the **1e-6 – 1e-8** level |
| Same run, saved checkpoints at 4,000 steps | both versions | **every file differs, in every byte** |

So the drift is not in the arena and not in the RNG — the variant parameters match to the last digit, which
rules out a stream change — but in floating-point accumulation inside the PPO update. A perturbation of one
part in a million at the first optimisation step is amplified by the chaotic dynamics of training into
completely different weights, and therefore a different replicate.

**Consequence, stated as the limitation it is:** installing `requirements.txt` and re-running the study
will not reproduce the recorded replicates. It will produce runs that are statistically *inside* the spread
this chapter measures — which is not a consolation but the study's own conclusion arriving from a second
direction. A difference of 1e-6 changes a replicate, and the between-seed SD is 0.12–0.22, so the ranking
is not a stable property of this system. Anyone quoting a specific win rate must quote the environment that
produced it, and that environment is the table above, not the pin file.

`requirements-measured.lock.txt` now records the environment above — 34 exact pins (both trees have the same
34; an earlier revision of this sentence said 37, which was the number of lines containing `==`, three of
them comments — `scripts/audit_dependencies.py` parses the pins and is what counts them now), generated by
`pip freeze` inside an isolated virtualenv, plus a separate block for the rendering packages. It is
verified rather than merely listed: `train_rl.py --seed 42 --steps 4000` produces byte-identical
checkpoints and `training_log.json` in the clean environment and in the one that ran the study.

What it does **not** do is produce hashes. `pip freeze` gives versions, not digests, and the torch
entry is a local-version wheel (`2.5.1+cu121`) served from PyTorch's own index rather than PyPI, so a
`--require-hashes` install is not achievable from this file alone. That is the remaining half of the
gap, and it is still item 4 of [§ 9.6](#96-what-is-and-is-not-reproducible).

Two notes on the local machines. The host's own `.venv/` is broken: its launcher points at a removed
`C:\Python313`, so it cannot start at all, and every measurement above came from a separate working
environment. The second NumPy environment used for this comparison was `--system-site-packages` with
NumPy 2.2.6 installed over PyTorch 2.5.1, so the two runs share a Torch version and only NumPy differs.

## 9.2 Setup

Checkpoints (`*.pt`, `*.npz`) are stored in Git LFS since the commit that introduced it; earlier history holds
them as ordinary blobs. Install Git LFS before cloning, or run `git lfs pull` afterwards -- without it the
checkout holds 130-byte pointer files and every test that loads a policy fails on them. The inventory of
[§ 9.7](#97-artefact-inventory) publishes the real files' SHA-256, which is the LFS object id.

```bash
git lfs install                   # once per machine
git clone <repository> && cd <repository>
# The engine is one package, src/marl_arena; ctde_arena/src is a shim onto it, so one environment and one
# Rust build serve both experiments.
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate  on POSIX
pip install -r requirements.txt   # ctde_arena/requirements.txt adds mlflow for experiment 2's train_rl.py
pip install pandas                # only needed by scripts/plot_metrics.py
pip install maturin==1.15.0
cd native/arena_core && maturin develop --release   # after installing Rust stable
cd ../..
cp .env.example .env              # .env is tracked and identical to .env.example
```

To rebuild the environment the committed numbers actually came from, install the lock instead. The two
files are not interchangeable, and [§ 9.1](#verification-environment-actually-used) measures what the
difference costs:

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements-measured.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
pip install pandas                # not a dependency of anything measured
```

Each committed `.env` lists 61 of the 62 environment variables `CONFIG` reads — the two trees run one
engine and read the same names, and the reward and termination keys of
[§ 4.4.2](04_mdp_formalisation.md#442-the-reward-and-the-termination-rule-are-configurable) and the remedy
switches of [§ 6.8](06_optimisation_procedure.md#68-the-docs10-remedies-as-switches) are in the file so that
reading it describes the whole instrument. Only three values differ from the code default in
`src/marl_arena/config.py` — `RANDOM_SEED`, which selects the slot rotation the shipped checkpoints need;
`RL_TRAIN_TOTAL_STEPS`, which the file pins to 1,000,000 against a code default of 3,000,000
([§ 6.7](06_optimisation_procedure.md#67-hyperparameters-as-trained)); and, in `ctde_arena/.env` only,
`PARADIGM_CYCLE`, which names experiment 2's three arms — so the file can be checked against the published
numbers without opening the source. `tests/test_arena_config.py` enforces all of it. The single deliberate omission is
`ARENA_DATA_DIR`: it re-roots the entire data tree,
`run_experiment.py` sets it to a fresh directory per replicate to guarantee random initialisation
([A-4](11_code_audit.md#silent-warm-start)), and pinning it in a file that every run shares would put all
of them in the same place.

## 9.3 Commands

| Command | Experiment | What it does |
|---|---|---|
| `python scripts/train_rl.py --seed S` | 1 | Headless JAX PPO training; seeded smoke path verified, and the full study re-trained on it ([§ 8.14](08_results.md#v3)) |
| `python main.py` | 1 | Ursina 3D window, greedy policies loaded from `data/checkpoints/`; demo matches record under `data/demo/`, not the versioned metrics tree |
| `python scripts/plot_metrics.py` | 1 | Reads `data/metrics/summary.json` → `exports/metrics/` |
| `python -m pytest tests/ -q`<!--auto:cmd-exp1--> | 1 | 217 passed, 1 skipped (verified 2026-11-02) |
| `python scripts/check_runtime.py` | either | Refuses an interpreter that is not the JAX pin + Rust extension; campaign scripts call it through `scripts/_runtime.sh` |
| `cd ctde_arena && python scripts/train_rl.py` | 2 | Same, plus MLflow scalar and checkpoint-artefact logging |
| `cd ctde_arena && mlflow ui --backend-store-uri file:./mlruns` | 2 | Tracking UI |
| `cd ctde_arena && python main.py` | 2 | 3D window for the CTDE variants |
| `cd ctde_arena && python -m pytest tests/ -q`<!--auto:cmd-exp2--> | 2 | 90 passed, 1 skipped (verified 2026-11-02) |
| `docker build -t ctde-arena -f ctde_arena/Dockerfile . && docker run --rm -v ${PWD}/ctde_arena/data:/repo/ctde_arena/data -v ${PWD}/ctde_arena/mlruns:/repo/ctde_arena/mlruns ctde-arena` | 2 | Containerised training; built from the repository root because the engine is shared |
| `bash scripts/run_item16_campaign.sh` | both | docs/11 item 16 as a resumable queue: the v3 checkpoints on the round-robin, random and scripted scales (66, 9 and 13 min), then `SPAWN_CYCLE_PER_MATCH` and `ARENA_LAYOUT=symmetric` one at a time at seeds 1-10 (83.7 and 151.2 min at four concurrent replicates), then versioning their checkpoints ([§ 8.14](08_results.md#which-remedy)) |
| `bash scripts/run_remedy_readout.sh` | both | Re-reads the JAX, single-key and v3 checkpoints at T = 0 with a slot remedy flipped in the evaluation only, into `results/readout_remedies/`; its two controls reproduce the published per-seed numbers exactly |
| `bash scripts/run_v3_campaign.sh` | both | The resumable queue behind [§ 8.14](08_results.md#v3): symmetric-arena baseline, JAX replication, the v3 study, scripted opponents, capacity arms at ten seeds, the `no_dr` random pool |
| `python scripts/report_study.py --sampled-readout results/readout/T0.5 --json-out results/study/coupled_tests.json` | both | The study table under both readouts, Friedman and sign-flip tests, bootstrap intervals on the secondary metrics |
| `python scripts/update_doc_counts.py` | both | Runs both suites under coverage and rewrites every marked test count and coverage figure in the docs; `--check` is the CI gate |
| `python scripts/benchmark_vec.py --arenas 1,4,16 --matches 32` | either | The vectorised evaluator against the one-arena path ([A-52](11_code_audit.md#a-wider-inference-batch-changes-greedy-outcomes)) |
| `python scripts/run_experiment.py --seed S --steps N --out DIR` | either | One seeded replicate: train, then greedy held-out evaluation |
| `python scripts/run_study.py --steps 500000 --seeds 1..10 --jobs 20` | both | Fan replicates out across processes and collect `results/study/` |
| `python scripts/run_study.py --collect-only --out data/runs_v2 --results-dir results/study_v2` | both | Rebuild an aggregate from run summaries already on disk, launching nothing |
| `python scripts/report_study.py --results-dir results/study_v2 --label "arena v2"` | both | Render the study tables for any campaign directory, not just the committed one |
| `python scripts/plot_study.py --runs-dir data/runs_v2 --out results/study_v2` | both | Draw `learning_curves.png` for any campaign directory |
| `python scripts/report_study.py [--results-dir DIR] [--compare-to CONTROL]` | both | Render the study tables, paired tests and forward power; with `--compare-to`, the per-seed paired table for an ablation arm or arena-v2 against a control campaign |
| `python scripts/plot_study.py` | both | Draw `results/study/learning_curves.png` |
| `python scripts/promote_run.py --experiment exp1 --list` | 1 or 2 | Show every replicate with its L1 distance to the study means |
| `python scripts/promote_run.py --experiment exp1 --seed median` | 1 or 2 | Copy one replicate into the versioned `data/` and write `PROVENANCE.json` |
| `python scripts/random_baseline.py --matches 300` | either | Untrained reference point: outcome mix, durations and per-arm metrics from random policies |
| `python scripts/run_cross_arena.py --matches-per-triple 30` | both | Play every three-architecture combination among the six versioned policies |
| `python scripts/run_cross_arena.py --exp1-replicate S --exp2-replicate M --seed H --out results/cross_arena_blocks/e1sS_e2sM_hH` | run from `ctde_arena/` | One block x one match stream of the round-robin, over replicate checkpoints instead of the promoted ones |
| `python scripts/report_cross_arena.py --in results/cross_arena/blocks.csv --json-out results/cross_arena/blocks.json` | root | McNemar pairwise tests, the leave-one-training-block-out column, the slot control and both stability tables |
| `python scripts/combine_cross_arena.py results/cross_arena_blocks/*` | root | Pool replicated round-robin runs, stamping each row with its training block and harness seed |
| `python scripts/run_cross_arena.py --campaign no_dr --exp1-replicate S --exp2-replicate M --seed H --out results/cross_arena_campaign/no_dr__e1sS_e2sM_hH` | run from `ctde_arena/` | One block of the round-robin over an **ablation** checkpoint set from `data/campaign_replicates/`; the capacity guard runs first, so `RL_HIDDEN_DIM` must equal the arm's width |
| `python scripts/version_campaign_checkpoints.py` (or `--check`) | root | Copy each ablation arm's completed runs into `data/campaign_replicates/` and write the manifest that records each set's override, hidden width, seed coverage and hashes; the seeds an arm covers are discovered on disk, so an extended arm cannot be left half-versioned, and `run_cross_arena.py --campaign` refuses an unversioned set rather than reading the scratch |
| `python scripts/version_training_replicates.py --pairs 5:6,6:8,7:9,8:10` | root | Version further study blocks into `data/training_replicates/`, resolving each replicate from the git-ignored scratch in whichever tree it was written; `--check --pairs …` verifies the bytes and writes nothing |
| `bash scripts/run_pool_v2_campaign.sh` | both | Queue the two pool re-measurements: the versioned checkpoints replayed under the arena-v2 termination and reward rule (30 runs at launch, later extended to all ten blocks by `run_pool_v2_ten_block.sh`), then the `no_dr` arm's checkpoints played against the committed pool (15 runs), each pooled and reported |
| `bash scripts/run_ten_block_campaign.sh` | both | Queue the four extra blocks (20 runs, 12,000 matches) behind the pool campaign, then re-pool all fifty run directories into `results/cross_arena/blocks10.csv` and its report — the merge refuses to run unless every new directory wrote a `matches.csv` |
| `bash scripts/run_pool_v2_ten_block.sh` | both | The same extension for the arena-v2 arm: four more blocks under the exported v2 keys, re-pooled into `results/cross_arena/v2_blocks.csv` at ten blocks, so § 8.12's rule control is measured against the headline pool's population rather than a six-block subset. Chained behind the readout campaign, and its merge refuses anything that is not `10 training blocks x 5 harness seeds` |
| `bash scripts/run_capacity_pool_campaign.sh` | both | Play the two capacity arms (`narrow_64`, `wide_256`) on the common opponent pool — 3 blocks × 5 streams each, the most those arms can give. `RL_HIDDEN_DIM` is read out of `MANIFEST.json` per arm rather than typed, because the round-robin's capacity guard is what stops one arm's weights being played at another's width. Chained behind the v2 extension |
| `python scripts/eval_readout.py --temperature 1 --seeds 1,…,10 --matches 150` | either tree, on its own replicates | Re-evaluate existing checkpoints under the sampled readout (`--temperature 0` replays the published argmax as the control), writing `per_seed_metrics.csv` in the study schema so `report_study.py --results-dir … --compare-to results/study` pairs it against the published numbers |
| `python scripts/eval_regimes.py --seeds 1,…,10 --matches 150` | either tree, on its own replicates | Replay the study's held-out protocol over the **versioned** replicates, recording one row per team per match *including* the duration the original runs never wrote |
| `bash scripts/run_regime_split.sh` | root | Both trees' regime replays in parallel, the 9,000-row merge into `results/study/eval_match_regimes.csv`, then the report |
| `python scripts/report_regimes.py` | root | The by-regime win-rate split of [§ 8.1](08_results.md#two-regimes), gated by the published-win-rate control (52/60 cells exact, worst 2/150 under the current runtime); writes `results/study/regime_split.json` |
| `bash scripts/run_readout_campaign.sh` | both | Queue all three readout arms (T = 0, 0.5, 1) over both experiments' ten seeds each, merge the per-tree parts, and report each arm against `results/study/`; chained behind the pool campaigns so no cost figure in § 9.5 is measured under two campaigns at once |
| `python scripts/run_absolute_eval.py --matches 150 --seeds 1,2,3` | run from `ctde_arena/` | Absolute strength on the promoted checkpoints: each trained policy against two uniform-random controllers, trained slot rotated |
| `python scripts/run_absolute_eval.py --blocks 1:7,10:5,9:2 --matches 50 --seeds 1,2,3 --out results/absolute_blocks --null-json results/baseline/random_baseline_uniform.json --compare-pool results/cross_arena/blocks.json` | run from `ctde_arena/` | The same over three training blocks, with the slot-matched null and the agreement with the peer pool; `--campaign <arm>` takes the ablation sets instead, behind the same capacity guard |
| `python scripts/run_absolute_eval.py --blocks 1:7,10:5,2:1,3:3,4:4,9:2 --matches 25 --seeds 1,2,3 --out results/absolute_six --null-json results/baseline/random_baseline_uniform.json --compare-pool results/cross_arena/blocks6.json` | run from `ctde_arena/` | The six-block predecessor of the headline artefact, still quoted because § 8.11's replication ladder is its finding |
| `bash scripts/run_traces_campaign.sh` | both | Re-train the twenty study replicates into the git-ignored `data/runs_traces/` so each carries a `training_log.json`, then join those traces to `results/study/` with `scripts/analyse_traces.py`. The campaign writes its own two controls into the record and both came back clean: 60/60 checkpoints hash-match `data/training_replicates/`, and the re-trained held-out rates sit within 3.3e-05 of the published ones |
| `bash scripts/run_no_dr_extension.sh` | both | Train the `no_dr` arm out to seeds 4-10 under the same single key change, version the new sets, and replay the arm over ten blocks so its common-pool result is measured at the same replication count as § 8.10. Chained behind the traces campaign. Re-launching it does not re-train: the training block is skipped once all fourteen run summaries exist, and the replay refuses to start unless the manifest lists 20 `no_dr` sets — [A-43](11_code_audit.md#an-arm-was-extended-and-the-versioning-step-silently-versioned-none-of-it) is why that guard exists |
| `bash scripts/run_absolute_ten_block.sh` | run from `ctde_arena/`, via the script | The headline [§ 8.11](08_results.md#absolute-strength) artefact: all ten blocks at the same cell size, against the peer pool at the *same* replication count — which is what the −0.04 story says the comparison requires. The script verifies the record's shape (4,500 rows, 750 per architecture, ten blocks x three streams) and that the summary names `blocks10.json` as its peer pool before calling it done |
| `bash scripts/run_three_million_campaign.sh` | both | Open-work item 7: re-train the same twenty replicates at the 3,000,000-step budget the code defaults to — `--steps` is the only difference from `results/study/` — into `data/runs_3m/` and `results/study_3m/`, then report the pairing and run `analyse_budget.py`. Refuses to train twice (skips when all twenty run summaries exist) and refuses to analyse a run that stopped short of its budget |
| `python scripts/analyse_budget.py --three-m data/runs_3m --half-m data/runs --three-m-results results/study_3m --out results/study_3m/budget_analysis.json` | root | The budget comparison, and the curve question `report_study` cannot answer: held-out rates paired on seed number, the ranking at each budget, and each run's held-out curve at 25/50/75/100 % of **its own** budget, with the controls the record has to carry (the steps each side actually trained, the shared cell count, and the 500 k side's parity against the committed study) |
| `python scripts/run_absolute_eval.py --report-only --out results/absolute_blocks …` | run from `ctde_arena/` | Re-derive a summary from its recorded `matches.csv`, playing nothing |
| `bash scripts/run_arena_v2_campaign.sh` | both | Queue the arena-v2 study: 10 seeds x 2 experiments at 500 k steps under the decayed alive bonus and draw-on-cap, waiting for whatever is already on the machine |
| `bash scripts/run_ablation_campaign.sh` | both | Queue the three configuration ablations (`no_dr`, `narrow_64`, `wide_256`), each one key different from the committed study, after the v2 campaign |
| `python scripts/report_inventory.py` | root | Regenerate the § 9.7 artefact table from `git ls-files` and live SHA-256 |
| `python scripts/render_trailer.py --frames 240` | either | Render a gameplay clip from the promoted policies via Ursina offscreen, and the JSON sidecar beside it that records what the caption asserts |

**The thirty-run round-robin loop.** `results/cross_arena/blocks6.csv` is the concatenation of
`run_cross_arena.py` over the six versioned training blocks x five harness seeds; the directory names are
what carry the labels, so the loop is:

```bash
# The round-robin runs inside ctde_arena/; the pooling and the report run at the repository root.
cd ctde_arena
for pair in "1 7" "10 5" "2 1" "3 3" "4 4" "5 6" "6 8" "7 9" "8 10" "9 2"; do
  set -- $pair; S=$1; M=$2
  for H in 1 2 3 4 5; do
    python ../scripts/run_cross_arena.py --exp1-replicate $S --exp2-replicate $M --seed $H \
      --out results/cross_arena_blocks/e1s${S}_e2s${M}_h${H}
  done
done
cd ..
python scripts/combine_cross_arena.py results/cross_arena_blocks/* --out results/cross_arena/blocks10.csv
python scripts/report_cross_arena.py --in results/cross_arena/blocks10.csv \
  --json-out results/cross_arena/blocks10.json
```

`--out` and `--in` resolve against the **repository root**, not the working directory, because these scripts
drive both trees; that is why the loop above runs from `ctde_arena/` and still writes to `results/`.
`--exp1-replicate/--exp2-replicate` read `data/training_replicates/{experiment}_seedN/` first and fall back
to the git-ignored `data/runs/`. The ten pairings above are the versioned blocks, and no seed of either
experiment appears twice; `run_cross_arena.py` without those flags plays the promoted `data/checkpoints/`
instead, which is the same networks as `exp1_seed9+exp2_seed2`.

`bash scripts/run_six_block_campaign.sh` and `bash scripts/run_ten_block_campaign.sh` run that loop with the
guards the first version of it lacked: every one of the run directories must contain a `matches.csv` before
anything is merged, and the merge must report the block count the script was written for. Both were written
after that script silently produced nothing -- a
repository-root-relative interpreter path inside a `cd ctde_arena` subshell -- and then re-merged the
fifteen existing directories as if they were the new thirty. The three-block `blocks.csv` and the six-block
`blocks6.csv` are still in the
repository because two of the three agreements between the two opponent pools, reported in
[§ 8.11](08_results.md#absolute-strength) at one, three and six sets, are quoted from them and from their
`blocks.json`; the six-set figure is quoted from `blocks6.json`, at its own replication count, and
`blocks6.csv` is also the peer pool of [§ 8.12](08_results.md#arena-v2-pool)'s six-vs-six control. Those counts have
to be matched: the −0.04 that section reports as its own past error was produced by comparing a one-set
measurement against a three-set peer table.

`run_experiment.py`, `random_baseline.py` and `render_trailer.py` are byte-identical in both trees, and
`tests/test_tree_parity.py` fails the suite if they drift; which engine they bind to is decided by the
working directory, exactly like `train_rl.py`. `run_study.py`, `report_study.py`, `plot_study.py`,
`promote_run.py`, `report_inventory.py`, `report_cross_arena.py`, `combine_cross_arena.py`,
`run_absolute_eval.py`, `run_arena_v2_campaign.sh` and `run_ablation_campaign.sh` exist only at
the root and drive both trees — `promote_run.py` writes experiment 2's artefacts into `ctde_arena/data/`,
and both `run_cross_arena.py` and `run_absolute_eval.py` are *run from* `ctde_arena/` because that tree's
package is the superset holding all six architectures. `tests/test_tree_parity.py` enforces which files must stay
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

Current JAX/Rust verification. The block below, the coverage rows further down and every "Current tests"
line in the READMEs are written by `scripts/update_doc_counts.py` from an actual run of both suites under
coverage, and CI's artefact job fails when the documented totals stop matching what pytest collects -- they
used to be typed, and went stale with every new test:

<!--auto:pytest-block-->
```
$ python -m pytest tests/ -q
217 passed, 1 skipped in 141.35s
$ cd ctde_arena && python -m pytest tests/ -q
90 passed, 1 skipped in 28.24s
```
<!--/auto-->

Three consecutive runs of each suite returned those totals unchanged. One earlier run of the root suite failed
a single case, and the summary line was the only thing captured, so the case is unnamed here; it did not
reproduce in the sixteen root-suite runs that followed, and the suite contains no time-dependent assertion.
Runtimes are quoted from the idle machine, and they are dominated by whatever else is on it.

```
$ python -m pytest tests/ -q            # experiment 1, before the JAX/Rust port
140 passed, 1 skipped
$ cd ctde_arena && python -m pytest tests/ -q
75 passed, 1 skipped
```

The skipped test in each suite is `test_render.py`'s offscreen path, which needs `ursina`: the pinned
requirement installs it, the verification environment above does not, so those assertions only run where the
commented rendering block of `requirements-measured.lock.txt` has been installed. The clips in
`docs/assets/` were rendered in such an environment, and the test numbers here are not from it — which means
the render path is verified by inspection of its output, not by the suite that is quoted above.

Current statement and branch coverage over `src/marl_arena`, measured with `coverage run --branch --source=src/marl_arena -m pytest`
(`pytest-cov` is not installed, and the two suites must still be run separately), alongside the figures the
previous implementation recorded:

| Tree | Statements | Missing | Statement | Branch | Runtime measured |
|---|---:|---:|---:|---:|---|
| Experiment 1<!--auto:cov-exp1--> | 2,275 | 234 | **90 %** | **81 %** | JAX/Rust |
| Experiment 2 (`ctde_arena/`)<!--auto:cov-exp2--> | 2,275 | 481 | **79 %** | **66 %** | JAX/Rust |
| Experiment 1 | 1,548 | 91 | 94 % | — | PyTorch, historical |
| Experiment 2 (`ctde_arena/`) | 1,660 | 187 | 89 % | — | PyTorch, historical |

The port lowered measured coverage in both trees, by eight and one point, without adding a feature that is
untested: the two files that grew are `rl/networks.py`, whose 65 missing statements are the CTDE-VD and
CTDE-Comm forward and `state_dict` branches plus the unknown-kind and shape-mismatch raises, and
`controllers/batching.py` at 14. A second implementation of every function is, by construction, a second set of
paths, and the suites train the paradigms they name rather than every branch of each. Three consecutive runs
of each suite gave the current totals unchanged. Earlier in the project the same
measurement did move by a couple of missing statements between identical invocations — four runs of
experiment 1 gave 90, 84, 88 and 90, which is why this table carried ranges for a while — and CI re-runs both
suites on Linux with `requirements.txt` ([§ 9.4](#continuous-integration)), where a statement or two is
expected to differ from the host numbers quoted here.

Twelve files hold experiment 1's 264 missing statements: `simulation.py` (73), `rl/networks.py` (65),
`controllers/rl_controller.py` (39), `controllers/base.py` (32), `controllers/batching.py` (14),
`rl/actions.py` (11), `systems/metrics.py` (9), `systems/plotting.py` (8), `rl/ppo.py` (7),
`systems/match_variant.py` (3), `runtime.py` (2) and `config.py` (1). Experiment 2 shares nine of them at
the same counts, and the 35-statement difference between the two trees is its own three:
`rl/networks.py` (−34) and `controllers/batching.py` (−4) are less exercised there, `rl/ppo.py` (+3) more.
`ui/dashboard.py`, which was the single
largest uncovered file for most of this repository's life, is at 100 % in both trees.

What the 188 root-tree and 91 `ctde_arena` test cases do **not** cover is catalogued in
[§ Coverage](11_code_audit.md#coverage). The collision solver is now geometrically verified against
hand-computed values, but no test checks a *metric* against an independent reference — see the note at the
end of that section on coverage not being correctness.

### Continuous integration <a name="continuous-integration"></a>

`.github/workflows/tests.yml` runs five job families on every push and pull request:

| Job | Gate |
|---|---|
| `suite (experiment 1/2, py3.10/3.11)` | `pytest tests/ -q` in each tree on two Python versions, then branch + statement coverage of `src/marl_arena` with `--fail-under=83` in experiment 1 and `74` in experiment 2, whose suite measures the whole shared engine but exercises only its own arms |
| `native extension parity` | builds `native/arena_core` with maturin and Rust, asserts the extension imports, and runs `tests/test_native_parity.py` so the test cannot skip |
| `docker image builds` | `docker build` of `ctde_arena/` and an import smoke test inside the container |
| `suite (windows, py3.11)` | the root suite on Windows, where A-42's path bug lived |
| `documentation and derived-artefact gates` | tree parity; documentation cross-references; regenerate `docs/09` § 9.7 and fail on a diff; re-derive **every** pooled round-robin report from its committed CSV and every absolute-strength summary from its own `matches.csv`, and fail on a diff; `bash -n` every campaign script; and check that `results/dependency_audit.json` covers exactly the pins of `requirements-measured.lock.txt` |

The two artefact gates are the interesting ones, because they turn claims this chapter makes in prose into
something a machine checks: § 9.6 says `blocks6.json` is derived data reproducible from `blocks6.csv`, and CI
now re-derives it on a different operating system than the one that produced it. Two checks work in the
other direction — `tests/test_docs_quote_the_pool.py` recomputes § 8.10's published tables and § 8.12's
rule-control table from `blocks10.json` and `v2_blocks.json`, and
`tests/test_docs_quote_the_readout.py` recomputes § 8.1's sampled-readout table *from* the artefacts behind
it, so a document cannot quietly keep a number an earlier pool produced. `bash -n` on the campaign
scripts exists after two of them "completed" while launching nothing (see [§ 9.3](#93-commands) and the
six-block campaign's post-mortem); a syntax check does not catch that, but it does catch the family of
quoting and variable-name errors in the same scripts.

**Five runs were red, and the gate was right.** Both suite jobs passed throughout; the artefact job died
re-deriving `results/absolute*/summary.json`, because those records name their matched null and their peer
pool with Windows separators, and a Linux runner cannot open `results\baseline\…`. The failure arrived with
`43429ed`, the commit that made `--report-only` inherit the inputs a summary was derived from: the same
inheritance that makes a re-derivation faithful is what reads back a path spelled for the host that wrote it,
so runs #11 to #15 were red. This is the chapter's own promise — a published table re-derives from a clone —
being checked on a machine that is not this one.
[A-42](11_code_audit.md#a-committed-artefact-recorded-windows-paths) is the post-mortem, and
`tests/test_artefact_path_portability.py` is the gate that keeps it from coming back. The writers now emit
POSIX paths, the affected records were regenerated, and run #16 is green on all three jobs — the re-derivation
that failed on the runner is the one that has since passed on it.

**What CI does not do, and why that is deliberate.** It installs `requirements.txt` (NumPy 2.2.6, JAX
0.4.38), not `requirements-measured.lock.txt` — the question it answers is whether the code does what its
tests say, and [A-31](11_code_audit.md#the-pinned-environment-does-not-reproduce-the-recorded-runs) measured
that the environment that would reproduce *the numbers* is not reachable from a CI container without also
re-running the study. It excludes `ursina`, so `test_render.py`'s capture case skips for the same reason it
skips on the host (a runner has no libGL, and a half-installed Panda3D would fail rather than skip). It
never trains a campaign: the fastest training test in either suite is a 4,000-step update, the native job
plays a 32-step parity test, and the Docker job imports the engine rather than running it. The Rust and
Docker jobs prove the artefacts build and import; they do not replace a measured run on the repository
runtime, which `scripts/_runtime.sh` enforces for campaigns.

## 9.5 Cost of a run

The measurements below are retained as records of the earlier PyTorch implementation. The JAX/Rust
runtime is measured separately in the runtime-update note at the end of this section because JAX's first
call compiles the update and must be excluded from steady-state throughput.

Measured on this machine (32-core host, RTX 4070 Laptop GPU), CPU only, domain randomisation enabled,
all three teams training, one process:

```
torch.set_num_threads(1) -> 368 steps/s
torch.set_num_threads(2) -> 261 steps/s
torch.set_num_threads(4) -> 259 steps/s
CUDA, same loop          -> 150 steps/s
```

These are historical measurements of the scalar policy path. The current code batches live teammates by
controller; a paired 1,500-step CPU run reached 599.5 steps/s against 272.4 for the original path, with
the same 17 matches and the same team totals. The tests and exact workload are recorded in
[A-44](11_code_audit.md#per-agent-policy-dispatch-was-the-training-bottleneck). The 20-way study figures
below are still the measurements from before batching; a concurrent campaign has not been re-run.
The subsequent vectorized nearest-enemy and clear-obstacle paths improved isolated hot-function timings;
whole-training timings were too noisy to replace the paired 599.5 steps/s measurement.

### Current JAX/Rust runtime sample (2026-10-18) <a name="runtime-jax-rust-sample"></a>

The environment is the one the port actually runs on here: Python 3.10.11, `jax`/`jaxlib` 0.4.38, NumPy
1.26.4 and `marl_arena_rust` 0.1.0 -- the versions A-45 records below, and the reason `requirements.txt`
pins this JAX build rather than the one an unverified reading once put in it.

Measured from the repository root with `python scripts/benchmark_runtime.py --steps 2000
--warmup-steps 2000 --repeats 4`, CPU backend, seed 42, domain randomisation off, all teams training.
The script runs both paths in one process, warms each path before timing, and checks match boundaries,
winners and final parameter hashes; `exact_outcome_and_weight_parity` was true again.

| Route | Four warmed 2,000-step repeats (steps/s) | Median |
|---|---|---:|
| Rust/PyO3 core | 291.6, 309.7, 275.0, 290.0 | **290.8** |
| Python fallback | 241.3, 262.9, 245.6, 265.7 | **254.3** |

The Rust route is ahead in every repeat, so the port is worth about **+14 % end to end** at this team size.
That reverses the first pair recorded here (410.6 Rust against 420.9 Python, one repeat each), which was a
single draw rather than a measurement: its own level did not recur in any of the four repeats, and the
machine was not idle while it ran. The warm-up rates are intentionally not compared: the first route includes
JAX's compile cost, and the second route reuses the compiled programs.

**Why a campaign does not scale with processes.** The same warm-then-measure route, run as `N` concurrent
processes on an idle machine (steady 2,000-step windows, Rust core):

| N | steps/s per process | Aggregate steps/s | CPU ms per env step |
|---:|---:|---:|---:|
| 1 | 335.4 | 335.4 | 4.65 |
| 2 | 286.4 | 572.8 | 7.21 |
| 4 | 196.3 | **785.4** | 7.88 |
| 8 | 76.2 | 609.8 | 21.96 |
| 16 | 40.6 | 649.8 | 42.21 |

Aggregate throughput peaks at four processes and then falls, while the CPU burned per step rises nine-fold --
wasted scheduling, not lost clock. The host is not the limit: a control with no repository code in it, eight
concurrent NumPy matmul loops with their BLAS and OpenMP pools capped to one thread, gives 1,197.8 iterations/s
alone against 878–888 each at eight (≈7,030 aggregate, **5.9× from eight processes**), and its CPU time per
iteration grows only 1.39×. The difference is what each process carries. A bare Python process here has 6
threads; NumPy's BLAS pool takes it to 29; after JAX's CPU backend initialises it is 104, and 127 if
`marl_arena.config`'s environment caps are skipped. Those caps are therefore doing something -- 127 to 104 --
but they do not make the process single-threaded, and the flag that would is not available: putting
`--intra_op_parallelism_threads=1` in `XLA_FLAGS` aborts the child at
`external/xla/xla/parse_flags_from_env.cc:226` with "Unknown flags in XLA_FLAGS". Eight arena processes are
~830 runnable threads on this machine's 32 logical cores.

Two things follow. A JAX campaign should be queued about **four deep**, which is where the measured aggregate
peaks, rather than the twelve or twenty the committed campaign scripts ran: on this instrument four processes
deliver about **1.2× the aggregate throughput of sixteen**, and they do it with a quarter of the threads in
flight. Those windows are early training, where matches are short, so no wall-clock hour figure may be read
off them -- the 3 M-step study below measured 49.1 steps/s per replicate at twenty-way because a learned
policy takes longer to kill than a random one. And no JAX number in this document may be compared with a
PyTorch one as if the two runtimes were the same instrument: the port's per-step work, optimizer and action RNG
all changed, and the older records below were measured with one thread per process while these are measured
with a hundred.

None of this closes the replication question. A JAX/Rust replication campaign is still required before any
chapter-8 number may be described as reproducible by this runtime. On this Windows x86_64 host JAX used CPU;
the [JAX platform table](https://docs.jax.dev/en/latest/installation.html) marks that CPU combination
experimental and provides no Windows GPU support, so this machine's discrete GPU is not a lever the runtime can
pull without changing operating system.

### The step's own host overhead (2026-10-20) <a name="host-dispatch-overhead"></a>

The 2026-10-18 sample above measures the whole step; a profile of where the step's wall clock goes found that
most of the JAX cost was not the small matmuls but the *host dispatches around them*. One `decide_many` call —
2.4 of them run per environment step on average — paid for: an eager `jnp.asarray(...)[None, :]` expand of the
global row, an eager `jnp.broadcast_to` of the critic's value (and for CTE of the policy rows), two eager
device-side slices `[:len(agents)]` of the actor outputs, and **two** `jax.device_get` syncs (logits, then
values). The PPO update paid a `jax.device_get` of its four scalars **per minibatch**, serialising the host
against every one of the ~12 dispatches an update makes. The fixes change no value the networks ever see:

* slicing, expanding and broadcasting moved to the host — a float32 row copied before or after the device
  transfer is the same bytes, and the padded batch shapes the jitted programs receive are unchanged;
* the two syncs became one `jax.device_get((logits, values))`;
* the per-minibatch PPO syncs were deferred to the end of the update — only the last minibatch's numbers are
  ever reported, and they are the same numbers (`src/marl_arena/rl/ppo.py`);
* `JAXNetwork.forward` no longer re-wraps inputs that are already float32 device arrays.

**Verification, bit for bit.** The pre-change and post-change sources were run as paired, same-seed training
windows. Experiment 1, 8,000 steps from scratch: match boundaries, winners, per-match reward sums and all
three teams' final parameter digests were **equal**. Experiment 2, the same window: every PPO update's
per-leaf parameter digests were equal across all 46 updates, and a first-update parameter dump matched
exactly. `exact_outcome_and_weight_parity` stayed true in every benchmark run before and after. The one
caveat this verification surfaced is not about the change: under concurrent load the *pre-change* runtime
itself is not bit-frozen — [A-47](11_code_audit.md#the-jax-training-update-is-not-bit-deterministic-under-concurrent-load)
records four identical-seed processes producing two weight sets that differ by ≤ 3e-08 after the first
update, and the same flip appearing under the post-change code at higher load. Solo, both runtimes were
byte-stable in every run.

**Measurement, interleaved.** Because this host's background state moves absolute rates by ±10 % between
sessions (the 2026-10-18 note measured the same effect), the pre- and post-change sources were benchmarked
alternately in one session with the committed protocol (`--steps 2000 --warmup-steps 2000 --repeats 4`):

| Protocol run, in order | Rust/PyO3 median steps/s | Python fallback median steps/s |
|---|---:|---:|
| pre-change source | 223.1 | 213.0 |
| **post-change source** | **298.2** | **262.8** |
| pre-change source | 224.5 | 206.5 |
| **post-change source** | **291.4** | **263.2** |

The paired medians put the change at **+30 % on the Rust route and +25 % on the fallback** (298.2/223.1 and
291.4/224.5; 262.8/213.0 and 263.2/206.5). Two days earlier the same instrument had measured the pre-change
Rust route at 290.8 on an idle host, so the absolute level of any one session is not the claim — the
interleaved ratio is.

**Concurrency re-measured.** The scaling ladder above was re-run twice with the post-change source (steady
2,000-step windows, Rust core). The two ladders disagree everywhere except at the depth the guidance
already recommends: at N=4 both measured **347–370 steps/s per process — solo speed — for an aggregate of
≈ 1,400–1,470**, where the committed ladder measured 196.3 and 785.4. At N=1, N=2 and N=8 the two ladders
differ by up to 80 % between waves run minutes apart (per-process 121–217 at N=8), so those cells are host
state, not code, and the committed table is left as the record of the pre-change instrument. The queue-four-deep
guidance survives — both ladders put N=4 at or near the aggregate peak with a quarter of the threads of N=16 —
and its economics improved: a four-deep queue of post-change replicates delivers ≈ 1.8× the pre-change
aggregate at that depth.

For the historical PyTorch runtime, one thread was the selected setting because each model is small and
each team batch has at most three rows. The original profile and the per-agent dispatch cost are in
[§ GPU and thread scaling](11_code_audit.md#gpu-and-thread-scaling).

| Budget | Wall time, one process | Notes |
|---|---:|---|
| 100,000 steps (historical runs) | ≈ 4.5 min | 463 / 456 matches |
| 500,000 steps | 90–101 min | one study replicate, **measured under 20-way concurrency**; ≈ 25 min alone on the previous scalar path |
| 1,000,000 steps | ≈ 45 min | historical estimate for the superseded 5-seed budget |
| 3,000,000 steps (code default) | ≈ 1.4 h | rough estimate scaled from the paired batched-path measurement |
| 500 k × 10 seeds × 2 experiments | **117.5 min total** | 20 processes at once on 32 cores; 31.5 CPU-h training + 4.7 h evaluation |
| 3 M × 10 seeds × 2 experiments (open-work item 7) | **1251.1 min** | `scripts/run_three_million_campaign.sh`: the committed study's twenty replicates re-trained at the budget `main.py` ships with, 49.1 steps/s per run under twenty-way load against the 85–92 the 500 k study measured. Six times the steps cost 10.6× the wall clock, and the gap is the finding: matches get longer as the policies learn to stay alive, so a step gets more expensive while nothing about the arena changes ([§ 8.13](08_results.md#budget-3m)) |
| same again as arena v2 | **104.3 min total** | `scripts/run_arena_v2_campaign.sh`; 20 runs, same budget, three arena keys changed ([§ 8.12](08_results.md#arena-v2)) |
| 500 k × 3 seeds × 2 experiments, one arm | **69–81 min per arm** | `scripts/run_ablation_campaign.sh`; 6 concurrent, so 24–34 min of training per replicate at 243–350 steps/s (against 85–92 steps/s under 20-way load) plus the held-out evaluation. Three arms in series: 3.9 h |
| Round-robin replay, 6 blocks × 5 seeds, arena-v2 rule | **89 min** | `scripts/run_pool_v2_campaign.sh` phase 1: 18,000 matches, five concurrent runs per block, 17–20 min per wave of 3,000 matches. Evaluation only — the 36 checkpoints already existed |
| 500 k × 7 new seeds × 2 experiments, one arm (`no_dr` to ten seeds) | **170.8 min** | `scripts/run_no_dr_extension.sh` phase 1: fourteen concurrent replicates under `DOMAIN_RANDOMIZATION=false`, each recording 96–132 min of train-plus-held-out time — 25.8 CPU-hours of training in 2.85 h of wall clock, the same ratio the twenty-run study row reports. This is the one re-measurement in this table that could not be bought with evaluation time alone: a pool cannot be extended past the seeds that exist, and item 13 of the audit's open list waited on it |
| Round-robin replay, 10 `no_dr` blocks × 5 seeds | **3 h 49 min** | `scripts/run_no_dr_extension.sh`: 35 new runs and 21,000 matches (63,000 team-matches) at twelve concurrent, 58–78 min per run, first finish 20:20 and last 23:11 after a 19:21 launch. Evaluation only — the 20 checkpoint sets were versioned first. The per-run time is the finding, not the machine: policies trained without domain randomisation end only 44.8 % of their matches by wipeout against 58.7 % for the randomised ones, so more matches run to the 90 s cap. The campaign's first shape was five concurrent in a wave per block, which measured 3 min per triple of architectures and left 27 of this machine's 32 cores idle; the queue width is why this row is hours rather than the ≈ 85 min its three-block predecessor took |
| Round-robin extension, 4 blocks × 5 seeds, committed rule, then re-pool | **79 min** | `scripts/run_ten_block_campaign.sh`: 20 new runs and 12,000 matches at 16–27 min per wave of 3,000, then the fifty-directory merge and its report. Evaluation only — all sixty checkpoints were versioned before it started. Chained behind the two rows above, so each campaign's wall clock is its own |
| Round-robin extension, 4 blocks × 5 seeds, arena-v2 rule | **5 h 08 min (upper bound, not a machine cost)** | `scripts/run_pool_v2_ten_block.sh`: 12,000 matches in four waves of 52, 110, 85 and 57 min, against the 17–20 min per wave the same script's first six blocks took nine hours earlier the same day. The work is identical per match — the two pools share the same 17,614 wipeout matches and the same 81.3 s mean duration — no sleep event is in the system log, and a 20-match probe run at the end of the campaign clocked 2.2 s/match, the afternoon rate. So the night window carried host load this repository cannot attribute, and the honest figure for the campaign is its own wave times rather than a throughput claim |
| Absolute strength, 10 blocks × 3 streams × 25 matches per arm | **27 min** | `scripts/run_absolute_ten_block.sh`: 4,500 matches in one process, ~0.36 s per match — random opponents never act, so the mean match runs 10 s of the 90 s cap instead of the 81 s the peer pool's matches take. The cheapest extension in this table by a wide margin, and the one that removed the last replication-count mismatch between two published numbers |
| Capacity arms on the common pool, 2 × 3 blocks × 5 seeds | **1 h 59 min** | `scripts/run_capacity_pool_campaign.sh`: 18,000 matches, 15 runs per arm, five concurrent. The two arms are not comparable in cost even though they play the same number of matches — `narrow_64` took 35 min (9–14 min per wave) and `wide_256` 94 min (25–35 min per wave) on the same machine an hour apart, which is the network width showing up in the evaluation harness rather than in the result. Both ran at the rate the afternoon probe measured, so unlike the row above this is a machine cost |
| Re-training the study for optimiser traces, 20 runs × 500 k steps | **134.8 min** | `scripts/run_traces_campaign.sh`: twenty concurrent replicates, the same seeds and budget as `results/study/`, then `analyse_traces.py` over the 60 cells. The reproduction is exact — every checkpoint hash matches the versioned set and every held-out rate matches the published one to 3.3e-05 — which is what lets a negative result about the traces be a statement about the study's own networks |
| Readout re-measurement, 60 cells × 3 temperatures | **4 h 43 min** | `scripts/run_readout_campaign.sh`: 27,000 held-out matches (150 per cell), two processes at a time — one per tree. The arms are not equal: 246 → 485 → 684 s per cell, i.e. 1.6 → 3.2 → 4.6 s per match, and the cause is the action rule's own compute rather than longer play — the sampled matches are *shorter* (median duration 30.3 → 18.4 s, mean survival 21.6 → 13.9 s). Evaluation only, and the T = 0 arm is the control that makes the other two interpretable |
| Regime replay, 20 replicates × 150 held-out matches | **14.9 min** | `scripts/run_regime_split.sh`: 3,000 matches (9,000 team-rows), two processes — one per tree (805.5 s and 890.9 s), evaluation only, over the versioned replicates. Closes audit item 8: the per-match durations the study never recorded, replayed under its own protocol; the current runtime reproduces the published win rates on 52 of 60 cells, worst 2/150 ([§ 8.1](08_results.md#two-regimes)). The readout campaign's T = 0 arm played the same 9,000 team-match volume in 82 min — on a machine five round-robin runs were already loaded onto, before the A-46 host-dispatch cut — so the two figures are not a controlled pair |

Before batching, measured per-replicate throughput under full 20-way load was 82.6–92.9 steps/s against the
scalar path's 368 steps/s in isolation, so the published study's twenty replicates cost 36.2 CPU-hours.
Those are historical study measurements; concurrent throughput has not been measured again after batching.
The original profile attributed the load to shared memory bandwidth in the NumPy hot loop, with 32 cores
for 20 single-threaded processes providing headroom.

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

The reproduction records in this section document the earlier PyTorch implementation. The current JAX
runtime is seeded and smoke-tested, and same-seed training windows have now been byte-compared — the
2026-10-20 host-overhead change was verified by running the pre- and post-change sources over identical
8,000-step windows and finding equal match boundaries, winners and final parameter digests in both
experiment trees ([§ 9.5](#host-dispatch-overhead)). What that comparison also found is a limit the runtime
has independently of that change: under concurrent load the PPO update is not bit-deterministic
([A-47](11_code_audit.md#the-jax-training-update-is-not-bit-deterministic-under-concurrent-load)), so
byte-identity is a property of *solo* runs on this instrument and may not be promised for a concurrent JAX
campaign. What [A-45](11_code_audit.md#the-runtime-swap-orphaned-the-committed-checkpoint-archive)
settled is the part that needed no new study: every committed policy now exists twice, a `.pt` for the
runtime that produced the tables and a byte-verified `.npz` this one can load, so § 8.10 to § 8.13 are
replayable again. A-45 measured what that means in practice: replaying one committed `no_dr` block
through the converted archive reproduced all 1,800 team-matches with **no difference in any recorded
column**, while the forward pass itself differs by up to 4e-07 on logits (`scripts/parity_probe.py` is the
instrument behind that figure) -- enough that exact
reproduction may be claimed for what was sampled and may not be claimed as a property of the runtime.

**Reproducible bit-for-bit from the repository:** every table in [§ Results](08_results.md), because all
of them are recomputed from committed CSV/JSON files with the commands quoted in each section.

**Historical PyTorch replay, verified before the port:** `scripts/train_rl.py --seed S`, in both trees.
The old implementation was checked with two 4,000-step runs at seed 42 and one at seed 43, each into its own
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
`src/marl_arena/config.py` then seeded `random`, NumPy's global RNG and Torch, and `RolloutBuffer` owned an
explicit `np.random.Generator`. The current code seeds its action generator per controller and initializes
JAX networks with explicit keys; byte-identical full-run replay has not yet been checked for this runtime.

In the historical implementation, experiment 2 was reproducible **only with `--no-mlflow`**: an MLflow
run id and wall-clock timestamps were written into the store, so the log was not byte-stable even when
the checkpoints were.

`ARENA_DATA_DIR` isolates each run's artefacts, so a replicate cannot silently warm-start from another's
checkpoints, and `run_experiment.py` passes `load_checkpoints=False`.

**Not reproducible:**

1. **The two historical 100 k-step runs.** They were produced before the control-law fix, before the
   observation-encoding change, and before `seed_all` existed. Their artefacts are versioned, so every
   number in [§ 8.2](08_results.md#historical-single-seed-runs) can be *recomputed* from them, but the run
   itself cannot be reproduced.
2. **Any run made through MLflow.** The store records wall-clock timestamps and generates run ids, so its
   run logs are not byte-stable.
3. **No code version is recorded inside the artefacts.** Checkpoints store `{paradigm, actor, critic}` and
   nothing else; `data/PROVENANCE.json` names the promoted replicate and the study, and Git supplies the
   commit, but a checkpoint file does not carry the hash that produced it.
4. **The historical measured lock differs from current requirements.** The published measurements were
   made on NumPy 1.26.4 / Torch 2.5.1, which is what `requirements-measured.lock.txt` (both trees, the same
   34 pins) records. Current requirements pin NumPy 2.2.6 and JAX 0.4.38. The historical lock does not fix
   CUDA/CPU selection, which stays a `.env` value, and [A-31](11_code_audit.md#the-pinned-environment-does-not-reproduce-the-recorded-runs)
   measured that swapping to the declared NumPy changes every saved checkpoint.
5. **The JAX port has no full-run replay result yet.** Its action generator and model initialisation are
   seeded, but changes in compiler, hardware and update batching have not been compared across complete
   training runs.

**Derived artefacts are recomputable, and that was checked rather than assumed.** Three of the committed
outputs are pure functions of other committed files, and each was re-derived and byte-compared:

| Artefact | Re-derived from | Command | Result |
|---|---|---|---|
| `results/cross_arena/blocks10.csv` | the 50 run-directory CSVs (git-ignored scratch) | `scripts/combine_cross_arena.py results/cross_arena_blocks/*` | identical SHA-256 (`7fa3029f8410…`) over all 90,000 team-match rows |
| `results/cross_arena/blocks10.json` | `blocks10.csv` | `scripts/report_cross_arena.py --in results/cross_arena/blocks10.csv` | deterministic, no RNG |
| `results/cross_arena/blocks6.csv` | the first 30 of those run-directory CSVs | `scripts/combine_cross_arena.py` over the six-block subset | identical SHA-256 (`06b23f4d1419…`) |
| `results/cross_arena/blocks6.json` | `blocks6.csv` | `scripts/report_cross_arena.py --in results/cross_arena/blocks6.csv` | deterministic, no RNG |
| `results/cross_arena/v2_blocks.json`, `no_dr_blocks.json`, `no_dr_blocks3.json` | their own pooled CSVs | the same report over `results/cross_arena_v2_blocks/*` and `results/cross_arena_campaign/*` | deterministic; CI re-derives every pooled JSON in `results/cross_arena/` in a loop over `*blocks*.csv`, so a new campaign — or a superseded pool kept for a correction, which is what `no_dr_blocks3.*` is — is covered without editing the workflow |
| `results/study/learning_curves.png` | `data/runs/*/evaluation/run_summary.json` | `scripts/plot_study.py --runs-dir data/runs` | identical SHA-256 (`1915d48d…`) |
| `results/absolute_six/summary.json` | `results/absolute_six/matches.csv` | `scripts/run_absolute_eval.py --report-only --out results/absolute_six` | no matches replayed |
| `results/absolute_ten/summary.json` | `results/absolute_ten/matches.csv` | `scripts/run_absolute_eval.py --report-only --out results/absolute_ten` | no matches replayed; this is the headline artefact, and CI re-derives every `results/absolute*` directory by glob rather than by a list |

Two of these four are re-derived on every push by CI, on Linux, from the committed inputs — the round-robin
reports against their CSVs, and the artefact inventory against `git ls-files`
([§ 9.4](#continuous-integration)). That is the difference between having checked once and being unable to
drift.

The last row is the point of `--report-only`: the headline aggregation of chapter 8's absolute-strength
section can be re-derived from the raw per-match record in seconds, on a machine with no GPU and no
inclination to replay 4,500 matches.

## 9.7 Artefact inventory

Every versioned experiment artefact (791 files), with SHA-256 for integrity checking. Regenerate this section with `python scripts/report_inventory.py`; it is built from `git ls-files`, so it cannot drift from what is actually committed, and each hash is of the blob Git stores rather than of the file on disk — a text artefact checked out with CRLF on Windows hashes differently from the same committed bytes on Linux, and a reader on any platform should be able to reproduce this column.

**Promoted replicate.** The artefacts that match the current code. Chosen by `promote_run.py --seed median`, which is the replicate with the least L1 distance to the per-arm study means; `PROVENANCE.json` records the choice and the full run summary behind it.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/data/PROVENANCE.json` | 6,372 | `dd239c8294b1cbdbb42d130026627e4b1f2d5e85b1e8c9c55c6a875b05592bfb` |
| `ctde_arena/data/checkpoints/team_1_ctde-comm.npz` | 289,236 | `d4fb91bf64f3954ebb2534dd636c8cb9de748c8bad16943b686da2920b565c95` |
| `ctde_arena/data/checkpoints/team_1_ctde-comm.pt` | 313,778 | `cb3fee053d68a7a3cc6e4cadbd3fdfc05a356e5a9542babad143a1aa2eb85ff1` |
| `ctde_arena/data/checkpoints/team_2_ctde-vd.npz` | 199,531 | `cc96d2a5bf8b947b64f3ff760ef40ec2405897ff0694e87fdf925ec859b512e7` |
| `ctde_arena/data/checkpoints/team_2_ctde-vd.pt` | 217,330 | `bf261e17c010ee0bb8f69b9d7e9ca1c4214cc9cfe0d8775038379dd71547d5f5` |
| `ctde_arena/data/checkpoints/team_3_ctde-cac.npz` | 280,666 | `8ec889b87fe509c6c139a93a96f97027e2fee3bba42aeb0288c57d0e3ceab8ef` |
| `ctde_arena/data/checkpoints/team_3_ctde-cac.pt` | 304,400 | `451b14b4c8aae906c3e7b135f9e814426b268354e8eb222d420b42dcb981cccb` |
| `ctde_arena/data/exports/comparative_dashboard.png` | 269,089 | `474621e93c7817c93abedc1404cd4461b70d7258609706051e746e867438886e` |
| `ctde_arena/data/metrics/agent_match_metrics.csv` | 18,418 | `81df004cae74a1217278898ee5dceb8eaa320d4a40508c4e103b5a6814be4281` |
| `ctde_arena/data/metrics/summary.json` | 942 | `7b87782481a08ea4c142d97b44714de18cb32d9aae8ade88f519e8e7a11c18e0` |
| `ctde_arena/data/metrics/team_match_metrics.csv` | 10,988 | `13d568cbd67b0a5a926f56e743b00426e764dc8a9f721d13b4b4c1c4cfdf449a` |
| `ctde_arena/data/metrics/trajectory_metrics.csv` | 1,184,497 | `8549a5746b2d34cb21e013416d420260d88a1074d0bd5e2ddd2280d38181275d` |
| `data/PROVENANCE.json` | 6,298 | `5174eaa34a3949d669145c0088f2bdd9106158f93e292d68d626bc8fe50f2727` |
| `data/checkpoints/team_1_cte.npz` | 297,965 | `2d3ee1bd6a0f04270ac5ad16e3f292c43e241ba0c8b077391fd7bfc8ade3e8ed` |
| `data/checkpoints/team_1_cte.pt` | 323,170 | `c978273409e60e45b9555d763d65c9ff13ba7774ffaf332ac4eccd11ffe59111` |
| `data/checkpoints/team_2_dte.npz` | 133,746 | `670ecaa077da12d0b0e24b756ef8dc981cca93f0a774134e08d7cefde3cdc347` |
| `data/checkpoints/team_2_dte.pt` | 145,762 | `93fc387fbfbd1cb935a271672954675230e1326565ae2680333a4ded4e4e3897` |
| `data/checkpoints/team_3_ctde.npz` | 280,383 | `400520e13e447d3a7d71ea72e9987fea886d90d32164b105e65723638a928447` |
| `data/checkpoints/team_3_ctde.pt` | 304,248 | `5a5273f3c73942e8fb0964bbb6a848734c58fee9cf0b52b4986f84cf8f2a4fae` |
| `data/exports/comparative_dashboard.png` | 272,710 | `d3daf6ac475005675a5411e39173c8ff53d5349ac39fb6c1655e9193753ab783` |
| `data/metrics/agent_match_metrics.csv` | 16,977 | `3495d7a1da60106cdeb60e3e8f7d50cc76911dc4f9e3bbd4289cc47e0a8aa2ba` |
| `data/metrics/summary.json` | 926 | `938fc507ce241286086fefce9f1628b3c05cd9438cffd5994607e08abb334c99` |
| `data/metrics/team_match_metrics.csv` | 10,809 | `25a31953f7198bd676752f18d4aa7dca6b305307eed05866fcfdfc99d2649b2f` |
| `data/metrics/trajectory_metrics.csv` | 1,343,793 | `e3539dd0a2667159b63559869c9ca391d670406850e2af69c7e5102a3f6db59f` |

**Study aggregates.** Recomputed by `report_study.py`, `plot_study.py` and `report_cross_arena.py` from the per-replicate and per-match summaries. The round-robin rows are the pooled tables chapter 8 quotes: `blocks10.csv` and `blocks10.json` (ten checkpoint sets x five match streams, the current result -- every replicate either experiment trained), `blocks6.csv` (six sets, the peer pool of the arena-v2 control and of the headline two-scale agreement, kept at the count those comparisons are matched at), `blocks.csv` (the first three sets, kept because two of the three agreement statistics in [§ 8.11](08_results.md#absolute-strength) are computed against its `blocks.json`), the v2 and capacity pools, and the `no_dr` pools at both replication counts -- `no_dr_blocks.csv` at ten sets is the current result, `no_dr_blocks3.csv` is the three-set pool it replaced, kept because [§ 8.13](08_results.md#no-dr-pool) quotes its numbers as the correction.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/cross_arena/blocks.csv` | 2,389,102 | `8891db57a270a5b32c187e3d086171736ce113c0cf772172f401ec34e5e04ca7` |
| `results/cross_arena/blocks.json` | 16,864 | `e624dbe87eb505c038685ffe915167fa4d8702b9c506a1ad51b298c13777525c` |
| `results/cross_arena/blocks10.csv` | 7,960,390 | `a98278cf226a0c9f239c574f1970a5aea5571ab97d007d444f575fef158fd747` |
| `results/cross_arena/blocks10.json` | 21,745 | `1a86501c13ea5f38d76d2e77e3006e2f6ff8219acd0345689a756076991a0bd6` |
| `results/cross_arena/blocks6.csv` | 4,772,724 | `fec107ccfba9b8ad3f681659f1161659523878804c39be399987883891e2c89f` |
| `results/cross_arena/blocks6.json` | 18,588 | `4b4b5e32d1919089d7567355dc27715ba2715d6a2f266344e9ecaf4abf244537` |
| `results/cross_arena/narrow_64_blocks.csv` | 2,653,460 | `49e3add91d4582ba98c2c29d5e2ab856911ce2de95a273a2c41e63b8b3aed447` |
| `results/cross_arena/narrow_64_blocks.json` | 15,877 | `a6ea0ba6ac6cd8eb21ee069f433e3f52a3e3f886b802b3232c77c838afe75e69` |
| `results/cross_arena/no_dr_blocks.csv` | 8,487,835 | `40cad9bc389d7202759155b78a5de661a2b9233c7dee0a7b717618b019190e6c` |
| `results/cross_arena/no_dr_blocks.json` | 36,003 | `706d39ef0ba9a296ef2f7c3f0899a8f956f2398bf529c325f141d529a1739501` |
| `results/cross_arena/no_dr_blocks3.csv` | 2,533,758 | `e9689e27298e9b50d84b76bd7efc0760f7983aa00980702ae7771f6c005d0e48` |
| `results/cross_arena/no_dr_blocks3.json` | 18,021 | `357982487cd87184e0e834ba52f86d3f79460ddb85b308c081de2a5f2903145a` |
| `results/cross_arena/v2_blocks.csv` | 8,172,817 | `49aad95f0f96c2343e45d590174a5ac1705c32114302a116bda777f2366691d5` |
| `results/cross_arena/v2_blocks.json` | 30,490 | `d521631979c371743d84532e045c0274289b8b7f8db7a31861680760030ba485` |
| `results/cross_arena/wide_256_blocks.csv` | 2,626,611 | `a4cf32ed064c04e5789222ec13dcf55420a5289c2f0b38f4a2936da90f794375` |
| `results/cross_arena/wide_256_blocks.json` | 16,659 | `9e25a582e234ebbe41ff12b1025d1f8af7b068acf1845ab047f0ad6b0eb1a7c5` |
| `results/study/analysis.json` | 4,226 | `0761e5fa08f9be1cbee21f515b28eb17ad48e65b7207868bc161ce933f87527b` |
| `results/study/coupled_tests.json` | 4,791 | `89de1c028157dfd189f81a23a8cfe89f23166f94cc84edab8c2a0bf7db99a686` |
| `results/study/eval_match_regimes.csv` | 589,145 | `235eef5a14766b04d1abc739602caa1ae9b145761393961689aac03805d29ff7` |
| `results/study/learning_curves.png` | 137,532 | `1915d48d117e8a96730b3659e0259e5ad6446c0fb7103d0642a6a5e7bd601cbe` |
| `results/study/per_seed_metrics.csv` | 5,226 | `a69098a4e88eef37c1291f73b1cb2526f0d31a98c0c2c6f0da27af0263b30d17` |
| `results/study/regime_split.json` | 5,777 | `ef7cb800f4a2f9aea949550e593eb443c60eacd74175b0ab27273bff4ba73638` |
| `results/study/regime_summary.json` | 2,666 | `6aacb485dd644a4b0a35046f7f345f4c8463b2d4285bf12edd444a8844862608` |

**The v3 campaign.** Everything `scripts/run_v3_campaign.sh` measured: the study re-trained on the JAX runtime (`study_jax/`) and with every docs/10 remedy on (`study_v3/`), each with `coupled_tests.json` ([§ 8.14](08_results.md#v3)); the capacity arms at ten seeds on the same runtime ([§ 8.13](08_results.md#capacity-ten)); the `no_dr` blocks against the random pool ([§ 8.13](08_results.md#no-dr-absolute)); the study's blocks against the scripted opponent ([§ 8.11](08_results.md#scripted-opponents)); and the uniform-random baseline in the symmetric arena ([§ 8.9](08_results.md#symmetric-baseline)). The training runs behind the four studies are git-ignored scratch under `data/runs_*/`.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/ablation_jax_narrow_64/analysis.json` | 4,223 | `8a612e7f0514d58507cbbb82402362b5294950a77e3eec675e05df415f0fbf36` |
| `results/ablation_jax_narrow_64/per_seed_metrics.csv` | 5,189 | `5bd2a702d0071d2b04dcb7210a44f5b59c3910f875c266bc16b7b019f830bf56` |
| `results/ablation_jax_narrow_64/report.md` | 7,963 | `20fcd08d89690f2c42a4e99e2b9fa0f5679da67466cbe1df15f76e23a776cb6c` |
| `results/ablation_jax_wide_256/analysis.json` | 4,215 | `c24fb9c12dad0cee444e416a65757cd6795398480178676c54b06ef633f9bd90` |
| `results/ablation_jax_wide_256/per_seed_metrics.csv` | 5,250 | `de3c00ff0bff18a4b644e087b6dc47c055308067559ca450d9e50b98f043d07a` |
| `results/ablation_jax_wide_256/report.md` | 7,966 | `971cddf0f11c8b168ddf902227825246954156c85c3f4df67c5c7960c885f10f` |
| `results/absolute_no_dr/matches.csv` | 261,884 | `9cb2d72daf11d52b0b8a394fa1a7ad0e6245124a5e40b9453ecacc5c4eb2ab11` |
| `results/absolute_no_dr/summary.json` | 45,242 | `3383b097f3a9fe378a6ccd89c0ed626cd71e360280826f66d8772979cf77b613` |
| `results/baseline_symmetric/random_baseline_uniform.json` | 1,192 | `1f72f99e735f7cbfdbe20066c72d5546588b402616b8acf24ccf8a0ac73ea0ea` |
| `results/scripted_ten/matches.csv` | 233,683 | `cc0ee953ae4f70e68fe56392e7a5bf404d4726b0142b1d92e664cd86f3039f6c` |
| `results/scripted_ten/summary.json` | 43,807 | `2a57eb6daa5cbc9d516359b693d11d7d3849c4fa89a8573549a7de34bd36457a` |
| `results/study_jax/analysis.json` | 4,197 | `77da184e99bf24fa69369883cdfd82f1b7b6943af27a3ef768e0fdc0021ca644` |
| `results/study_jax/coupled_tests.json` | 4,768 | `636b4ed6156fcf30574ccbe8e38eb343f2c658c4ec3b5861294c546051a17318` |
| `results/study_jax/per_seed_metrics.csv` | 5,204 | `bc4ca761c78881a7bb9738b78f97c7e2ccd6d85ae9f783cd896e68af24bfb210` |
| `results/study_jax/report.md` | 7,984 | `a9613e2e43ed4a5d25650ce9466e4526b25c96cecef756ab059d9b3b771eda06` |
| `results/study_v3/analysis.json` | 4,220 | `e4e82da9bb2f510b19a83fdcac50d622513f2b4b08528fcd5153b5ba7be914c1` |
| `results/study_v3/coupled_tests.json` | 4,781 | `ab532343e3aff5164886b636cfa231bdcbe166e8dc0ee8fb92822bc353772155` |
| `results/study_v3/per_seed_metrics.csv` | 5,258 | `8427d1c7ed08b746a2f3a645451d1092c788bc11a98cfe34ab15fe8d61e251f5` |
| `results/study_v3/report.md` | 7,987 | `f4fbc1ac1bf0cc89571a41ffb5c47df815947e75bd8006f6f1a40c45a3a90055` |

**docs/11 item 16.** What `scripts/run_item16_campaign.sh` and `scripts/run_remedy_readout.sh` measured: the v3 checkpoints on the round-robin (`cross_arena/v3_blocks.*`, its fifty run directories under `cross_arena_v3_blocks/`), the random and the scripted scale ([§ 8.14](08_results.md#v3-pools)); the two slot remedies one key at a time on the JAX runtime; and the same checkpoints re-read with the key flipped in the evaluation only (`readout_remedies/`, [§ 8.14](08_results.md#which-remedy)).

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/ablation_jax_spawn_cycle/analysis.json` | 4,225 | `d3f01749fb4cf8b8e817c27c41746bb8d18e82d1af5d3b5e7ce48fc3736f19b2` |
| `results/ablation_jax_spawn_cycle/coupled_tests.json` | 4,788 | `f1ef6e54dd0fdef845a61bbeec8023f8f020e68dc7a467b6b0d3905d701648dd` |
| `results/ablation_jax_spawn_cycle/per_seed_metrics.csv` | 5,256 | `4db6025f88cf9a6856f4a70cfffe9b51b1f09993077e67529c1c63b3b444d08e` |
| `results/ablation_jax_spawn_cycle/report.md` | 8,076 | `82163504b88bf04c9953559ef14388aac023ad3c8c6e580c402be48a0a390da9` |
| `results/ablation_jax_symmetric/analysis.json` | 4,208 | `4874f5e2d17489cd7599f55d8f6eba926dcf209aac3d8500aee3ad120e0414d7` |
| `results/ablation_jax_symmetric/coupled_tests.json` | 4,781 | `e64c96aae94939cdf1d83d4cb0dca9a7dc53f4d80e6745b559186947b5c68a73` |
| `results/ablation_jax_symmetric/per_seed_metrics.csv` | 5,269 | `3df541d5fa5d18fb726203ccb2feedb57d161bd0339a6aaf5024cf110a416063` |
| `results/ablation_jax_symmetric/report.md` | 8,051 | `19008c26a5b2d6c443debf858c6a03af203bd851656c09ef2e572d45a9303db7` |
| `results/absolute_v3/matches.csv` | 246,479 | `600d454e4d0ab3cfa91a41f62a3e2793c30788d2a1ac951f4059ed4474a96f31` |
| `results/absolute_v3/summary.json` | 44,992 | `9616adc881f18b6a7d3f98fef82336fb0f2fe3e8e0bfa937319ade69c7fab7d4` |
| `results/cross_arena/v3_blocks.csv` | 8,195,529 | `5ccd6cee672b914d67c67c3ee705caa1c211c8f75996a946dc2629f6e67af277` |
| `results/cross_arena/v3_blocks.json` | 39,610 | `668008628411b6028fe8bcbc2715a72fa3bb69b998f18b884225caeb09ee9e46` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h1/matches.csv` | 114,717 | `08ac136a7c6ca5f80e000a0ff8a88d37b93b39135ba41f4e040126ee32aaa20f` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h1/summary.json` | 3,415 | `f8c774868b1aa92e09e0f321f26c32a8ad2548df4f425d9f61ea577b78a75b5c` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h2/matches.csv` | 114,840 | `7be10ca152335c8520f7035f0f2f25ed13c535da50685ff1bc4aabd37818e630` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h2/summary.json` | 3,410 | `3b3f5fb6810c83ba9206101e3011fc14c0cf1119fe5d6a6c1122b3ae4817c4a4` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h3/matches.csv` | 114,388 | `3b0f54933ce0c90407a444c9dd54418508a1de02ea721b86dfa43adf64642d05` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h3/summary.json` | 3,424 | `627082213534272f6a3f4dafd53c7b1bffdf694d4c8025e9e7b9a0d7b8cf0e52` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h4/matches.csv` | 114,964 | `91f9053971e44050ca0535612d6e0dadec1ce48ab48c1eeda6f523fc4b64a409` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h4/summary.json` | 3,414 | `314c272b98030f7b456887e6a93c654e5bdfef5a5c26d956fe8889f3bad70f20` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h5/matches.csv` | 114,946 | `32c8a8c92fbe44304641ef2995a64aac339a4b0a699a718fec8655f87d282acf` |
| `results/cross_arena_v3_blocks/v3__e1s10_e2s5_h5/summary.json` | 3,409 | `ef5b32a451b657a41c07b22ce8b2195d497f862da461a0bed8f45e1e577283e5` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h1/matches.csv` | 114,673 | `4d6ec3f5d75563d4069a224682871edead3882f96859dc7bfe7dc96a4721925e` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h1/summary.json` | 3,416 | `47670ca321ab8f770707f6d7f1328bef93021478a2e6a1a2ad5afc867f2260f4` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h2/matches.csv` | 114,819 | `b46c0e58b05bdf857aade4dae5af9d0e3b5421553f8e8b9ea2619f97413d479a` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h2/summary.json` | 3,409 | `0f0155326ffd1acf28f60f09f7085a682bf8423dc6133b6228c9fb88806b0d8f` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h3/matches.csv` | 114,497 | `939e0c031025a2f4b62dec9f7c7070fd38ed1614103b091127e5586666dde839` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h3/summary.json` | 3,418 | `0a73913db0b07d58d2dcf73f17735bdb918c5b93c37f44ffff4b5536b6d393ab` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h4/matches.csv` | 114,763 | `28f15827c2d4bb66db9c0d16a73ca25c34c2c7a0ad123781204827c84e8da901` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h4/summary.json` | 3,412 | `54a5c52340d684c954b728be360d8bc2ea549f0365e88593e1b27dd858530a25` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h5/matches.csv` | 114,636 | `4f0181873869d96c05aa187b3fb8d738945c27564b6d615bf247d51bc3c2623c` |
| `results/cross_arena_v3_blocks/v3__e1s1_e2s7_h5/summary.json` | 3,414 | `794d77fd4112475220b685fdb01d7ae0c471015f397848d4921be2e49bdf4051` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h1/matches.csv` | 114,919 | `1ccd6d047c5edbc67f162921e943e7cfaca34cea27bc4d3983fb9374a091e1d6` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h1/summary.json` | 3,404 | `3e68acd5296f188e99f9e9cc0a297b92e7bbabc689cf6ff3e02f5831a0360a25` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h2/matches.csv` | 115,097 | `e0c93c4053d8062a3d9de611db3d7fd89da3e2f1898642f145b0becb823884dc` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h2/summary.json` | 3,416 | `1d929541d393100415a008f9dfe33c4a7cf10660b369a18a55eb5ca6e919e8ee` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h3/matches.csv` | 115,253 | `784a684c711cd4111dd97cc8554d4068ec432ec21e9c3203a679a574033fe14b` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h3/summary.json` | 3,414 | `675d65e2d1e3abbc5f5c6169d7c330705dad064ddad7421a62a53025043b3b27` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h4/matches.csv` | 114,573 | `f8ddee6636387712bb183b27e995f2a46d70423b93db71120df56e6e78f8876f` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h4/summary.json` | 3,414 | `08be2bc747e58b5b63ac92b590d89a321bc14f0480a3b589bfb46b75742332e2` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h5/matches.csv` | 115,018 | `25f1419352df5c8fa91c05293630ed37436838b38002c0e919a78c71604bbce1` |
| `results/cross_arena_v3_blocks/v3__e1s2_e2s1_h5/summary.json` | 3,412 | `5250f2276d85e5cce5632cd4b43fe6734f0c4b1597c023b8dbcec65d165d29e4` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h1/matches.csv` | 114,648 | `d597472a36b8f66c73edfded5a7382dde5ce9e4035961b9b1706a85bef5b6f6e` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h1/summary.json` | 3,412 | `78ad9f35ba2fd047503851b9d2e10b4c09a4371fca73778065aaf9a95e84f394` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h2/matches.csv` | 114,683 | `b565c2dc06cb098c5bfd66143f890d257ac99dddffa7e3379842d2bfe6844781` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h2/summary.json` | 3,406 | `ad8a178e5d6474848b87cb8482d92a978c80aa0adace92e96b509a373e154d06` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h3/matches.csv` | 114,947 | `c8905dd1bdf5225890f81704776ec2a76caa924fb8cd5cbee0a0d8a463dacea7` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h3/summary.json` | 3,412 | `e65fb9fa8a1ba8897b44fbfc1826b75c0469955c5bd1a9a8c948b165a0e1c35b` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h4/matches.csv` | 114,837 | `0f5a8d3356e24ccf5f7b5aac09aa691c3604e601f5e7860e764e8f292f8fe9b9` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h4/summary.json` | 3,409 | `07c53137a3377f060ecc3a4ee386bddacee91fb27abc68d69cc5bbf138d6fce1` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h5/matches.csv` | 114,660 | `a6152009a1ac47d14ce86ea8153a3eae91381846de5fc0f4f9765e0df650f47e` |
| `results/cross_arena_v3_blocks/v3__e1s3_e2s3_h5/summary.json` | 3,420 | `866fc9668173c78a7bf805eeef0aec074b58d976b86dcf7cadf1ed3061f5ae3a` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h1/matches.csv` | 116,032 | `665e8be12e47fcc5e7db7853e41fddf33bca73be29b437c8ae055caa1cafc35a` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h1/summary.json` | 3,415 | `63e406e73a0a3c900dab7f1513dec839dffab40e9ab853e5d500264a0e80d09d` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h2/matches.csv` | 116,013 | `ef22a4e4ad5facd628aff724ea9e509c76b0730493bcf7474aa8dd33401ba4e7` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h2/summary.json` | 3,411 | `d1623b0b55c3e03dcaeb10aa173960fa25e3af03960c29403d9da0b836cd3de9` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h3/matches.csv` | 115,782 | `afb14519682c15b622391890e390174af5da336c7d8bd3f69fa77ecfd7673cee` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h3/summary.json` | 3,417 | `8c09ee60f789d16d9b6d15b81189e812aaf490eec16fb0c631a7827c039518c9` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h4/matches.csv` | 116,027 | `1074ec3ff79835f608ca90bea62d30b13c04134738c1862ee8e31f918ee65007` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h4/summary.json` | 3,410 | `7f8020015c7544631948024682226c34871db56f2233619732c46486bcb038ee` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h5/matches.csv` | 115,766 | `d79d633f02765d1736f5890dd75377dc36a4dac982d68017b2f9940bb7f5ce43` |
| `results/cross_arena_v3_blocks/v3__e1s4_e2s4_h5/summary.json` | 3,418 | `c087c21029a80fe07adfc581d1b0e3e38f59dc9c79b5f83b69e56536b91cd144` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h1/matches.csv` | 114,976 | `bb723031a91b79e8bd8924eb0fac97560b4ed67b9f865432722e6072d837cd36` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h1/summary.json` | 3,413 | `c7455c296de9ee7877df24a003119bc1e6a83cf8cd9b7786424a6429aa3e3183` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h2/matches.csv` | 115,061 | `fa52e89c9dbcb25c8a8e7f7382afe1d7dec0836892696d72fd4fbb9cbb1856af` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h2/summary.json` | 3,416 | `457a59e9ca4e0a7995ccd8e47d35b1d1055b586812e9f0b7842b2b34a1631fff` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h3/matches.csv` | 115,049 | `677dd31724b2aebfb7a830c56af3f6c1c5e8c22fb55f96ca817062070820a323` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h3/summary.json` | 3,406 | `9c8f27649676f186ae72b915d8983f04ba99e2084d58600c915041386fbb5d4b` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h4/matches.csv` | 115,210 | `41b586e2e276e3447f9f3f28d889ec1e8f8c7e6e28a2dab950a8df8161e66f21` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h4/summary.json` | 3,406 | `0e0bcc6e9be04acb6ab746f40f58d01a7c05ce406cbf76b05f0a663bd7408a5b` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h5/matches.csv` | 115,102 | `7a523b537bcd6bc63390873865cb767a442b65b01d3533f7ff7564d5433dcfb0` |
| `results/cross_arena_v3_blocks/v3__e1s5_e2s6_h5/summary.json` | 3,415 | `7cc8c53653ca11399588c8d06d554e88885e96bc9de87ce449f17c1e161ee1ed` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h1/matches.csv` | 115,228 | `44521cda647b2a887fe55de8c8b08017387e60fbd34386041d63487ea55c0e37` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h1/summary.json` | 3,410 | `a14b8f42c7fb32ae397a6547f78f5f35f547164b7707cbe2f7cb47413d84a482` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h2/matches.csv` | 114,987 | `8f915018357b6362844e055b5f4f8775a967e7fcc4533dc2ed7ab169271b7b3d` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h2/summary.json` | 3,404 | `8b5d202f450bc9586125870eb7eebbcc97c66e11e6e75d4023d7a45fc0fb35c4` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h3/matches.csv` | 114,937 | `05a1de67e6af5e4a63b13bb02c3cc3966b6519f968c703ea69be7f698420f61d` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h3/summary.json` | 3,414 | `f525b6808ad5db890688c890cfdc4d9dca6d7a1573739bfc4d2aca55dfc8f935` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h4/matches.csv` | 114,900 | `b9fb87d56803ae68b24ec2216487e8a3da493785dccc381d12ad8a9ec976025b` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h4/summary.json` | 3,414 | `f8954502a56dfa7b0af21b9bc6a17cd162c9187c6a239a58b5d1fb27f147e4d7` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h5/matches.csv` | 114,981 | `ff5c39f231d7a96d6ad865ef0db48df9bd46108337f43befe28310c4322ce711` |
| `results/cross_arena_v3_blocks/v3__e1s6_e2s8_h5/summary.json` | 3,405 | `24f3a3f5161dbf16153e375ff7462bcd694166400cae363c99d96066f142da54` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h1/matches.csv` | 115,625 | `f89a5f151638176a424691f836388c8fe275702acd1410937c6c8cd30ce59946` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h1/summary.json` | 3,416 | `57b731dde38e60c19d0260500f5271e5e9b002008acadd6c4ac4490d621d5006` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h2/matches.csv` | 115,400 | `2fc1fdd6540e4b5d59632ce66ce0f354f3af3a5a39c937aa5a6c118a0d2e9ce0` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h2/summary.json` | 3,403 | `70f498e7a336f66eafa3e3e862658bc81ab701e4fb6c2621b8bc80f1de804f47` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h3/matches.csv` | 115,131 | `0d911e698db32707d2004453ef77af43a4dfdcd18bde75096d32dfc3988f52a1` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h3/summary.json` | 3,420 | `a6bec9ef5ea1f93a9cbb53621232afa0c77736e3ef91865706c105af005208e2` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h4/matches.csv` | 115,364 | `d2ccea945904fd950b8f54de33a169cb3dc92d95073b84f8e8d842f9f47f533b` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h4/summary.json` | 3,413 | `610df9ad3683c57bcc96677a88314ef2b11fd74138a3508bd10cf6a8935f4281` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h5/matches.csv` | 115,112 | `d224f4c7aec4022c4f09a2821f71ccb3c813643fbdbb1aee6fdeeac195a08f08` |
| `results/cross_arena_v3_blocks/v3__e1s7_e2s9_h5/summary.json` | 3,415 | `569615930e5199924de8b1623af6a345f8a24e5aa15819ba5fdc8d12931d69cb` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h1/matches.csv` | 115,042 | `e78cf70c2a8d90de1790d52e84bec479bec20f6d2edb9f27a9d702926729e986` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h1/summary.json` | 3,416 | `8b2cd776fc9e9c94d2cf2afede7ae23159ce1479f000199e144c243c094543ea` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h2/matches.csv` | 115,009 | `6645748b6bfb437417c460176852a07b92bac3c8817d9e16f3faa5cf5cf80759` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h2/summary.json` | 3,409 | `7dd2bc38bef5e2f0f4bc18c643e73abb5ec48c879c0fda5a39fde415e4256119` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h3/matches.csv` | 114,929 | `8ed85731e2749daa59fe07358a5aa352eff959d889ef31f94bfaa884c29fb27b` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h3/summary.json` | 3,415 | `e9f4840080ba24d8781211ad925e327b8f6fbabb9f5a290754e12520bdf355b3` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h4/matches.csv` | 115,432 | `5cabd0452f01ede8c69bb81ce01a59488be10b91dd2a2e3066754f3619ec4816` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h4/summary.json` | 3,418 | `12cb6ae4734f86004716b00cff508ff5581252d7cb79b356b2c68d2341810e22` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h5/matches.csv` | 114,918 | `001b1c97dcff951e64175a0f854f1713a16d22a929fab15804414b54701c5d42` |
| `results/cross_arena_v3_blocks/v3__e1s8_e2s10_h5/summary.json` | 3,417 | `48e0aff66f979d72c37c4f08d41ed5b07b3019f2d6b88422982cfd7ca063e3ab` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h1/matches.csv` | 115,002 | `1f55e850f2a0abb8286fe24e591986fcd6159c2796467b4f61b85f262a0186f4` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h1/summary.json` | 3,416 | `35960b4ae17e2fdc907b2026110e4d282491ac7713527ab3ed35233cf4425bbf` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h2/matches.csv` | 114,963 | `400d8fd44d83e9c5c45864baca6317e6b4497840b1c7111a7a64cccb01ae1911` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h2/summary.json` | 3,415 | `f0234e2fca8871d539269ca6922af7ba3c346fe78996965de70889609a635f4a` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h3/matches.csv` | 115,451 | `33609474b694355a0d045a9bfe9fa1c5047c8016ad4bd52cd20541b1e1e15159` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h3/summary.json` | 3,410 | `dab48fa4b1c7047a09fcc2747032fb04e4ebd317a4f9ef8fca47f7859e17ffdb` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h4/matches.csv` | 115,031 | `1e1c35b16934e9cd4d55c5253af70098b10b6440899cf92fc4337d7506f7a7f1` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h4/summary.json` | 3,415 | `5cab489c42293b1767870058eff9eccb54bf39581c77dde7876a5e9cedeebcab` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h5/matches.csv` | 115,337 | `915206ac66a0cce1a99764393a110c3af8735ca40bd7cb271921bb4c921826d4` |
| `results/cross_arena_v3_blocks/v3__e1s9_e2s2_h5/summary.json` | 3,413 | `797bce07011ad937f533f09dc4167082fe17b52c8055adcd513c04706ce9a34e` |
| `results/readout_remedies/jax_on/coupled_tests.json` | 4,789 | `419e9362766370ab70cf8256ddd02c3d12ee4247d08f1002742abff99a1275b7` |
| `results/readout_remedies/jax_on/exp1/per_seed_metrics.csv` | 3,246 | `1d52fa6c144c529c19c799732f56a20cc9fce85bd2d27dbf30f9f5a660e9fb25` |
| `results/readout_remedies/jax_on/exp1/summary.json` | 836 | `f8bf6df64c2dd086605a411cff9590a4a40ad5a1b6a15ef229a4ab29e64e8c39` |
| `results/readout_remedies/jax_on/exp2/per_seed_metrics.csv` | 3,347 | `8d305ec49ef5a10cef37e48f41da7b33f3e7e2a061adb68426631ce2dd0bcd5e` |
| `results/readout_remedies/jax_on/exp2/summary.json` | 836 | `f4958cda17def11d07f402ac96dc38bc4199e360126a4d97cca0faa9a1fbe2cc` |
| `results/readout_remedies/jax_on/per_seed_metrics.csv` | 6,325 | `41b3069bc9009b29a462da20bddbc96f99c3386d1f6fb65e26fd4e6fb98abac1` |
| `results/readout_remedies/jax_on/report.md` | 7,996 | `41cbe586cafa7d6d14c1baf3d9d2dd4e6ec435b694ac15ebd21d4050c238e250` |
| `results/readout_remedies/jax_symmetric/coupled_tests.json` | 4,777 | `48232b0841e9ef061fa634ee073f11c3494e73cbaf8a5bdd10ea1c8b05d404c8` |
| `results/readout_remedies/jax_symmetric/exp1/per_seed_metrics.csv` | 3,204 | `6eb57260565429b3f3818bb3ae9cf598c24156a80358ee7141277cc5275420c2` |
| `results/readout_remedies/jax_symmetric/exp1/summary.json` | 840 | `e8421ca5fc58e3198ef3a80ce28fda902cb2b87b4fdb600c03256848df4b428a` |
| `results/readout_remedies/jax_symmetric/exp2/per_seed_metrics.csv` | 3,311 | `25deba5909a106c4b76e05a7a51bbfdeb56e860eb921d42078543e683361ff8d` |
| `results/readout_remedies/jax_symmetric/exp2/summary.json` | 840 | `dd3fa63b962ad6a5d25592324a4b0e7a61bf10c5e66b71a20887592f74325fde` |
| `results/readout_remedies/jax_symmetric/per_seed_metrics.csv` | 6,247 | `bfc4ebccf94a9dc160ecca525a82aa378e6404c1f7c8b337384f4119e4731857` |
| `results/readout_remedies/jax_symmetric/report.md` | 8,022 | `e19c16ec67a801d8c65ba8e9e1f4a03dd7cd8f770dc3ba3707af7ee2bd7c6dd8` |
| `results/readout_remedies/spawn_cycle_off/coupled_tests.json` | 4,794 | `3e78db3bec79d754263287a42fd91b7df1b2c0aeabdc8f534dc6a80473da405b` |
| `results/readout_remedies/spawn_cycle_off/exp1/per_seed_metrics.csv` | 3,213 | `4dc45f1f18cfc67737b260277578e17cfbe62efe1b0a59fd9e13b595a526b455` |
| `results/readout_remedies/spawn_cycle_off/exp1/summary.json` | 837 | `592232d50e4dd58058d543c07339606fa28167dbaab1035a8e4fc9b0a7a73499` |
| `results/readout_remedies/spawn_cycle_off/exp2/per_seed_metrics.csv` | 3,380 | `6d8f3b18ad282608b9eb282c898607327d8220d2f18495bf676517c1a45d63bb` |
| `results/readout_remedies/spawn_cycle_off/exp2/summary.json` | 837 | `57603c0c7f7beb7fd4bd9aac3320459b8a5e88c5d2c2fd0b20c40e708fcc19d9` |
| `results/readout_remedies/spawn_cycle_off/per_seed_metrics.csv` | 6,325 | `ac67c8eeefe74fb91df06d2c90e00abda27cff5ddc0533aa02fb9626993372ae` |
| `results/readout_remedies/spawn_cycle_off/report.md` | 8,046 | `ff11332be71850d5d0d9f794d51d388ef08d425035e992961d9623b94c2bd89a` |
| `results/readout_remedies/spawn_cycle_on/exp1/per_seed_metrics.csv` | 3,216 | `2244c4e67230984d6b498512ac7ff845373ef6a3c172a1cdd1c28646c8a33947` |
| `results/readout_remedies/spawn_cycle_on/exp1/summary.json` | 836 | `6ed0df844035e7215c4635731ee477eb0bf00fa9db4f47b5c3748e3b256de73b` |
| `results/readout_remedies/spawn_cycle_on/exp2/per_seed_metrics.csv` | 3,383 | `3421dac967fb397fded36966aabc4a35b6ec96ffe88d0ba2268107dcf8119c2c` |
| `results/readout_remedies/spawn_cycle_on/exp2/summary.json` | 836 | `0a883965634a39a612b927c172b48d56fe952f3248f5ab87c031717a9c17930f` |
| `results/readout_remedies/spawn_cycle_on/per_seed_metrics.csv` | 6,331 | `5df99844a49aca8eb4a7148197ac8d0876718057494943e041813d9d8501317a` |
| `results/readout_remedies/symmetric_off/coupled_tests.json` | 4,770 | `cc7a444aff3056f629eba2daeb1bbc0c66e151a0421743f5bda1146f00aa1aac` |
| `results/readout_remedies/symmetric_off/exp1/per_seed_metrics.csv` | 3,207 | `02f9427e27ce316e813a40a24143d63b36c927a9b2d93ae898fc56ae49f3dbaf` |
| `results/readout_remedies/symmetric_off/exp1/summary.json` | 837 | `96a54ac30f94d2946a2b5cb3c624f27a0042e1cb52915c6936e8989aaf69e98f` |
| `results/readout_remedies/symmetric_off/exp2/per_seed_metrics.csv` | 3,363 | `2c250a596d6a9117dbd2db6303626be2ece4a0c1adf82ee03f6de5d8b353fe05` |
| `results/readout_remedies/symmetric_off/exp2/summary.json` | 837 | `e252fa57ccedf4b3b064e614f201f41fd870cfdc5ff9bd241c418c5c0ccbc6c0` |
| `results/readout_remedies/symmetric_off/per_seed_metrics.csv` | 6,302 | `c238c86409518e759446363b34d67e60a5ac853c92809ca0ae55cf3ef55e4888` |
| `results/readout_remedies/symmetric_off/report.md` | 8,016 | `bc7076f26ee057d13406bdc1bb06817b13826bbc1273432f44d1edb1b0d2e781` |
| `results/readout_remedies/symmetric_on/exp1/per_seed_metrics.csv` | 3,213 | `33c39deea73132e0a22a6377cad38ef8da930e50e045047d08788ac2746393a7` |
| `results/readout_remedies/symmetric_on/exp1/summary.json` | 840 | `ae5fc135ac0bcc43fef51b26809c4335c6279151e91f0f6972142d17d6775342` |
| `results/readout_remedies/symmetric_on/exp2/per_seed_metrics.csv` | 3,355 | `49d7e5464b25e23aacb80bd00dff0e8abf645396d0830db95b8a9615b405c434` |
| `results/readout_remedies/symmetric_on/exp2/summary.json` | 840 | `0bb3fafa9c48b933b5bb59fc08a8392cff27b6f12ca5981ddf1b19317e9e4448` |
| `results/readout_remedies/symmetric_on/per_seed_metrics.csv` | 6,300 | `1e56dab9ca3338ae21ae233e6c438d056874a23301775eec0617637a15587b3a` |
| `results/readout_remedies/v3_off/coupled_tests.json` | 4,770 | `cdda1b6c8d9f56c551e21e979a4f5905599445ab094f5f3a8e088c7bdec3a175` |
| `results/readout_remedies/v3_off/exp1/per_seed_metrics.csv` | 3,201 | `394334e9b41323d1254cc282a1abea16e92766bda6c9ebf30148d5ef450ad670` |
| `results/readout_remedies/v3_off/exp1/summary.json` | 840 | `10f67dc4cdcb7f104b7f9d10467c6bfd30bf124c16410aa8d06b949c4b42a6e1` |
| `results/readout_remedies/v3_off/exp2/per_seed_metrics.csv` | 3,364 | `3d06cea57ce9cd208226278072e74651ba64890ac24d367e6250d42c4ac5545c` |
| `results/readout_remedies/v3_off/exp2/summary.json` | 840 | `5349fb988aa3cd9dbbee5a22631fd97c654c4183204d6874dc07ffe8f56bcd2c` |
| `results/readout_remedies/v3_off/per_seed_metrics.csv` | 6,297 | `955f49f65a71ef28e29c6e51eb12a79b10bc025cbda0d7175d3e69f9ab63924b` |
| `results/readout_remedies/v3_off/report.md` | 7,969 | `37dc6a74093372ac28023147cd7a7e0c03b59f170a989c2f77817063371c459b` |
| `results/scripted_v3/matches.csv` | 246,518 | `df818e7c474c24c1a2c63364ab5f66b32dab3766b794df027549a85aa3bf2665` |
| `results/scripted_v3/summary.json` | 43,993 | `3f5a9978d4d1d1af4a7dc5dbc33d1c46bfabb3e3d526e0ef79e545ef02ce13ab` |

**Optimiser traces and the analysis of them.** `results/trace_analysis.json`: the KL, entropy and value-loss traces of the re-trained twenty replicates joined to the published per-replicate reads, with the two controls the analysis rests on recorded alongside it — every produced checkpoint hash-matches `data/training_replicates/`, and the re-trained held-out rates match `results/study/` to 3.3e-05. The per-run `training_log.json` files themselves are git-ignored scratch under `data/runs_traces/`; this is the derived summary, reproducible with `scripts/analyse_traces.py` from a machine that has re-run the campaign.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/trace_analysis.json` | 48,529 | `dfe996ed0e5bff04bebc6c55f003dde6d57ca45bffc094e4778080a1929a6b39` |

**Readout re-measurement.** The same replicates played twice, under argmax and under a fixed-temperature sample of the policy's own logits: `results/readout/T0` is the published readout replayed as a control, and `T0.5` / `T1` are the sampled arms. The schema is `run_study.py`'s, which is what lets `report_study.py --compare-to results/study` pair them against the study without a new instrument.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/readout/T0.5/per_seed_metrics.csv` | 6,253 | `8dd06a4f383f3504dece6b378e7ab5776310bee4adb1bbe64524b4dd95605ef8` |
| `results/readout/T0.5/summary.json` | 768 | `b74c6371ad86660f9f29d6af6849b7d924e004a6305d43e979e20493b10e6427` |
| `results/readout/T0/per_seed_metrics.csv` | 6,301 | `1f1d328c8e96458e31d7cf75fbc530479c4f609280c0a95b82a9b2b76684a959` |
| `results/readout/T0/summary.json` | 744 | `5e3aed9b3f58064d172ecfa7506fd8e464452eefb402f37df9361046db73ad04` |
| `results/readout/T1/per_seed_metrics.csv` | 6,272 | `aa8a41376889dd121efc21f439cfd16705dd6ee53b59f059cd6077f104a1f4d5` |
| `results/readout/T1/summary.json` | 764 | `9e9b1b22f063c9a7f58a21462be60dd579d87f87828021bcb9e3b1804920e661` |

**Superseded round-robin pools.** The earlier replication counts of the same instrument, kept because chapter 8's history is quoted from them and not from the current table: `pairwise_first_run_600.json` is the first single-checkpoint run (600 matches) whose verdict was 'no pair separates'; `seeds/` holds the five 600-match runs that made up `replicates.csv` (3,000 matches, one checkpoint set, and the pooled analysis whose missing `harness_seed` key left every pair meeting 600 times -- [A-39](11_code_audit.md#the-pooled-round-robin-analysis-dropped-a-replication-column-from-its-match-key)); `replicates.json` is that 3,000-match report. None of them is evidence about the current ranking, and the chain 600 -> 3,000 -> 9,000 -> 18,000 -> 30,000 is the reason every separation in [§ 8.10](08_results.md#cross-arena) is quoted with its replication count.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/cross_arena/pairwise_first_run_600.json` | 5,563 | `99bb2ca332c212669a899b0c8d31f89a6e35fab71791df91f22d91efbd2f80ce` |
| `results/cross_arena/replicates.csv` | 594,744 | `df1b3bd556f700a49ffa24015f62a37594523cbdb2b1f58898fe7e5f02393f09` |
| `results/cross_arena/replicates.json` | 7,164 | `2b31001ac1ab1f1a5965b4aa48061fa2289cb34d88bc2391c332695dd1279104` |
| `results/cross_arena/seeds/summary_seed1.json` | 2,822 | `4cd52a7b62fb38eaccebb6b9fa90b4393a6e7d9d1bfcb1eb7558ef2b612e74dd` |
| `results/cross_arena/seeds/summary_seed2.json` | 2,823 | `8ea817f19aa93baa6d96bd82ef9efb6fec2a6abf99c6cee06529fc0806c76b81` |
| `results/cross_arena/seeds/summary_seed3.json` | 2,811 | `e5e847f5d525f94f4d7ad19f1365c9879eaec64d0c5fabb77463bd0fa1d1dd90` |
| `results/cross_arena/seeds/summary_seed4.json` | 2,820 | `272adeb3a59349573431fe8fd3e71bb8b4fb77c906f1d13ffe8325fa73b232ee` |
| `results/cross_arena/seeds/summary_seed5.json` | 2,820 | `fc0028d8d7811f32d5483da563ec57cc6b294ad3006d267a4d2e01c11700a63c` |

**Untrained baselines.** `random_baseline.py` output, 300 fixed-variant headless matches per mode. The uniform files are byte-identical between the trees because that mode never queries a network.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/results/baseline/random_baseline_greedy.json` | 1,163 | `5836778e11b5c4809f51670a3edf859db660c114282cb92cf63fe4ba31a12f63` |
| `ctde_arena/results/baseline/random_baseline_uniform.json` | 1,153 | `55c5dfda0f8f5108b4bb382d1847d821fcfc64a6902c13be43c7901c6d3b3fa5` |
| `results/baseline/random_baseline_greedy.json` | 1,102 | `d86ef73ae93808b28c9a950c7fc70dabefee7d5295dd6011658c6a86bdff59a4` |
| `results/baseline/random_baseline_uniform.json` | 1,139 | `908e2f4e379d30c0d8333d4e164f0711cadb02a66d0588c6219f7716480cb5d9` |

**Historical single-seed runs.** Superseded by the study but retained because [§ 8.2](08_results.md#historical-single-seed-runs) quotes them and their checkpoints no longer load into the current observation encoding.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/data/historical_100k/checkpoints/team_1_ctde-vd.npz` | 197,916 | `0350153c98c5ac7ae1730b53e60621416fe4f8630e3033e50a765a34bec93eea` |
| `ctde_arena/data/historical_100k/checkpoints/team_1_ctde-vd.pt` | 216,346 | `c0ab135db4674649e16db19e4ab901e4895196e298d891e62487cb1353f4bb24` |
| `ctde_arena/data/historical_100k/checkpoints/team_2_ctde-cac.npz` | 274,222 | `23f68d440170b04044c8de1dafccdc32ee500c0206fb6c9cc4cf9c41cd6fb5e2` |
| `ctde_arena/data/historical_100k/checkpoints/team_2_ctde-cac.pt` | 299,324 | `6352a1f04d74165fcfc08e632d5b52ecc69f3346eaa250b35fab4fae3dee29ed` |
| `ctde_arena/data/historical_100k/checkpoints/team_3_ctde-comm.npz` | 282,455 | `2c465eb6efa5fd2a896b9fda391972b67bcfb2201f2ae16ab230088cffea7b9b` |
| `ctde_arena/data/historical_100k/checkpoints/team_3_ctde-comm.pt` | 308,194 | `a80855ba555d8579623deb07591983c1fe3cd8c9eef1b5fd346106ae89b461a4` |
| `ctde_arena/data/historical_100k/checkpoints/training_log.json` | 1,594 | `7b15f63a1136277c561769a779e1f39bf4f6d369e5cd148c0d07ebdc1a3842f7` |
| `ctde_arena/data/historical_100k/exports/comparative_dashboard.png` | 272,740 | `3d0248bf4c843540393bd36b0c476a0165522419ca08732c6bb88c3f00c4c7c7` |
| `ctde_arena/data/historical_100k/metrics/agent_match_metrics.csv` | 19,882 | `4137008304294274e7b022ed7cda573a0bfc1f9a13206555b5de524bde673105` |
| `ctde_arena/data/historical_100k/metrics/summary.json` | 918 | `922f9ed5d1ece6a6a0007f89d1be78d95920231d92c302ff56c8fb1f072598d2` |
| `ctde_arena/data/historical_100k/metrics/team_match_metrics.csv` | 11,583 | `da37159c5de0f935c44f1f17d17f674fd9fbe778fc41a7e0c7473e557792101c` |
| `ctde_arena/data/historical_100k/metrics/trajectory_metrics.csv` | 2,081,943 | `b3b4685d1b69c6e4fcd96c1e71f0b1efb61c572ac8b82146d20e5a9c0dbdb252` |
| `data/historical_100k/checkpoints/training_log.json` | 1,577 | `f0a95b42ce1302ba97a18d76dff395c32aa356d5cc476e5a01570ca7ab4d813e` |
| `data/historical_100k/exports/comparative_dashboard.png` | 285,844 | `0815af67bff72de2db06f0b7a42974c4e5bd1a5c2632ca766eb10e1ee4a0c321` |
| `data/historical_100k/metrics/agent_match_metrics.csv` | 177,672 | `30930d6c39aeeb3d6e3db02e0213aa1570845e4079e1b5060c4e20dad1078e25` |
| `data/historical_100k/metrics/summary.json` | 920 | `acabf37218a246c2336d77766463c8ad415a56b4c69541c05ccb204d66a0c55a` |
| `data/historical_100k/metrics/team_match_metrics.csv` | 10,936 | `491117a1b37256e621bd0130b36a976cc55d748afd3845617e0be387aaea1a07` |
| `data/historical_100k/metrics/trajectory_metrics.csv` | 1,595,699 | `c7142281233095d790647f25059cfd2ac3d62e2d8a10a05384eb2ff0925a7bee` |

**Training replicates behind the round-robin.** The checkpoint sets the round-robin is played over: one experiment-1 replicate and one experiment-2 replicate per block, ten blocks in `blocks10.csv`, the first six in `blocks6.csv` and the first three in `blocks.csv`. Versioned because the round-robin's claims rest on them while `**/data/runs*/` is git-ignored. `exp1_seed9+exp2_seed2` is byte-identical to the promoted checkpoints above, which is what makes that block a parity control. All ten are played: they are the twenty study replicates, each used exactly once, so no further block exists to add to this pool.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `data/training_replicates/exp1_seed1/team_1_dte.npz` | 133,760 | `a4035d1fb6c5148a220731ba7c3ecb5175d38c5dd876187d1de48eeaa5275b76` |
| `data/training_replicates/exp1_seed1/team_1_dte.pt` | 145,762 | `e74177b814dbb3dbe2ed2e9f5af67eadb9e94583f3d5b289d025a64532a84a92` |
| `data/training_replicates/exp1_seed1/team_2_ctde.npz` | 280,269 | `5b497de79f9a21693adf83f68721bb3ddfdcd89d78493cdf1cd9527d2eaa7880` |
| `data/training_replicates/exp1_seed1/team_2_ctde.pt` | 304,248 | `d46a09bb4b044bb5915c82b454eda3a1c7ff741d30113477f6789c3f7dd6b3ae` |
| `data/training_replicates/exp1_seed1/team_3_cte.npz` | 298,043 | `15dbb0e4b5beaa9cd42ac37a622e6f2cbf73bfb9e0aff22a2c24167143aae9a3` |
| `data/training_replicates/exp1_seed1/team_3_cte.pt` | 323,170 | `e7613fa2c40276695a35672016a2d5c170c018164f2ccef10cb8b3a92c82b22d` |
| `data/training_replicates/exp1_seed10/team_1_dte.npz` | 133,709 | `32e8b5aaba02beabbc3048309a1a746675246e6e8506ec4afdb758d1916b7063` |
| `data/training_replicates/exp1_seed10/team_1_dte.pt` | 145,762 | `1abf8cc62a757bc4bc61dbec67142a4159fb676eac4f7a856bd3803423715348` |
| `data/training_replicates/exp1_seed10/team_2_ctde.npz` | 280,505 | `611827fdf8ad42252adecc41fa11bf45813be03a117e64c60ad375867280bc8d` |
| `data/training_replicates/exp1_seed10/team_2_ctde.pt` | 304,248 | `274e65e176ba5c07b356095b0bb1317bc1a77680df1629bd0344ec3f8b80fc6c` |
| `data/training_replicates/exp1_seed10/team_3_cte.npz` | 297,943 | `c5dac63af0426dcf1270841bd0990ef4692ca2feb3f033d5a9184ccbbb7f832b` |
| `data/training_replicates/exp1_seed10/team_3_cte.pt` | 323,170 | `3c9665d812d3719542413a00f5a7dc9da5e4001f41cd8df93599105e824595ca` |
| `data/training_replicates/exp1_seed2/team_1_ctde.npz` | 280,275 | `d764a5712b181b429c8908598c7b7b0552c406d6b545aeeff8576e0d2d3517f4` |
| `data/training_replicates/exp1_seed2/team_1_ctde.pt` | 304,248 | `b41e0d05894b970efecb7ecd10fc90a0f8ebc260a7f5f4de845c2b1e95b2f8bc` |
| `data/training_replicates/exp1_seed2/team_2_cte.npz` | 298,051 | `fc6187577d6439c5562faaa71e3c37ac4ab72a963d7c0a18a4e0fd948455316f` |
| `data/training_replicates/exp1_seed2/team_2_cte.pt` | 323,170 | `fa5584c0366c673c0b20bbe4958f295b59160eb24144e69fb90a129874f872a7` |
| `data/training_replicates/exp1_seed2/team_3_dte.npz` | 133,772 | `639ff7f558d50279e270c3f19dc0b43b70572cde8cc8aa89d903157fda090d4b` |
| `data/training_replicates/exp1_seed2/team_3_dte.pt` | 145,762 | `6cf716c85db237143de4c1ffda5bd0bfff7c0ed3cda43e53878365f4f24285ae` |
| `data/training_replicates/exp1_seed3/team_1_cte.npz` | 297,969 | `06cc40f9bf87690b5b08811c465c33fc32fcc3af568ec892fd57583a51c1e15a` |
| `data/training_replicates/exp1_seed3/team_1_cte.pt` | 323,170 | `26bec8463a68eab36e8b8181cfd97f40be8d8e05b258375de62b2a8005e75cf1` |
| `data/training_replicates/exp1_seed3/team_2_dte.npz` | 133,730 | `db03920b205dbcbcf8ed66ea4e82c2f579f822f51345e5d16a182a4529b0effd` |
| `data/training_replicates/exp1_seed3/team_2_dte.pt` | 145,762 | `31d610801fcf770282dea3276e8a6a0951277f286e60ceee89c801ef0192bad0` |
| `data/training_replicates/exp1_seed3/team_3_ctde.npz` | 280,559 | `b61858755987b05fd17a4143a2805844fd675a6ab8775cd82fc61e200fb41ba8` |
| `data/training_replicates/exp1_seed3/team_3_ctde.pt` | 304,248 | `b01513cf107717860ee511dcf692eacfcc4b7f6aa02f39caa078c44be6367b28` |
| `data/training_replicates/exp1_seed4/team_1_dte.npz` | 133,716 | `d861bfd852df03ffcfb8a7f91bd3e9b5748c83300b67b548b177d6547b62ae42` |
| `data/training_replicates/exp1_seed4/team_1_dte.pt` | 145,762 | `c8af14b70143e669c270335cd12db21b1771271c938b2b2493e29c8b3ccc1590` |
| `data/training_replicates/exp1_seed4/team_2_ctde.npz` | 280,351 | `800edb6032ccdc3b213c094e356ec2199c1bff0403538b3c1c22a4ab491fa44a` |
| `data/training_replicates/exp1_seed4/team_2_ctde.pt` | 304,248 | `0666cfb5d45545b45b41d76733e850b46bc63378b597eafb9ca5c886368e2718` |
| `data/training_replicates/exp1_seed4/team_3_cte.npz` | 297,970 | `6335f4141f87b8d697a9c00535700b6467360f84f0074e8731950e921e7540df` |
| `data/training_replicates/exp1_seed4/team_3_cte.pt` | 323,170 | `e91e83c806d4e368126d572343be1c4e66d5079339a8396116d1c10b6ecf01c7` |
| `data/training_replicates/exp1_seed5/team_1_ctde.npz` | 280,309 | `fd65e995cd46fe70a2f3fd6bfb4dd6f06c8d6ad812d5df017046333e38fb89ba` |
| `data/training_replicates/exp1_seed5/team_1_ctde.pt` | 304,248 | `1e278dd3bb5509f1b95a155531232a7d8bfbe3f086eca202e374f8e968273dd7` |
| `data/training_replicates/exp1_seed5/team_2_cte.npz` | 298,029 | `cc5215da20d0ce9fd65a92f97f7bfe003e53daec538c96ffba8b29cc9b289f14` |
| `data/training_replicates/exp1_seed5/team_2_cte.pt` | 323,170 | `595a178634ea529c5f55b05ddf6a67e249479d111aa1abdccac3ae63d7977f25` |
| `data/training_replicates/exp1_seed5/team_3_dte.npz` | 133,784 | `af8cac30d416aa29a7ee5205af07deb326a200368ccac82dc2d4b9d1ebf40d8d` |
| `data/training_replicates/exp1_seed5/team_3_dte.pt` | 145,762 | `22b72dc327e273c862089228321e797e2d4b116f3d770c201ef1f18bf9e77939` |
| `data/training_replicates/exp1_seed6/team_1_cte.npz` | 298,134 | `5cb07b6446aff3faf1579b321fe96b66bc9a5dc94e181842338c51da16c233a6` |
| `data/training_replicates/exp1_seed6/team_1_cte.pt` | 323,170 | `8a02c8963f8a608b316daadda3f4064eacf2955b3b4cc3b4be46d8b88e593b77` |
| `data/training_replicates/exp1_seed6/team_2_dte.npz` | 133,722 | `4b03dfbdf1bfdda49b16efc7b2fa5573de22468bcd06406025abd04da50f0b16` |
| `data/training_replicates/exp1_seed6/team_2_dte.pt` | 145,762 | `5e4ea3bf157826833cc577e51fa80623178e94e5cc979260892cc1b5d7bc445f` |
| `data/training_replicates/exp1_seed6/team_3_ctde.npz` | 280,408 | `253de60c63b074c677c7dcd3e71be7f2fca404cb29f97d870153028f5210f608` |
| `data/training_replicates/exp1_seed6/team_3_ctde.pt` | 304,248 | `a6b23a421642da04dc4257956615280d28acd885185c7da43e676f3eeb077396` |
| `data/training_replicates/exp1_seed7/team_1_dte.npz` | 133,764 | `d004eaeddce8f970e1220f9e477b82b2258669a9289b045e95f8b5bed31c3de5` |
| `data/training_replicates/exp1_seed7/team_1_dte.pt` | 145,762 | `9b72962182b4547d9c0dbd5baa15890b3eb62279534c6c0d437bbbdd6745ab74` |
| `data/training_replicates/exp1_seed7/team_2_ctde.npz` | 280,411 | `9535ff3814f0e72946756ffa003293853abe0a1c7c2e7b451e2411795b62d97f` |
| `data/training_replicates/exp1_seed7/team_2_ctde.pt` | 304,248 | `420d02ca60393ef2433e650c6df76aecbcb785c11d6f4b06b3db0764ad8eaa6f` |
| `data/training_replicates/exp1_seed7/team_3_cte.npz` | 297,959 | `fa83b61cdf61fbff14a986152b06f4278b4ee6cb96c53c3fd48f88852b54db7e` |
| `data/training_replicates/exp1_seed7/team_3_cte.pt` | 323,170 | `22a6392d8a63319aceacc39857c1636bdb15cb147988cd3fb7aadbaee4b05887` |
| `data/training_replicates/exp1_seed8/team_1_ctde.npz` | 280,279 | `e49715dd4b8e2b2e3e61fa8bafd0da516970d1611031b487281c3ee253db3351` |
| `data/training_replicates/exp1_seed8/team_1_ctde.pt` | 304,248 | `65765a1ee0b6976e40674dce1f82576bc20fc2f7e39adc86f1f50a208ec16ddb` |
| `data/training_replicates/exp1_seed8/team_2_cte.npz` | 298,096 | `8d11b51653abcca4fa82dc1bb28f064f968624ac34d2906ffcbb83136d86bf1c` |
| `data/training_replicates/exp1_seed8/team_2_cte.pt` | 323,170 | `d3da6724d37a26c763a084c49035dd1357b097299a4305efb3bc3587f2fce3db` |
| `data/training_replicates/exp1_seed8/team_3_dte.npz` | 133,784 | `e66854bb2b64437313d101ef50a8a66160e9115817301ce55a5ec120107d0d3f` |
| `data/training_replicates/exp1_seed8/team_3_dte.pt` | 145,762 | `e7a6663165b19c85e7dfcf72259d4dcd2720dd7071dcc7c7258698ab1a2550d1` |
| `data/training_replicates/exp1_seed9/team_1_cte.npz` | 297,965 | `2d3ee1bd6a0f04270ac5ad16e3f292c43e241ba0c8b077391fd7bfc8ade3e8ed` |
| `data/training_replicates/exp1_seed9/team_1_cte.pt` | 323,170 | `c978273409e60e45b9555d763d65c9ff13ba7774ffaf332ac4eccd11ffe59111` |
| `data/training_replicates/exp1_seed9/team_2_dte.npz` | 133,746 | `670ecaa077da12d0b0e24b756ef8dc981cca93f0a774134e08d7cefde3cdc347` |
| `data/training_replicates/exp1_seed9/team_2_dte.pt` | 145,762 | `93fc387fbfbd1cb935a271672954675230e1326565ae2680333a4ded4e4e3897` |
| `data/training_replicates/exp1_seed9/team_3_ctde.npz` | 280,383 | `400520e13e447d3a7d71ea72e9987fea886d90d32164b105e65723638a928447` |
| `data/training_replicates/exp1_seed9/team_3_ctde.pt` | 304,248 | `5a5273f3c73942e8fb0964bbb6a848734c58fee9cf0b52b4986f84cf8f2a4fae` |
| `data/training_replicates/exp2_seed1/team_1_ctde-cac.npz` | 280,568 | `15eb3fc55514589fc3705f2143991897f790525e0d9f642f96b3d82f88151a61` |
| `data/training_replicates/exp2_seed1/team_1_ctde-cac.pt` | 304,400 | `7c3eb6de6b1388f1b37eb3c334f8185ffce543829a4db796d907f4ae3498c054` |
| `data/training_replicates/exp2_seed1/team_2_ctde-comm.npz` | 289,296 | `72de3e21dd18f330731e5be285c93def4471c63908e9a76789cae6338a39b8ef` |
| `data/training_replicates/exp2_seed1/team_2_ctde-comm.pt` | 313,778 | `baae4dda211f526da971587b58fbba379f1f9b2a06b3530d3310f4d3bda38b2f` |
| `data/training_replicates/exp2_seed1/team_3_ctde-vd.npz` | 199,471 | `af910f4245e4372233a05933cfadd0c4dc0214a42b28bbc95caaf4f5d1f864aa` |
| `data/training_replicates/exp2_seed1/team_3_ctde-vd.pt` | 217,330 | `e7d7790c164ee0cbfb6c2346a2f947e4c9ac2b1c976d6ddededb0a0c286d4c8d` |
| `data/training_replicates/exp2_seed10/team_1_ctde-cac.npz` | 280,386 | `6ea5be85178eb9d6a6341d0a84ba1ab90c8091b8ffafad8048d76c89aa6442d7` |
| `data/training_replicates/exp2_seed10/team_1_ctde-cac.pt` | 304,400 | `587aba36f719d9a9af3c01fec30866d0fc403682f12b15e2e40cc2e371db577a` |
| `data/training_replicates/exp2_seed10/team_2_ctde-comm.npz` | 289,271 | `da82fed8365e89b8c3ef0b06e3fe99dee777dbc98866f6fe2bab59fa7bea5f44` |
| `data/training_replicates/exp2_seed10/team_2_ctde-comm.pt` | 313,778 | `43f12f9a7b62380042640f27a1230f03991ed81c301d4b95bac6b3242a238a15` |
| `data/training_replicates/exp2_seed10/team_3_ctde-vd.npz` | 199,634 | `1baa4d18188f921406c335c17c2014178e3497a55aa3aa05f2ec7b237b854e4a` |
| `data/training_replicates/exp2_seed10/team_3_ctde-vd.pt` | 217,330 | `8bef5ccb2d26c2857c9918393004b3daa6de731bb0a4e0061b0f1c86c7309f6c` |
| `data/training_replicates/exp2_seed2/team_1_ctde-comm.npz` | 289,236 | `d4fb91bf64f3954ebb2534dd636c8cb9de748c8bad16943b686da2920b565c95` |
| `data/training_replicates/exp2_seed2/team_1_ctde-comm.pt` | 313,778 | `cb3fee053d68a7a3cc6e4cadbd3fdfc05a356e5a9542babad143a1aa2eb85ff1` |
| `data/training_replicates/exp2_seed2/team_2_ctde-vd.npz` | 199,531 | `cc96d2a5bf8b947b64f3ff760ef40ec2405897ff0694e87fdf925ec859b512e7` |
| `data/training_replicates/exp2_seed2/team_2_ctde-vd.pt` | 217,330 | `bf261e17c010ee0bb8f69b9d7e9ca1c4214cc9cfe0d8775038379dd71547d5f5` |
| `data/training_replicates/exp2_seed2/team_3_ctde-cac.npz` | 280,666 | `8ec889b87fe509c6c139a93a96f97027e2fee3bba42aeb0288c57d0e3ceab8ef` |
| `data/training_replicates/exp2_seed2/team_3_ctde-cac.pt` | 304,400 | `451b14b4c8aae906c3e7b135f9e814426b268354e8eb222d420b42dcb981cccb` |
| `data/training_replicates/exp2_seed3/team_1_ctde-vd.npz` | 199,533 | `7d33ed284592dbc4abb9ea0f1549bf808fc6dbd95d16ec8fb03bc5753f62754e` |
| `data/training_replicates/exp2_seed3/team_1_ctde-vd.pt` | 217,330 | `14feaae8a70a214b4eba09ac9cf41d14ef2ea1837e8d8c4bf3dbdc7524950e1a` |
| `data/training_replicates/exp2_seed3/team_2_ctde-cac.npz` | 280,407 | `e2d512e7a7548ef306437d6659a5c76742902db602e629940d31f2fe14725034` |
| `data/training_replicates/exp2_seed3/team_2_ctde-cac.pt` | 304,400 | `77c98639e3d2e86d64c1640a266127fd255549e2709007ade734f9562b95682e` |
| `data/training_replicates/exp2_seed3/team_3_ctde-comm.npz` | 289,332 | `61cf9af6e0f5414a4f01f5bef0979f28df2a916ad2e79483f0b7381df0e5aadf` |
| `data/training_replicates/exp2_seed3/team_3_ctde-comm.pt` | 313,778 | `d06e8316b86063471a03b28bab4a1c3652b09a80e8666c8202dfcc3d1dab2c70` |
| `data/training_replicates/exp2_seed4/team_1_ctde-cac.npz` | 280,321 | `d82d0ef839abb4e725f210d2f839eda8383cf07057c627a0275174b9451f3184` |
| `data/training_replicates/exp2_seed4/team_1_ctde-cac.pt` | 304,400 | `1cd86ec8561ae603768bb1f37baa23484d1599cd96bc7f9668d46b4768948106` |
| `data/training_replicates/exp2_seed4/team_2_ctde-comm.npz` | 289,305 | `52e8f19b1fba80f5271decf4455af9d9cdcdd9729abec3d94de5a88a78151f01` |
| `data/training_replicates/exp2_seed4/team_2_ctde-comm.pt` | 313,778 | `1ee645fd0afe4da4a574b92ba72427221778555135ccef12b1729c0de7279136` |
| `data/training_replicates/exp2_seed4/team_3_ctde-vd.npz` | 199,617 | `883b9772c3c2de901a3b42d8362834e831c4616f71fbc94f77124f3a7d6e547d` |
| `data/training_replicates/exp2_seed4/team_3_ctde-vd.pt` | 217,330 | `319e3e0beb661142d823dcce8e38c7356c0cc49833aa88b1d0bba307f8f83a9d` |
| `data/training_replicates/exp2_seed5/team_1_ctde-comm.npz` | 289,353 | `72ada863a3d08ac83cc8ed05ab234d2323c2427b2d7aa0e828848d0d0bc8e72b` |
| `data/training_replicates/exp2_seed5/team_1_ctde-comm.pt` | 313,778 | `3d9bb400a0694e5deda58bca87e6965c2699760bee62fe3b1dd03c5387be1667` |
| `data/training_replicates/exp2_seed5/team_2_ctde-vd.npz` | 199,502 | `6517f11487d3773efc9ff7337f362d4f99a1d47294c86e81226d36955910bad3` |
| `data/training_replicates/exp2_seed5/team_2_ctde-vd.pt` | 217,330 | `c8f3caa5094976f0bffc34aa4c8a0e1d6b5c653cb9ee65fbd5a30b8f345198de` |
| `data/training_replicates/exp2_seed5/team_3_ctde-cac.npz` | 280,357 | `039d24246395cb1e80c70ef84a81af6b63961d18ec527871430e3cd0c8d96884` |
| `data/training_replicates/exp2_seed5/team_3_ctde-cac.pt` | 304,400 | `3898cd7a3ae8a1877c08e41a620e49cc5533be420cd0f10a7773147cc12cdca0` |
| `data/training_replicates/exp2_seed6/team_1_ctde-vd.npz` | 199,431 | `3a2efd9923280786496b9d8076b849afc0a332f9db4afa95d25749cfa76a595c` |
| `data/training_replicates/exp2_seed6/team_1_ctde-vd.pt` | 217,330 | `fa1ccd9518a7b121b31bcd3ec2c6d2f21a50839a90deb83fe20d9f8631e22aa8` |
| `data/training_replicates/exp2_seed6/team_2_ctde-cac.npz` | 280,428 | `33376e2b5a5c63a399078a48e2a2ec9e6322a8f2418368a0fd791cb183851cf1` |
| `data/training_replicates/exp2_seed6/team_2_ctde-cac.pt` | 304,400 | `15331b2abd0aa3312261ecf9cdee069bbff0fcff52119d7047a42c4918f09e08` |
| `data/training_replicates/exp2_seed6/team_3_ctde-comm.npz` | 289,229 | `6fcdc17ab1048712ea80d3df84cedf83f94bea226dd6011a38e269df122a75d8` |
| `data/training_replicates/exp2_seed6/team_3_ctde-comm.pt` | 313,778 | `a39b067f11a28d3316f184248d262ce2c9bca044f8287fb4140c19805d051ed6` |
| `data/training_replicates/exp2_seed7/team_1_ctde-cac.npz` | 280,421 | `05701d0cfae84ae5a84dc10783b5c5adaf9a83ad4199e220e89301e9422dffda` |
| `data/training_replicates/exp2_seed7/team_1_ctde-cac.pt` | 304,400 | `bcb973516fc3e1141a24dda3243373de730ca6f52476f6227dc7162152506bf1` |
| `data/training_replicates/exp2_seed7/team_2_ctde-comm.npz` | 289,402 | `3ce224636cd54006da3d8399268ce36cf2990c29d5da0e6808e0db13682dcb5b` |
| `data/training_replicates/exp2_seed7/team_2_ctde-comm.pt` | 313,778 | `5a22ef5f10fb96b78892ffb58e351876072573fd958f2874d8e7877c1794d14b` |
| `data/training_replicates/exp2_seed7/team_3_ctde-vd.npz` | 199,607 | `e51b4c5df2ad3e9783aa4e6decba447b334fd8cc92f65ad8c1add8bd8b806da3` |
| `data/training_replicates/exp2_seed7/team_3_ctde-vd.pt` | 217,330 | `d49f1fdfdd579ea443e6e253df7d814650b736551f51cd8f32a7e980b93f31c7` |
| `data/training_replicates/exp2_seed8/team_1_ctde-comm.npz` | 289,243 | `e725a9613cb6c185c43a8fb9bf06a09e4ce6b3ea521725b996e43dd229347c6c` |
| `data/training_replicates/exp2_seed8/team_1_ctde-comm.pt` | 313,778 | `a93f6b1000f5034e7bcd7b5349daf3529b3f63c1f91d14091bdc4b958d0818d8` |
| `data/training_replicates/exp2_seed8/team_2_ctde-vd.npz` | 199,794 | `ad3821f2b4bdff5e7d501aaa8d67c4294207ab7c5e7cebeabb4ff32578245a5a` |
| `data/training_replicates/exp2_seed8/team_2_ctde-vd.pt` | 217,330 | `a414c55167c467d3670d54e6987fa5143919a672355d6037e68fa698d591975d` |
| `data/training_replicates/exp2_seed8/team_3_ctde-cac.npz` | 280,666 | `284f55ebd0fc96ae9e5505367cf6a331e9a1391a5f97be6679a50dcaf3c6580c` |
| `data/training_replicates/exp2_seed8/team_3_ctde-cac.pt` | 304,400 | `2d0195ccf746bcca740148ab91f0a42e616370a369ede29a074fa5f1eb9bc761` |
| `data/training_replicates/exp2_seed9/team_1_ctde-vd.npz` | 199,717 | `ff77860fea8117d360e6cf5945f1f62495b7bb926dafcf0a0d2d2317e65a6481` |
| `data/training_replicates/exp2_seed9/team_1_ctde-vd.pt` | 217,330 | `a1be765533b7a16476aef4a4cbfd53bb347ba62e78159ab3104a104176cb0651` |
| `data/training_replicates/exp2_seed9/team_2_ctde-cac.npz` | 280,372 | `c8e32d3efc7aa56d805d32983c384553546e5ebec2e5e1793ddbd3c16d0e16c6` |
| `data/training_replicates/exp2_seed9/team_2_ctde-cac.pt` | 304,400 | `9d5fd224b499af62b0a28de54e5a14ed9db402ebc808f085c939b9629ccd0318` |
| `data/training_replicates/exp2_seed9/team_3_ctde-comm.npz` | 289,321 | `56ec5b7380833e4bb19335837da15fdaace6c55e93a1eae647b17934a8796339` |
| `data/training_replicates/exp2_seed9/team_3_ctde-comm.pt` | 313,778 | `6f9355bd19f43e84b5d18519ebc175e26048d00eb5cbb67f0b5a9ed13bcc1760` |

**Campaign checkpoints behind the arm tables.** The three configuration-ablation sets of [§ 8.13](08_results.md#ablations), and the v3 and single-key sets of [§ 8.14](08_results.md#which-remedy), versioned for the same reason as the round-robin blocks -- the scratch they were trained into is git-ignored -- so an arm can be replayed without retraining it: twenty sets each for `no_dr`, `v3`, `spawn_cycle` and `symmetric` (seeds 1-10 of both experiments) and six for each capacity arm (seeds 1-3). `MANIFEST.json` beside them records each arm's overrides and the study it is paired with, the hidden width and actor parameter count read out of each file's own weights, the arm's seed coverage, and a SHA-256 per file; no checkpoint format here stores capacity or configuration metadata, and the replay guard checks the manifest instead (A-56).

| Path | Bytes | SHA-256 |
|---|---:|---|
| `data/campaign_replicates/MANIFEST.json` | 139,112 | `8298197e99f0c5b24a81059f771810119f4502be6d83beae9cd508fd72daccff` |
| `data/campaign_replicates/narrow_64/exp1_seed1/team_1_dte.npz` | 38,042 | `4c43345e94c8b36b096d34820d6633b7fa5146970608776d50d0e77e6fea7cdb` |
| `data/campaign_replicates/narrow_64/exp1_seed1/team_1_dte.pt` | 42,082 | `7ad507f6be709f8d6e1b68b5fc3fe08d24fe152d0b88f14f4365b30e1716d92e` |
| `data/campaign_replicates/narrow_64/exp1_seed1/team_2_ctde.npz` | 82,014 | `9425241a04115c10265473e7805d2603b7713855af534880113ad5373bead76a` |
| `data/campaign_replicates/narrow_64/exp1_seed1/team_2_ctde.pt` | 89,720 | `ec9e1b3f931f6bbeecbf7400944b9a4da3bbd855a93facb33c516f9a5959f3b5` |
| `data/campaign_replicates/narrow_64/exp1_seed1/team_3_cte.npz` | 90,638 | `45ac1f4cf94540af80499ec3304e0d33011aeea2aaa5ab228801e296be4a6d6a` |
| `data/campaign_replicates/narrow_64/exp1_seed1/team_3_cte.pt` | 98,914 | `02f49b03fcd359ec64b1b0fe8e02edc0723fae487bb421705f41c7c8dbe08970` |
| `data/campaign_replicates/narrow_64/exp1_seed2/team_1_ctde.npz` | 81,998 | `df2646a83e04ba1fc8c1e1093a5bde65c98dd91c3c5c9cea23e66513339a9aaa` |
| `data/campaign_replicates/narrow_64/exp1_seed2/team_1_ctde.pt` | 89,720 | `c32c300d2ba57e0bb5feada1576adb1b7eafc6ff2b2ac53d44c3f9103a287426` |
| `data/campaign_replicates/narrow_64/exp1_seed2/team_2_cte.npz` | 90,554 | `12b760939e1342681249a3119a5839d77956fa76c24b2d90130919b9d9d4c375` |
| `data/campaign_replicates/narrow_64/exp1_seed2/team_2_cte.pt` | 98,914 | `52c43344a1da513ff1dc92ead7d56f8cba30511b60b115c4964aa5c90d009914` |
| `data/campaign_replicates/narrow_64/exp1_seed2/team_3_dte.npz` | 38,076 | `742b082f384e8329df2ffc860099eddd21411810cbf4557736fac192fc67d439` |
| `data/campaign_replicates/narrow_64/exp1_seed2/team_3_dte.pt` | 42,082 | `be61390d4e5e5decf56ee9654885e7eaeca6e80aac3aaded40a595db3100fc4b` |
| `data/campaign_replicates/narrow_64/exp1_seed3/team_1_cte.npz` | 90,544 | `fd4ed6068c1eadf9423e0dec66131bdc24d60a8590fd74cac58a569bf7617fdd` |
| `data/campaign_replicates/narrow_64/exp1_seed3/team_1_cte.pt` | 98,914 | `770e3e7cf90eac5b5127946b531ee29a7391d2e830dc9accc313e4e2c30cbcef` |
| `data/campaign_replicates/narrow_64/exp1_seed3/team_2_dte.npz` | 38,014 | `63c388d4ce281a3ae34288b639a392309cbed5c5d9d58ec3c71ec77ee1b23bcc` |
| `data/campaign_replicates/narrow_64/exp1_seed3/team_2_dte.pt` | 42,082 | `e59b0c10386cc9906b9b044c0f60d3d7935197cbd530ee6760e40efb357d1601` |
| `data/campaign_replicates/narrow_64/exp1_seed3/team_3_ctde.npz` | 82,092 | `68ea8960f5fbb1046412bf08b46ade147bc8c77cab5f85cebd98dc7fccfd1476` |
| `data/campaign_replicates/narrow_64/exp1_seed3/team_3_ctde.pt` | 89,720 | `433da5b57ff30f754d515bdae47d992bd82bb2477b123a60d8b01982ffa81a30` |
| `data/campaign_replicates/narrow_64/exp2_seed1/team_1_ctde-cac.npz` | 82,082 | `2fc9acc0f2201d77e5d62a88d99c58092210dc2dbf8beae8df6530beec3fdfd5` |
| `data/campaign_replicates/narrow_64/exp2_seed1/team_1_ctde-cac.pt` | 89,872 | `fdef47d77ecb22ff259705ecfc0ac8cf64ed35256c9bb28d5e77cfd68fe99a6b` |
| `data/campaign_replicates/narrow_64/exp2_seed1/team_2_ctde-comm.npz` | 86,660 | `3bbc7f1be95145d40d6f151ae81006b6d223e38c9ab4fc36226021c356695388` |
| `data/campaign_replicates/narrow_64/exp2_seed1/team_2_ctde-comm.pt` | 94,898 | `a615b5ff2f9eb1b974d970c4d3fb8d87ddc7c1096eecf90017826b5c5017ef5c` |
| `data/campaign_replicates/narrow_64/exp2_seed1/team_3_ctde-vd.npz` | 56,680 | `908ba18bee744cd38cb26c25399b901b9ad96918a189490bfa77ccb5b8f025f4` |
| `data/campaign_replicates/narrow_64/exp2_seed1/team_3_ctde-vd.pt` | 62,450 | `418c4c9f2265299f91c7cabc7f78faae6e2131989fe3f4b19e3dd625a2f91d71` |
| `data/campaign_replicates/narrow_64/exp2_seed2/team_1_ctde-comm.npz` | 86,652 | `a73ff0ddfcdad817050e4cc3aba434e0e1e09db233b7051b010285a7606ce023` |
| `data/campaign_replicates/narrow_64/exp2_seed2/team_1_ctde-comm.pt` | 94,898 | `5678d2b4e1c3a3dfc5e214495cc78c0eca6a93ebdd165be75131addf9bf0d84b` |
| `data/campaign_replicates/narrow_64/exp2_seed2/team_2_ctde-vd.npz` | 56,668 | `d159c985473d50305faa61542cb9be72aed4fe2160190e5ba195fd7e92c63d16` |
| `data/campaign_replicates/narrow_64/exp2_seed2/team_2_ctde-vd.pt` | 62,450 | `7ac8263c8bd783e6fe6b9e31363d8eeceb61dddc796c9a71e43152e2d8a761a6` |
| `data/campaign_replicates/narrow_64/exp2_seed2/team_3_ctde-cac.npz` | 82,070 | `364fdd2d6b60eed2ffbf78897393c86043124d82eb8b788644303bd4d27a6020` |
| `data/campaign_replicates/narrow_64/exp2_seed2/team_3_ctde-cac.pt` | 89,872 | `1f49b23d10e4d03a6efd35f35180cc7e6742039a94be011969a7a86c1ac8336b` |
| `data/campaign_replicates/narrow_64/exp2_seed3/team_1_ctde-vd.npz` | 56,641 | `e73db06920ff10ac7939fe2a1e78d828dea395a49a88914b708973d0839ac51c` |
| `data/campaign_replicates/narrow_64/exp2_seed3/team_1_ctde-vd.pt` | 62,450 | `0e3ba7e0cf75416ed4d3c2830d89b1748c01f78837d2fb4bb92045c016996adf` |
| `data/campaign_replicates/narrow_64/exp2_seed3/team_2_ctde-cac.npz` | 82,055 | `4e05416eece360dc12fde1ae77525f94b56411dbfa7311958bcb6636eef18aa9` |
| `data/campaign_replicates/narrow_64/exp2_seed3/team_2_ctde-cac.pt` | 89,872 | `557da54109e7f0d3969b5d2c95fd264a7d251efc4015bd1e530522ce2ac4defe` |
| `data/campaign_replicates/narrow_64/exp2_seed3/team_3_ctde-comm.npz` | 86,693 | `88f27b6c01bcc060c67704d8695c07dcb5e685fb20b025f0a3ced4d828cd50c5` |
| `data/campaign_replicates/narrow_64/exp2_seed3/team_3_ctde-comm.pt` | 94,898 | `2a02f732c2a8fcf03ad12666742617bcff7f45699f08bd08a925f9e014036ebf` |
| `data/campaign_replicates/no_dr/exp1_seed1/team_1_dte.npz` | 133,541 | `80a0c8e7c4f9bee6d31bf38a0dda5805e3058f39e24be533014d696f0b608deb` |
| `data/campaign_replicates/no_dr/exp1_seed1/team_1_dte.pt` | 145,762 | `34ac57fbacba389eb9d8243ed714419a18ed81f9087f2aae241879595e416861` |
| `data/campaign_replicates/no_dr/exp1_seed1/team_2_ctde.npz` | 280,120 | `3aaa8065fcc8bb45328b59ba52ef7eae9e2aae8e48a6ac588b7acf50e3e26b3c` |
| `data/campaign_replicates/no_dr/exp1_seed1/team_2_ctde.pt` | 304,248 | `3c557870975f0d2a4b3fb8e0d3b719a4aa26034f7d436799c00f3d3efd3b726c` |
| `data/campaign_replicates/no_dr/exp1_seed1/team_3_cte.npz` | 297,664 | `e29a90d591272632b7af62c14440a6b89306e2688c779a53db13820e1d5307e9` |
| `data/campaign_replicates/no_dr/exp1_seed1/team_3_cte.pt` | 323,170 | `e349b6f9c20e2983a89a33115607d00cb0816b26028aa05c6b717e38d91d89e0` |
| `data/campaign_replicates/no_dr/exp1_seed10/team_1_dte.npz` | 133,588 | `b3d032bce583c2c08aed838f80eaf4558343485df0b23182b010e1308b6a2adc` |
| `data/campaign_replicates/no_dr/exp1_seed10/team_1_dte.pt` | 145,762 | `2d0f5159a3eed63c308a654cbf14c337cd7c14f873281f57161470a07510b75c` |
| `data/campaign_replicates/no_dr/exp1_seed10/team_2_ctde.npz` | 279,940 | `502842c05aac239833e974f4da58c2df10a6bfdf5458987f1abe2751e0bf716e` |
| `data/campaign_replicates/no_dr/exp1_seed10/team_2_ctde.pt` | 304,248 | `e6ff77ce09246966896ddb7f00445e38887ac5c42b7d115c4d0b1373e779754e` |
| `data/campaign_replicates/no_dr/exp1_seed10/team_3_cte.npz` | 297,081 | `ad783666b47ab649aa389f0d497634c5d533bd84fbe6c615781f4255b3a2ce52` |
| `data/campaign_replicates/no_dr/exp1_seed10/team_3_cte.pt` | 323,170 | `0ed90b7a85dff91a401997f775ec246d77513421d1b624c384e58f817b2f772e` |
| `data/campaign_replicates/no_dr/exp1_seed2/team_1_ctde.npz` | 280,140 | `a276d846cfbae2287a48947e0f634a345d7cfb15398031f63370184810c4dacb` |
| `data/campaign_replicates/no_dr/exp1_seed2/team_1_ctde.pt` | 304,248 | `b0cd2b2128ef62b4d2be90386ed51cad5288d9b602a3d124e3f0dbc6a1101d5b` |
| `data/campaign_replicates/no_dr/exp1_seed2/team_2_cte.npz` | 297,560 | `ca535c08cceb5e8605fd54fc90b8992cd414959e01e2cf2fb4e37935de529448` |
| `data/campaign_replicates/no_dr/exp1_seed2/team_2_cte.pt` | 323,170 | `5b79caffd167889ecae76a1d09e0452d5e361dd6dd531d4f89c0259db2c4eebf` |
| `data/campaign_replicates/no_dr/exp1_seed2/team_3_dte.npz` | 133,638 | `fc8ac4947c989522bf407365bf02b239f900564d9440739d7776135a722d9478` |
| `data/campaign_replicates/no_dr/exp1_seed2/team_3_dte.pt` | 145,762 | `5b6376c077aaaea762a18f6a68ad71cc262a56c03b6dad735004a4c0a362e9e0` |
| `data/campaign_replicates/no_dr/exp1_seed3/team_1_cte.npz` | 297,637 | `f95bf8a8c60ea893ea73328437acfdfd2e369e7bbe2a610e05e7343b975a2c6f` |
| `data/campaign_replicates/no_dr/exp1_seed3/team_1_cte.pt` | 323,170 | `9e0a636f0abf14fe6698522d294a0690c98f38e34b78dc14f32ce50106c413af` |
| `data/campaign_replicates/no_dr/exp1_seed3/team_2_dte.npz` | 133,608 | `c9e4f539e68d29a4b01c06d2318a89064f9a1e044966083003a879040a418e99` |
| `data/campaign_replicates/no_dr/exp1_seed3/team_2_dte.pt` | 145,762 | `94d0bce0fa8e04e37274cc66ac81699d0397e67568f00f6d6f06931a38cbcec4` |
| `data/campaign_replicates/no_dr/exp1_seed3/team_3_ctde.npz` | 279,968 | `3371a78419d4ca92a8a88f5ee143cfc21f8df632123467e802e79440e5c4b751` |
| `data/campaign_replicates/no_dr/exp1_seed3/team_3_ctde.pt` | 304,248 | `83c0fddb58c6a02776709a992019ff8e87c95b288613793fcdfbc055c4345f59` |
| `data/campaign_replicates/no_dr/exp1_seed4/team_1_dte.npz` | 133,523 | `05b10fb8a0c7379597cf0199d8eebd8d989b05c11a0d803803bea5a7d0ae155c` |
| `data/campaign_replicates/no_dr/exp1_seed4/team_1_dte.pt` | 145,762 | `f346d872ff5c975de82b23309beb19d6c857ed467ba701e42a38d39197fe87e2` |
| `data/campaign_replicates/no_dr/exp1_seed4/team_2_ctde.npz` | 280,047 | `2de406308e60fb50b9b03f6166e26b76d1c135769b296bceb7a9ea2a4fe466eb` |
| `data/campaign_replicates/no_dr/exp1_seed4/team_2_ctde.pt` | 304,248 | `ce2ee0234b6c37bfa37e86f3c3a9b30171458347f861613fcec6c30d82a5228d` |
| `data/campaign_replicates/no_dr/exp1_seed4/team_3_cte.npz` | 297,569 | `a512c215d9de62502db2d9484626162033cdeb23e63c40b8ebfec90a98b30f2a` |
| `data/campaign_replicates/no_dr/exp1_seed4/team_3_cte.pt` | 323,170 | `4018d384ffb24694c0179fd7575202704ced48cba9b3fdc674e69206207db202` |
| `data/campaign_replicates/no_dr/exp1_seed5/team_1_ctde.npz` | 280,122 | `3aa346acd91a7b1b49ff235f976708bc9dc7f34bc3d79609ad053ce964dc33ab` |
| `data/campaign_replicates/no_dr/exp1_seed5/team_1_ctde.pt` | 304,248 | `4164ca4617d040b4a966c8017c4c8461f1b3a105f4ae1d10f7b5b59b52885e75` |
| `data/campaign_replicates/no_dr/exp1_seed5/team_2_cte.npz` | 297,676 | `d948c8e67140c42c9e425a0c5c730deab3b68266b6f4436e13fb949e90326055` |
| `data/campaign_replicates/no_dr/exp1_seed5/team_2_cte.pt` | 323,170 | `03d22829b748a44e8e0dfbca5cf06337991d8bcba7dd538f7f8b7365b611cdf2` |
| `data/campaign_replicates/no_dr/exp1_seed5/team_3_dte.npz` | 133,697 | `3c6921326c08418fd4f9b7860e5664aa032d58cd317d3b0262da1b77a3e85695` |
| `data/campaign_replicates/no_dr/exp1_seed5/team_3_dte.pt` | 145,762 | `94d0328b6c9fea56adb603fada9864824026a1ce72fcb01c0d9db8186f356e02` |
| `data/campaign_replicates/no_dr/exp1_seed6/team_1_cte.npz` | 297,603 | `5fea9475a8d98e2d0e742ba44fe9e90b23cd1b0bf97fec33a866f9a39c45233a` |
| `data/campaign_replicates/no_dr/exp1_seed6/team_1_cte.pt` | 323,170 | `aacc86c4bec34c7d7442e38f143f0c843303222df92487db4d16b365b84337f9` |
| `data/campaign_replicates/no_dr/exp1_seed6/team_2_dte.npz` | 133,542 | `a12288ab3f062f5615a375cdfbbf328e02763744421917b07d6b82e8373aa805` |
| `data/campaign_replicates/no_dr/exp1_seed6/team_2_dte.pt` | 145,762 | `8867383e3df02d747320a6faa1af523625d23e3d698b62fdbf3a03cf03e37b3c` |
| `data/campaign_replicates/no_dr/exp1_seed6/team_3_ctde.npz` | 280,144 | `854f81aa5d5c13c6614cf7672686ae0fce3375c16e0598242dc8223a2dc9fd05` |
| `data/campaign_replicates/no_dr/exp1_seed6/team_3_ctde.pt` | 304,248 | `e50d5983cda9c9977c520345581c26086fcb3d82b76520399cc7a288e26f2ea8` |
| `data/campaign_replicates/no_dr/exp1_seed7/team_1_dte.npz` | 133,581 | `a231f06df51ba526c418ef8da09c64cf2be974abfe156f41a0c24c8a3203ac06` |
| `data/campaign_replicates/no_dr/exp1_seed7/team_1_dte.pt` | 145,762 | `93d30270806e93d30b02789a87bb4aa30a41ab75d8599d77e11f0f54dbfe4141` |
| `data/campaign_replicates/no_dr/exp1_seed7/team_2_ctde.npz` | 280,070 | `1a7485734a406161ddd34a64199fc63ff0343781e7111bbe019be43e434b05f6` |
| `data/campaign_replicates/no_dr/exp1_seed7/team_2_ctde.pt` | 304,248 | `680705b5e7ac155f9f678d12fe6f270820d7b43d9613691a038ef64941ca856b` |
| `data/campaign_replicates/no_dr/exp1_seed7/team_3_cte.npz` | 297,582 | `6c1a754e703adf160703067f950ee40806606b56ce2c8255fe195f5f76fb3f1a` |
| `data/campaign_replicates/no_dr/exp1_seed7/team_3_cte.pt` | 323,170 | `1dba455057b6a0d3a82266e60687c8e00db498ae862bb9d2fc29c92f1191fc74` |
| `data/campaign_replicates/no_dr/exp1_seed8/team_1_ctde.npz` | 280,009 | `4880943f5c52be49dcad05ce3cba722e887bec7ae6097468780fbc5db8aaf5f5` |
| `data/campaign_replicates/no_dr/exp1_seed8/team_1_ctde.pt` | 304,248 | `6c0740ad0cfea1103b3fb81442a7fd048d28daee530d5d0ef15cf0b232da49b7` |
| `data/campaign_replicates/no_dr/exp1_seed8/team_2_cte.npz` | 297,525 | `065fd19cb1c7ed557b445315153a28902dbbb52c75619669996ecd6be4b5f254` |
| `data/campaign_replicates/no_dr/exp1_seed8/team_2_cte.pt` | 323,170 | `a55ffe06a231fd177cb08cd7c84b2fd24330001e5b1d7abbc28bb84ae8d924db` |
| `data/campaign_replicates/no_dr/exp1_seed8/team_3_dte.npz` | 133,626 | `d4d97252eb1ad1508d08f92436110750605544f29a83537d3fcf825b69bff2e1` |
| `data/campaign_replicates/no_dr/exp1_seed8/team_3_dte.pt` | 145,762 | `9a1fa58947538630920c53e9379a5e36ac771a7e996aea37c52736c8d6507aed` |
| `data/campaign_replicates/no_dr/exp1_seed9/team_1_cte.npz` | 297,422 | `ed54a8eff29cf251c76eec41819204c798e6e97420723b7f16ce3c0eb4fb75db` |
| `data/campaign_replicates/no_dr/exp1_seed9/team_1_cte.pt` | 323,170 | `7e245e7435ff4342eb9a206fb22f6b4e022e46096ab8bb55ccd5b32646f9feb8` |
| `data/campaign_replicates/no_dr/exp1_seed9/team_2_dte.npz` | 133,653 | `49670fa56b9ef7d02af8c980bd7330dd8f194d86e59896ec18fac5e66b00eb92` |
| `data/campaign_replicates/no_dr/exp1_seed9/team_2_dte.pt` | 145,762 | `a676d850181d01749e2c266274780aafd135d2f45b83f2f857d78da37353f59f` |
| `data/campaign_replicates/no_dr/exp1_seed9/team_3_ctde.npz` | 280,112 | `6d6deb25d659e411844c9e1d0f9bc57693e8e6d67ea5d7fda210a14e5b8d6189` |
| `data/campaign_replicates/no_dr/exp1_seed9/team_3_ctde.pt` | 304,248 | `a31b189939ab81683692a4fc29659a6e356325e81a02ead85078dd9bb59c8fbf` |
| `data/campaign_replicates/no_dr/exp2_seed1/team_1_ctde-cac.npz` | 280,110 | `a13dfbfdb55e015a85afa4a69449411c21f1f06c99c8ebeb3246e320650c8fea` |
| `data/campaign_replicates/no_dr/exp2_seed1/team_1_ctde-cac.pt` | 304,400 | `df5d5cfefc509477ecb7b71191c8bb457afaa7f3c40e6c4e2e66b4a3490c165c` |
| `data/campaign_replicates/no_dr/exp2_seed1/team_2_ctde-comm.npz` | 288,628 | `e9bd40617dd3b7dc990099558a81dd759871ad0e2d77b7f2366a3987c1eeb822` |
| `data/campaign_replicates/no_dr/exp2_seed1/team_2_ctde-comm.pt` | 313,778 | `564b544c004dad03765cc03d77507bcfe85242c48dfeb6514e03ca6e433559db` |
| `data/campaign_replicates/no_dr/exp2_seed1/team_3_ctde-vd.npz` | 199,380 | `4dde7f5a7679f716e9392aba4e07b107c4636e1a2ec1c3c76cc886fd185672c8` |
| `data/campaign_replicates/no_dr/exp2_seed1/team_3_ctde-vd.pt` | 217,330 | `432d2d0808e08c58e4b81ee0fb0b923f9d23870fb85bf7bda6b9ef315305af9d` |
| `data/campaign_replicates/no_dr/exp2_seed10/team_1_ctde-cac.npz` | 280,088 | `95d997b034ed363332681ae6b3304e97692b407378372ea9e47074a9f1b5a067` |
| `data/campaign_replicates/no_dr/exp2_seed10/team_1_ctde-cac.pt` | 304,400 | `fee0226dd6dfc5034c4f37db3da277969be5ba7a84f7f3bfd21a92934a06d23a` |
| `data/campaign_replicates/no_dr/exp2_seed10/team_2_ctde-comm.npz` | 288,893 | `c3e879f114a5c71ddd2f6b9835a10fc474a238e130bd9f0b62f2738d4fec20a1` |
| `data/campaign_replicates/no_dr/exp2_seed10/team_2_ctde-comm.pt` | 313,778 | `1754b346759f4fdfcab4b487a4b1546723f332177e8704b8e8fde8ae8943650e` |
| `data/campaign_replicates/no_dr/exp2_seed10/team_3_ctde-vd.npz` | 199,402 | `022d1ddaf92c4ab9886d41d67ac361b2c233ccc35185e3c8f0186b6a32bf6a2d` |
| `data/campaign_replicates/no_dr/exp2_seed10/team_3_ctde-vd.pt` | 217,330 | `bb13e97d7a74656234338b66db1d4eff36b9f7871a45a6e68eba30385029abc2` |
| `data/campaign_replicates/no_dr/exp2_seed2/team_1_ctde-comm.npz` | 288,885 | `773f76f8f4b7a5e00b6b133fcd9b0356505f9a9f9da32b25748cfd259691d7b3` |
| `data/campaign_replicates/no_dr/exp2_seed2/team_1_ctde-comm.pt` | 313,778 | `0d5c44dd9b6444ce5936aec67d03d08dab3eb098567567065c2ab339c4308685` |
| `data/campaign_replicates/no_dr/exp2_seed2/team_2_ctde-vd.npz` | 199,446 | `1bdf63771634ab39bb48b81dba45b64930d2804331b79003ad48f35fb051dbb2` |
| `data/campaign_replicates/no_dr/exp2_seed2/team_2_ctde-vd.pt` | 217,330 | `fa888e1f414b6d126bf02ad2031dd6724b712524e76408a27e41befa9b8fb8b5` |
| `data/campaign_replicates/no_dr/exp2_seed2/team_3_ctde-cac.npz` | 280,021 | `f08836cbeba5e8ae3f7ed4c8e19607aa37cdcb097f2b4716b66edc9db8ad37ec` |
| `data/campaign_replicates/no_dr/exp2_seed2/team_3_ctde-cac.pt` | 304,400 | `6b7b7ec27657fd381981c6418d9293560dd5bebe63a615205a1bdf5ece77d586` |
| `data/campaign_replicates/no_dr/exp2_seed3/team_1_ctde-vd.npz` | 199,668 | `fe5a68b21ab4c03f8f7175c9af26d589bf815bc6bca9075a8138433cd471180c` |
| `data/campaign_replicates/no_dr/exp2_seed3/team_1_ctde-vd.pt` | 217,330 | `ff1e848b2a9170991dd714da09b5025cc1260eff9fc5a2c00d74e6bb05bd3afc` |
| `data/campaign_replicates/no_dr/exp2_seed3/team_2_ctde-cac.npz` | 280,036 | `fb9f357fc160db45a34c7e3f22b50cf91fad83ead42aec027e51a8490cc61471` |
| `data/campaign_replicates/no_dr/exp2_seed3/team_2_ctde-cac.pt` | 304,400 | `a7636e6f9f4054fe437dbebfc6bf6b7b0880c0c2ec9edf6f66eaa989ccbd7349` |
| `data/campaign_replicates/no_dr/exp2_seed3/team_3_ctde-comm.npz` | 288,692 | `1bcec0d2f2443d748bc215ba466252917cfd6218b1c548a0335f48a4ae4e4283` |
| `data/campaign_replicates/no_dr/exp2_seed3/team_3_ctde-comm.pt` | 313,778 | `41b245d02ca4f27ab5b631e2676003718bd7bc5613576293119112f425769627` |
| `data/campaign_replicates/no_dr/exp2_seed4/team_1_ctde-cac.npz` | 279,745 | `9abef1104596f1a6b5c24f58a7e1e54ddb114118c9aea0a152054db5c1190790` |
| `data/campaign_replicates/no_dr/exp2_seed4/team_1_ctde-cac.pt` | 304,400 | `70bde01fbe0e938db312e757237e0654dd3eae32699cd66f6398ceed35175fd8` |
| `data/campaign_replicates/no_dr/exp2_seed4/team_2_ctde-comm.npz` | 288,884 | `fb92eb69fb42ca1b8dfbc8cc244e970ddbbed5a6577a2b5143dce3007b2ad213` |
| `data/campaign_replicates/no_dr/exp2_seed4/team_2_ctde-comm.pt` | 313,778 | `c78579fe46ffbabedb8bc4d8a553232a3965ea3e8f6ec36f9e350e8a89b07bdd` |
| `data/campaign_replicates/no_dr/exp2_seed4/team_3_ctde-vd.npz` | 199,538 | `5795547282ff0a7f9f934955b0c23dea5c46907ec2d28ffd2a2315666a51e73b` |
| `data/campaign_replicates/no_dr/exp2_seed4/team_3_ctde-vd.pt` | 217,330 | `2f8be6988743fe07d5c43970b16c410aba771994192aba37e9d529167a49500e` |
| `data/campaign_replicates/no_dr/exp2_seed5/team_1_ctde-comm.npz` | 288,700 | `938bf7210ab0f854bc3203ffe56b41eca93693cf402356de5102e479aa8b99d6` |
| `data/campaign_replicates/no_dr/exp2_seed5/team_1_ctde-comm.pt` | 313,778 | `b71dfaab6e517fa668083ebc622c1b4bba6e12a4af23fd400b951a31d4f5f510` |
| `data/campaign_replicates/no_dr/exp2_seed5/team_2_ctde-vd.npz` | 199,411 | `8c475c51d70379a78191b728a152f559b404c281c338934cbcdf9d0971c2a8e3` |
| `data/campaign_replicates/no_dr/exp2_seed5/team_2_ctde-vd.pt` | 217,330 | `c54e4d5e4538e8ef98d23bbd67090e25163916d6015683b65278008454e2151b` |
| `data/campaign_replicates/no_dr/exp2_seed5/team_3_ctde-cac.npz` | 280,119 | `222d67bd02972d6d55ea9c25233084a392b40bd8abb5c58022329e1a77726b27` |
| `data/campaign_replicates/no_dr/exp2_seed5/team_3_ctde-cac.pt` | 304,400 | `9b6e641d8195207cc49e1a6ddebace03fd09c87bc0daf3ad08a5f8fc254b4c9a` |
| `data/campaign_replicates/no_dr/exp2_seed6/team_1_ctde-vd.npz` | 199,631 | `8225705f1e9bfe9d0ae373caa81cf89fe62bcb672f7b1340d29e26149413da20` |
| `data/campaign_replicates/no_dr/exp2_seed6/team_1_ctde-vd.pt` | 217,330 | `8f9f1e4f01285883b0131f5b1444b899960b6bc077ea4fd888e12b6da604d913` |
| `data/campaign_replicates/no_dr/exp2_seed6/team_2_ctde-cac.npz` | 279,729 | `76da9b9a5c466efff947838085a4b8bd2f30ff84f6e652c965ff35bbb7af955e` |
| `data/campaign_replicates/no_dr/exp2_seed6/team_2_ctde-cac.pt` | 304,400 | `1b820be4ff6932ec2a2d63e5ecb34d705625978b8b3e4b4fda753603aad2e1e9` |
| `data/campaign_replicates/no_dr/exp2_seed6/team_3_ctde-comm.npz` | 288,664 | `447c3303d66de7c4b1bb0bbb7c7d179f235655550ecb28ce11605297d9e8d4b2` |
| `data/campaign_replicates/no_dr/exp2_seed6/team_3_ctde-comm.pt` | 313,778 | `0bcb750d63532324dd9034960b7000a1923a4848640ff0e057a4c3e4067331b6` |
| `data/campaign_replicates/no_dr/exp2_seed7/team_1_ctde-cac.npz` | 279,956 | `9e8defe32684964d7b9a5b21574f30e46937d2a922c3ee37dd9760e49cafe6fe` |
| `data/campaign_replicates/no_dr/exp2_seed7/team_1_ctde-cac.pt` | 304,400 | `415f9674cc109b33c980a4f3f1d8d0c9c24521e4313e1e279fbdc6fc1cc76d64` |
| `data/campaign_replicates/no_dr/exp2_seed7/team_2_ctde-comm.npz` | 288,735 | `4adb202dfa8a0e38f52bd9a3c955434196e9518f294a3b52d4752fde933ee0f3` |
| `data/campaign_replicates/no_dr/exp2_seed7/team_2_ctde-comm.pt` | 313,778 | `044a301ff94acf4af29b0e070feb3bdf191b3a4bc4d812c5649aa56e66e156ed` |
| `data/campaign_replicates/no_dr/exp2_seed7/team_3_ctde-vd.npz` | 199,450 | `16464568ae1c9afe788998f546408536cff12e73491e7c2f48159ed321ee0c69` |
| `data/campaign_replicates/no_dr/exp2_seed7/team_3_ctde-vd.pt` | 217,330 | `291196260ca39255319b12f35df084a2a50c2749fe076beb1a084243701dfdd6` |
| `data/campaign_replicates/no_dr/exp2_seed8/team_1_ctde-comm.npz` | 288,955 | `25316c04ac66a87b6db39eb7d33d6279afb87cc15e8ab4ca1089f367a0dda150` |
| `data/campaign_replicates/no_dr/exp2_seed8/team_1_ctde-comm.pt` | 313,778 | `e7764bc917957979c44b293eadfc788dfaa81310f8a3ba26a4455b3beae4a7d7` |
| `data/campaign_replicates/no_dr/exp2_seed8/team_2_ctde-vd.npz` | 199,137 | `a9ee805c2e8ecb8dc5233bf41bcecdde0b849eb797e98bf0f0711ec7d1f72398` |
| `data/campaign_replicates/no_dr/exp2_seed8/team_2_ctde-vd.pt` | 217,330 | `f627418fca163af8215a1263b5e3c49d1b4772d8330b23b7b3475d566b38fa5c` |
| `data/campaign_replicates/no_dr/exp2_seed8/team_3_ctde-cac.npz` | 279,942 | `5d8c26f0fad48f1f77c4ed34cdda34d23ced390ef72d8b5b571cd74bb2bbd89b` |
| `data/campaign_replicates/no_dr/exp2_seed8/team_3_ctde-cac.pt` | 304,400 | `cc647fe3b9ae5af3e85624f34809b9635c92d31d1fb8d5474720f2a3f2f79be9` |
| `data/campaign_replicates/no_dr/exp2_seed9/team_1_ctde-vd.npz` | 199,602 | `1c20557d5ebd154055288d81625ccd766eb7c1f38a0bfed347e5df2ca4027580` |
| `data/campaign_replicates/no_dr/exp2_seed9/team_1_ctde-vd.pt` | 217,330 | `18a6661512a4145d40138009f9fc9b0599675100dca532c764f0724dc642e654` |
| `data/campaign_replicates/no_dr/exp2_seed9/team_2_ctde-cac.npz` | 280,058 | `a79350cf24277a296dd1b3f7b71b7a7712797e7aace1e2dae53f8d57f72cd43b` |
| `data/campaign_replicates/no_dr/exp2_seed9/team_2_ctde-cac.pt` | 304,400 | `712fd1f2de85ad2022b2252fb7ae78aa6158986eb2867ef7fda7269c95d78ebd` |
| `data/campaign_replicates/no_dr/exp2_seed9/team_3_ctde-comm.npz` | 288,101 | `23566471d681f2600877b112e663de94f2b3ffca0784f9fd83bf2614bd6ed447` |
| `data/campaign_replicates/no_dr/exp2_seed9/team_3_ctde-comm.pt` | 313,778 | `0bd4cc857e4cd1ceb5105fafb61f973dd0baf812226d45803a3cee83abec8967` |
| `data/campaign_replicates/spawn_cycle/exp1_seed1/team_1_dte.npz` | 133,846 | `1ccb658477bbd07873ea7e8317708b65ad895dc4d6becb6d8155195a3cbef1af` |
| `data/campaign_replicates/spawn_cycle/exp1_seed1/team_2_ctde.npz` | 280,366 | `f1e36d8e0a8b88f852b614f3f3fa2b0ca9fb0cc01b7383311d7205c9f70fc89a` |
| `data/campaign_replicates/spawn_cycle/exp1_seed1/team_3_cte.npz` | 297,920 | `0c7079a0ee3e0ea2d8ad5ddef7cffc18f8f8adc81889ea6b828ade56eedc459d` |
| `data/campaign_replicates/spawn_cycle/exp1_seed10/team_1_dte.npz` | 133,805 | `32874991fcf525d94e88a6f5242eb70a5b125d190e43c7239adc83b07cfacbcf` |
| `data/campaign_replicates/spawn_cycle/exp1_seed10/team_2_ctde.npz` | 280,536 | `cbeb90099e8f79eb4b15c1a9545285c7e6fa12f7b9ef51899e4c973b40b2c24c` |
| `data/campaign_replicates/spawn_cycle/exp1_seed10/team_3_cte.npz` | 297,994 | `d69b3f70853031736e3856caf6dd4bdc4d5c3095b4d8d9f6259671584d468181` |
| `data/campaign_replicates/spawn_cycle/exp1_seed2/team_1_ctde.npz` | 280,549 | `ca3511e2e39935caeea41bacd15d3f800751100461120e33639a34a93e29eb25` |
| `data/campaign_replicates/spawn_cycle/exp1_seed2/team_2_cte.npz` | 297,999 | `f5c312c553f4300a73d48e8eba897cccc9bc9fa70a6f3b549e84b9eefc316374` |
| `data/campaign_replicates/spawn_cycle/exp1_seed2/team_3_dte.npz` | 133,742 | `da60bba171edda7b26bb9d442f55b13abe62f26526c7d482421f3a2bf8445577` |
| `data/campaign_replicates/spawn_cycle/exp1_seed3/team_1_cte.npz` | 297,824 | `c051c0a9aeea95120ce8dbcb68e89e81e7391a088875ba80a034f605644a171c` |
| `data/campaign_replicates/spawn_cycle/exp1_seed3/team_2_dte.npz` | 133,730 | `d0d8ac8e94cedde87b6dd075256ff2f828b32fc248ce3a20a9f59bafd09c80cf` |
| `data/campaign_replicates/spawn_cycle/exp1_seed3/team_3_ctde.npz` | 280,419 | `084910b47626817b8e03ae58da113b58dd9bff87582df2af7898e8142ad48323` |
| `data/campaign_replicates/spawn_cycle/exp1_seed4/team_1_dte.npz` | 133,773 | `a0e941fe0b50f63fed3e5639ff56050e3d508e3ca9de50bbbdff5b55a50be20f` |
| `data/campaign_replicates/spawn_cycle/exp1_seed4/team_2_ctde.npz` | 280,497 | `8d98cab3a36eb8431e76a18fa68d2b159ff8b933e622de9e8401c1feec3a98da` |
| `data/campaign_replicates/spawn_cycle/exp1_seed4/team_3_cte.npz` | 298,006 | `cd0539cf6d7b59a93c1fdf136521b3a0a2151bf58fb52e3b414b16bd7ad3dd4b` |
| `data/campaign_replicates/spawn_cycle/exp1_seed5/team_1_ctde.npz` | 280,543 | `56be41e7e042a5af4d6e11d1781002e508bea195165284768b7297f666c6467b` |
| `data/campaign_replicates/spawn_cycle/exp1_seed5/team_2_cte.npz` | 298,069 | `973853f44ce52f4608e78eed132236ca0aa68a154660b35fae484705b79e7a16` |
| `data/campaign_replicates/spawn_cycle/exp1_seed5/team_3_dte.npz` | 133,751 | `f0ddf66a2d398964aff286963d6bfa071dc8fde61b3021d4eccccda21582a42e` |
| `data/campaign_replicates/spawn_cycle/exp1_seed6/team_1_cte.npz` | 297,891 | `2d9c631a05334f58a4de6a451415c2e9dd57ad20e37738e7dc7cc705e0e7d5a4` |
| `data/campaign_replicates/spawn_cycle/exp1_seed6/team_2_dte.npz` | 133,669 | `844dc6cb28564be857e9d15412b8362ca549cf15d9cb9d66250b7f38bb2a4563` |
| `data/campaign_replicates/spawn_cycle/exp1_seed6/team_3_ctde.npz` | 280,516 | `8d979588291277cf2695d882474ce7828770c1dbcb0d9f7d67642d8fcb1d590b` |
| `data/campaign_replicates/spawn_cycle/exp1_seed7/team_1_dte.npz` | 133,723 | `ce2fb2120aa3a10602586b5158032f0a332e7626c83bb36565d8b1bcfa2611e1` |
| `data/campaign_replicates/spawn_cycle/exp1_seed7/team_2_ctde.npz` | 280,513 | `3a00b15d5acfdfb16f57ed4a9f76ae590237033609ac0b577aee90012341d3d4` |
| `data/campaign_replicates/spawn_cycle/exp1_seed7/team_3_cte.npz` | 297,944 | `00f0776590a8b4ddbc1600d12895ad8408603f5115a56fffcf4042b99bfd5eec` |
| `data/campaign_replicates/spawn_cycle/exp1_seed8/team_1_ctde.npz` | 280,513 | `8e361314c2fbdf8b34f1e478c39b60ec4a264adec6e105623c41978af6bfd4a1` |
| `data/campaign_replicates/spawn_cycle/exp1_seed8/team_2_cte.npz` | 297,962 | `1dc4e77b7cdd8180a0443370e6eb88016d46586f081b578f08c4dcb670bda428` |
| `data/campaign_replicates/spawn_cycle/exp1_seed8/team_3_dte.npz` | 133,753 | `6245242be4903a508e38e05e6e7e8541d8047e6f11690f06e2ef7b60fbf05785` |
| `data/campaign_replicates/spawn_cycle/exp1_seed9/team_1_cte.npz` | 297,973 | `4b63b95f0c9a94345b546e5d20e05e49a5dfad165c2af18adec1819426c2d865` |
| `data/campaign_replicates/spawn_cycle/exp1_seed9/team_2_dte.npz` | 133,773 | `398f021307100f4968fff3d30bf2ffa062125b76d751422e73da9bdc6d4d8d95` |
| `data/campaign_replicates/spawn_cycle/exp1_seed9/team_3_ctde.npz` | 280,362 | `92c4b457be2780a504b1a73b233fc1403b87f39e90796d29365883e096f68f4a` |
| `data/campaign_replicates/spawn_cycle/exp2_seed1/team_1_ctde-cac.npz` | 280,226 | `1ad678573669e0ff6c589d404174bfdac68c0efeb958fc06ac38ea576e5a6dd2` |
| `data/campaign_replicates/spawn_cycle/exp2_seed1/team_2_ctde-comm.npz` | 289,429 | `e069822a9adb6a4e3a63e1f8191fb3e0193b73979a1295c15d14aa06360e4047` |
| `data/campaign_replicates/spawn_cycle/exp2_seed1/team_3_ctde-vd.npz` | 199,563 | `5e1f9e567a8db5ffbc4833f15fd319a869bf5d505da205443a54ba3742a35545` |
| `data/campaign_replicates/spawn_cycle/exp2_seed10/team_1_ctde-cac.npz` | 280,347 | `a82f7389c211f9b0b32668f9aa03f88aa539aa0251dfb4eb02bb9b1ce672e380` |
| `data/campaign_replicates/spawn_cycle/exp2_seed10/team_2_ctde-comm.npz` | 289,380 | `108ac083beed3d9039b62f000d520340b79e6ceba852373c47654f000d9e33ae` |
| `data/campaign_replicates/spawn_cycle/exp2_seed10/team_3_ctde-vd.npz` | 199,664 | `20f3e78494f50b15d646676ed5173b91d611a97042018c711ea928d2b4a3847d` |
| `data/campaign_replicates/spawn_cycle/exp2_seed2/team_1_ctde-comm.npz` | 289,372 | `bc0493a15ade31c40a5cd6592d8a74530e68987aabb6ae040d7ec658b830b8ff` |
| `data/campaign_replicates/spawn_cycle/exp2_seed2/team_2_ctde-vd.npz` | 199,637 | `c44879ac68b27e61d1d97fce23ef5f59ab30c4956959b33103f4e99ccb53a15b` |
| `data/campaign_replicates/spawn_cycle/exp2_seed2/team_3_ctde-cac.npz` | 280,365 | `02d6e025eeeebf57b43267bc0a1207a816177d9e19392df80e34ffe8d6072368` |
| `data/campaign_replicates/spawn_cycle/exp2_seed3/team_1_ctde-vd.npz` | 199,532 | `3bdc467451cbc81ed6ea5b754e2008c532801eef0680c0376a3f7a3eca7b0377` |
| `data/campaign_replicates/spawn_cycle/exp2_seed3/team_2_ctde-cac.npz` | 280,489 | `1ac67ac770218b340db1bec504cc209d8615c9029848e9748074bd8f862a88b1` |
| `data/campaign_replicates/spawn_cycle/exp2_seed3/team_3_ctde-comm.npz` | 289,318 | `ffc3967cc92cf43d7347208d22666bcc93c879f34509090b56deab3ec1188648` |
| `data/campaign_replicates/spawn_cycle/exp2_seed4/team_1_ctde-cac.npz` | 280,457 | `2a3ec371c5db813ec4aa33b9e37ce64959c48ba4c773457c143b414ff44d27be` |
| `data/campaign_replicates/spawn_cycle/exp2_seed4/team_2_ctde-comm.npz` | 289,246 | `eec8c7d9bba7641e50829bb791e4c1954ddb8b7dec81b7aeda62a8a6dd19a973` |
| `data/campaign_replicates/spawn_cycle/exp2_seed4/team_3_ctde-vd.npz` | 199,625 | `3e5a1ea073cad23fc14edd74b750bc5d2261c3df24d5fce5de0b050162b673a8` |
| `data/campaign_replicates/spawn_cycle/exp2_seed5/team_1_ctde-comm.npz` | 289,375 | `d87b6155ac66f36b835035dc1c575e7df11de563a3ce266b40a94208d9a79f7f` |
| `data/campaign_replicates/spawn_cycle/exp2_seed5/team_2_ctde-vd.npz` | 199,597 | `86513272df5e39a60a92076376a78790b75f198ce2daa423066aa0226ca07376` |
| `data/campaign_replicates/spawn_cycle/exp2_seed5/team_3_ctde-cac.npz` | 280,329 | `167d88563d5f8c113f25d5a4addc34d47570ede3ca8a918d60feff5614f023a4` |
| `data/campaign_replicates/spawn_cycle/exp2_seed6/team_1_ctde-vd.npz` | 199,487 | `cad827ad7144c9c44280982d9bcdb45feb3ca3267179398d1433cacbfa4b056e` |
| `data/campaign_replicates/spawn_cycle/exp2_seed6/team_2_ctde-cac.npz` | 280,204 | `a009d60b1f6aa201f5644782ee9fe43ea80cf07cd0e6f9520bbc62a1413971e4` |
| `data/campaign_replicates/spawn_cycle/exp2_seed6/team_3_ctde-comm.npz` | 289,329 | `e9ffa8354125d056ec9fa450508a0354c585784e56d18ea209022d0c108112f1` |
| `data/campaign_replicates/spawn_cycle/exp2_seed7/team_1_ctde-cac.npz` | 280,247 | `7f0aa7c75578d96657e7e5690b963089d0f9ab48adbfa8b2674dbc17459bd0d2` |
| `data/campaign_replicates/spawn_cycle/exp2_seed7/team_2_ctde-comm.npz` | 289,137 | `6b3ecf898755fd1f0fdabe36470960500880775d9451059e67ecfb2156f9ccad` |
| `data/campaign_replicates/spawn_cycle/exp2_seed7/team_3_ctde-vd.npz` | 199,538 | `9bc8cff97effe37843ea6b6886862028c68341259a99d5bc36743431a49db0da` |
| `data/campaign_replicates/spawn_cycle/exp2_seed8/team_1_ctde-comm.npz` | 289,365 | `04cb608105dad9671128e011b5e50f835a8640c08da8eef07d55330a86c15a8d` |
| `data/campaign_replicates/spawn_cycle/exp2_seed8/team_2_ctde-vd.npz` | 199,473 | `6b86f94e6bf05256bf334cb7472c0b2e6895864362d2d8a2b170c82c44c17102` |
| `data/campaign_replicates/spawn_cycle/exp2_seed8/team_3_ctde-cac.npz` | 280,342 | `630c0eb979ddb221780d57bd96ba27968410d4dff70a427bea87029336a6a6e7` |
| `data/campaign_replicates/spawn_cycle/exp2_seed9/team_1_ctde-vd.npz` | 199,593 | `114dbb05be82b7f514454aa51f09d42fdc717b9a78e4f314129e17567146f611` |
| `data/campaign_replicates/spawn_cycle/exp2_seed9/team_2_ctde-cac.npz` | 280,651 | `48d7ff2328bcd1c9fdc330a22374f764325e88431afcabf5ce0d62b3c2659ad5` |
| `data/campaign_replicates/spawn_cycle/exp2_seed9/team_3_ctde-comm.npz` | 289,243 | `a18beb9bffe1ba85113107040cac13f1e9432570803dd327f3c915a766db510c` |
| `data/campaign_replicates/symmetric/exp1_seed1/team_1_dte.npz` | 133,784 | `b08abc59f7b9a4d478e0e7a781cc750da8e02dfde4cfa59079432c041a06cd71` |
| `data/campaign_replicates/symmetric/exp1_seed1/team_2_ctde.npz` | 280,467 | `dce2d0081217273bc6793d7b83f74f94f9fe4fcdb9c49f6ba27d332ce4bdb547` |
| `data/campaign_replicates/symmetric/exp1_seed1/team_3_cte.npz` | 297,980 | `46db8b0134fe947018c0be67442c3b1e33c2abb84595f0f54e6b68e41d28a308` |
| `data/campaign_replicates/symmetric/exp1_seed10/team_1_dte.npz` | 133,781 | `55b47389ee0ec2fa7a6bd54ce58eda4425c57f5078b79d3a53121863015214a9` |
| `data/campaign_replicates/symmetric/exp1_seed10/team_2_ctde.npz` | 280,578 | `db977d69b16437d66984cd2b0230d8313770a81a9a532b9b25eab72c1ee4c420` |
| `data/campaign_replicates/symmetric/exp1_seed10/team_3_cte.npz` | 297,949 | `33e2abf55dd0a48d481e4e21719cfb985e0fb4c82e21ca42a5ee2c9dc4540318` |
| `data/campaign_replicates/symmetric/exp1_seed2/team_1_ctde.npz` | 280,490 | `e255609c3cf8aa73c770f2d2f101773b8fef35007bb6c34952353fcbcfb9ca39` |
| `data/campaign_replicates/symmetric/exp1_seed2/team_2_cte.npz` | 297,823 | `2f9e995f70c17ed48fcfb2323ca8facd6fe61eca839d7efede76303587c4500e` |
| `data/campaign_replicates/symmetric/exp1_seed2/team_3_dte.npz` | 133,759 | `0e8f7df3e2735bdcdeeb74404eb4c3464a97a84f2d18d8c05999e1623b43acc5` |
| `data/campaign_replicates/symmetric/exp1_seed3/team_1_cte.npz` | 297,986 | `d1a46c05e16d59f1492b18ce7af8f55c5026f301a103f171b040a2c174092e87` |
| `data/campaign_replicates/symmetric/exp1_seed3/team_2_dte.npz` | 133,744 | `7e684bfed885f281aa9a944c37ce81274a860bc1ed13b39e44f944d29fc300c2` |
| `data/campaign_replicates/symmetric/exp1_seed3/team_3_ctde.npz` | 280,534 | `e02d37fe4184db092c0c480c0960e360a89b4484f39eec59f85c68d8e9a46f0e` |
| `data/campaign_replicates/symmetric/exp1_seed4/team_1_dte.npz` | 133,757 | `75412928725dfb0d93a51fd7913368b3917acfc1ff49e9c5e4ffe122ed17a804` |
| `data/campaign_replicates/symmetric/exp1_seed4/team_2_ctde.npz` | 280,622 | `ccb80e9731dfc5c49aa942923950eec44492ec93daef6ddc9f66b68604bd8c16` |
| `data/campaign_replicates/symmetric/exp1_seed4/team_3_cte.npz` | 297,907 | `9462d642feddd1963a34f7cf289c4db31e56c6876eb9c7d01e82d080f6b86487` |
| `data/campaign_replicates/symmetric/exp1_seed5/team_1_ctde.npz` | 280,347 | `43c38df96d6ab9beac495e5deacc1268bf6ea54039b267c2a4a3b33853165774` |
| `data/campaign_replicates/symmetric/exp1_seed5/team_2_cte.npz` | 297,913 | `2b46bc746195e7bd30298948e299ba3a0d7db13c2e5952b9bbb477aaab1bb5ac` |
| `data/campaign_replicates/symmetric/exp1_seed5/team_3_dte.npz` | 133,766 | `d1b4aaf372f91a6d12f247139e86f6bb771a06b21758d7a0953068382120a830` |
| `data/campaign_replicates/symmetric/exp1_seed6/team_1_cte.npz` | 297,871 | `f163ffb38b0d3d002e38c766aa1e3c1780ae8f4810260b4958afeaedc167712b` |
| `data/campaign_replicates/symmetric/exp1_seed6/team_2_dte.npz` | 133,857 | `ddc1b203b511cc04235614453fad110c12b9f8f34b953529e2aa82d313110d72` |
| `data/campaign_replicates/symmetric/exp1_seed6/team_3_ctde.npz` | 280,348 | `f01802853cc709b0ef990b956866db3e58125febaa15efc204f9ad7877c6e035` |
| `data/campaign_replicates/symmetric/exp1_seed7/team_1_dte.npz` | 133,777 | `2968fd5c00f139a80c86c3a5e0c4f7b4f9f81e0b5a184dd3d92238093f74b74c` |
| `data/campaign_replicates/symmetric/exp1_seed7/team_2_ctde.npz` | 280,354 | `16f8947d63fc65d7486ab5b828183b4256e8231f2a54fffa8fb8b013487cb573` |
| `data/campaign_replicates/symmetric/exp1_seed7/team_3_cte.npz` | 298,034 | `2474f57e8aeeccdd9f028d24362f6a30b4c9465c95a6e36b4e9667f62cfef6e7` |
| `data/campaign_replicates/symmetric/exp1_seed8/team_1_ctde.npz` | 280,428 | `931929a85d82d830755966452d425d6a5c1e6efbf9bf656c89db5f5b7ec43909` |
| `data/campaign_replicates/symmetric/exp1_seed8/team_2_cte.npz` | 298,005 | `f779bb9904dd5ccb347a9d9a3cf4274f7c30dd59cbd29b3f2e55754ae1b9531f` |
| `data/campaign_replicates/symmetric/exp1_seed8/team_3_dte.npz` | 133,776 | `53310ba39390d395aca3593673f40136ab4399b37582177de8295049d8e86df8` |
| `data/campaign_replicates/symmetric/exp1_seed9/team_1_cte.npz` | 297,947 | `00655b47eb5d557cff119e711e17a02127ecffdc8d55bd71b7036f58334f00c9` |
| `data/campaign_replicates/symmetric/exp1_seed9/team_2_dte.npz` | 133,758 | `aca749443be84953b553ba172ebf614d3cf3e553079ed4a3938ab81312487758` |
| `data/campaign_replicates/symmetric/exp1_seed9/team_3_ctde.npz` | 280,612 | `41f75adb908db629c59a61fb6d21a018af904b5204584b24fb7a12c75315bafc` |
| `data/campaign_replicates/symmetric/exp2_seed1/team_1_ctde-cac.npz` | 280,360 | `2852cbf2d0b2b9750e638b33fa591995da029fb4dc9c4315033b56b834932066` |
| `data/campaign_replicates/symmetric/exp2_seed1/team_2_ctde-comm.npz` | 289,401 | `85681c8cce57d48337ff8f3328afad53a224ea529fd291087d6188a738a6c43b` |
| `data/campaign_replicates/symmetric/exp2_seed1/team_3_ctde-vd.npz` | 199,649 | `49f18348bec946c68afeeb60c9ecca7b751189f44686143c41689ab83c1ac3f1` |
| `data/campaign_replicates/symmetric/exp2_seed10/team_1_ctde-cac.npz` | 280,378 | `15cb95a1cec893ab7e1d2a697fab88325f6fd5d3933ae1dfe50c0de9a8ed00db` |
| `data/campaign_replicates/symmetric/exp2_seed10/team_2_ctde-comm.npz` | 289,467 | `91b15853f64ca2d98fee264ae628649db2b8b2a3d476a60b3abfabf4a61897f1` |
| `data/campaign_replicates/symmetric/exp2_seed10/team_3_ctde-vd.npz` | 199,663 | `83f99e969e8fab833f0f978850c4da9975e12d4df014397300e2f55303dbf0f8` |
| `data/campaign_replicates/symmetric/exp2_seed2/team_1_ctde-comm.npz` | 289,282 | `ad7246bbd8dc8b46c9f60cdab7187daf37a144ebe8d692afe0f87f6bc98a56d0` |
| `data/campaign_replicates/symmetric/exp2_seed2/team_2_ctde-vd.npz` | 199,536 | `784b35828e667823399d8d2dbf0bc01324c2aa341372481109dbb6af2275cd4a` |
| `data/campaign_replicates/symmetric/exp2_seed2/team_3_ctde-cac.npz` | 280,381 | `3322293536f9c42a3001c6b57761460d60106a2baeb5cb54168d9646e8c4fdf2` |
| `data/campaign_replicates/symmetric/exp2_seed3/team_1_ctde-vd.npz` | 199,500 | `f2c8eb75ae77acd2001942aa296bf70d86cc8be726d7240301109e33a864c4c6` |
| `data/campaign_replicates/symmetric/exp2_seed3/team_2_ctde-cac.npz` | 280,586 | `59c10fd5c329b57b73e15d1013526cc0c35a970cce0d1f2c879955e9ae4edac5` |
| `data/campaign_replicates/symmetric/exp2_seed3/team_3_ctde-comm.npz` | 289,252 | `9a1bc3da04a25b64fc51927aba78a66b8e5a779a1c8df3211cb0455e9bec79fd` |
| `data/campaign_replicates/symmetric/exp2_seed4/team_1_ctde-cac.npz` | 280,316 | `eaccbe62759f76a5f6e2f813924621cfd2123ef0957a997a5d66f7ab404d4e3f` |
| `data/campaign_replicates/symmetric/exp2_seed4/team_2_ctde-comm.npz` | 289,330 | `86c377a5926cd55dd8bdc807deb089e28915a0004e0f4a56fa17880baf5f7f5c` |
| `data/campaign_replicates/symmetric/exp2_seed4/team_3_ctde-vd.npz` | 199,575 | `9cacc24759b0c0165dfa7c4d536f7189a0c5ab2a89bb63b029065661006e9b19` |
| `data/campaign_replicates/symmetric/exp2_seed5/team_1_ctde-comm.npz` | 289,280 | `be0c3370b3c437a1ca5b327badf468819ccf5992833325dce1ed73195847a349` |
| `data/campaign_replicates/symmetric/exp2_seed5/team_2_ctde-vd.npz` | 199,374 | `bad9f2097c9cfecf57c104f3ac71b5003fd26d65964d56945a011eece1718946` |
| `data/campaign_replicates/symmetric/exp2_seed5/team_3_ctde-cac.npz` | 280,414 | `7d46c6865deab4898f1e0fb67f032a705caace922e9d38e6ae54b4c9f4ba5519` |
| `data/campaign_replicates/symmetric/exp2_seed6/team_1_ctde-vd.npz` | 199,561 | `022381eec647dd7e010581f20cf260bd1950f57bb86eaee5a56cc7b9eaaedbdc` |
| `data/campaign_replicates/symmetric/exp2_seed6/team_2_ctde-cac.npz` | 280,344 | `c786c2a3b670600a01ec2b9171d3a49f22fef70b3783207f1e98c2befbefec85` |
| `data/campaign_replicates/symmetric/exp2_seed6/team_3_ctde-comm.npz` | 289,448 | `d016e779bfd5d64b91e6b61848bd16eafd9ed0fba74d3b48abed0bc428823436` |
| `data/campaign_replicates/symmetric/exp2_seed7/team_1_ctde-cac.npz` | 280,308 | `dfeeca20a15e9506fd8ca5dc53998b35cd180d6b98eee4028cbcacea9ec13321` |
| `data/campaign_replicates/symmetric/exp2_seed7/team_2_ctde-comm.npz` | 289,401 | `98186454aed5c009fc84bb2d1960c1a5dcc9f36a6bb0dacf34b64f452e45570f` |
| `data/campaign_replicates/symmetric/exp2_seed7/team_3_ctde-vd.npz` | 199,620 | `eb2928298c690cbfee18baf8d084de22ecdf9f2732a6a00bc36fc2fd5cef3685` |
| `data/campaign_replicates/symmetric/exp2_seed8/team_1_ctde-comm.npz` | 289,269 | `dddd5daf257a6c2c2a6993f5a9b048a485217cc524247737bfdad1fea257b0b5` |
| `data/campaign_replicates/symmetric/exp2_seed8/team_2_ctde-vd.npz` | 199,448 | `9d055373630e24a463f2875cb5b3b2e895985c3ec900c197089695173e03a4f3` |
| `data/campaign_replicates/symmetric/exp2_seed8/team_3_ctde-cac.npz` | 280,146 | `987d7d580512db7784eaef88ae5325b8d7e903a47d3096729f7c42211b8bd21a` |
| `data/campaign_replicates/symmetric/exp2_seed9/team_1_ctde-vd.npz` | 199,541 | `efd9efbe2be83437a96a8b4bfb7f2fba9647d3e2807befcb7dc864c3773c6598` |
| `data/campaign_replicates/symmetric/exp2_seed9/team_2_ctde-cac.npz` | 280,460 | `a87a626d27672f43b92d3fbf31a634fcbf93aeff3fc7f64386059edecc45996c` |
| `data/campaign_replicates/symmetric/exp2_seed9/team_3_ctde-comm.npz` | 289,252 | `73176c7426818e86442d2995dd480c7cf63992df6b9611d395d5e819a0171e40` |
| `data/campaign_replicates/v3/exp1_seed1/team_1_dte.npz` | 402,190 | `e08bcb3ceae0439c7aabb06905beabc990f17b41f98b382bb6f3fec11c0a8a06` |
| `data/campaign_replicates/v3/exp1_seed1/team_2_ctde.npz` | 402,535 | `99e851cf44daa678929ad1dbea2501552205634f537b9cb670a4bdb681a0e43c` |
| `data/campaign_replicates/v3/exp1_seed1/team_3_cte.npz` | 405,852 | `12ae8d762df4f5ec3307a2b6143adecd5e9096e6f96911d8d451673e0955c837` |
| `data/campaign_replicates/v3/exp1_seed10/team_1_dte.npz` | 402,099 | `1f79f0885e06018a82e1b5f09f92eecb2ede274e2070984f3d75f89b898d55ee` |
| `data/campaign_replicates/v3/exp1_seed10/team_2_ctde.npz` | 402,536 | `1f0061a16980ae1476e7108e1952569f4e99cdd07c2bf4b0009edcc36ab0cf8e` |
| `data/campaign_replicates/v3/exp1_seed10/team_3_cte.npz` | 405,876 | `ab032be9e53b54ba1e54c81a3ac99bae78bbaf85992127ee48a5db50676d209a` |
| `data/campaign_replicates/v3/exp1_seed2/team_1_ctde.npz` | 402,397 | `643f8691f8ec3ce0d2abd76ec5552e9350f62ab459dcde17c701ef5f57d47453` |
| `data/campaign_replicates/v3/exp1_seed2/team_2_cte.npz` | 405,923 | `88fb105853ae917c451356a67701fc1b0fa15b0fb11fe1e1909fe47ac559545e` |
| `data/campaign_replicates/v3/exp1_seed2/team_3_dte.npz` | 402,164 | `83cdbd9d45130895351553ae9d91c6a6f5e8b88989161176f19f2a2dcfd5cf0a` |
| `data/campaign_replicates/v3/exp1_seed3/team_1_cte.npz` | 405,809 | `2346e54fe33a3e2aeb8cf76c334412f4f5092197ff7dee07feb403966de86a52` |
| `data/campaign_replicates/v3/exp1_seed3/team_2_dte.npz` | 401,892 | `0547212bc8ead8d0ec0528ef67c436e9c0b3e34fe65d8861f1a2d7398244be00` |
| `data/campaign_replicates/v3/exp1_seed3/team_3_ctde.npz` | 402,570 | `a2fe43fec594bf4ae46645946d431c985113e7c77debfc5b16773de7e051cfcb` |
| `data/campaign_replicates/v3/exp1_seed4/team_1_dte.npz` | 402,311 | `db3c990350efdf2518182db91e77690ebfd9b8ca81407ee91b66388dc3aac6d3` |
| `data/campaign_replicates/v3/exp1_seed4/team_2_ctde.npz` | 402,520 | `72734cc2c30b8331826551120002cf854a4fd16bcaa0819d8aa9a0eba227c58e` |
| `data/campaign_replicates/v3/exp1_seed4/team_3_cte.npz` | 405,968 | `8846ac40da067f82983d1c01c0c19978cd12b41e28ff3d23de73483504776d14` |
| `data/campaign_replicates/v3/exp1_seed5/team_1_ctde.npz` | 402,584 | `c12fc22b9e403e4adcf65cb5f749bca8c73c6890db228eb0c0ab8ee680225e0b` |
| `data/campaign_replicates/v3/exp1_seed5/team_2_cte.npz` | 405,898 | `369f6b3dc71cc0fefcafae65d4b4fe06eb1ead154af10e5e430df688f7841a41` |
| `data/campaign_replicates/v3/exp1_seed5/team_3_dte.npz` | 402,104 | `2f0fffb2845ee4fa904857e194476868352e8a37e99bf7f4c5422992af879af6` |
| `data/campaign_replicates/v3/exp1_seed6/team_1_cte.npz` | 405,994 | `46b28377cab2618dd016881e2aae5b8ced4f6fe1e7db5fc85068838361d91926` |
| `data/campaign_replicates/v3/exp1_seed6/team_2_dte.npz` | 402,133 | `98796469253a2bcc4ba8faebbf321a29defb1c2e20c20be2cfe2858ac7510b68` |
| `data/campaign_replicates/v3/exp1_seed6/team_3_ctde.npz` | 402,918 | `88d303bd6a6a6456033daf77be84cc6cdd3300939676489a8557ded03d0c5756` |
| `data/campaign_replicates/v3/exp1_seed7/team_1_dte.npz` | 402,155 | `b8330d6177aab6cb14111351db60b7ee92b4ba195e7dab5a5837048ce38de928` |
| `data/campaign_replicates/v3/exp1_seed7/team_2_ctde.npz` | 402,669 | `af52382e6f3b4ecd32e677446f59fedcd1dc061b8143e23cec231ed1b81f9930` |
| `data/campaign_replicates/v3/exp1_seed7/team_3_cte.npz` | 405,914 | `94d50d7ffc62ca63f45fbdcd7fe76eb87d49083c315252fa2d1a6e6435fa4c27` |
| `data/campaign_replicates/v3/exp1_seed8/team_1_ctde.npz` | 402,472 | `3afad9baff0bbade9e256e0556647193ee1983fe2043622aa5ba82f0c45ec0e2` |
| `data/campaign_replicates/v3/exp1_seed8/team_2_cte.npz` | 405,923 | `7a0b49be11f70c95902bcd79a6ef9e0a7c31f197de3cf8e91e5abd74b3028622` |
| `data/campaign_replicates/v3/exp1_seed8/team_3_dte.npz` | 402,156 | `2e023ac27d6b44b587c42b4572c6e66a08ae956e8707e179b6a7fdcea5f4518b` |
| `data/campaign_replicates/v3/exp1_seed9/team_1_cte.npz` | 405,826 | `6f77150c34f9a9d4e98dd71536150eac3ebbeb40abb8b662e0bee4bc94f057f8` |
| `data/campaign_replicates/v3/exp1_seed9/team_2_dte.npz` | 402,168 | `d927c1352e2b1ead0a32a8bdec8d52e98fb3f6b3571b38041fbf86f2b3913c30` |
| `data/campaign_replicates/v3/exp1_seed9/team_3_ctde.npz` | 402,667 | `48e377f2d98f96f5b09047a6270907eef97bd988bfc9b92f78ed90156bba4837` |
| `data/campaign_replicates/v3/exp2_seed1/team_1_ctde-cac.npz` | 402,635 | `528a76a408f33094fc15c4c55e465b951ced73c559816224ee5341998ab3e951` |
| `data/campaign_replicates/v3/exp2_seed1/team_2_ctde-comm.npz` | 407,143 | `37e60cc10d5c77393116c1b9eba5051081f752238634358aa5cf0a9cc3604cc9` |
| `data/campaign_replicates/v3/exp2_seed1/team_3_ctde-vd.npz` | 406,056 | `727309efc8c4ea8c43c80d83e9d1800c4f9bd88d78fa5340f1082d85441c37d5` |
| `data/campaign_replicates/v3/exp2_seed10/team_1_ctde-cac.npz` | 402,574 | `5b8cdaf4d5889c9a18d1f22438f27a6b4445cad09ec8b440ef91d5fa8e00df62` |
| `data/campaign_replicates/v3/exp2_seed10/team_2_ctde-comm.npz` | 407,268 | `f6b881b7da893e4a2db0b110c6625a2eba7b27267fcd6b8258a14aae2e523ce4` |
| `data/campaign_replicates/v3/exp2_seed10/team_3_ctde-vd.npz` | 405,879 | `b7b82aeecc715058e564abad3ae329c577cd15a3628d91f178e43efd0ccb98fd` |
| `data/campaign_replicates/v3/exp2_seed2/team_1_ctde-comm.npz` | 406,968 | `0e1724bb81a358a9ed02ba24deb59da2992719a87eb5c25535c5bcec469299f5` |
| `data/campaign_replicates/v3/exp2_seed2/team_2_ctde-vd.npz` | 405,585 | `f3c414e1990f1ad14b900c39cabd780725a7d1edcc0219eec767a4b8689110ee` |
| `data/campaign_replicates/v3/exp2_seed2/team_3_ctde-cac.npz` | 402,563 | `4e1e3120974b0a117d98d840dc538133874c35e9d94ad4ba1d43473eb363eb6d` |
| `data/campaign_replicates/v3/exp2_seed3/team_1_ctde-vd.npz` | 406,075 | `954be396a20dd5e72d616bbd2032d15c5b9ab5310a64eefa42fe24d3f95231cd` |
| `data/campaign_replicates/v3/exp2_seed3/team_2_ctde-cac.npz` | 402,605 | `964839906194f6e8181f92db680178cc4ba13b4a075af8896766f31669d60943` |
| `data/campaign_replicates/v3/exp2_seed3/team_3_ctde-comm.npz` | 407,213 | `9013b8fb4ddde0bfd974c44a6e2e236dfa02e2d77f3e7145bec81a094eeb0054` |
| `data/campaign_replicates/v3/exp2_seed4/team_1_ctde-cac.npz` | 402,556 | `1ccb3d9568f835dbb2db09677744a67562241474991e1bad36e0efc61f1ec0ef` |
| `data/campaign_replicates/v3/exp2_seed4/team_2_ctde-comm.npz` | 406,967 | `bda8d335ed16461ae204f4cb0f982ae2d2c2c62de2c6577874ce367b511d6675` |
| `data/campaign_replicates/v3/exp2_seed4/team_3_ctde-vd.npz` | 405,756 | `eafd07639308bd5e0f351b8cd6c042f1f1a0c9140e76eb6558345e3a1985d8bb` |
| `data/campaign_replicates/v3/exp2_seed5/team_1_ctde-comm.npz` | 407,269 | `7533f7602e34b07bb58b0dfd3c5fb0420aab8c3254bfd2d4e5a80f2f230a6d08` |
| `data/campaign_replicates/v3/exp2_seed5/team_2_ctde-vd.npz` | 405,745 | `6a09eab6159ce85e5b7b48c8260151e9157915c46d39450e3962e35b99d95d53` |
| `data/campaign_replicates/v3/exp2_seed5/team_3_ctde-cac.npz` | 402,827 | `887c377c3a25810e7504024b5305142f52a92441fdbb376127f92e03eb0a0d42` |
| `data/campaign_replicates/v3/exp2_seed6/team_1_ctde-vd.npz` | 405,802 | `901a20e0cfc8f446fea9f5afefe1fbec2e385bc501979021cdb4177150add48a` |
| `data/campaign_replicates/v3/exp2_seed6/team_2_ctde-cac.npz` | 402,387 | `d611f2039ab347ca1aa0a0eaa76c539ff919e6aa0f5ea6e7d2bfb82c14731847` |
| `data/campaign_replicates/v3/exp2_seed6/team_3_ctde-comm.npz` | 406,974 | `bd0a98385b215323f6c5272d341f3d1e2194af8ee82b42063e92745c93dd444a` |
| `data/campaign_replicates/v3/exp2_seed7/team_1_ctde-cac.npz` | 402,691 | `a31e8b7f6f01fe73fc4e021ca0ef529473585503e63ac26aa860228e0b345614` |
| `data/campaign_replicates/v3/exp2_seed7/team_2_ctde-comm.npz` | 407,268 | `9af80e63a980851ea9bb30d70f93c45519cd4f90f685ae585a410b7505ee67d1` |
| `data/campaign_replicates/v3/exp2_seed7/team_3_ctde-vd.npz` | 405,928 | `5ef230b1eb0505216339fadcadfaddd30050614892da522986ef9de67a30eaa4` |
| `data/campaign_replicates/v3/exp2_seed8/team_1_ctde-comm.npz` | 407,086 | `6e490cc8b7606d3efddc25de47467ae991309bd6bf5c2614f4b7adb2a4bcffcd` |
| `data/campaign_replicates/v3/exp2_seed8/team_2_ctde-vd.npz` | 405,857 | `51f2a0bf987d2be35be88ca9360d7a554f4ed14ed46e7df00af5a48608188e78` |
| `data/campaign_replicates/v3/exp2_seed8/team_3_ctde-cac.npz` | 402,587 | `4eec5ff7a075cd29379d0128e3edef34fc787fae818e04db55eb1fa0d3fba289` |
| `data/campaign_replicates/v3/exp2_seed9/team_1_ctde-vd.npz` | 405,915 | `aac8234be5e5a4d54ae27331bd897e4d8514e462abe014ecb8adeed00d29f218` |
| `data/campaign_replicates/v3/exp2_seed9/team_2_ctde-cac.npz` | 402,831 | `5b4cce3a7d05e8b7e929c49c743d6298321d80ee077ccc736f6a9d1f859a99fc` |
| `data/campaign_replicates/v3/exp2_seed9/team_3_ctde-comm.npz` | 407,172 | `d07c6311cf4f9382f10933e59a188ed1d579f794f0da8c3c9e7983e481cbad49` |
| `data/campaign_replicates/wide_256/exp1_seed1/team_1_dte.npz` | 508,259 | `9eae088d0e047b3f1007a12e9f981125277e9ba7d3495a916e752fb3358dd133` |
| `data/campaign_replicates/wide_256/exp1_seed1/team_1_dte.pt` | 549,730 | `4b9fd74f1cfb3b7c01cdecc34f4c568d355f49229cd51cdcfdb4b419439e0805` |
| `data/campaign_replicates/wide_256/exp1_seed1/team_2_ctde.npz` | 1,042,192 | `468308cb80fc87d5d03d85f0cc8e6a29c69884265022581d4e67025d7d28ac7b` |
| `data/campaign_replicates/wide_256/exp1_seed1/team_2_ctde.pt` | 1,126,584 | `fd0ba963dddc738a585a75b65f2f6908e4af0e38b1e1c70c22eb676febccfb39` |
| `data/campaign_replicates/wide_256/exp1_seed1/team_3_cte.npz` | 1,077,647 | `ec88e0389e77003a68c73a48a7e09f9093b266c3593b209660920810fa84c711` |
| `data/campaign_replicates/wide_256/exp1_seed1/team_3_cte.pt` | 1,164,898 | `c91c63842ad3037827e8371994881cf3a5f93d68ad4cab6ac1f00790baebbc3f` |
| `data/campaign_replicates/wide_256/exp1_seed2/team_1_ctde.npz` | 1,042,437 | `ee9b3552c8d5f33aa14d9df501c5adb4460c6bc2b44bddfe9c9169cad4359a57` |
| `data/campaign_replicates/wide_256/exp1_seed2/team_1_ctde.pt` | 1,126,584 | `36cbc432a43b59f324f3044afdcb66e1bd292a92deef5765a6c6e196517df3f9` |
| `data/campaign_replicates/wide_256/exp1_seed2/team_2_cte.npz` | 1,077,898 | `a883171d47556024aae03b35a42207a5143db70c7dc1a09798163de6d7a10475` |
| `data/campaign_replicates/wide_256/exp1_seed2/team_2_cte.pt` | 1,164,898 | `dd43339e8215491461cc49d6c1816c1fea74fc77e5d01099f361334a623d5eda` |
| `data/campaign_replicates/wide_256/exp1_seed2/team_3_dte.npz` | 508,249 | `77b2a40a6afba66b349fec07d965f1150404fe0bccc054d1fa1195537dc4c13a` |
| `data/campaign_replicates/wide_256/exp1_seed2/team_3_dte.pt` | 549,730 | `48f0bc4071384954c8a887f650dc13a0ffe7272f2f0d82123128b783cb8e85b2` |
| `data/campaign_replicates/wide_256/exp1_seed3/team_1_cte.npz` | 1,077,955 | `330b50b9cc2d185ea0ed9f4dcd1313243f0843b379b0afc17b87e59ac8f44c50` |
| `data/campaign_replicates/wide_256/exp1_seed3/team_1_cte.pt` | 1,164,898 | `0d84fa6fa111a6fcc96f19f7f50dc75e7b72873ec9c143e46dee67702d6ca063` |
| `data/campaign_replicates/wide_256/exp1_seed3/team_2_dte.npz` | 508,211 | `1b2273159fb1604dfb8078b1f16c0149182702435ce9c49e23ad2a086ca29b5f` |
| `data/campaign_replicates/wide_256/exp1_seed3/team_2_dte.pt` | 549,730 | `625af4a66871cff68616e9a884fcd9f2ec187dc5ea9ca1bd091afc724b0ff7bf` |
| `data/campaign_replicates/wide_256/exp1_seed3/team_3_ctde.npz` | 1,042,089 | `cf6c79d1c88e03d18cf927a6da15e37c20ea7cfb9fe4b5de2c99fba1c75be360` |
| `data/campaign_replicates/wide_256/exp1_seed3/team_3_ctde.pt` | 1,126,584 | `471d85bf4ad9a643f34bcbc6ab50d1270bc3fa243a18577eb013ad192cf77527` |
| `data/campaign_replicates/wide_256/exp2_seed1/team_1_ctde-cac.npz` | 1,042,289 | `3fd7d53f2e62556181f635d0c7f5a9518cf1b7abdd153063bfd294a599226caa` |
| `data/campaign_replicates/wide_256/exp2_seed1/team_1_ctde-cac.pt` | 1,126,672 | `d80fcfb116f359abfc14fc445b15c5e2223a7b0bb504989e118c55931ae9b74b` |
| `data/campaign_replicates/wide_256/exp2_seed1/team_2_ctde-comm.npz` | 1,058,597 | `e182604d7c7c27db9a6f6d456d921c8b25ae67f72967b443ad898b90685fe08b` |
| `data/campaign_replicates/wide_256/exp2_seed1/team_2_ctde-comm.pt` | 1,144,818 | `b2971445bbfeb0f7e109286e8c2b7d6b1942a72cb7b186028e508c8f3ab400bd` |
| `data/campaign_replicates/wide_256/exp2_seed1/team_3_ctde-vd.npz` | 759,581 | `1d2c8ae19c3bfd6121251511046806beadeb775af04cd4a49766bc158a556674` |
| `data/campaign_replicates/wide_256/exp2_seed1/team_3_ctde-vd.pt` | 822,002 | `d100fcf1938f44f76ebefb9361430f59c1e28bb36995c91b7655b695dbda1d6e` |
| `data/campaign_replicates/wide_256/exp2_seed2/team_1_ctde-comm.npz` | 1,058,982 | `e7b2cf74f6c9ccbd0e4694d97d15cf42f60c82e3f4041f9b901e9dca837fd755` |
| `data/campaign_replicates/wide_256/exp2_seed2/team_1_ctde-comm.pt` | 1,144,818 | `57c1a70dc696cd5e9390d62a092a3fc9f11f9f0a1fe7fcf8979b8c20ed030359` |
| `data/campaign_replicates/wide_256/exp2_seed2/team_2_ctde-vd.npz` | 759,194 | `cd589bab0efca3a6aa42fdde72c93789278456d8edd2b50fdb6a30289f5dfab6` |
| `data/campaign_replicates/wide_256/exp2_seed2/team_2_ctde-vd.pt` | 822,002 | `93dcd6dee69344733c676aaa777cb4df727be6c30545a4b112803622394c22b4` |
| `data/campaign_replicates/wide_256/exp2_seed2/team_3_ctde-cac.npz` | 1,041,960 | `3e44e8b0ac984784810bc09eddac5b70784c3cc0e2d7e8937213c9fdd519ac45` |
| `data/campaign_replicates/wide_256/exp2_seed2/team_3_ctde-cac.pt` | 1,126,672 | `b81a45c88804f0a80598e59832565e4df1a5357ebea03bdd20b5fa3c546e5cdf` |
| `data/campaign_replicates/wide_256/exp2_seed3/team_1_ctde-vd.npz` | 759,545 | `d310cc351ccadbbd121f878da100e4a3ef5ef948fd326fbc2a324e9e0d156f4d` |
| `data/campaign_replicates/wide_256/exp2_seed3/team_1_ctde-vd.pt` | 822,002 | `6ed9b897bcb358daa4a225a83dee13e9b2e3a1dc123a403c0f48cf1c329a1ea4` |
| `data/campaign_replicates/wide_256/exp2_seed3/team_2_ctde-cac.npz` | 1,041,969 | `fcc9e80efa1c19307349518e2054badf0323bccfb3ff700d773331e3588f26db` |
| `data/campaign_replicates/wide_256/exp2_seed3/team_2_ctde-cac.pt` | 1,126,672 | `334ae23dbe0196d78181e149be08d1bfaf927108764d4cee1d52176df6819e9c` |
| `data/campaign_replicates/wide_256/exp2_seed3/team_3_ctde-comm.npz` | 1,058,811 | `d3f778f494cf15037e29f31cbca1019844c67b47d23fe28328825d54e70f3da5` |
| `data/campaign_replicates/wide_256/exp2_seed3/team_3_ctde-comm.pt` | 1,144,818 | `ea8e5a78869772c416737d1983a51ab8e584553bf3b51183a07eaba57c751dc7` |

**Environment provenance and governance.** The environment every measurement was taken in and the record of what is known about it: the two identical `pip freeze` locks of [§ 9.1](#verification-environment-actually-used), OSV's per-pin answer from [§ 9.10](#dependency-audit) -- for the measured lock and, since A-48, for each tree's runtime `requirements.txt` as well -- and the MIT grant over the whole repository.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `LICENSE` | 1,075 | `fd4599d12f7345289a2684a68d852eac9e0e7a68f0ed17298ebfe502c11f95f6` |
| `ctde_arena/requirements-measured.lock.txt` | 2,448 | `13a9ead65d70af5c9c42671c50477a8b736b864b790b25e38d61bee78a1f55bd` |
| `ctde_arena/results/dependency_audit_runtime.json` | 1,123 | `ce368e0bf38bb0406a38361d78ec261a5d8ca335e65eb19d158130734742a560` |
| `requirements-measured.lock.txt` | 2,448 | `13a9ead65d70af5c9c42671c50477a8b736b864b790b25e38d61bee78a1f55bd` |
| `results/dependency_audit.json` | 11,288 | `5de3f482896f88e80460bda28378eca0a48b0f9661d270e6f92b0c8e705ec3dd` |
| `results/dependency_audit_runtime.json` | 1,028 | `1cfacb10332d5f909403cd69f43b89be4153c319eb2a1f458f95448d4b709e19` |

**Ablation and arena-v2 campaigns.** `run_arena_v2_campaign.sh` and `run_ablation_campaign.sh` output, each aggregated by `run_study.py` from 500 k-step replicates: `no_dr` on seeds 1-10 (extended by `run_no_dr_extension.sh`), the two capacity arms on seeds 1-3, arena v2 on seeds 1-10, and `results/study_3m/` -- the same twenty replicates re-trained at the 3 M-step budget `main.py` ships with, which is the one arm whose single key is the budget. Every arm differs from `results/study/` by exactly one configuration key.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/ablation_narrow_64/analysis.json` | 3,478 | `c419dde1780a142055687fcedd15cf36a611179038c857db7f3c69d4bda8ed61` |
| `results/ablation_narrow_64/per_seed_metrics.csv` | 1,731 | `1c141c4c9911a0b51bdb4a59c8bc0d524e6b5347127d62dc19a44e717833f2bc` |
| `results/ablation_no_dr/analysis.json` | 4,200 | `8945d5b4881e4dcee76b3757d438e573b2b704fa1e48ff9feced43f06df9d656` |
| `results/ablation_no_dr/per_seed_metrics.csv` | 5,208 | `3f6a735c527816947c34b1253742a36accce62e8936a6f589cc1fb5d81307346` |
| `results/ablation_wide_256/analysis.json` | 3,476 | `a760c1e3b92c53ac0a09158fafd4a27527f76e7e9a5bf1e9f94c71bff66028c0` |
| `results/ablation_wide_256/per_seed_metrics.csv` | 1,723 | `26c751cf5cd3a70aec4fe0e79ad6eb369a1de3ad18465ef4f16c7d18f9429e80` |
| `results/study_3m/analysis.json` | 4,193 | `ea0cab2021656a313b355b70226aae4ddedd150c9623cd5ab8bd326a82194f38` |
| `results/study_3m/budget_analysis.json` | 6,775 | `5eca38fea20e1e45de18fc120be857f118eca1e8f835d7c2f7d244ff0116e317` |
| `results/study_3m/per_seed_metrics.csv` | 5,315 | `ee6ac0ae4318e8ba0e5b6119cac6232f9b46d0263f60311d93615718db21bdb4` |
| `results/study_v2/analysis.json` | 4,222 | `44242faee5785f288be1c4eb754ddd38eca0d41d4a00822a966638772425dec8` |
| `results/study_v2/learning_curves.png` | 70,161 | `a2f19be70612de222a47c329f44730d4de78bb29b909db5e531ea3b3219274a0` |
| `results/study_v2/per_seed_metrics.csv` | 5,278 | `76e50a118474572b85fcaaab1d39751cdec7ff7c4aec39bad6e32754490618b3` |

**Absolute-strength evaluation.** `run_absolute_eval.py` output: one trained policy against two uniform-random controllers, trained slot rotated. `absolute_ten/` is the headline — ten checkpoint sets, 750 matches per architecture, the same population as the round-robin — and `absolute/`, `absolute_blocks/` and `absolute_six/` are its one-, three- and six-set predecessors, kept because the replication ladder is the section's finding. Each `summary.json` re-derives from its `matches.csv` with `--report-only`.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `results/absolute/matches.csv` | 106,691 | `9267675ed8f90a0da3efbb2f9d93dc89e9ec2ccdc03938efefc1f8fee7a9b0dd` |
| `results/absolute/summary.json` | 14,142 | `9f5b1b2798d7e2f53f13d5b8fe7a211598226eff2a745353a999fef6500c0014` |
| `results/absolute_blocks/matches.csv` | 141,452 | `ce745d9df991588cce5f8d7b34db4e963f813ccbe43d2b5eba9e49c9654946ac` |
| `results/absolute_blocks/summary.json` | 23,708 | `eaff0097f88b98816689bae8c8e2f96ba1df65eabd12ba272f8c6b6782f1f98d` |
| `results/absolute_six/matches.csv` | 140,415 | `33a771cc57f2562ca7821040442e28723541be57d5c04a7bf64f5e3dff311c11` |
| `results/absolute_six/summary.json` | 32,520 | `ce5bf03714ed4af11c3d84eecab2f0b23c2b909438001fc07ba45815589ca777` |
| `results/absolute_ten/matches.csv` | 234,107 | `6d9466a0b0bd3137de4c010bd322bc0b65f01bc380ad3664cbe9c17ce0d99f53` |
| `results/absolute_ten/summary.json` | 44,744 | `f6e8f52dbb7ee660c0aa02bde66d1f5ede381cd4762049879e57311089a6e609` |

**Gameplay clips.** Rendered by `render_trailer.py` from the promoted replicate's trained policies; the JSON beside each clip records the seed, rotation, paradigm per slot and the per-frame alive and shot counts the caption asserts.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `docs/assets/gameplay_exp1.gif` | 701,949 | `9fb159c403a737cb78b723d9945a55475f90135b7f90fb24a761361a5ec8e307` |
| `docs/assets/gameplay_exp1.json` | 45,652 | `ccef910a688e078c482bb23daab9071de386fea80fbcb6a34539e35e5157fca2` |
| `docs/assets/gameplay_exp2.gif` | 531,137 | `fd2bd6429861d9c23398659c284e615996bfb57d639fd28c1911c0f676e09653` |
| `docs/assets/gameplay_exp2.json` | 45,806 | `4681ffbf26888759644fc2e04ee216b1ecaaf56b8cb16f767e057e2b75f04c2c` |

**MLflow export and derived charts.** The 48 points that reached MLflow, plus the `plot_metrics.py` output for each tree.

| Path | Bytes | SHA-256 |
|---|---:|---|
| `ctde_arena/data/mlflow_export/mlflow_logged_metrics.csv` | 4,445 | `1cd7ce114f2473b6e5e1452da0436c5c6b583070d3337e3866f8048945e16c10` |
| `ctde_arena/data/mlflow_export/mlflow_runs.json` | 1,954 | `722f194366bc2ac1710023aa2e1dfa37ff8fb0278d54886415874ce5fe62d7d3` |
| `ctde_arena/exports/metrics/metrics_comparison.png` | 21,200 | `b5ae4c3368f3e4b665fc5f9f76982a69feebfee824da037ddfd36e43d30065a5` |
| `ctde_arena/exports/metrics/metrics_summary.csv` | 421 | `f36e70d0368778c5d74d12662be84cfdb8fb247bc7e8da7c749962b515e88dcc` |
| `exports/metrics/metrics_comparison.png` | 21,707 | `7e3a2eab959e1452b141c5563949f7064df3610b3afb47cd571ad04ff2c1368e` |
| `exports/metrics/metrics_summary.csv` | 405 | `8b1a93048cccfa41630cd9daf531ec033d816526cedd63b75d944f364911570c` |

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
| `**/data/runs*/` | per-replicate scratch for every campaign — `runs`, `runs_v2`, `runs_ablation_*` — holding checkpoints and per-step trajectory CSVs; the aggregated analysis in `results/` is versioned instead, and the two checkpoint sets the analysis rests on are versioned beside it: `data/training_replicates/` (the round-robin blocks) and `data/campaign_replicates/` (the ablation arms, with their manifest) |

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

## 9.10 Dependency vulnerabilities <a name="dependency-audit"></a>

The audit and JSON immediately below cover only the historical measured PyTorch lock. The current
runtime requirements no longer include PyTorch; they use JAX and load `.npz` checkpoints. PyTorch remains
an optional temporary dependency for converting legacy `.pt` files and for the historical campaign
archiver.

`scripts/audit_dependencies.py` asks the OSV database about every exact pin in
`requirements-measured.lock.txt` — the environment the numbers in this document were produced in — and
writes `results/dependency_audit.json`. It queries OSV directly rather than running `pip audit` because
pip-audit resolves through PyPI's own advisory feed, which is a subset of OSV and lags behind it: a green
`pip audit` would be evidence that PyPI had not been told, not evidence that nothing is known. The script
takes the pins as they are written, so the answer is about the artefact that produced the measurements and
not about a fresh resolution of the dependency graph.

Audit run recorded in the artefact (`generated_utc` 2026-10-16T18:04:41+00:00): **33 of the 34 pins have no
known vulnerability.** The one that does is `torch==2.5.1+cu121`, with 22 OSV records — 9 `GHSA` ids and 13
`PYSEC` ids, which alias 19 distinct CVEs — graded 1 CRITICAL, 3 MODERATE, 5 LOW and 13 with no severity
field. The fixes land between 2.6.0 and 2.13.0.

One of those records concerns this repository's own mitigation: `GHSA-53q9-r3pm-6pq6` / `CVE-2025-32434`,
"**`torch.load` with `weights_only=True` leads to remote code execution**", fixed in 2.6.0.
[A-25](11_code_audit.md#unsafe-checkpoint-load) records that `load_checkpoint` was moved to
`weights_only=True`, and that is still the right change — every load site in both trees uses it, and
`scripts/version_campaign_checkpoints.py` does too — but on the pinned torch it is a mitigation with a known
hole in it.

**Historical pin decision.** The preceding comparison explained why the old measured environment kept
PyTorch 2.5.1 while its then-current requirements used 2.6.0. The current runtime removes PyTorch from
`requirements.txt`; this does not rewrite the provenance of the published results, which remain attached
to the historical lock and original `.pt` files.

**Current exposure.** The training and inference path reads non-pickle `.npz` checkpoints. Torch's pickle
loader is confined to `scripts/convert_torch_checkpoints.py` and the historical campaign archiver; those
tools should only be used with trusted legacy files.

**What would change the decision:** any entry point that accepts a checkpoint path from an untrusted source,
or shipping the checkpoints as data a client supplies. Either would justify both the torch upgrade and a
fresh measurement campaign, whose cost § 9.1 quantifies.

### The runtime locks, audited (2026-10-20) <a name="runtime-dependency-audit"></a>

**Upgraded on `main` (2026-10-31, merged 2026-11-01).** The three advised pins were moved to the newest
releases -- `python-dotenv` 1.0.1 -> 1.2.4, `pytest` 8.3.5 -> 9.2.0 and, in experiment 2, `mlflow` 2.17.2 ->
3.17.2 -- and the audit re-run: **8/8 and 9/9 pins with no known OSV advisory**. The change was re-verified
the way this section says an upgrade must be, in a fresh virtualenv with NumPy 2.2.6 and the Rust extension
built: both suites pass (211 + 2 skipped, 90 + 1 skipped, and again after the merge onto the v3 code), `benchmark_runtime.py`'s exact outcome-and-weight
parity gate holds (Rust 342.8 / fallback 319.6 steps/s, inside the host's session drift), and experiment 2's
`train_rl.py` logs to MLflow. MLflow 3 refuses the file store unless `MLFLOW_ALLOW_FILE_STORE=true`, which
`train_rl.py` now sets by default because this repository tracks to a local `./mlruns` and never runs a
tracking server. What the upgrade does not change: the published numbers were produced under the measured
lock and keep that provenance; the PyTorch-era advisory of A-37 is a property of that lock, not of these pins.

The record below is the pre-upgrade audit, kept as the state it describes.

Everything above is about `requirements-measured.lock.txt` — the environment that produced the published
numbers. The environments that *run* the code today are `requirements.txt` in each tree, and until now no
record said what OSV knows about them; the audit script always took the measured lock as its default and
nothing gated the runtime pins at all. Both are now recorded and CI-gated the same way the measured lock is
(`results/dependency_audit_runtime.json` in each tree, re-verified on every push), so a pin can no longer
change — or acquire a known advisory nobody has read — without the gate noticing the record is stale.

| Lock | Pins | Clean | With advisories |
|---|---:|---:|---|
| `requirements.txt` (experiment 1) | 8 | 6 | `python-dotenv==1.0.1`, `pytest==8.3.5` |
| `ctde_arena/requirements.txt` (experiment 2) | 9 | 6 | the same two, plus `mlflow==2.17.2` |

The applicability of each, and the decision:

* **`python-dotenv==1.0.1`** — `GHSA-mf9w-mj56-hr94` / CVE-2026-28684, MODERATE: `set_key()` and
  `unset_key()` follow symlinks when rewriting `.env` files, allowing a local attacker to overwrite
  arbitrary files. This repository never calls either function — `git grep` finds no `set_key` or
  `unset_key` anywhere — it only reads, via `load_dotenv(ROOT_DIR / ".env")`, and that file is tracked and
  identical to `.env.example`. The advisory's code path does not exist here. Fixed in 1.2.2.
* **`pytest==8.3.5`** — `GHSA-6w46-j5rx-g56g` / CVE-2025-71176, MODERATE: pytest through 9.0.2 relies on
  `/tmp/pytest-of-{user}` directories on UNIX, allowing a hostile local user to interfere with test runs.
  A development tool on developer machines and ephemeral CI runners, where the local-user threat model is
  empty; no repository artefact is produced by pytest in production. Fixed in 9.0.3.
* **`mlflow==2.17.2`** — 58 OSV records: 7 CRITICAL, 15 HIGH, 7 MODERATE, 2 LOW and 31 unspecified,
  aliasing the tracking-server advisory family (path traversal to arbitrary file read/write, SSRF and
  deserialization reachable through the REST API), fixed across 2.19.0–2.22.4. The attack surface those
  records describe is a *network-reachable tracking server*. This repository never starts one: experiment
  2's training logs to a local file store (`mlflow.set_tracking_uri("file:./mlruns")`, or `--no-mlflow`,
  which is what the historical verification used), and the only server the documentation ever tells a
  reader to run is `mlflow ui` against that local store — a loopback browser session, not a service. The
  decision that follows: the pin stays, because the version is what the tracked MLflow artefacts were
  produced with and no workflow here reaches the vulnerable surface. **What would change it:** exposing
  `mlflow ui` beyond loopback, pointing `set_tracking_uri` at a remote server, or any entry point that
  accepts a tracking URI from outside the repository.

The pins are not upgraded: A-37 set the precedent that an environment pin is part of a measurement's
provenance, and each upgrade would be a change nobody has re-verified the suites or the throughput sample
on. The records above are what lets that decision be made with the advisories in view rather than in
ignorance of them.

```bash
python scripts/audit_dependencies.py                 # needs network; writes nothing if OSV is unreachable
python scripts/audit_dependencies.py --print-only    # query without touching the artefact
```

An unreachable OSV is a non-zero exit, not a clean report — the script refuses to write a verdict it could not
verify. `tests/test_dependency_audit.py` pins the parser (a dropped pin would be queried as "nothing known")
and the severity reader (a misread CRITICAL would be a footnote).
