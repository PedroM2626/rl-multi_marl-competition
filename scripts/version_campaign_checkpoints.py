"""Version the configuration-ablation checkpoints, with a manifest that says what each set was trained as.

docs/08_results.md#ablations measures three arms that differ from the committed study by exactly one
configuration key each. Its aggregates are versioned (`results/ablation_*`), but the policies behind them
were not: they sat under the git-ignored `**/data/runs*/` scratch, so reproducing an arm meant retraining it
(~70 min of wall clock per three-seed arm, measured; 170.8 min for the seven seeds `no_dr` grew by) and a
reader could not replay a single match of it.

This script copies each arm's checkpoints into `data/campaign_replicates/<arm>/<experiment>_seedN/` and
writes `MANIFEST.json` beside them. The manifest matters more than the copies: a `.pt` in this repository
stores only `{"paradigm", "actor"}`, so the capacity a checkpoint was trained at is *not* in the file, and
the ablations are precisely the arms that vary capacity. Loading a 256-wide policy at `RL_HIDDEN_DIM=128`
raises a shape error rather than silently measuring the wrong thing, but the manifest means a reader learns
the right value before running anything.

Idempotent: a destination whose bytes already match the source is left alone, and a mismatch aborts instead
of overwriting, so the versioned set cannot drift from the scratch it came from.

Which seeds an arm covers is *discovered from the scratch*, not listed here. A constant seed tuple was a
defect: `run_no_dr_extension.sh` trained seeds 4 to 10, this script copied only 1-3 because that is what it
was told to look for, and the pool replay then read the git-ignored scratch -- an arm whose published
checkpoints were not the ones that produced its numbers. A run is only picked up if it has three
checkpoints *and* an evaluation summary, so a training run that died halfway cannot be versioned silently.

    python scripts/version_campaign_checkpoints.py            # copy + manifest
    python scripts/version_campaign_checkpoints.py --check    # hashes only, writes nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "campaign_replicates"

# Exactly the overrides each arm's campaign script exported, the scratch directory it trained into, and the
# study its aggregate is paired against. The PyTorch-era arms are one key from results/study; v3 is the
# whole docs/10 bundle (scripts/run_v3_campaign.sh) and the two single-key arms are one key each from the
# JAX replication (scripts/run_item16_campaign.sh). Each arm covers as many seeds as its scratch holds --
# see completed_seeds.
V3_OVERRIDES = {
    "RL_GAE_PER_AGENT": "true", "RL_UPDATE_MINIBATCHES": "16", "RL_TARGET_KL": "0.02", "RL_PARAM_TARGET": "-1",
    "COMM_ROUNDS": "2", "COMM_MASK_DEAD": "true", "VD_LOCAL_CRITIC": "true", "OBS_OBSTACLE_RAYS": "8",
    "ARENA_LAYOUT": "symmetric", "SPAWN_CYCLE_PER_MATCH": "true",
    "RL_OPPONENT_POOL_PROB": "0.2", "RL_OPPONENT_POOL_EVERY_MATCHES": "50", "RL_OPPONENT_POOL_SIZE": "20",
}
ARMS = {
    "no_dr": {"overrides": {"DOMAIN_RANDOMIZATION": "false"},
              "scratch": "runs_ablation_no_dr", "baseline": "results/study"},
    "narrow_64": {"overrides": {"RL_HIDDEN_DIM": "64"},
                  "scratch": "runs_ablation_narrow_64", "baseline": "results/study"},
    "wide_256": {"overrides": {"RL_HIDDEN_DIM": "256"},
                 "scratch": "runs_ablation_wide_256", "baseline": "results/study"},
    "v3": {"overrides": V3_OVERRIDES, "scratch": "runs_v3", "baseline": "results/study_jax"},
    "spawn_cycle": {"overrides": {"SPAWN_CYCLE_PER_MATCH": "true"},
                    "scratch": "runs_jax_spawn_cycle", "baseline": "results/study_jax"},
    "symmetric": {"overrides": {"ARENA_LAYOUT": "symmetric"},
                  "scratch": "runs_jax_symmetric", "baseline": "results/study_jax"},
}
EXPERIMENTS = ("exp1", "exp2")
# Experiment 2's scratch tree is inside ctde_arena/, which is where its runs were written.
SCRATCH_ROOT = {"exp1": ROOT, "exp2": ROOT / "ctde_arena"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_verified(source: Path, target: Path) -> str:
    """Copy one checkpoint and refuse to overwrite a versioned file whose bytes differ from the source.

    Shared with `version_training_replicates.py`: the point of versioning a checkpoint set is that a reader
    gets *these* weights, so a silent mismatch between the copy and the scratch it came from is the one
    failure mode neither script may have.
    """
    if target.is_file() and sha256(target) != sha256(source):
        raise SystemExit(f"refusing to overwrite {target}: its bytes differ from {source}")
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return sha256(target)


def policy_files(directory: Path) -> list[Path]:
    """The three policy files of one set, in the format this directory actually holds.

    `.npz` is the runtime's format since A-44 and the one a reader can load without PyTorch; `.pt` is what
    the scratch of a run trained before the port still contains, and what the published tables were produced
    from. Preferring `.npz` means a set copied after a conversion carries the loadable file, and a set that
    only exists as `.pt` is still versionable.
    """
    npz = sorted(directory.glob("*.npz"))
    return npz or sorted(directory.glob("*.pt"))


def actor_shape(path: Path) -> tuple[int | None, int]:
    """(hidden width, parameter count) of the actor stored in this checkpoint.

    Every actor MLP in either tree is built by `_mlp`, whose first layer is `Linear(input, hidden)`, so the
    row count of each `*.0.weight` is the hidden width -- `backbone.0` for CTE/DTE/CAC/VD, `msg_net.0` and
    `policy_head.0` for Comm. The widths have to agree or the file is not one width, which the ablations
    are precisely about: neither checkpoint format records capacity metadata, so this is the only place a
    reader can learn that a policy needs `RL_HIDDEN_DIM=256` instead of the committed 128.

    Both formats store the weights in Torch's `[out, in]` layout -- the JAX loader transposes on the way in
    (A-45) -- so the width is read the same way from either. The `.pt` branch is the only place this script
    needs PyTorch, and it is never taken for a converted archive.
    """
    if path.suffix == ".pt":
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=True)
        actor = {name: tensor for name, tensor in payload.get("actor", {}).items()}
    else:
        import numpy as np

        with np.load(path) as stored:
            prefix = "actor::"
            actor = {name[len(prefix):]: stored[name] for name in stored.files if name.startswith(prefix)}
    widths = {int(tensor.shape[0]) for name, tensor in actor.items()
              if name.endswith(".0.weight") and hasattr(tensor, "shape") and tensor.ndim == 2}
    # A Torch tensor counts with numel(), a NumPy array with .size; reading only the first made every
    # converted archive record 0 parameters.
    parameters = sum(int(tensor.numel()) if hasattr(tensor, "numel") else int(tensor.size)
                     for tensor in actor.values() if hasattr(tensor, "numel") or hasattr(tensor, "size"))
    width = widths.pop() if len(widths) == 1 else None
    return width, parameters


def scratch_root(arm: str, experiment: str) -> Path:
    return SCRATCH_ROOT[experiment] / "data" / ARMS[arm]["scratch"]


def source_dir(arm: str, experiment: str, seed: int) -> Path:
    return scratch_root(arm, experiment) / f"{experiment}_seed{seed}"


def completed_seeds(arm: str, experiment: str) -> list[int]:
    """Every seed of this arm whose training run finished, found on disk rather than from a list.

    A set counts as finished only if it wrote three checkpoints *and* an evaluation summary: an arm killed
    mid-training leaves checkpoints behind, and versioning those would publish weights that no aggregate in
    docs/08_results.md was computed from.
    """
    root = scratch_root(arm, experiment)
    if not root.is_dir():
        return []
    seeds = []
    for run_dir in sorted(root.glob(f"{experiment}_seed*")):
        name = run_dir.name
        if not name.removeprefix(f"{experiment}_seed").isdigit():
            continue
        checkpoints = policy_files(run_dir / "checkpoints")
        if len(checkpoints) == 3 and (run_dir / "evaluation" / "run_summary.json").is_file():
            seeds.append(int(name.removeprefix(f"{experiment}_seed")))
    return sorted(seeds)


def versioned_seeds(arm: str, experiment: str) -> list[int]:
    """The seeds this arm+experiment already has committed under data/campaign_replicates/."""
    root = OUT / arm
    if not root.is_dir():
        return []
    seeds = []
    for run_dir in sorted(root.glob(f"{experiment}_seed*")):
        suffix = run_dir.name.removeprefix(f"{experiment}_seed")
        if suffix.isdigit() and len(policy_files(run_dir)) == 3:
            seeds.append(int(suffix))
    return sorted(seeds)


def record(run_dir: Path, files: list[Path]) -> dict:
    """Everything the run wrote about itself, plus what the weights say about their own shape."""
    summary_path = run_dir / "evaluation" / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    held_out = summary.get("held_out_greedy", {})
    per_file = {}
    widths: set[int] = set()
    for file in files:
        width, parameters = actor_shape(file)
        if width is not None:
            widths.add(width)
        per_file[file.name] = {
            "sha256": sha256(file),
            "bytes": file.stat().st_size,
            "hidden_dim": width,
            "actor_parameters": parameters,
        }
    return {
        "seed": summary.get("seed"),
        "completed_steps": summary.get("completed_steps"),
        "target_steps": summary.get("target_steps"),
        "train_seconds": summary.get("train_seconds"),
        "domain_randomization": summary.get("domain_randomization"),
        "paradigm_rotation": summary.get("paradigm_rotation"),
        "paradigms": summary.get("paradigms"),
        "held_out_greedy_matches": held_out.get("matches"),
        "hidden_dim": widths.pop() if len(widths) == 1 else None,
        "files": per_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the versioned set against the scratch without writing anything")
    parser.add_argument("--arms", default=",".join(ARMS),
                        help="comma-separated arms to (re)version; the manifest keeps every other arm's entry "
                             "as it was, so an arm still training does not have to exist yet")
    args = parser.parse_args()
    selected = [arm for arm in args.arms.split(",") if arm]
    unknown_arms = sorted(set(selected) - set(ARMS))
    if unknown_arms:
        raise SystemExit(f"unknown arms {unknown_arms}; known: {sorted(ARMS)}")
    previous = (json.loads((OUT / "MANIFEST.json").read_text(encoding="utf-8"))
                if (OUT / "MANIFEST.json").is_file() else {"arms": {}})

    manifest: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Checkpoints of the configuration ablations in docs/08_results.md#ablations, versioned "
                   "so an arm can be replayed without retraining it and its capacity is recorded somewhere "
                   "other than the campaign script.",
        "produced_by": "no_dr / narrow_64 / wide_256: bash scripts/run_ablation_campaign.sh, then bash "
                       "scripts/run_no_dr_extension.sh for the seeds no_dr carries past three; v3: bash "
                       "scripts/run_v3_campaign.sh; spawn_cycle / symmetric: bash scripts/run_item16_campaign.sh",
        "baseline": "each arm's 'baseline' field: results/study (PyTorch, both experiments, 500000 steps, "
                    "domain randomisation on, RL_HIDDEN_DIM=128, seeds 1-10) for the PyTorch-era arms, "
                    "results/study_jax (the same on the JAX runtime) for v3 and the single-key arms",
        "seed_coverage": "each arm's 'seeds' field lists what its scratch actually holds, discovered per "
                         "experiment rather than assumed; report_study.py pairs on seed number, so an arm "
                         "with fewer seeds than the control is compared on the shared ones only",
        "reproduce_training": "bash scripts/run_ablation_campaign.sh, or one arm on its own: "
                              "env RL_HIDDEN_DIM=64 python scripts/run_study.py --steps 500000 "
                              "--seeds 1,2,3 --jobs 3 --eval-matches 150 "
                              "--out data/runs_ablation_narrow_64 --results-dir results/ablation_narrow_64",
        "arms": {},
    }
    arms: dict[str, object] = manifest["arms"]  # type: ignore[index]
    arms.update({arm: entry for arm, entry in previous.get("arms", {}).items() if arm not in selected})
    copied = 0
    checked = 0

    for arm in selected:
        overrides = ARMS[arm]["overrides"]
        sets: dict[str, object] = {}
        discovered: dict[str, list[int]] = {}
        for experiment in EXPERIMENTS:
            seeds = completed_seeds(arm, experiment)
            discovered[experiment] = seeds
            if not seeds:
                raise SystemExit(f"{arm}: no completed {experiment} run found under "
                                 f"{scratch_root(arm, experiment)} -- the "
                                 f"scratch is git-ignored, so this script can only run on the machine that "
                                 f"trained the arm")
            on_disk = set(seeds)
            stale = sorted(set(versioned_seeds(arm, experiment)) - on_disk)
            if stale:
                raise SystemExit(f"{arm}: {experiment}_seed{stale} is versioned but its scratch is gone or "
                                 f"incomplete; the manifest would describe weights it cannot re-derive")
            for seed in seeds:
                run_dir = source_dir(arm, experiment, seed)
                source = run_dir / "checkpoints"
                files = policy_files(source)
                if len(files) != 3:
                    raise SystemExit(f"{arm}: expected three checkpoints in {source}, found "
                                     f"{[f.name for f in files]}")
                destination = OUT / arm / f"{experiment}_seed{seed}"
                destination.mkdir(parents=True, exist_ok=True)
                for file in files:
                    target = destination / file.name
                    if args.check:
                        if not target.is_file():
                            raise SystemExit(f"{arm}: {target.relative_to(ROOT)} is not versioned")
                        if sha256(target) != sha256(file):
                            raise SystemExit(f"{arm}: {target.relative_to(ROOT)} differs from its source "
                                             f"{file.relative_to(ROOT)}")
                        checked += 1
                        continue
                    if target.is_file() and sha256(target) == sha256(file):
                        continue
                    if target.is_file():
                        raise SystemExit(f"{arm}: refusing to overwrite {target.relative_to(ROOT)}, which "
                                         f"differs from {file.relative_to(ROOT)}")
                    shutil.copyfile(file, target)
                    copied += 1
                # The manifest records the committed bytes: since A-45 a set can hold both the .pt that
                # produced the published numbers and the .npz this runtime loads, and it is the latter a
                # reader can actually open.
                sets[f"{experiment}_seed{seed}"] = {
                    "trained_as": {**overrides},
                    "source_scratch": run_dir.relative_to(ROOT).as_posix(),
                    **record(run_dir, policy_files(destination)),
                }
        arms[arm] = {"overrides": overrides, "baseline": ARMS[arm]["baseline"],
                     "seeds": {k: v for k, v in discovered.items()}, "sets": sets}

    if args.check:
        print(f"verified {checked} versioned checkpoints against their scratch; nothing written")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # A parameter-matched arm (RL_PARAM_TARGET) trains each architecture at its own width, so its set has no
    # single width; every file still has to state one.
    unknown = [f"{arm}/{name}/{file}" for arm, payload in arms.items()
               for name, entry in payload["sets"].items()  # type: ignore[index]
               for file, shape in entry["files"].items() if shape["hidden_dim"] is None]
    unknown += [f"{arm}/{name}" for arm, payload in arms.items()
                for name, entry in payload["sets"].items()  # type: ignore[index]
                if entry["hidden_dim"] is None and "RL_PARAM_TARGET" not in payload["overrides"]]
    if unknown:
        raise SystemExit(f"could not read a single hidden width out of {unknown}; the manifest would be "
                         f"lying about the capacity these sets were trained at")
    print(f"copied {copied} checkpoints; manifest lists "
          f"{sum(len(a['sets']) for a in arms.values())} sets -> {OUT / 'MANIFEST.json'}")
    for arm, payload in arms.items():
        widths = sorted({shape["hidden_dim"] for entry in payload["sets"].values()  # type: ignore[index]
                         for shape in entry["files"].values()})
        coverage = ", ".join(f"{e} seeds {v}" for e, v in payload["seeds"].items())  # type: ignore[index]
        sets = len(payload["sets"])  # type: ignore[arg-type]
        print(f"  {arm}: hidden width {widths} over {sets} sets ({coverage})")


if __name__ == "__main__":
    sys.exit(main())
