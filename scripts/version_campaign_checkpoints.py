"""Version the configuration-ablation checkpoints, with a manifest that says what each set was trained as.

docs/08_results.md#ablations measures three arms that differ from the committed study by exactly one
configuration key each. Its aggregates are versioned (`results/ablation_*`), but the policies behind them
were not: they sat under the git-ignored `**/data/runs*/` scratch, so reproducing an arm meant retraining it
(~70 min of wall clock per arm, measured) and a reader could not replay a single match of it.

This script copies each arm's checkpoints into `data/campaign_replicates/<arm>/<experiment>_seedN/` and
writes `MANIFEST.json` beside them. The manifest matters more than the copies: a `.pt` in this repository
stores only `{"paradigm", "actor"}`, so the capacity a checkpoint was trained at is *not* in the file, and
the ablations are precisely the arms that vary capacity. Loading a 256-wide policy at `RL_HIDDEN_DIM=128`
raises a shape error rather than silently measuring the wrong thing, but the manifest means a reader learns
the right value before running anything.

Idempotent: a destination whose bytes already match the source is left alone, and a mismatch aborts instead
of overwriting, so the versioned set cannot drift from the scratch it came from.

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

# Exactly the overrides scripts/run_ablation_campaign.sh exported per arm; the baseline for all three is
# results/study (domain randomisation on, hidden width 128, seeds 1-3, 500,000 steps).
ARMS = {
    "no_dr": {"DOMAIN_RANDOMIZATION": "false"},
    "narrow_64": {"RL_HIDDEN_DIM": "64"},
    "wide_256": {"RL_HIDDEN_DIM": "256"},
}
SEEDS = (1, 2, 3)
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


def actor_shape(path: Path) -> tuple[int | None, int]:
    """(hidden width, parameter count) of the actor stored in this checkpoint.

    Every actor MLP in either tree is built by `_mlp`, whose first layer is `Linear(input, hidden)`, so the
    row count of each `*.0.weight` is the hidden width -- `backbone.0` for CTE/DTE/CAC/VD, `msg_net.0` and
    `policy_head.0` for Comm. The widths have to agree or the file is not one width, which the ablations
    are precisely about: a `.pt` records no capacity metadata, so this is the only place a reader can learn
    that a policy needs `RL_HIDDEN_DIM=256` instead of the committed 128.
    """
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    actor = payload.get("actor", {})
    widths = {int(tensor.shape[0]) for name, tensor in actor.items()
              if name.endswith(".0.weight") and hasattr(tensor, "shape") and tensor.ndim == 2}
    parameters = sum(int(tensor.numel()) for tensor in actor.values() if hasattr(tensor, "numel"))
    width = widths.pop() if len(widths) == 1 else None
    return width, parameters


def source_dir(arm: str, experiment: str, seed: int) -> Path:
    return SCRATCH_ROOT[experiment] / "data" / f"runs_ablation_{arm}" / f"{experiment}_seed{seed}"


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
    args = parser.parse_args()

    manifest: dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Checkpoints of the configuration ablations in docs/08_results.md#ablations, versioned "
                   "so an arm can be replayed without retraining it and its capacity is recorded somewhere "
                   "other than the campaign script.",
        "produced_by": "bash scripts/run_ablation_campaign.sh",
        "baseline": "results/study (seeds 1-3 of both experiments, 500000 steps, domain randomisation on, "
                    "RL_HIDDEN_DIM=128)",
        "reproduce_training": "bash scripts/run_ablation_campaign.sh, or one arm on its own: "
                              "env RL_HIDDEN_DIM=64 python scripts/run_study.py --steps 500000 "
                              "--seeds 1,2,3 --jobs 3 --eval-matches 150 "
                              "--out data/runs_ablation_narrow_64 --results-dir results/ablation_narrow_64",
        "arms": {},
    }
    arms: dict[str, object] = manifest["arms"]  # type: ignore[index]
    copied = 0
    checked = 0

    for arm, overrides in ARMS.items():
        sets: dict[str, object] = {}
        for experiment in EXPERIMENTS:
            for seed in SEEDS:
                run_dir = source_dir(arm, experiment, seed)
                source = run_dir / "checkpoints"
                if not source.is_dir():
                    raise SystemExit(f"{arm}: missing {source} -- the scratch is git-ignored, so this "
                                     f"script can only run on the machine that trained the arm")
                files = sorted(source.glob("*.pt"))
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
                sets[f"{experiment}_seed{seed}"] = {
                    "trained_as": {**overrides},
                    "source_scratch": run_dir.relative_to(ROOT).as_posix(),
                    **record(run_dir, files),
                }
        arms[arm] = {"overrides": overrides, "sets": sets}

    if args.check:
        print(f"verified {checked} versioned checkpoints against their scratch; nothing written")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    unknown = [f"{arm}/{name}" for arm, payload in arms.items()
               for name, entry in payload["sets"].items() if entry["hidden_dim"] is None]  # type: ignore[index]
    if unknown:
        raise SystemExit(f"could not read a single hidden width out of {unknown}; the manifest would be "
                         f"lying about the capacity these sets were trained at")
    print(f"copied {copied} checkpoints; manifest lists "
          f"{sum(len(a['sets']) for a in arms.values())} sets -> {OUT / 'MANIFEST.json'}")
    for arm, payload in arms.items():
        widths = sorted({entry["hidden_dim"] for entry in payload["sets"].values()})  # type: ignore[index]
        print(f"  {arm}: hidden width {widths} over {len(payload['sets'])} sets")  # type: ignore[arg-type]


if __name__ == "__main__":
    sys.exit(main())
