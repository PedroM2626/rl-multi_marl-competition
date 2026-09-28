"""Promote one replicate's artefacts into the versioned data/ directories.

The study writes each replicate to data/runs/<name>/, which is git-ignored. The repository still has to
ship a set of checkpoints and metrics that match the code that produced them, so one replicate is chosen
explicitly and copied into the tracked location rather than left implicit.

Usage:
    python scripts/promote_run.py --experiment exp1 --seed 1
    python scripts/promote_run.py --experiment exp2 --seed 1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {"exp1": ROOT, "exp2": ROOT / "ctde_arena"}
SUBDIRS = ["checkpoints", "metrics", "exports"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(TARGETS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--list", action="store_true", help="show available replicates and exit")
    args = parser.parse_args()

    runs = ROOT / "data" / "runs"
    if args.list:
        for summary in sorted(runs.glob("*/evaluation/run_summary.json")):
            payload = json.loads(summary.read_text(encoding="utf-8"))
            name = summary.parents[1].name
            wins = {p: round(s["win_rate"], 3) for p, s in payload["held_out_greedy"]["per_team"].items()}
            print(f"  {name:16s} rot={payload['paradigm_rotation']} steps={payload['completed_steps']:,} "
                  f"{payload['train_seconds']/60:.0f}min held-out={wins}")
        return

    source = runs / f"{args.experiment}_seed{args.seed}"
    if not source.is_dir():
        raise SystemExit(f"no such replicate: {source}")

    destination = TARGETS[args.experiment] / "data"
    for sub in SUBDIRS:
        src = source / sub
        if not src.is_dir():
            continue
        dst = destination / sub
        dst.mkdir(parents=True, exist_ok=True)
        for item in sorted(src.iterdir()):
            if item.is_file():
                shutil.copy2(item, dst / item.name)
        print(f"copied {sub}: {sorted(p.name for p in dst.iterdir())}")

    marker = destination / "PROVENANCE.json"
    marker.write_text(
        json.dumps(
            {
                "promoted_run": f"{args.experiment}_seed{args.seed}",
                "note": "These artefacts are one replicate of the study in results/study/. The full "
                        "per-replicate data stays under data/runs/ and is not versioned.",
                "summary": json.loads(
                    (source / "evaluation" / "run_summary.json").read_text(encoding="utf-8")
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {marker.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
