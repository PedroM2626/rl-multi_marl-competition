"""Promote one replicate's artefacts into the versioned data/ directories.

The study writes each replicate to data/runs/<name>/, which is git-ignored. The repository still has to
ship a set of checkpoints and metrics that match the code that produced them, so one replicate is chosen
explicitly and copied into the tracked location rather than left implicit.

Usage:
    python scripts/promote_run.py --experiment exp1 --list
    python scripts/promote_run.py --experiment exp1 --seed median
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


def summaries(experiment: str) -> dict[int, dict]:
    runs = ROOT / "data" / "runs"
    out: dict[int, dict] = {}
    for path in runs.glob(f"{experiment}_seed*/evaluation/run_summary.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[int(payload["seed"])] = payload
    return out


def median_seed(payloads: dict[int, dict]) -> int:
    """The replicate whose per-paradigm held-out win rates sit closest to the experiment mean.

    Promoting a seed by name invites picking one that happens to confirm the hypothesis; the shipped
    artefacts should look like the study, so the choice is made by L1 distance to the per-arm means.
    """
    arms = {p["paradigms"][team] for p in payloads.values() for team in p["held_out_greedy"]["per_team"]}
    means = {
        arm: sum(
            p["held_out_greedy"]["per_team"][team]["win_rate"]
            for p in payloads.values()
            for team in p["held_out_greedy"]["per_team"]
            if p["paradigms"][team] == arm
        )
        / max(sum(1 for p in payloads.values() if arm in p["paradigms"].values()), 1)
        for arm in arms
    }
    def distance(payload: dict) -> float:
        return sum(
            abs(payload["held_out_greedy"]["per_team"][team]["win_rate"] - means[payload["paradigms"][team]])
            for team in payload["held_out_greedy"]["per_team"]
        )
    return min(payloads, key=lambda seed: distance(payloads[seed]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=sorted(TARGETS), required=True)
    parser.add_argument(
        "--seed",
        help="replicate to promote, or 'median' to pick the replicate closest to the study means"
    )
    parser.add_argument("--list", action="store_true", help="show available replicates and exit")
    args = parser.parse_args()
    if not args.list and not args.seed:
        parser.error("--seed is required unless --list is given")

    payloads = summaries(args.experiment)
    if not payloads:
        raise SystemExit(f"no replicates found under {ROOT / 'data' / 'runs'} for {args.experiment}")

    if args.list:
        samples: dict[str, list[float]] = {}
        for payload in payloads.values():
            for team, stats in payload["held_out_greedy"]["per_team"].items():
                samples.setdefault(payload["paradigms"][team], []).append(stats["win_rate"])
        means = {arm: sum(v) / len(v) for arm, v in samples.items()}
        print(f"  study means: { {arm: round(m, 3) for arm, m in sorted(means.items())} }")
        for seed, payload in sorted(payloads.items()):
            wins = {
                payload["paradigms"][team]: round(stats["win_rate"], 3)
                for team, stats in payload["held_out_greedy"]["per_team"].items()
            }
            l1 = sum(abs(wins[arm] - means[arm]) for arm in wins)
            print(f"  seed {seed:<3d} rot={payload['paradigm_rotation']} "
                  f"steps={payload['completed_steps']:,} {payload['train_seconds']/60:.0f}min "
                  f"held-out={wins} L1={l1:.3f}")
        print(f"  median (closest to the means): seed {median_seed(payloads)}")
        return

    seed = median_seed(payloads) if args.seed == "median" else int(args.seed)
    if seed not in payloads:
        raise SystemExit(f"no such replicate: {args.experiment}_seed{seed}")
    source = ROOT / "data" / "runs" / f"{args.experiment}_seed{seed}"

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
                "promoted_run": f"{args.experiment}_seed{seed}",
                "selection": "explicit --seed" if args.seed != "median" else "median: least L1 distance to the per-arm study means",
                "note": "These artefacts are one replicate of the study in results/study/. A single "
                        "replicate is not evidence about the ranking: see the per-seed matrix in "
                        "docs/08_results.md before reading the win rates below as a result.",
                "summary": payloads[seed],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {marker.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
