"""Version another training block: copy its six checkpoints out of the git-ignored study scratch.

docs/08_results.md#cross-arena quotes a spread across checkpoint sets, and every block it can quote has to be
something a reader can obtain. `data/training_replicates/` holds the ones the pooled tables were played over;
this script adds to it, in the same directory layout `replicate_policies()` resolves
(`<experiment>_seed<N>/<team>_<paradigm>.pt`), so a new block becomes playable without any further change.

The study trained ten seeds per experiment, of which the blocks currently versioned use six of each. The rest
exist only under `data/runs/`, which `.gitignore` excludes, so a claim like "ten independently trained networks
per architecture" would be unverifiable from a clone.

    python scripts/version_training_replicates.py --pairs 5:6,6:8,7:9,8:10
    python scripts/version_training_replicates.py --check --pairs 5:6,6:8,7:9,8:10   # bytes, writes nothing
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "training_replicates"


def _shared():
    """`copy_verified` from the campaign script -- one copy-and-verify rule, not two."""
    spec = importlib.util.spec_from_file_location("vcc", ROOT / "scripts" / "version_campaign_checkpoints.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def scratch_dir(experiment: str, seed: int) -> Path:
    """Where a study replicate's checkpoints sit, in whichever tree the campaign wrote them."""
    for base in (ROOT, ROOT / "ctde_arena"):
        candidate = base / "data" / "runs" / f"{experiment}_seed{seed}" / "checkpoints"
        if candidate.is_dir():
            return candidate
    raise SystemExit(f"no scratch directory for {experiment}_seed{seed}; the study scratch is git-ignored, "
                     f"so this only runs on the machine that trained it")


def parse_pairs(spec: str) -> list[tuple[int, int]]:
    pairs = []
    for item in spec.split(","):
        exp1, _, exp2 = item.partition(":")
        if not exp2:
            raise SystemExit(f"--pairs wants exp1:exp2 seed pairs, got {item!r}")
        pairs.append((int(exp1), int(exp2)))
    if len({pair[0] for pair in pairs}) != len(pairs) or len({pair[1] for pair in pairs}) != len(pairs):
        raise SystemExit(f"a seed repeats across pairs ({pairs}); one replicate per experiment per block, or "
                         f"the block label stops meaning 'one independently trained network per architecture'")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="", help="exp1:exp2 seed pairs to version, e.g. 5:6,6:8")
    parser.add_argument("--check", action="store_true", help="verify the versioned bytes, write nothing")
    args = parser.parse_args()
    pairs = parse_pairs(args.pairs)
    if not pairs:
        raise SystemExit("--pairs is required (or --check with the same pair list)")
    shared = _shared()
    copied = checked = 0
    for exp1, exp2 in pairs:
        for experiment, seed in (("exp1", exp1), ("exp2", exp2)):
            source = scratch_dir(experiment, seed)
            files = sorted(source.glob("*.pt"))
            if len(files) != 3:
                raise SystemExit(f"{experiment}_seed{seed}: expected three checkpoints in {source}, "
                                 f"found {[f.name for f in files]}")
            target_dir = OUT / f"{experiment}_seed{seed}"
            for file in files:
                target = target_dir / file.name
                if args.check:
                    if not target.is_file():
                        raise SystemExit(f"{target.relative_to(ROOT)} is not versioned")
                    if shared.sha256(target) != shared.sha256(file):
                        raise SystemExit(f"{target.relative_to(ROOT)} differs from its scratch source")
                    checked += 1
                else:
                    shared.copy_verified(file, target)
                    copied += 1
        print(f"block exp1_seed{exp1}+exp2_seed{exp2}: "
              f"{'verified' if args.check else 'versioned'} 6 checkpoints")
    print(f"{'verified' if args.check else 'copied'} {copied + checked} files into {OUT}")


if __name__ == "__main__":
    sys.exit(main())
