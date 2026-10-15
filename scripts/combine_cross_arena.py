"""Merge replicated cross-arena runs into one CSV carrying both replication columns.

`run_cross_arena.py` writes one directory per (training block, harness seed) pair. The pooled analysis in
`report_cross_arena.py` needs them as a single table, and the question replication answers -- is the
ordering stable, or did one match stream or one set of checkpoints produce it -- needs both kept, so they
are added as columns rather than dropped by concatenation.

Usage:
    python scripts/combine_cross_arena.py results/cross_arena_seed1 results/cross_arena_seed2 ...
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESERVED = ("training_block", "harness_seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="run directories, each holding a matches.csv")
    parser.add_argument("--out", type=Path, default=Path("results/cross_arena/replicates.csv"))
    return parser.parse_args()


def labels(directory: Path) -> tuple[str, str]:
    """(training block, harness seed) from a run directory name.

    `e1s10_e2s5_h3` -> ("exp1_seed10+exp2_seed5", "3"). Both are needed: the block is the training
    replicate and the harness seed is the match stream, and conflating them is what made the first
    round-robin analysis silently drop four fifths of its pairs.
    """
    name = directory.name
    match = re.fullmatch(r"e1s(\d+)_e2s(\d+)_h(\d+)", name)
    if match:
        return f"exp1_seed{match.group(1)}+exp2_seed{match.group(2)}", match.group(3)
    return "promoted", name.rsplit("seed", 1)[-1]


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows: list[dict[str, object]] = []
    fields: list[str] | None = None

    for directory in args.dirs:
        src = directory if directory.suffix == ".csv" else directory / "matches.csv"
        if not src.is_file():
            raise SystemExit(f"no matches.csv under {directory}")
        with src.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise SystemExit(f"{src} has no header row")
            block, harness_seed = labels(directory)
            if fields is None:
                fields = [*RESERVED, *reader.fieldnames]
            elif list(reader.fieldnames) != fields[2:]:
                raise SystemExit(
                    f"{src} has columns {reader.fieldnames}, expected {fields[2:]} -- "
                    "runs from different code versions cannot be pooled"
                )
            # A run directory that already carries these columns would have its own labels silently
            # preferred over the ones derived from its name, and the replication tables would then
            # group by something other than the replication.
            clash = [name for name in RESERVED if name in reader.fieldnames]
            if clash:
                raise SystemExit(f"{src} already has column(s) {clash}; they are assigned from the "
                                 f"run directory name")
            for row in reader:
                rows.append({"training_block": block, "harness_seed": harness_seed, **row})

    out = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    blocks = sorted({row["training_block"] for row in rows})
    seeds = sorted({row["harness_seed"] for row in rows})
    print(f"merged {len(blocks)} training blocks x {len(seeds)} harness seeds "
          f"= {len(blocks) * len(seeds)} runs -> {len(rows)} team-matches, wrote {out}")


if __name__ == "__main__":
    main()
