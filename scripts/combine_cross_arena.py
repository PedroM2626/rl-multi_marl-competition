"""Merge replicated cross-arena runs into one CSV with a harness-seed column.

`run_cross_arena.py` writes one directory per harness seed. The pooled analysis in
`report_cross_arena.py` needs them as a single table, and the question replication answers -- is the
ordering stable, or did one stream produce it -- needs the seed kept, so it is added as a column rather
than dropped by concatenation.

Usage:
    python scripts/combine_cross_arena.py results/cross_arena_seed1 results/cross_arena_seed2 ...
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dirs", nargs="+", type=Path, help="run directories, each holding a matches.csv")
    parser.add_argument("--out", type=Path, default=Path("results/cross_arena/replicates.csv"))
    return parser.parse_args()


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
            if fields is None:
                fields = ["harness_seed", *reader.fieldnames]
            elif list(reader.fieldnames) != fields[1:]:
                raise SystemExit(
                    f"{src} has columns {reader.fieldnames}, expected {fields[1:]} -- "
                    "runs from different code versions cannot be pooled"
                )
            seed = directory.name.rsplit("seed", 1)[-1]
            for row in reader:
                rows.append({"harness_seed": seed, **row})

    out = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    seeds = sorted({row["harness_seed"] for row in rows})
    print(f"merged {len(seeds)} runs ({', '.join(seeds)}) -> {len(rows)} team-matches, wrote {out}")


if __name__ == "__main__":
    main()
