"""Regenerate the docs/09 artefact inventory table from the files Git actually tracks."""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "09_reproducibility.md"

GROUPS = [
    (
        "Promoted replicate",
        "The artefacts that match the current code. Chosen by `promote_run.py --seed median`, which is the "
        "replicate with the least L1 distance to the per-arm study means; `PROVENANCE.json` records the "
        "choice and the full run summary behind it.",
        ("data/checkpoints/", "data/metrics/", "data/exports/", "data/PROVENANCE.json",
         "ctde_arena/data/checkpoints/", "ctde_arena/data/metrics/", "ctde_arena/data/exports/",
         "ctde_arena/data/PROVENANCE.json"),
    ),
    (
        "Study aggregates",
        "Recomputed by `report_study.py` and `plot_study.py` from the per-replicate summaries; every table in "
        "[§ 8.1](08_results.md#replicated-study) is derived from these three files.",
        ("results/study/",),
    ),
    (
        "Untrained baselines",
        "`random_baseline.py` output, 300 fixed-variant headless matches per mode. The uniform files are "
        "byte-identical between the trees because that mode never queries a network.",
        ("results/baseline/", "ctde_arena/results/baseline/"),
    ),
    (
        "Historical single-seed runs",
        "Superseded by the study but retained because [§ 8.2](08_results.md#historical-single-seed-runs) "
        "quotes them and their checkpoints no longer load into the current observation encoding.",
        ("data/historical_100k/", "ctde_arena/data/historical_100k/"),
    ),
    (
        "MLflow export and derived charts",
        "The 48 points that reached MLflow, plus the `plot_metrics.py` output for each tree.",
        ("ctde_arena/data/mlflow_export/", "exports/metrics/", "ctde_arena/exports/metrics/"),
    ),
]


def tracked_artefacts() -> list[str]:
    listing = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout
    return [
        p
        for p in listing.splitlines()
        if p.startswith(
            ("data/", "ctde_arena/data/", "exports/", "ctde_arena/exports/",
             "results/", "ctde_arena/results/")
        )
    ]


def rows_for(paths: list[str]) -> list[str]:
    table = ["| Path | Bytes | SHA-256 |", "|---|---:|---|"]
    for path in sorted(paths):
        blob = (ROOT / path).read_bytes()
        table.append(f"| `{path}` | {len(blob):,} | `{hashlib.sha256(blob).hexdigest()}` |")
    return table


def main() -> None:
    tracked = tracked_artefacts()
    claimed: set[str] = set()
    blocks: list[str] = []
    for title, blurb, prefixes in GROUPS:
        members = [p for p in tracked if p.startswith(prefixes)]
        claimed.update(members)
        blocks.append("\n".join([f"**{title}.** {blurb}", ""] + rows_for(members)))
    ungrouped = sorted(set(tracked) - claimed)
    if ungrouped:
        raise SystemExit(f"artefacts match no group: {ungrouped}")

    header = [
        "## 9.7 Artefact inventory",
        "",
        f"Every versioned experiment artefact ({len(tracked)} files), with SHA-256 for integrity checking. "
        "Regenerate this section with `python scripts/report_inventory.py`; it is built from `git ls-files`, "
        "so it cannot drift from what is actually committed.",
    ]
    body = "\n\n".join(blocks)
    start = re.search(r"^## 9\.7 Artefact inventory.*?(?=^## 9\.8)", DOC.read_text(encoding="utf-8"),
                      re.S | re.M)
    if not start:
        raise SystemExit("section 9.7 not found")
    text = DOC.read_text(encoding="utf-8")
    replacement = "\n".join(header) + "\n\n" + body + "\n\n"
    DOC.write_text(text[: start.start()] + replacement + text[start.end() :], encoding="utf-8")
    print(f"wrote {len(tracked)} artefact rows into docs/09_reproducibility.md")


if __name__ == "__main__":
    main()
