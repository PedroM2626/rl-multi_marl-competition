"""Regenerate the docs/09 artefact inventory from the files Git tracks, hashing their *committed* bytes.

The hash column is `git cat-file blob HEAD:<path>`, not the file on disk: this host checks text files out
with CRLF while the repository stores LF, so working-tree hashes are unreproducible from a Linux clone."""

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
        "Recomputed by `report_study.py`, `plot_study.py` and `report_cross_arena.py` from the per-replicate "
        "and per-match summaries. The round-robin rows are the pooled tables chapter 8 quotes: `blocks6.csv` "
        "and `blocks6.json` (six checkpoint sets x five match streams, the current result), `blocks.csv` "
        "(the first three sets, kept because two of the three agreement statistics in [§ 8.11]"
        "(08_results.md#absolute-strength) are computed against it), and the v2 and ablation pools the "
        "re-measurement campaign writes.",
        ("results/study/", "results/cross_arena/blocks.csv", "results/cross_arena/blocks.json",
         "results/cross_arena/blocks6.csv", "results/cross_arena/blocks6.json",
         "results/cross_arena/v2_blocks.csv", "results/cross_arena/v2_blocks.json",
         "results/cross_arena/no_dr_blocks.csv", "results/cross_arena/no_dr_blocks.json"),
    ),
    (
        "Superseded round-robin pools",
        "The earlier replication counts of the same instrument, kept because chapter 8's history is quoted "
        "from them and not from the current table: `pairwise_first_run_600.json` is the first single-"
        "checkpoint run (600 matches) whose verdict was 'no pair separates'; `seeds/` holds the five 600-match "
        "runs that made up `replicates.csv` (3,000 matches, one checkpoint set, and the pooled analysis whose "
        "missing `harness_seed` key left every pair meeting 600 times -- [A-39](11_code_audit.md#the-pooled-"
        "round-robin-analysis-dropped-a-replication-column-from-its-match-key)); `replicates.json` is "
        "that 3,000-match report. None of them is evidence about the current ranking, and the chain 600 -> "
        "3,000 -> 9,000 -> 18,000 is the reason every separation in [§ 8.10](08_results.md#cross-arena) is "
        "quoted with its replication count.",
        ("results/cross_arena/pairwise_first_run_600.json", "results/cross_arena/replicates.csv",
         "results/cross_arena/replicates.json", "results/cross_arena/seeds/"),
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
        "Training replicates behind the round-robin",
        "The checkpoint sets the round-robin is played over: one experiment-1 replicate and one "
        "experiment-2 replicate per block, six blocks in `blocks6.csv` and the first three in `blocks.csv`. "
        "Versioned because the round-robin's claims rest on them while `**/data/runs*/` is git-ignored. "
        "`exp1_seed9+exp2_seed2` is byte-identical to the promoted checkpoints above, which is what makes "
        "that block a parity control. Ten blocks are now versioned -- all twenty study replicates -- and the "
        "four beyond `blocks6.csv` are waiting for `scripts/run_ten_block_campaign.sh`, so they are listed "
        "here as available checkpoints, not as a pooled result.",
        ("data/training_replicates/",),
    ),
    (
        "Ablation checkpoints behind the arm tables",
        "The three configuration-ablation sets of [§ 8.13](08_results.md#ablations), versioned for the same "
        "reason as the round-robin blocks -- the scratch they were trained into is git-ignored -- so an arm "
        "can be replayed without retraining it. `MANIFEST.json` beside them records each set's single "
        "override, the hidden width read out of its own weights, and a SHA-256 per file; a `.pt` in this "
        "repository stores no capacity metadata, which is the one thing that distinguishes `narrow_64` and "
        "`wide_256` from the control.",
        ("data/campaign_replicates/",),
    ),
    (
        "Environment provenance and governance",
        "The environment every measurement was taken in and the record of what is known about it: the two "
        "identical `pip freeze` locks of [§ 9.1](#verification-environment-actually-used), OSV's per-pin "
        "answer from [§ 9.10](#dependency-audit), and the MIT grant over the whole repository.",
        ("requirements-measured.lock.txt", "ctde_arena/requirements-measured.lock.txt",
         "results/dependency_audit.json", "LICENSE"),
    ),
    (
        "Ablation and arena-v2 campaigns",
        "`run_arena_v2_campaign.sh` and `run_ablation_campaign.sh` output, each aggregated by "
        "`run_study.py` from 500 k-step replicates on seeds 1-3 (v2: 10 seeds). Every arm differs from "
        "`results/study/` by exactly one configuration key.",
        ("results/study_v2/", "results/ablation_no_dr/", "results/ablation_narrow_64/",
         "results/ablation_wide_256/"),
    ),
    (
        "Absolute-strength evaluation",
        "`run_absolute_eval.py` output: one trained policy against two uniform-random controllers, trained "
        "slot rotated. `absolute/` is the promoted checkpoint set, `absolute_blocks/` three sets and "
        "`absolute_six/` six, each at 450 matches per architecture; each `summary.json` re-derives from its "
        "`matches.csv` with `--report-only`.",
        ("results/absolute/", "results/absolute_blocks/", "results/absolute_six/"),
    ),
    (
        "Gameplay clips",
        "Rendered by `render_trailer.py` from the promoted replicate's trained policies; the JSON beside "
        "each clip records the seed, rotation, paradigm per slot and the per-frame alive and shot counts "
        "the caption asserts.",
        ("docs/assets/",),
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
             "results/", "ctde_arena/results/", "docs/assets/",
             "requirements-measured.lock.txt", "ctde_arena/requirements-measured.lock.txt", "LICENSE")
        )
    ]


def committed_blob(path: str) -> bytes:
    """The bytes as Git will store them, not the bytes on this disk.

    The repository is committed with LF line endings and this host checks text files out with CRLF
    (`core.autocrlf=true`), so hashing the working tree published hashes that no Linux or macOS clone can
    reproduce -- for the text artefacts the two differ, and the table cannot be an integrity check while it
    describes one checkout configuration. A clone's `git verify-file` and a reader's own hash then agree
    with this table on every platform. Binary files are unaffected: Git does not convert them.

    The object is read from the *index* (`git cat-file blob :path`), not from `HEAD`, because the normal
    sequence here is stage the artefact, regenerate the table, commit: a `HEAD:` lookup fails on a file that
    is staged but not yet committed, and an unstaged edit must not leak into a published hash either. In CI,
    where the index mirrors the fresh checkout's commit, the two forms are the same object.
    """
    result = subprocess.run(["git", "cat-file", "blob", f":{path}"], cwd=ROOT,
                            capture_output=True, check=True).stdout
    return result


def rows_for(paths: list[str]) -> list[str]:
    table = ["| Path | Bytes | SHA-256 |", "|---|---:|---|"]
    for path in sorted(paths):
        blob = committed_blob(path)
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
        "so it cannot drift from what is actually committed, and each hash is of the blob Git stores rather "
        "than of the file on disk — a text artefact checked out with CRLF on Windows hashes differently from "
        "the same committed bytes on Linux, and a reader on any platform should be able to reproduce this "
        "column.",
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
