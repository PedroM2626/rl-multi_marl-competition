"""Write the test counts and coverage figures into the documentation, instead of typing them.

Every published "N passed" and coverage percentage used to be copied by hand from a terminal, and the audit
kept finding them stale. The lines that carry those numbers now hold an invisible marker,
`<!--auto:KEY-->`; this script runs both suites under coverage and regenerates every marked line (and the
fenced block between `<!--auto:pytest-block-->` and `<!--/auto-->`) from what it measured.

    python scripts/update_doc_counts.py            # run both suites with coverage, rewrite the docs
    python scripts/update_doc_counts.py --check    # CI: the documented totals equal what pytest collects

`--check` does not run the suites; it compares the documented passed + skipped total of each tree with
`pytest --collect-only`, which is cheap and platform-independent. Coverage percentages are only written,
never checked, because they move by a statement or two between Python versions (docs/11, A-6).
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "marl_arena"
TREES = {"exp1": ROOT, "exp2": ROOT / "ctde_arena"}
DOCS = [ROOT / "README.md", ROOT / "ctde_arena" / "README.md", ROOT / "docs" / "09_reproducibility.md"]
MARKER = re.compile(r"<!--auto:([a-z0-9-]+)-->")


def run_suite(tree: Path) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        data = Path(scratch) / ".coverage"
        report = Path(scratch) / "coverage.json"
        completed = subprocess.run(
            [sys.executable, "-m", "coverage", "run", f"--data-file={data}", "--branch", f"--source={ENGINE}",
             "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
            cwd=tree, capture_output=True, text=True,
        )
        summary = completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else ""
        counts = {kind: int(n) for n, kind in re.findall(r"(\d+) (passed|skipped|failed|error)", summary)}
        if completed.returncode != 0 or counts.get("failed") or counts.get("error"):
            raise SystemExit(f"{tree.name} suite did not pass; refusing to publish its numbers:\n{summary}")
        subprocess.run([sys.executable, "-m", "coverage", "json", f"--data-file={data}", "-o", str(report)],
                       cwd=tree, check=True, capture_output=True)
        totals = json.loads(report.read_text(encoding="utf-8"))["totals"]
    statements, missing = totals["num_statements"], totals["missing_lines"]
    return {
        "passed": counts.get("passed", 0), "skipped": counts.get("skipped", 0),
        "seconds": re.search(r"in ([\d.]+)s", summary).group(1) if "in " in summary else "?",
        "statements": statements, "missing": missing,
        "statement_pct": round(100 * (statements - missing) / statements),
        "branch_pct": round(100 * totals["covered_branches"] / max(totals["num_branches"], 1)),
    }


def lines_for(key: str, m: dict[str, dict], today: str) -> str:
    def status(tree: str) -> str:
        r = m[tree]
        return (f"{r['passed']} passed, {r['skipped']} skipped, **{r['statement_pct']} %** statements / "
                f"**{r['branch_pct']} %** branches")

    def coverage_row(tree: str, label: str) -> str:
        r = m[tree]
        return (f"| {label}<!--auto:cov-{tree}--> | {r['statements']:,} | {r['missing']} | "
                f"**{r['statement_pct']} %** | **{r['branch_pct']} %** | JAX/Rust |")

    return {
        "readme-status": f"| Current tests<!--auto:readme-status--> | {status('exp1')} | {status('exp2')} |",
        "ctde-status": f"| Current tests<!--auto:ctde-status--> | {status('exp2')} |",
        "cov-exp1": coverage_row("exp1", "Experiment 1"),
        "cov-exp2": coverage_row("exp2", "Experiment 2 (`ctde_arena/`)"),
        "cmd-exp1": (f"| `python -m pytest tests/ -q`<!--auto:cmd-exp1--> | 1 | {m['exp1']['passed']} passed, "
                     f"{m['exp1']['skipped']} skipped (verified {today}) |"),
        "cmd-exp2": (f"| `cd ctde_arena && python -m pytest tests/ -q`<!--auto:cmd-exp2--> | 2 | "
                     f"{m['exp2']['passed']} passed, {m['exp2']['skipped']} skipped (verified {today}) |"),
        "pytest-block": "\n".join([
            "<!--auto:pytest-block-->",
            "```",
            "$ python -m pytest tests/ -q",
            f"{m['exp1']['passed']} passed, {m['exp1']['skipped']} skipped in {m['exp1']['seconds']}s",
            "$ cd ctde_arena && python -m pytest tests/ -q",
            f"{m['exp2']['passed']} passed, {m['exp2']['skipped']} skipped in {m['exp2']['seconds']}s",
            "```",
            "<!--/auto-->",
        ]),
    }[key]


def rewrite(path: Path, measured: dict[str, dict], today: str) -> int:
    text = path.read_text(encoding="utf-8")
    changed = 0
    block = re.compile(r"<!--auto:pytest-block-->.*?<!--/auto-->", re.DOTALL)
    if block.search(text):
        text = block.sub(lambda _: lines_for("pytest-block", measured, today), text)
        changed += 1
    out = []
    for line in text.split("\n"):
        match = MARKER.search(line)
        if match and match.group(1) != "pytest-block":
            line = lines_for(match.group(1), measured, today)
            changed += 1
        out.append(line)
    path.write_text("\n".join(out), encoding="utf-8", newline="\n")
    return changed


def documented_totals() -> dict[str, int]:
    """passed + skipped per tree, as the README status row states them."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if "<!--auto:readme-status-->" in line)
    pairs = re.findall(r"(\d+) passed, (\d+) skipped", row)
    return {"exp1": sum(map(int, pairs[0])), "exp2": sum(map(int, pairs[1]))}


def collected(tree: Path) -> int:
    out = subprocess.run([sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
                         cwd=tree, capture_output=True, text=True, check=True).stdout
    return int(re.search(r"(\d+) tests? collected", out).group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        documented = documented_totals()
        stale = {tree: (documented[tree], collected(path)) for tree, path in TREES.items()
                 if documented[tree] != collected(path)}
        if stale:
            raise SystemExit("documented test totals are stale (documented, collected): "
                             f"{stale}; run scripts/update_doc_counts.py and commit")
        print(f"documented test totals match the collected suites: {documented}")
        return
    measured = {tree: run_suite(path) for tree, path in TREES.items()}
    today = datetime.date.today().isoformat()
    for path in DOCS:
        print(f"{path.relative_to(ROOT)}: {rewrite(path, measured, today)} marked line(s) rewritten")
    print(json.dumps(measured, indent=2))


if __name__ == "__main__":
    main()
