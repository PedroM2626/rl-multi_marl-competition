"""Section 8.1's regime tables must be the replay's artefacts, not a transcription.

Audit item 8's open half was closed by replaying the study's held-out protocol with the per-match
duration recorded (`results/study/eval_match_regimes.csv`). The claims that replay licenses are strong:
that the pooled order is the timeout order in both experiments, that the fight winners are CTDE and
Comm, and that the replay reproduces the published win rates on 52 of 60 cells with a worst cell of 2
wins of 150. Every one of those is recomputed here from the committed rows, so the section cannot keep a
number an earlier replay produced -- or one no replay produced at all.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results" / "study"
RESULTS = ROOT / "docs" / "08_results.md"
REGIMES = ("wipeout", "timeout", "draw")


def replay_rows() -> list[dict[str, str]]:
    with (STUDY / "eval_match_regimes.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def match_outcomes(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], str]:
    campaign = json.loads((STUDY / "regime_summary.json").read_text(encoding="utf-8"))
    cap = float(campaign["protocol"]["arena"]["match_duration_seconds"])
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["experiment"], row["seed"], row["match_index"])].append(row)
    outcomes = {}
    for key, team_rows in grouped.items():
        if not any(int(row["winner"]) for row in team_rows):
            outcomes[key] = "draw"
        elif float(team_rows[0]["duration_seconds"]) >= cap - 0.5:
            outcomes[key] = "timeout"
        else:
            outcomes[key] = "wipeout"
    return outcomes


def per_arm_tallies(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, list[int]]]:
    outcomes = match_outcomes(rows)
    tally: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: {r: [0, 0] for r in REGIMES})
    for row in rows:
        outcome = outcomes[(row["experiment"], row["seed"], row["match_index"])]
        cell = tally[(row["experiment"], row["paradigm"])][outcome]
        cell[1] += 1
        cell[0] += int(row["winner"])
    return tally


def per_arm_rates(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, float]]:
    return {
        key: {
            regime: (cell[0] / cell[1] if cell[1] else None)
            for regime, cell in regimes.items()
        }
        for key, regimes in per_arm_tallies(rows).items()
    }


def section_text() -> str:
    text = RESULTS.read_text(encoding="utf-8")
    start = text.index("### Two regimes: wipeouts and timeouts")
    end = text.index("### The historical conclusions do not survive")
    return text[start:end]


def test_the_by_regime_tables_are_the_replays_numbers() -> None:
    rows = replay_rows()
    rates = per_arm_rates(rows)
    tallies = per_arm_tallies(rows)
    parsed = 0
    for line in section_text().splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 6 or not cells[0] or cells[0] in ("Exp. 1", "Exp. 2", "architecture"):
            continue
        paradigm = cells[0].strip("*")
        if not re.fullmatch(r"\*{0,2}\d\.\d{3}\*{0,2} \(\d+\)", cells[1]):
            continue
        parsed += 1
        experiment = "exp1" if paradigm in {"CTE", "DTE", "CTDE"} else "exp2"
        record = rates[(experiment, paradigm)]
        for index, regime in enumerate(REGIMES):
            quoted_rate, quoted_count = cells[1 + index].rstrip(")").split(" (")
            assert abs(float(quoted_rate.strip("*")) - record[regime]) < 0.0005, (paradigm, regime)
            assert int(quoted_count) == tallies[(experiment, paradigm)][regime][1], (paradigm, regime)
        cell_tallies = tallies[(experiment, paradigm)]
        total_wins = sum(cell_tallies[r][0] for r in REGIMES)
        total_matches = sum(cell_tallies[r][1] for r in REGIMES)
        pooled = total_wins / total_matches
        assert abs(float(cells[4].strip("*")) - pooled) < 0.0005, f"{paradigm}: pooled {cells[4]} vs {pooled:.4f}"
        gap = record["wipeout"] - record["timeout"]
        assert abs(float(cells[5].replace("−", "-").strip("*")) - gap) < 0.0011, \
            f"{paradigm}: gap {cells[5]} vs {gap:.4f}"
    assert parsed == 6, f"the by-regime tables parsed {parsed} rows, expected six"


def test_the_regime_shares_and_clock_replicates_are_the_replays_counts() -> None:
    rows = replay_rows()
    outcomes = match_outcomes(rows)
    text = section_text()

    counts: dict[str, dict[str, int]] = {e: dict.fromkeys(REGIMES, 0) for e in ("exp1", "exp2")}
    per_replicate: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: dict.fromkeys(REGIMES, 0))
    for (experiment, seed, _), outcome in outcomes.items():
        counts[experiment][outcome] += 1
        per_replicate[(experiment, seed)][outcome] += 1

    shares = re.search(
        r"\*\*(\d+\.\d) %\*\*\s+of\s+experiment\s+1's\s+and\s+\*\*(\d+\.\d) %\*\*\s+of\s+experiment\s+2's\s+"
        r"end\s+by\s+elimination;\s+\*\*(\d+\.\d) %\s+/\s+(\d+\.\d) %\*\*\s+run\s+to\s+the\s+90\s+s\s+"
        r"clock;\s+\*\*(\d+\.\d) %\s+/\s+(\d+\.\d) %\*\*\s+are\s+draws", text)
    assert shares, "the opening share sentence was reworded; this test reads its numbers out of it"
    for experiment, group in (("exp1", (1, 3, 5)), ("exp2", (2, 4, 6))):
        for regime, group_index in zip(REGIMES, group):
            quoted = float(shares.group(group_index))
            measured = counts[experiment][regime] / 1500 * 100
            assert abs(measured - quoted) < 0.05, (experiment, regime, quoted, measured)

    clock_replicates = {("exp1", "6"), ("exp1", "8"), ("exp2", "2"), ("exp2", "4"), ("exp2", "9")}
    named_rows = 0
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 4 or "_seed" not in cells[0]:
            continue
        experiment, _, seed = cells[0].partition("_seed")
        named_rows += 1
        record = per_replicate[(experiment, seed)]
        for index, regime in enumerate(REGIMES):
            assert int(cells[1 + index]) == record[regime], (cells[0], regime)
    assert named_rows == 5, f"the clock-dominant table parsed {named_rows} named rows, expected five"

    quoted = re.search(r"\| the other fifteen \| (\d+)[–-](\d+) \| (\d+)[–-](\d+) \| (\d+)[–-](\d+) \|",
                       text)
    assert quoted, "the 'other fifteen' range row is missing"
    pairs = list(zip((quoted.group(1), quoted.group(3), quoted.group(5)),
                     (quoted.group(2), quoted.group(4), quoted.group(6))))
    others = [counts for key, counts in per_replicate.items() if key not in clock_replicates]
    for index, regime in enumerate(REGIMES):
        low, high = (int(pairs[index][0]), int(pairs[index][1]))
        values = [record[regime] for record in others]
        assert min(values) >= low and max(values) <= high, (regime, min(values), max(values), low, high)


def test_the_control_statement_matches_the_artefacts() -> None:
    rows = replay_rows()
    published = {(r["experiment"], r["seed"], r["team"]): float(r["eval_win_rate"])
                 for r in csv.DictReader((STUDY / "per_seed_metrics.csv").open(newline="", encoding="utf-8"))}
    wins: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        wins[(row["experiment"], row["seed"], row["team_name"])] += int(row["winner"])
    exact = sum(1 for key, count in wins.items()
                if abs(count - published[key] * 150) < 0.5)
    worst = max(abs(count - published[key] * 150) for key, count in wins.items())
    text = section_text()
    quoted_cells = re.search(r"on \*\*(\d+) of\s+60 cells\*\*", text)
    quoted_worst = re.search(r"\*\*(\d+) wins of\s+150\*\*", text)
    assert quoted_cells and quoted_worst, "the control sentence was reworded; update this test with it"
    assert int(quoted_cells.group(1)) == exact, f"the document says {quoted_cells.group(1)} exact, the rows say {exact}"
    assert int(quoted_worst.group(1)) == round(worst), f"the document says {quoted_worst.group(1)}, the rows say {worst:.2f}"
    split = json.loads((STUDY / "regime_split.json").read_text(encoding="utf-8"))
    assert split["control"]["cells_exact"] == exact
    assert abs(split["control"]["max_replayed_vs_published_wins"] - worst) < 1e-6


def test_the_baseline_tables_regime_cells_are_the_replays_numbers() -> None:
    """§ 8.9's 'Trained' column used to say 'mixed' for the regime split; it quotes the replay now."""
    rows = replay_rows()
    outcomes = match_outcomes(rows)
    shares: dict[str, dict[str, float]] = {"exp1": {}, "exp2": {}}
    for experiment in ("exp1", "exp2"):
        counts = dict.fromkeys(REGIMES, 0)
        for (replicate_experiment, _, _), outcome in outcomes.items():
            if replicate_experiment == experiment:
                counts[outcome] += 1
        for regime in REGIMES:
            shares[experiment][regime] = counts[regime] / 1500 * 100

    text = RESULTS.read_text(encoding="utf-8")
    for experiment, wipeout_quoted, cap_quoted, draw_quoted in (
            ("exp1", "57.9 % — [§ 8.1](#two-regimes)", "37.9 % / 4.2 %", None),
            ("exp2", "55.5 % — [§ 8.1](#two-regimes)", "42.4 % / 2.1 %", None)):
        assert wipeout_quoted in text, f"experiment {experiment}'s wipeout share is no longer quoted"
        assert cap_quoted in text, f"experiment {experiment}'s cap/draw share is no longer quoted"
        assert abs(float(wipeout_quoted.split(" ")[0]) - shares[experiment]["wipeout"]) < 0.05
        cap, draw = (float(part.rstrip(" %")) for part in cap_quoted.split(" / "))
        assert abs(cap - shares[experiment]["timeout"]) < 0.05
        assert abs(draw - shares[experiment]["draw"]) < 0.05
