"""Section 8.13's configuration-ablation tables must be recomputable from the artefacts that produced them.

Two of these tables were rewritten by the ten-seed `no_dr` campaign, and one of them changed its headline: the
three-seed arm table put plain CTDE 0.193 *below* its control, the ten-seed one puts it 0.074 *above*, and the
three-block pool table put CTE second where seven more blocks put it sixth. A section whose conclusions move
that much on more replication cannot rest on a transcription of a terminal. Every published cell here is
recomputed from `results/ablation_*` and the two pooled `results/cross_arena/*_blocks.json` reports, and the
prose counts and percentages -- how many pairs separate, how many survive leave-one-block-out, how the matches
end -- are derived from the same files.
"""

from __future__ import annotations

import importlib.util
import json
import re
import pytest
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
DOC = ROOT / "docs" / "08_results.md"
COMMITTED = ROOT / "results" / "cross_arena" / "blocks10.json"
NODR_POOL = ROOT / "results" / "cross_arena" / "no_dr_blocks.json"
NODR_SUPERSEDED = ROOT / "results" / "cross_arena" / "no_dr_blocks3.json"
NODR_BUDGET = ROOT / "results" / "study_3m" / "budget_analysis.json"
NODR_CSV = ROOT / "results" / "cross_arena" / "no_dr_blocks.csv"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_ab_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    rest = text[start + len(heading):]
    stop = re.search(r"^#{2,3} ", rest, re.M)
    return rest[: stop.start()] if stop else rest


def rows(block: str) -> list[list[str]]:
    out = []
    for line in block.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells and set("".join(cells)) <= {"-", ":", " "}:
            continue
        out.append(cells)
    return out


def number(cell: str) -> float:
    return float(cell.replace("**", "").replace("−", "-").replace("×", "").strip())


def nearly(document: float, recorded: float, places: int) -> bool:
    return round(document, places) == round(recorded, places)


def ranks_of(per_architecture: dict) -> dict[str, int]:
    order = sorted(per_architecture, key=lambda a: -per_architecture[a]["win_rate"])
    return {architecture: index + 1 for index, architecture in enumerate(order)}


def rank(cell: str) -> int:
    digits = re.match(r"\s*(\d+)", cell.replace("**", ""))
    assert digits, cell
    return int(digits.group(1))


def test_the_no_dr_pool_rate_table_is_both_reports() -> None:
    text = DOC.read_text(encoding="utf-8")
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    nodr = json.loads(NODR_POOL.read_text(encoding="utf-8"))
    table = rows(section(text, "### The no_dr checkpoints against the common pool"))
    body = [r for r in table if len(r) == 4 and r[0] in nodr["per_architecture"]]
    assert len(body) == 6, f"parsed {len(body)} rows of the rate table"
    for architecture, versus, under, positions in body:
        assert nearly(number(versus), committed["per_architecture"][architecture]["win_rate"], 3), architecture
        assert nearly(number(under), nodr["per_architecture"][architecture]["win_rate"], 3), architecture
        before, after = positions.split("→")
        assert rank(before) == ranks_of(committed["per_architecture"])[architecture], (architecture, positions)
        assert rank(after) == ranks_of(nodr["per_architecture"])[architecture], (architecture, positions)
    assert len(nodr["training_blocks"]) == 10, nodr["training_blocks"]
    assert len(nodr["per_architecture"]) == 6


def test_the_no_dr_pool_spread_table_is_both_reports() -> None:
    text = DOC.read_text(encoding="utf-8")
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    nodr = json.loads(NODR_POOL.read_text(encoding="utf-8"))
    table = rows(section(text, "### The no_dr checkpoints against the common pool"))
    body = [r for r in table if len(r) == 5 and r[0] in nodr["per_architecture"]]
    assert len(body) == 6, f"parsed {len(body)} rows of the spread table"
    for row in body:
        architecture = row[0]
        for column, report in ((1, committed), (2, nodr)):
            levels = list(report["training_block_stability"][architecture]["levels"].values())
            assert nearly(number(row[column]), statistics.pstdev(levels), 3), (architecture, row[column])
        levels = list(nodr["training_block_stability"][architecture]["levels"].values())
        low, high = row[4].split("–")
        assert nearly(number(low), min(levels), 3), architecture
        assert nearly(number(high), max(levels), 3), architecture
        ratio = (statistics.pstdev(levels)
                 / statistics.pstdev(list(committed["training_block_stability"][architecture]["levels"].values())))
        assert nearly(number(row[3]), ratio, 1), (architecture, row[3], ratio)


def test_the_no_dr_pool_prose_counts_and_percentages_are_the_reports() -> None:
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    nodr = json.loads(NODR_POOL.read_text(encoding="utf-8"))

    def separations(report: dict) -> tuple[int, int]:
        significant = sum(1 for v in report["pairwise_mcnemar"].values() if v["verdict"] == "**significant**")
        robust = sum(1 for v in report["block_sensitivity"].values() if v["robust_to_block"])
        return significant, robust

    assert separations(nodr) == (10, 4), separations(nodr)
    assert separations(committed) == (4, 3), separations(committed)
    for claim in ("Ten of fifteen pairs clear the Bonferroni bar (alpha = 0.00333) in this pool against "
                  "**four of fifteen** in the committed one, and four are robust to leaving a block out",
                  "against three in the committed pool"):
        assert claim in text, claim

    percentages = {}
    for name, report in (("no_dr", nodr), ("committed", committed)):
        outcomes, total = report["match_outcomes"], report["matches"]
        percentages[name] = {key: round(100 * outcomes[key] / total, 1)
                             for key in ("wipeout", "timeout", "draw")}
    assert percentages["no_dr"] == {"wipeout": 44.8, "timeout": 46.4, "draw": 8.8}, percentages["no_dr"]
    assert percentages["committed"] == {"wipeout": 58.7, "timeout": 38.2, "draw": 3.1}, percentages["committed"]
    assert ("44.8 % of the no_dr pool's matches ended in a wipeout against 58.7 % of the committed pool's, "
            "46.4 % were decided with the clock running out against 38.2 %, and 8.8 % were draws against "
            "3.1 %") in text

    # The largest move the pool instrument attributes to this key, quoted in the capacity subsection.
    # The difference of the two rates as the rate table prints them, not of the unrounded values -- the same
    # convention the rule-control gate uses, so a reader subtracting two printed cells arrives at this number.
    moves = [abs(round(nodr["per_architecture"][a]["win_rate"], 3)
                 - round(committed["per_architecture"][a]["win_rate"], 3)) for a in nodr["per_architecture"]]
    assert round(max(moves), 3) == 0.060, max(moves)
    assert "turning domain randomisation off moves an arm by at most 0.060" in text


def paired(study, control: dict, arm: dict, experiment: str, paradigm: str) -> tuple:
    """The published row for one (experiment, paradigm): means, paired difference, t, p, count below."""
    keys = [k for k in sorted(set(control) & set(arm)) if k[0] == experiment and k[1] == paradigm]
    pairs = [(control[k], arm[k]) for k in keys]
    differences = [float(a["eval_win_rate"]) - float(c["eval_win_rate"]) for c, a in pairs]
    t_stat, p_value = study.paired_t(differences)
    return (sum(float(c["eval_win_rate"]) for c, _ in pairs) / len(pairs),
            sum(float(a["eval_win_rate"]) for _, a in pairs) / len(pairs),
            sum(differences) / len(differences), t_stat, p_value,
            sum(1 for d in differences if d < 0), len(pairs))


def test_the_no_dr_arm_table_is_paired_against_the_study_on_ten_seeds() -> None:
    text = DOC.read_text(encoding="utf-8")
    study = load_script("report_study")
    control = {(r["experiment"], r["paradigm"], int(r["seed"])): r
               for r in study.read_metrics(ROOT / "results" / "study")}
    arm = {(r["experiment"], r["paradigm"], int(r["seed"])): r
           for r in study.read_metrics(ROOT / "results" / "ablation_no_dr")}
    assert len(set(control) & set(arm)) == 60, "expected ten seeds x six paradigms shared with the control"

    table = rows(section(text, "### A configuration key moves the outcome more than a paradigm does"))
    body = [r for r in table if len(r) == 7 and r[0].startswith("**exp")]
    assert len(body) == 6, body
    cells = {}
    for row in body:
        header = re.match(r"\*\*(exp\d)\*\*\s+(\S+)", row[0])
        assert header, row[0]
        key = (header.group(1), paradigm_of(header.group(2)))
        control_mean, arm_mean, diff, t_stat, p_value, below, seeds = paired(study, control, arm, *key)
        assert seeds == 10, (key, seeds)
        values = [c.replace("**", "") for c in row[1:]]
        assert nearly(float(values[0]), control_mean, 3), key
        assert nearly(float(values[1]), arm_mean, 3), key
        assert nearly(number(values[2]), diff, 3), key
        assert nearly(number(values[3]), t_stat, 2), key
        assert nearly(float(values[4]), p_value, 3), key
        assert values[5] == f"{below}/10", key
        cells[key] = diff

    largest = max(abs(value) for value in cells.values())
    assert nearly(largest, 0.161, 3), largest
    assert "largest single-key move is 0.161" in " ".join(text.split())
    # The correction itself has to stay stated: the three-seed table this one replaced put CTDE under its
    # control by 0.193, and the ten-seed table puts it above by 0.074.
    assert "−0.193" in text and "+0.074" in text


def test_the_capacity_arm_table_is_paired_on_the_three_seeds_it_has() -> None:
    text = DOC.read_text(encoding="utf-8")
    study = load_script("report_study")
    control_rows = study.read_metrics(ROOT / "results" / "study")
    table = rows(section(text, "### A configuration key moves the outcome more than a paradigm does"))
    body = [r for r in table if len(r) == 8 and r[0].startswith("**exp")]
    assert len(body) == 6, body
    for row in body:
        header = re.match(r"\*\*(exp\d)\*\*\s+(\S+)", row[0])
        experiment, paradigm = header.group(1), paradigm_of(header.group(2))
        for arm, rate_column, diff_column, p_column in (("narrow_64", 2, 3, 4), ("wide_256", 5, 6, 7)):
            arm_rows = study.read_metrics(ROOT / "results" / f"ablation_{arm}")
            keyed = {(r["experiment"], r["paradigm"], int(r["seed"])): r for r in arm_rows}
            shared = [k for k in sorted(set((r["experiment"], r["paradigm"], int(r["seed"])) for r in control_rows))
                      if k in keyed]
            assert len(shared) == 18, f"{arm}: {len(shared)} cells, expected three seeds x six paradigms"
            control = {(r["experiment"], r["paradigm"], int(r["seed"])): r for r in control_rows}
            pairs = [(control[k], keyed[k]) for k in shared if k[0] == experiment and k[1] == paradigm]
            mean_control = sum(float(c["eval_win_rate"]) for c, _ in pairs) / len(pairs)
            mean_arm = sum(float(a["eval_win_rate"]) for _, a in pairs) / len(pairs)
            differences = [float(a["eval_win_rate"]) - float(c["eval_win_rate"]) for c, a in pairs]
            _, p_value = study.paired_t(differences)
            if arm == "narrow_64":
                assert nearly(float(row[1].replace("**", "")), mean_control, 3), row
            assert nearly(float(row[rate_column].replace("**", "")), mean_arm, 3), (row, arm)
            assert nearly(number(row[diff_column]), sum(differences) / len(differences), 3), (row, arm)
            assert nearly(float(row[p_column].replace("**", "")), p_value, 3), (row, arm)


ABBREVIATED = {"CAC": "CTDE-CAC", "Comm": "CTDE-Comm", "VD": "CTDE-VD"}


def paradigm_of(cell: str) -> str:
    """The row label as the CSV spells it: § 8.13 abbreviates experiment 2's three variants."""
    return ABBREVIATED.get(cell, cell)


def cell_means(table: dict, keys: list[tuple], column: str) -> dict[tuple, float]:
    """Mean over the shared seeds, per (experiment, paradigm) -- which is what the published row is.

    The ranges in § 8.13 are spans of those row means, not of the sixty underlying cells, and the firing rate
    is computed from the means rather than averaged over per-cell rates, exactly as `report_study` does it.
    """
    out: dict[tuple, list[float]] = {}
    for key in keys:
        out.setdefault((key[0], key[1]), []).append(float(table[key][column]))
    return {cell: sum(values) / len(values) for cell, values in out.items()}


def test_the_behavioural_ranges_are_the_two_csvs() -> None:
    """The survival, elimination and firing ranges § 8.13 quotes for `no_dr` are columns of these files."""
    study = load_script("report_study")
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    control = {(r["experiment"], r["paradigm"], int(r["seed"])): r
               for r in study.read_metrics(ROOT / "results" / "study")}
    arm = {(r["experiment"], r["paradigm"], int(r["seed"])): r
           for r in study.read_metrics(ROOT / "results" / "ablation_no_dr")}
    keys = sorted(set(control) & set(arm))

    spans = {}
    for column in ("eval_mean_survival", "eval_eliminations_per_match", "eval_shots_per_match"):
        control_cells, arm_cells = cell_means(control, keys, column), cell_means(arm, keys, column)
        spans[column] = ((min(control_cells.values()), max(control_cells.values())),
                         (min(arm_cells.values()), max(arm_cells.values())))
        assert set(control_cells) == set(arm_cells) == {(e, p) for e, p, _ in keys}, column

    rate_control = {}
    rate_arm = {}
    control_shots = cell_means(control, keys, "eval_shots_per_match")
    control_life = cell_means(control, keys, "eval_mean_survival")
    arm_shots = cell_means(arm, keys, "eval_shots_per_match")
    arm_life = cell_means(arm, keys, "eval_mean_survival")
    for cell in control_shots:
        rate_control[cell] = study.shots_per_alive_second(control_shots[cell], control_life[cell])
        rate_arm[cell] = study.shots_per_alive_second(arm_shots[cell], arm_life[cell])
    ratios = [rate_control[c] / rate_arm[c] for c in rate_control]

    survival = spans["eval_mean_survival"]
    eliminations = spans["eval_eliminations_per_match"]
    shots = spans["eval_shots_per_match"]
    claims = (
        f"({survival[0][0]:.1f}–{survival[0][1]:.1f} s in the control to {survival[1][0]:.1f}–"
        f"{survival[1][1]:.1f} s)",
        f"({eliminations[0][0]:.2f}–{eliminations[0][1]:.2f} → {eliminations[1][0]:.2f}–"
        f"{eliminations[1][1]:.2f} per match)",
        f"({shots[0][0]:.0f}–{shots[0][1]:.0f} → {shots[1][0]:.0f}–{shots[1][1]:.0f} shots per match)",
        f"from {min(rate_control.values()):.2f}–{max(rate_control.values()):.2f} to "
        f"{min(rate_arm.values()):.2f}–{max(rate_arm.values()):.2f}",
        f"a drop of {min(ratios):.1f}× to {max(ratios):.1f}×",
    )
    for claim in claims:
        assert claim in text, claim


def test_the_superseded_three_block_numbers_are_the_rescued_record() -> None:
    """The replacement paragraph compares ten blocks against three, so the three-block record has to exist.

    The ten-block merge overwrote `no_dr_blocks.csv`, and a correction that quotes numbers from an artefact
    nobody can re-open is only one half of a correction. The superseded pool is kept as `no_dr_blocks3.*`, and
    every figure the paragraph attributes to it is checked against that file here.
    """
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    old = json.loads(NODR_SUPERSEDED.read_text(encoding="utf-8"))
    assert len(old["training_blocks"]) == 3 and old["matches"] == 9_000, old["training_blocks"]

    rates = {a: round(v["win_rate"], 3) for a, v in old["per_architecture"].items()}
    assert rates == {"CTDE": 0.386, "CTE": 0.316, "CTDE-VD": 0.300, "CTDE-CAC": 0.300,
                     "CTDE-Comm": 0.243, "DTE": 0.203}, rates
    spread = {a: statistics.pstdev(list(v["levels"].values()))
              for a, v in old["training_block_stability"].items()}
    assert round(max(spread.values()), 3) == 0.223, spread
    smallest = min(spread, key=lambda a: spread[a])
    assert smallest == "DTE" and round(spread[smallest], 3) == 0.018, spread
    assert round(statistics.pstdev(
        list(json.loads(NODR_POOL.read_text(encoding="utf-8"))["training_block_stability"][smallest]
             ["levels"].values())), 3) == 0.111

    assert sum(1 for v in old["pairwise_mcnemar"].values()
               if v["verdict"] == "**significant**") == 9
    assert sum(1 for v in old["block_sensitivity"].values() if v["robust_to_block"]) == 2
    outcomes = old["match_outcomes"]
    assert round(100 * outcomes["wipeout"] / old["matches"], 1) == 41.1
    assert round(100 * outcomes["draw"] / old["matches"], 1) == 12.6

    cte_levels = sorted(old["training_block_stability"]["CTE"]["levels"].values())
    assert sum(1 for level in cte_levels if level > 0.40) == 2, cte_levels

    for claim in ("CTDE 0.386 first, CTE 0.316 **second** (from sixth), CTDE-VD and CTDE-CAC tied at 0.300, "
                  "CTDE-Comm 0.243 fifth, DTE 0.203 last",
                  "between-block SD up to 0.223",
                  "nine of fifteen pairs separating and two surviving leave-one-block-out",
                  "its 0.316 came from two of three blocks landing above 0.400",
                  "DTE's SD, the smallest in the three-block table at 0.018, is 0.111 over ten blocks"):
        assert claim in text, claim

def test_the_budget_tables_are_the_pair_study_record() -> None:
    """Section 8.13's fourth arm: every cell of both budget tables is recomputed from budget_analysis.json.

    This is the one place in the chapter where the *same* arms are compared at two training budgets, so the
    numbers are load-bearing for the claim that the ordering is an estimate at a budget rather than a property
    of the architectures -- and the drift column is only meaningful because the curves are fractions of each
    run's own budget, which the record itself states.
    """
    text = " ".join(DOC.read_text(encoding="utf-8").split())
    record = json.loads(NODR_BUDGET.read_text(encoding="utf-8"))
    paired = record["paired_held_out"]
    assert len(paired) == 6

    rows = [[c.strip() for c in line.strip().strip("|").split("|")]
            for line in DOC.read_text(encoding="utf-8").splitlines() if line.startswith("| exp")]
    cells_of = [r for r in rows if len(r) == 8 and r[0].startswith("exp")]
    assert len(cells_of) == 6, rows
    for cells in cells_of:
        experiment, paradigm = cells[0], cells[1]
        key = f"{experiment}|{paradigm}"
        row = paired[key]
        assert float(cells[2].replace("**", "")) == pytest.approx(row["held_out_500k"], abs=5e-4), key
        assert float(cells[3].replace("**", "")) == pytest.approx(row["held_out_3m"], abs=5e-4), key
        assert abs(float(cells[4].replace("−", "-").replace("**", ""))) == pytest.approx(
            abs(row["paired_diff"]), abs=5e-4), key
        assert float(cells[5].replace("−", "-")) == pytest.approx(row["t"], abs=5e-3), key
        assert float(cells[6]) == pytest.approx(row["p"], abs=5e-3), key
        assert cells[7] == f"{row['seeds_below_500k']}/10", key
        assert row["seeds"] == 10, key

    assert "not one of them clears it" in text and "alpha = 0.00833" in text
    ranking = record["ranking"]
    assert [r["paradigm"] for r in ranking["500k"]["exp1"]] == ["DTE", "CTDE", "CTE"]
    assert [r["paradigm"] for r in ranking["3m"]["exp1"]] == ["CTDE", "DTE", "CTE"]
    assert [r["paradigm"] for r in ranking["500k"]["exp2"]] == ["CTDE-VD", "CTDE-Comm", "CTDE-CAC"]
    assert [r["paradigm"] for r in ranking["3m"]["exp2"]] == ["CTDE-Comm", "CTDE-CAC", "CTDE-VD"]
    for experiment, label in (("exp1", "DTE 0.381 > CTDE 0.326 > CTE 0.252"),
                              ("exp2", "VD 0.367 > Comm 0.331 > CAC 0.282")):
        assert label in text, experiment
    assert "| **CTDE 0.429 > DTE 0.365 > CTE 0.159** |" in text
    assert "| **Comm 0.395 > CAC 0.290 > VD 0.277** |" in text

    curves = record["held_out_curves"]
    for key, row in curves.items():
        experiment, paradigm = key.split("|")
        sequence = " → ".join(f"{row['3m'][fraction]:.3f}"
                                   for fraction in ("0.25", "0.5", "0.75", "1.0"))
        # The chapter prints negatives with a real minus sign, so the comparison has to as well.
        drifts = " | ".join(f"{row['drift'][side]:+.3f}".replace("-", "−") for side in ("500k", "3m"))
        claim = f"| {experiment} | {paradigm} | {sequence} | {drifts} |"

        assert claim in DOC.read_text(encoding="utf-8"), claim

    assert record["controls"]["budgets"]["3m"]["target_steps"] == [3_000_000]
    assert record["controls"]["budgets"]["500k"]["target_steps"] == [500_000]
    assert record["controls"]["half_m_parity_max_abs_delta_vs_committed_study"] <= 5e-5
