"""The budget instrument must be able to tell "the ranking moved" from "the curve kept going".

`scripts/analyse_budget.py` exists for one question -- open-work item 7, does the study's ranking survive the
3,000,000-step budget the code defaults to -- and it answers it with two readings of two sources: held-out win
rates from the study CSVs, and the held-out curve each run snapshots at fractions of *its own* budget. Both
are easy to get wrong in a way that publishes the opposite of what it means: comparing the curves as if 0.25
of one budget were 0.25 of the other, pairing arms on the wrong seed, or trusting a run that stopped short of
the budget it was asked for.

The campaign's scratch is git-ignored, so both sides are built here from synthetic runs, with exactly one arm
given growth at the long budget. The assertion that matters is that the instrument separates the two
questions: the arm that moved is named, the arms that did not are flat in the paired table *and* in their
drift, and an incomplete run is refused rather than averaged.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyse_budget", ROOT / "scripts" / "analyse_budget.py")
budget = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budget)

EXPERIMENTS = {"exp1": ("CTE", "CTDE", "DTE"), "exp2": ("CTDE-VD", "CTDE-CAC", "CTDE-Comm")}
BASE = {"exp1": {"CTE": 0.250, "CTDE": 0.330, "DTE": 0.380},
        "exp2": {"CTDE-VD": 0.370, "CTDE-CAC": 0.280, "CTDE-Comm": 0.330}}
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
SEEDS = list(range(1, 11))
GROWTH = 0.05


def grown(experiment: str, paradigm: str, steps: int) -> float:
    """The one manipulation the fixtures make: CTDE-Comm improves at the long budget, nothing else does."""
    return GROWTH if (experiment, paradigm, steps) == ("exp2", "CTDE-Comm", 3_000_000) else 0.0


def write_run(tree_root: Path, runs_dir: str, experiment: str, seed: int, steps: int) -> None:
    teams = {f"Team {index + 1}": paradigm for index, paradigm in enumerate(EXPERIMENTS[experiment])}
    final = {p: BASE[experiment][p] + ((seed % 3) - 1) * 0.02 + grown(experiment, p, steps)
             for p in EXPERIMENTS[experiment]}
    curve = {p: [final[p] - 0.30 * grown(experiment, p, steps),
                 final[p] - 0.15 * grown(experiment, p, steps),
                 final[p] - 0.05 * grown(experiment, p, steps),
                 final[p]] for p in EXPERIMENTS[experiment]}
    run = tree_root / runs_dir / f"{experiment}_seed{seed}" / "evaluation"
    run.mkdir(parents=True, exist_ok=True)
    summary = {
        "seed": seed, "target_steps": steps, "completed_steps": steps, "matches_played": 4000,
        "train_seconds": 100.0, "paradigm_rotation": {}, "paradigms": teams, "training_cumulative": {},
        "held_out_greedy": {"matches": 150,
                            "per_team": {t: {"win_rate": final[p]} for t, p in teams.items()}},
        "curve": [{"fraction": fraction, "env_steps": int(steps * fraction), "elapsed_seconds": 1.0,
                   "per_team": {t: {"win_rate": curve[p][index]} for t, p in teams.items()}}
                  for index, fraction in enumerate(FRACTIONS)],
    }
    (run / "run_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def write_results(tmp_path: Path, runs_dir: str, name: str) -> None:
    """Collapse the synthetic runs into one per_seed_metrics.csv, the way run_study's collector does."""
    directory = tmp_path / "results" / name
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for tree in (tmp_path, tmp_path / "ctde_arena"):
        for experiment in EXPERIMENTS:
            for run in sorted((tree / runs_dir).glob(f"{experiment}_seed*")):
                summary = json.loads((run / "evaluation" / "run_summary.json").read_text(encoding="utf-8"))
                for team, paradigm in summary["paradigms"].items():
                    rate = summary["held_out_greedy"]["per_team"][team]["win_rate"]
                    rows.append([experiment, summary["seed"], team, paradigm, f"{rate:.4f}"])
    with (directory / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["experiment", "seed", "team", "paradigm", "eval_win_rate"])
        writer.writerows(rows)


@pytest.fixture()
def two_budgets(tmp_path: Path):
    """Both trees at both budgets, plus the two results directories the analysis compares.

    Experiment 1 is trained in the root tree and experiment 2 in `ctde_arena/`, which is where each campaign
    actually writes them, so the analysis has to search both.
    """
    half, three = Path("data/runs"), Path("data/runs_3m")
    for runs_dir, steps in ((half, 500_000), (three, 3_000_000)):
        for experiment in EXPERIMENTS:
            tree = tmp_path if experiment == "exp1" else tmp_path / "ctde_arena"
            for seed in SEEDS:
                write_run(tree, str(runs_dir), experiment, seed, steps)
    write_results(tmp_path, str(half), "study")
    write_results(tmp_path, str(three), "study_3m")
    return tmp_path, half, three


def analyse(tmp_path: Path, monkeypatch, three, half):
    monkeypatch.setattr(budget, "ROOT", tmp_path)
    return budget.analyse(three, half, Path("results/study_3m"))


def test_the_paired_table_compares_the_same_seed_on_both_sides(two_budgets, monkeypatch) -> None:
    tmp, half, three = two_budgets
    record = analyse(tmp, monkeypatch, three, half)

    assert len(record["paired_held_out"]) == 6
    for key, row in record["paired_held_out"].items():
        assert row["seeds"] == 10, key
        assert row["held_out_3m"] == pytest.approx(row["held_out_500k"] + row["paired_diff"]), key
    movers = sorted(key for key, row in record["paired_held_out"].items() if abs(row["paired_diff"]) > 1e-9)
    assert movers == ["exp2|CTDE-Comm"], movers
    assert record["paired_held_out"]["exp2|CTDE-Comm"]["paired_diff"] == pytest.approx(GROWTH, abs=1e-4)
    assert record["paired_held_out"]["exp1|DTE"]["paired_diff"] == pytest.approx(0.0, abs=1e-9)
    assert record["paired_held_out"]["exp2|CTDE-Comm"]["seeds_below_500k"] == 0


def test_the_curve_is_read_as_drift_within_its_own_budget(two_budgets, monkeypatch) -> None:
    tmp, half, three = two_budgets
    record = analyse(tmp, monkeypatch, three, half)

    comm = record["held_out_curves"]["exp2|CTDE-Comm"]
    assert list(comm["500k"]) == ["0.25", "0.5", "0.75", "1.0"], comm["500k"]
    assert comm["drift"]["3m"] == pytest.approx(0.30 * GROWTH, abs=1e-4), comm["drift"]
    assert comm["drift"]["500k"] == pytest.approx(0.0, abs=1e-9), comm["drift"]
    flat = record["held_out_curves"]["exp1|DTE"]
    assert flat["drift"]["500k"] == flat["drift"]["3m"] == 0.0, flat["drift"]
    assert "0.25 of 3,000,000" in record["controls"]["note"], record["controls"]["note"]


def test_the_ranking_lists_are_the_two_budgets_own_orders(two_budgets, monkeypatch) -> None:
    tmp, half, three = two_budgets
    record = analyse(tmp, monkeypatch, three, half)
    before = [row["paradigm"] for row in record["ranking"]["500k"]["exp2"]]
    after = [row["paradigm"] for row in record["ranking"]["3m"]["exp2"]]
    assert before == ["CTDE-VD", "CTDE-Comm", "CTDE-CAC"], before
    assert after == ["CTDE-Comm", "CTDE-VD", "CTDE-CAC"], after
    assert [r["paradigm"] for r in record["ranking"]["500k"]["exp1"]] \
        == [r["paradigm"] for r in record["ranking"]["3m"]["exp1"]]


def test_the_controls_name_the_budget_each_side_actually_trained(two_budgets, monkeypatch) -> None:
    tmp, half, three = two_budgets
    record = analyse(tmp, monkeypatch, three, half)
    budgets = record["controls"]["budgets"]
    assert budgets["500k"] == {"runs": 20, "target_steps": [500_000], "completed_steps": [500_000],
                               "held_out_matches_per_arm": [150]}, budgets["500k"]
    assert budgets["3m"] == {"runs": 20, "target_steps": [3_000_000], "completed_steps": [3_000_000],
                             "held_out_matches_per_arm": [150]}, budgets["3m"]
    assert record["controls"]["shared_cells"] == 60
    # The 500 k side is read from the committed study's own runs, so parity is the CSV's rounding floor.
    assert record["controls"]["half_m_parity_max_abs_delta_vs_committed_study"] <= 5e-5
    for key, value in record["inputs"].items():
        assert "\\" not in value, (key, value)


def test_a_run_that_stopped_short_of_its_budget_is_refused(two_budgets, monkeypatch) -> None:
    tmp, half, three = two_budgets
    shorted = sorted((tmp / "ctde_arena" / three).glob("exp2_seed*/evaluation/run_summary.json"))[0]
    summary = json.loads(shorted.read_text(encoding="utf-8"))
    summary["completed_steps"] = 2_500_000
    shorted.write_text(json.dumps(summary), encoding="utf-8")

    record = analyse(tmp, monkeypatch, three, half)
    with pytest.raises(SystemExit, match="did not complete every run"):
        budget.check(record)
