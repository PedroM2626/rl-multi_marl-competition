"""Gates for the two aggregation steps a campaign depends on: collecting run summaries, and pairing arms.

A-35 is the reason the first group exists. `run_study.py` launches one child per replicate with `cwd` in that
experiment's tree, so experiment 2's runs land under `ctde_arena/data/` while experiment 1's land under
`data/`; the collector globbed only its own root and produced an arena-v2 aggregate of 30 rows where the
campaign had trained 60. Half a campaign, no failure, and a plausible table. `collect()` now returns the
paths it could not read and main reconciles them against the number launched -- both halves are pinned here,
on a scratch pair of trees that is removed again.

`report_study.compare_arms` is the instrument behind every ablation table in docs/08 section 8.13, and it
pairs on the seed: a row whose sign or pairing is wrong reverses a published conclusion, so the arithmetic is
checked on hand-built rows whose difference is known by construction.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
SCRATCH = Path("data") / "_collection_probe"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_sc_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_study = _load("run_study")
report_study = _load("report_study")


def _write_summary(base: Path, experiment: str, seed: int, paradigm: str, win_rate: float) -> Path:
    """One run summary with exactly the fields `collect()` reads."""
    directory = base / f"{experiment}_seed{seed}" / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    teams = {f"Team {i}": paradigm if i == 1 else f"other{i}" for i in (1, 2, 3)}
    payload = {
        "seed": seed,
        "matches_played": 2369,
        "train_seconds": 1720.5,
        "steps_per_second": 290.6,
        "paradigms": teams,
        "training_cumulative": {team: {"win_rate": 0.5} for team in teams},
        "held_out_greedy": {
            "matches": 150,
            "per_team": {
                team: {"win_rate": win_rate, "eliminations_per_match": 1.5,
                       "mean_survival_time": 40.0, "shots_per_match": 100.0, "shot_accuracy": 0.02}
                for team in teams
            },
        },
    }
    path = directory / "run_summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def probe_trees():
    """A scratch replicate in each tree, removed whatever the test does."""
    paths = [
        _write_summary(PROJECT_ROOT / SCRATCH, "exp1", 1, "CTE", 0.4),
        _write_summary(PROJECT_ROOT / "ctde_arena" / SCRATCH, "exp2", 1, "CTDE-VD", 0.2),
    ]
    yield paths
    for base in (PROJECT_ROOT, PROJECT_ROOT / "ctde_arena"):
        shutil.rmtree(base / SCRATCH, ignore_errors=True)


def test_summary_paths_reads_both_trees(probe_trees) -> None:
    """The A-35 regression: one launched campaign, two output roots, and the aggregate has to see both."""
    found = run_study.summary_paths(SCRATCH)
    assert len(found) == 2, found
    assert {Path(path).parents[1].name for path in found} == {"exp1_seed1", "exp2_seed1"}
    records, missing = run_study.collect(found)
    assert missing == []
    assert sorted({str(r["experiment"]) for r in records}) == ["exp1", "exp2"], \
        "an aggregate that sees one experiment is what A-35 shipped as"
    assert len(records) == 6, "three teams per tree"


def test_summary_paths_is_absolute_safe(probe_trees, tmp_path: Path) -> None:
    """An absolute output directory is read once, not once per tree, so paths cannot double-count."""
    _write_summary(tmp_path, "exp1", 7, "CTE", 0.3)
    found = run_study.summary_paths(tmp_path)
    assert len(found) == len(set(found)) == 1


def test_collect_reports_what_it_could_not_read(probe_trees) -> None:
    """A missing summary must be named, not skipped: silent skipping is how half a campaign vanished."""
    found = run_study.summary_paths(SCRATCH) + [
        str(PROJECT_ROOT / SCRATCH / "exp1_seed9" / "evaluation" / "run_summary.json")]
    records, missing = run_study.collect(found)
    assert len(missing) == 1 and "exp1_seed9" in missing[0], missing
    assert len(records) == 6, "the readable half still collects"


def test_collect_row_carries_the_lineage_of_its_run(probe_trees) -> None:
    """Experiment and paradigm come from the path and the payload, not from the file name alone."""
    records, _ = run_study.collect(run_study.summary_paths(SCRATCH))
    row = next(r for r in records if r["experiment"] == "exp1" and r["team"] == "Team 1")
    assert row["paradigm"] == "CTE" and row["seed"] == 1 and row["eval_matches"] == 150
    assert row["eval_win_rate"] == pytest.approx(0.4)
    assert row["train_win_rate"] == pytest.approx(0.5), "training and held-out are different numbers"


def _rows(experiment: str, paradigm: str, seeds: dict[int, float]) -> list[dict]:
    return [{"experiment": experiment, "paradigm": paradigm, "seed": seed, "eval_win_rate": rate,
             "eval_mean_survival": 30.0 + rate, "eval_eliminations_per_match": 2.0,
             "eval_shots_per_match": 50.0} for seed, rate in seeds.items()]


def test_read_metrics_types_the_seed_as_an_int(tmp_path: Path) -> None:
    """Everything else is a float; the seed is the pairing key and a string seed would never match."""
    with (tmp_path / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "paradigm", "team", "seed", "eval_win_rate"])
        writer.writeheader()
        writer.writerows([{"experiment": "exp1", "paradigm": "CTE", "team": "Team 1",
                           "seed": "3", "eval_win_rate": "0.25"}])
    rows = report_study.read_metrics(tmp_path)
    assert rows[0]["seed"] == 3 and isinstance(rows[0]["seed"], int)
    assert rows[0]["eval_win_rate"] == pytest.approx(0.25)
    assert rows[0]["experiment"] == "exp1"


def test_shots_are_priced_per_second_of_alive_life() -> None:
    """The normalisation behind the study's firing-volume column: three agents, each alive `mean_survival`."""
    assert report_study.shots_per_alive_second(60.0, 20.0) == pytest.approx(1.0)
    assert report_study.shots_per_alive_second(60.0, 0.0) == 0.0, "a team never alive cannot be normalised"
    assert report_study.shots_per_alive_second(120.0, 40.0) == pytest.approx(1.0), "scale, not raw counts"


def test_compare_arms_pairs_on_the_seed_and_reports_arm_minus_control(capsys) -> None:
    control = _rows("exp1", "CTE", {1: 0.40, 2: 0.30})
    arm = _rows("exp1", "CTE", {1: 0.20, 2: 0.15})
    report_study.compare_arms(control, arm, "synthetic arm")
    printed = capsys.readouterr().out
    assert "2 shared seeds per arm" in printed
    row = next(line for line in printed.splitlines() if line.startswith("| exp1 | CTE |"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[0] == "exp1" and cells[1] == "CTE"
    assert float(cells[2]) == pytest.approx(0.35), "control mean"
    assert float(cells[3]) == pytest.approx(0.175), "arm mean"
    assert float(cells[4]) == pytest.approx(-0.175), "the difference is arm minus control"
    assert cells[7] == "2/2", "both seeds fell below the control"
    assert float(cells[5]) < 0, "a uniform fall must give a negative t"


def test_compare_arms_refuses_to_pair_different_seeds(capsys) -> None:
    """Seeds 1-3 against seeds 4-6 are two unpaired samples, and pairing them would print a t anyway."""
    report_study.compare_arms(_rows("exp1", "CTE", {1: 0.4}), _rows("exp1", "CTE", {9: 0.4}), "arm")
    assert "no seeds in common" in capsys.readouterr().out


def test_compare_arms_keeps_paradigms_apart(capsys) -> None:
    control = _rows("exp1", "CTE", {1: 0.40}) + _rows("exp1", "DTE", {1: 0.60})
    arm = _rows("exp1", "CTE", {1: 0.10}) + _rows("exp1", "DTE", {1: 0.55})
    report_study.compare_arms(control, arm, "two paradigms")
    printed = capsys.readouterr().out
    assert sum(line.startswith("| exp1 |") for line in printed.splitlines()) == 2
    assert "1 shared seeds per arm" in printed, "the divisor is the seed count, not the row count"
