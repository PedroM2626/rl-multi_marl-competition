"""Regression tests for the round-robin statistics harness.

The pooled round-robin is the artefact every claim in chapter 8's cross-arena section rests on, and two of
its bugs were silent: a match key that left out a replication column merged five match streams onto the
same thirty keys, so the paired tests saw one fifth of the data and still printed a plausible table. Both
the key composition and the header count that exposed it are gated here, on synthetic runs whose true
match count is known by construction.
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

COLUMNS = ["triple", "slot", "paradigm", "match", "permutation", "winner",
           "eliminations", "remaining_agents", "survival_seconds", "shots", "shots_hit", "duration_seconds"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_ca_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


report = _load("report_cross_arena")
combine = _load("combine_cross_arena")


def _row(paradigm: str, winner: str, triple: str, match: int, slot: str = "Team 1") -> dict[str, object]:
    return {"triple": triple, "slot": slot, "paradigm": paradigm, "match": match, "permutation": 1,
            "winner": winner, "eliminations": 2, "remaining_agents": 3, "survival_seconds": 20.0,
            "shots": 40, "shots_hit": 1, "duration_seconds": 20.0}


def test_match_key_keeps_runs_apart_that_differ_only_by_replication() -> None:
    """The defect this guards: two runs writing the same triple and match indices are two runs, not one."""
    base = {"triple": "CTE|CTDE|DTE", "match": 7, "paradigm": "CTE", "winner": "CTE"}
    keys = {
        report.MatchKey.of({**base}),
        report.MatchKey.of({**base, "harness_seed": "1"}),
        report.MatchKey.of({**base, "harness_seed": "2"}),
        report.MatchKey.of({**base, "training_block": "exp1_seed1+exp2_seed7", "harness_seed": "1"}),
        report.MatchKey.of({**base, "training_block": "exp1_seed9+exp2_seed2", "harness_seed": "1"}),
    }
    assert len(keys) == 5


def test_match_key_triple_field_is_the_triple_not_a_replication_column() -> None:
    """The header once printed the harness-seed count and called it triples, after a column was inserted."""
    key = report.MatchKey.of({"triple": "CTE|DTE", "match": 3, "harness_seed": "5", "training_block": "b"})
    assert key.triple == "CTE|DTE"
    assert key.index == 3


def test_mcnemar_exact_matches_hand_computed_binomial_tails() -> None:
    assert report.mcnemar_exact(0, 0) == 1.0
    assert report.mcnemar_exact(1, 0) == 1.0                       # one discordant outcome carries no signal
    assert abs(report.mcnemar_exact(0, 5) - 2 * 0.5 ** 5) < 1e-12   # 0/5 -> 0.0625
    assert abs(report.mcnemar_exact(1, 9) - 2 * 11 / 1024) < 1e-12  # (C(10,0)+C(10,1)) / 2**10, doubled
    assert report.mcnemar_exact(6, 6) == 1.0
    assert report.mcnemar_exact(490, 660) == report.mcnemar_exact(660, 490)


def test_labels_split_training_block_from_harness_seed() -> None:
    assert combine.labels(Path("results/cross_arena_blocks/e1s10_e2s5_h3")) == ("exp1_seed10+exp2_seed5", "3")
    assert combine.labels(Path("results/cross_arena_blocks/e1s1_e2s7_h1")) == ("exp1_seed1+exp2_seed7", "1")
    # A run made from the promoted checkpoints has no training block to report, but still a seed.
    assert combine.labels(Path("results/cross_arena/seed5")) == ("promoted", "5")


def _write_runs(tmp_path: Path, blocks: dict[str, list[str]]) -> list[Path]:
    """One run directory per block, `blocks[name]` being the winning paradigm of each match in turn."""
    triple = "CTE|CTDE|DTE"
    run_dirs = []
    for name, winners in blocks.items():
        directory = tmp_path / f"e1s{name}_e2s1_h1"
        directory.mkdir()
        with (directory / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            for match, winner in enumerate(winners):
                for slot, paradigm in (("Team 1", "CTE"), ("Team 2", "CTDE"), ("Team 3", "DTE")):
                    writer.writerow(_row(paradigm, winner, triple, match, slot))
        run_dirs.append(directory)
    return run_dirs


def _pool(tmp_path: Path, run_dirs: list[Path]) -> tuple[str, dict]:
    merged = tmp_path / "pooled.csv"
    subprocess.run([sys.executable, str(SCRIPTS / "combine_cross_arena.py"),
                    *[str(d) for d in run_dirs], "--out", str(merged)],
                   check=True, capture_output=True, cwd=PROJECT_ROOT, text=True)
    json_path = tmp_path / "pooled.json"
    result = subprocess.run([sys.executable, str(SCRIPTS / "report_cross_arena.py"),
                             "--in", str(merged), "--json-out", str(json_path)],
                            check=True, capture_output=True, cwd=PROJECT_ROOT, text=True)
    import json
    return result.stdout, json.loads(json_path.read_text(encoding="utf-8"))


def test_pooling_replicated_runs_counts_every_match(tmp_path: Path) -> None:
    """End to end over both scripts, because the collapse happened at their seam, not inside either one."""
    run_dirs = _write_runs(tmp_path, {"1": ["CTE", "DTE"], "2": ["CTE", "DTE"], "3": ["CTE", "DTE"]})
    stdout, data = _pool(tmp_path, run_dirs)
    pooled = list(csv.DictReader((tmp_path / "pooled.csv").open(newline="", encoding="utf-8")))
    assert len(pooled) == 18                                     # 3 runs x 2 matches x 3 teams
    assert {r["training_block"] for r in pooled} == {f"exp1_seed{i}+exp2_seed1" for i in (1, 2, 3)}
    header = next(line for line in stdout.splitlines() if line)
    # 3 runs x 2 matches. A key without the harness seed or without the training block reports 2; both
    # wrong answers leave every other number in the table plausible.
    assert header == "6 matches over 1 triples, 3 architectures, 18 team-matches", header


def test_block_sensitivity_names_a_pair_whose_signal_is_one_block(tmp_path: Path) -> None:
    """A pooled p can come from a single set of checkpoints; the leave-one-block-out column has to say so."""
    # CTE wins everywhere in blocks 11 and 12 and half the time in 13, while CTDE never wins at all.
    run_dirs = _write_runs(tmp_path, {
        "11": ["CTE"] * 12,
        "12": ["CTE"] * 12,
        "13": ["CTE", "DTE"] * 6,
    })
    stdout, data = _pool(tmp_path, run_dirs)
    sensitivity = data["block_sensitivity"]
    # Both pairs pass Bonferroni pooled, so both are listed.
    assert set(sensitivity) == {"CTDE vs CTE", "CTE vs DTE"}, sensitivity
    # CTDE-vs-CTE has zero reverse discordants in every block: dropping any block keeps it significant.
    assert sensitivity["CTDE vs CTE"]["robust_to_block"] is True
    # CTE-vs-DTE is carried entirely by blocks 11 and 12; without either, the remaining pair of blocks is
    # 18/6, which is p = 0.023 and fails the corrected threshold.
    assert sensitivity["CTE vs DTE"]["robust_to_block"] is False
    assert abs(sensitivity["CTE vs DTE"]["without_block_p"]["exp1_seed11+exp2_seed1"]
               - report.mcnemar_exact(18, 6)) < 1e-9
    sensitivity_section = stdout.split("Leave-one-training-block-out")[1].split("Slot control")[0]
    line = next(row for row in sensitivity_section.splitlines() if row.startswith("| CTE vs DTE |"))
    assert line.rstrip().endswith("| no |"), line


def test_block_sensitivity_is_absent_for_a_single_block(tmp_path: Path) -> None:
    """The column only means something once there is more than one set of checkpoints to leave out."""
    run_dirs = _write_runs(tmp_path, {"7": ["CTE"] * 12})
    stdout, data = _pool(tmp_path, run_dirs)
    assert data["block_sensitivity"] == {}
    assert "Leave-one-training-block-out" not in stdout
