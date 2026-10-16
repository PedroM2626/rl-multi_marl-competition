"""Regression tests for the two evaluation instruments built on the round-robin harness.

The pooled round-robin is the artefact every claim in chapter 8's cross-arena section rests on, and two of
its bugs were silent: a match key that left out a replication column merged five match streams onto the
same thirty keys, so the paired tests saw one fifth of the data and still printed a plausible table. Both
the key composition and the header count that exposed it are gated here, on synthetic runs whose true
match count is known by construction. The absolute-strength aggregators ([§ 8.11](08_results.md)) share the
harness and the replication columns, so their counting and level-spread logic is gated on hand-counted rows.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

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
    # An ablation set of checkpoints keeps its arm in the label, or two different policies trained from the
    # same two seeds would pool as one training block.
    assert combine.labels(Path("no_dr__e1s1_e2s1_h3")) == ("no_dr:exp1_seed1+exp2_seed1", "3")
    assert combine.labels(Path("wide_256__e1s2_e2s3_h1")) == ("wide_256:exp1_seed2+exp2_seed3", "1")


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
    # Stored at six significant digits (sig6), not twelve decimal places, so a tiny p survives the JSON.
    # Six digits is a relative error up to 5e-6, which is what the tolerance below allows.
    assert sensitivity["CTE vs DTE"]["without_block_p"]["exp1_seed11+exp2_seed1"] \
        == pytest.approx(report.mcnemar_exact(18, 6), rel=1e-5)
    sensitivity_section = stdout.split("Leave-one-training-block-out")[1].split("Slot control")[0]
    line = next(row for row in sensitivity_section.splitlines() if row.startswith("| CTE vs DTE |"))
    assert line.rstrip().endswith("| no |"), line


def test_block_sensitivity_is_absent_for_a_single_block(tmp_path: Path) -> None:
    """The column only means something once there is more than one set of checkpoints to leave out."""
    run_dirs = _write_runs(tmp_path, {"7": ["CTE"] * 12})
    stdout, data = _pool(tmp_path, run_dirs)
    assert data["block_sensitivity"] == {}
    assert "Leave-one-training-block-out" not in stdout


class _StubHarness:
    """The absolute-eval aggregators take the round-robin harness for Wilson and the draw label."""
    DRAW_TEAM = "draw"
    POLICIES = {"CTE": Path("cte.pt"), "DTE": Path("dte.pt")}
    wilson = staticmethod(report.wilson)
    replicate_policies = None  # set by the module under test when needed


def _abs_row(paradigm: str, trained: str, winner: str, block: str = "b1", seed: int = 1) -> dict:
    return {"paradigm": paradigm, "trained_team": trained, "winner": winner, "duration_seconds": 10.0,
            "training_block": block, "harness_seed": seed}


def test_summarise_counts_wins_draws_losses_and_the_slot_split() -> None:
    absolute = _load("run_absolute_eval")
    rows = [_abs_row("CTE", "Team 1", "Team 1"), _abs_row("CTE", "Team 2", "Team 2"),
            _abs_row("CTE", "Team 3", "draw"), _abs_row("CTE", "Team 1", "Team 3"),
            _abs_row("DTE", "Team 1", "Team 1")]
    out = absolute.summarise(rows, _StubHarness)
    cte = out["CTE"]
    assert (cte["matches"], cte["wins"], cte["draws"], cte["losses"]) == (4, 2, 1, 1)
    assert cte["win_rate_vs_random"] == pytest.approx(0.5)
    assert cte["loss_rate"] == pytest.approx(0.25)
    assert cte["win_rate_by_trained_team"] == {"Team 1": 0.5, "Team 2": 1.0, "Team 3": 0.0}
    # A win is the trained team winning; a random team taking the match is a loss, not a draw.
    assert out["DTE"]["wins"] == 1 and out["DTE"]["losses"] == 0


def test_spread_needs_two_levels_and_reports_the_levels_it_found() -> None:
    absolute = _load("run_absolute_eval")
    rows = [_abs_row("CTE", "Team 1", "Team 1", block="b1", seed=1),
            _abs_row("CTE", "Team 1", "Team 3", block="b1", seed=2),
            _abs_row("CTE", "Team 1", "Team 1", block="b2", seed=1),
            _abs_row("CTE", "Team 1", "Team 1", block="b2", seed=2)]
    assert absolute.by_level(rows, "harness_seed", _StubHarness)  # two streams
    assert absolute.by_level(rows, "training_block", _StubHarness)  # two blocks
    single = [r for r in rows if r["training_block"] == "b1"]
    assert absolute.by_level(single, "training_block", _StubHarness) == {}
    blocks = absolute.by_level(rows, "training_block", _StubHarness)
    s = absolute.spread(blocks)["CTE"]
    assert s["levels"] == {"b1": 0.5, "b2": 1.0}
    assert s["sd"] == pytest.approx(0.25)


def test_parse_blocks_labels_and_rejects_malformed_pairs(tmp_path: Path) -> None:
    absolute = _load("run_absolute_eval")
    label, policies = absolute.parse_blocks("", _StubHarness)[0]
    assert label == "promoted" and set(policies) == {"CTE", "DTE"}
    seen: list[tuple[int, int, str]] = []
    _StubHarness.replicate_policies = lambda s, m, campaign="": (
        seen.append((s, m, campaign)) or {"CTE": Path("x.pt"), "DTE": Path("y.pt")})
    blocks = absolute.parse_blocks("1:7,10:5", _StubHarness)
    assert [label for label, _ in blocks] == ["exp1_seed1+exp2_seed7", "exp1_seed10+exp2_seed5"]
    assert seen == [(1, 7, ""), (10, 5, "")]
    with pytest.raises(SystemExit):
        absolute.parse_blocks("1-7", _StubHarness)


def test_parse_blocks_prefixes_an_ablation_set_and_checks_its_capacity() -> None:
    """A campaign set must be labelled as one, and must not be playable at the wrong hidden width.

    `data/campaign_replicates/<arm>` holds the checkpoints of the configuration ablations, two of which were
    trained at a capacity the default configuration cannot load. The label carries the arm because
    `no_dr/exp1_seed1+exp2_seed1` and the study's `exp1_seed1+exp2_seed1` are different policies that would
    otherwise pool under one training-block key.
    """
    absolute = _load("run_absolute_eval")
    calls: list[str] = []
    _StubHarness.replicate_policies = lambda s, m, campaign="": {"CTE": Path("x.pt")}
    _StubHarness.check_campaign_capacity = lambda arm: calls.append(arm)
    blocks = absolute.parse_blocks("1:1", _StubHarness, "no_dr")
    assert [label for label, _ in blocks] == ["no_dr:exp1_seed1+exp2_seed1"]
    assert calls == ["no_dr"]
    with pytest.raises(SystemExit):
        absolute.parse_blocks("", _StubHarness, "no_dr")


def test_replicate_policies_falls_back_to_the_scratch_of_the_right_tree(tmp_path: Path, monkeypatch) -> None:
    """A replicate that was never versioned must resolve from `data/runs`, in whichever tree it sits.

    The study wrote both experiments under the repository root; the ablation campaign wrote experiment 2 inside
    `ctde_arena/`. A fallback that assumed one tree raised "no such training replicate" for the other case --
    the failure a reader hits when asking for a block that was never versioned. Built on synthetic trees
    because every study seed is versioned now, and a real scratch would make the test machine-dependent.
    """
    harness = _load("run_cross_arena")
    root, other = tmp_path / "root", tmp_path / "ctde"
    monkeypatch.setattr(harness, "ROOT_ROOT", root)
    monkeypatch.setattr(harness, "CTDE_ROOT", other)

    def scratch(base: Path, experiment: str, seed: int) -> None:
        directory = base / "data" / "runs" / f"{experiment}_seed{seed}" / "checkpoints"
        directory.mkdir(parents=True)
        paradigms = ("cte", "dte", "ctde") if experiment == "exp1" else ("ctde-vd", "ctde-cac", "ctde-comm")
        for index, paradigm in enumerate(paradigms, 1):
            (directory / f"team_{index}_{paradigm}.pt").write_bytes(b"x")

    # Experiment 2's study replicate is in the ROOT tree; experiment 1's is in the versioned set.
    scratch(root, "exp2", 6)
    versioned = root / "data" / "training_replicates" / "exp1_seed5"
    versioned.mkdir(parents=True)
    for index, paradigm in enumerate(("cte", "dte", "ctde"), 1):
        (versioned / f"team_{index}_{paradigm}.pt").write_bytes(b"v")
    policies = harness.replicate_policies(5, 6)
    assert len(policies) == 6, policies
    from_versioned = [p for p in policies.values() if str(versioned) in str(p)]
    from_scratch = [p for p in policies.values() if str(root / "data" / "runs") in str(p)]
    assert len(from_versioned) == 3, from_versioned
    assert len(from_scratch) == 3, from_scratch
    assert all(p.is_file() for p in policies.values())

    # The ablation shape: experiment 2's scratch inside its own tree, under runs_ablation_<arm>/.
    ablation = other / "data" / "runs_ablation_no_dr" / "exp2_seed6" / "checkpoints"
    ablation.mkdir(parents=True)
    for index, paradigm in enumerate(("ctde-vd", "ctde-cac", "ctde-comm"), 1):
        (ablation / f"team_{index}_{paradigm}.pt").write_bytes(b"a")
    campaign_versioned = root / "data" / "campaign_replicates" / "no_dr"
    campaign_versioned.mkdir(parents=True)
    (campaign_versioned / "exp1_seed5").mkdir()
    for index, paradigm in enumerate(("cte", "dte", "ctde"), 1):
        (campaign_versioned / "exp1_seed5" / f"team_{index}_{paradigm}.pt").write_bytes(b"v")
    resolved = harness.replicate_policies(5, 6, "no_dr")
    assert any(str(ablation) in str(path) for path in resolved.values()), \
        "the campaign fallback did not look in the ctde tree"
    with pytest.raises(SystemExit):
        harness.replicate_policies(99, 6, "no_dr")


def test_the_pool_report_carries_a_firing_rate_per_unit_of_alive_life(tmp_path: Path) -> None:
    """shots / (3 x mean survival): an arm that is dead earlier also shoots less.

    The metric is the only way to separate the two, and it is computed by two different programs here --
    `run_cross_arena.summarise` per run and `report_cross_arena` over the pool -- so the values are checked
    against each other and against the columns they are derived from, not just against a fixture.
    """
    harness = _load("run_cross_arena")
    rows = [_row("CTE", "CTE", "CTE|CTDE|DTE", 1), _row("CTE", "DTE", "CTE|CTDE|DTE", 2)]
    for row in rows:
        row["survival_seconds"] = 20.0
        row["shots"] = 60
    stats = harness.summarise(rows)["CTE"]
    assert stats["mean_survival_seconds"] == pytest.approx(20.0)
    assert stats["shots_per_match"] == pytest.approx(60.0)
    assert stats["shots_per_alive_second"] == pytest.approx(60.0 / (3 * 20.0))

    pooled_dirs = _write_runs(tmp_path, {"a": ["CTE", "DTE"]})
    _, pooled = _pool(tmp_path, pooled_dirs)
    for paradigm, entry in pooled["per_architecture"].items():
        expected = entry["shots_per_match"] / (3 * entry["mean_survival_seconds"])
        assert entry["shots_per_alive_second"] == pytest.approx(expected, rel=1e-3), paradigm


def test_rank_pearson_and_the_two_scale_join() -> None:
    """compare_scales is the instrument behind 'strength is not one quantity' (docs/08 section 8.11)."""
    absolute = _load("run_absolute_eval")
    assert absolute.rank_of({"a": 0.5, "b": 0.7, "c": 0.5, "d": 0.1}) == {"b": 1.0, "a": 2.5, "c": 2.5, "d": 4.0}
    assert absolute.pearson([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert absolute.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert math.isnan(absolute.pearson([1, 2], [1, 2]))  # two points cannot be correlated

    peer = {"CTE": {"win_rate": 0.60, "eliminations_per_match": 2.0, "shots_per_match": 90.0,
                    "mean_survival_seconds": 20.0, "shot_accuracy": 0.02},
            "DTE": {"win_rate": 0.30, "eliminations_per_match": 1.0, "shots_per_match": 60.0,
                    "mean_survival_seconds": 25.0, "shot_accuracy": 0.03},
            "CTDE-Comm": {"win_rate": 0.45, "eliminations_per_match": 1.5, "shots_per_match": 75.0,
                          "mean_survival_seconds": 15.0, "shot_accuracy": 0.04}}
    pooled = {"CTE": {"win_rate_vs_random": 0.20}, "DTE": {"win_rate_vs_random": 0.80},
              "CTDE-Comm": {"win_rate_vs_random": 0.50}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump({"per_architecture": peer}, handle)
        path = Path(handle.name)
    try:
        out = absolute.compare_scales(pooled, path)
    finally:
        path.unlink()
    # Perfect inversion of the peer order -> Spearman exactly -1, and the join is on architecture name.
    assert out["spearman_rank_correlation"] == pytest.approx(-1.0)
    assert out["pearson_win_rate"] == pytest.approx(-1.0)
    assert set(out["per_arm"]) == {"CTE", "DTE", "CTDE-Comm"}
    assert out["per_arm"]["CTE"]["rank_random"] == 3.0 and out["per_arm"]["CTE"]["rank_peers"] == 1.0


def test_matched_null_requires_the_uniform_baseline(tmp_path: Path) -> None:
    absolute = _load("run_absolute_eval")
    greedy = tmp_path / "greedy.json"
    greedy.write_text(json.dumps({"action_policy": "greedy",
                                  "per_team": {"Team 1": {"win_rate": 0.167}}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        absolute.matched_null(greedy)
    uniform = tmp_path / "uniform.json"
    uniform.write_text(json.dumps({"action_policy": "uniform", "per_team": {
        "Team 1": {"win_rate": 0.2567}, "Team 2": {"win_rate": 0.27}, "Team 3": {"win_rate": 0.4633}}}),
        encoding="utf-8")
    null = absolute.matched_null(uniform)
    assert null == {"Team 1": 0.2567, "Team 2": 0.27, "Team 3": 0.4633}
    pooled = {"CTE": {"win_rate_vs_random": 0.7467,
                      "win_rate_by_trained_team": {"Team 1": 0.6733, "Team 2": 0.72, "Team 3": 0.8467}}}
    balanced = absolute.add_matched_null(pooled, null)
    assert balanced == pytest.approx(0.33)
    assert pooled["CTE"]["excess_vs_matched_null"] == pytest.approx(0.4167, abs=1e-4)
    # The Team 3 excess is smaller than the Team 1 one because Team 3 is a better corner to begin with.
    assert pooled["CTE"]["excess_by_trained_team"]["Team 3"] < pooled["CTE"]["excess_by_trained_team"]["Team 1"]


def test_a_published_p_value_is_never_zero() -> None:
    """`round(p, 12)` turned 1.1e-37 into 0.0; a table of zeros is not a statistical report.

    Checked over every pooled report that is committed, because the pools with the most decisive pairs are
    exactly the ones that produce the smallest p -- the failure hides in the strongest results.
    """
    assert report.sig6(1.1126038e-37) == 1.1126e-37
    assert report.sig6(0.217396595811) == 0.217397
    pooled = sorted((PROJECT_ROOT / "results" / "cross_arena").glob("*.json"))
    assert pooled, "no pooled reports to check"
    for path in pooled:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for name, entry in (payload.get("pairwise_mcnemar") or {}).items():
            b, c = entry["discordant"]
            if b + c:
                assert entry["p"] > 0.0, f"{path.name}: {name} stores p=0 for {b}/{c}"
        for name, sens in (payload.get("block_sensitivity") or {}).items():
            for key, value in sens["without_block_p"].items():
                assert value > 0.0, f"{path.name}: {name} without {key} stores p=0"
            for block, detail in sens["per_block"].items():
                assert detail["solo_p"] > 0.0, f"{path.name}: {name} in {block} stores p=0"


def test_the_versioning_scripts_share_one_copy_and_verify_rule() -> None:
    """`version_training_replicates` borrows `copy_verified` from the campaign script -- in the *commit*.

    A caller committed without the helper it imports runs on the machine that wrote it and nowhere else; this
    repository made that mistake once, and it is invisible until someone else runs the documented command. So
    the check reads `git show HEAD:`, not the working tree: both sides have to be committed together.
    """
    committed = {}
    for name in ("version_campaign_checkpoints.py", "version_training_replicates.py"):
        committed[name] = subprocess.run(["git", "show", f"HEAD:scripts/{name}"], cwd=PROJECT_ROOT,
                                         capture_output=True, text=True, check=True).stdout
    caller = committed["version_training_replicates.py"]
    helpers = re.findall(r"^def ([a-z_]+)\(", committed["version_campaign_checkpoints.py"], re.M)
    borrowed: set[str] = set()
    for variable in re.findall(r"(\w+) = _shared\(\)", caller):
        borrowed |= set(re.findall(rf"\b{variable}\.([a-z_]+)\(", caller))
    assert borrowed == {"copy_verified"}, f"unexpected cross-script calls: {sorted(borrowed)}"
    missing = sorted(borrowed - set(helpers))
    assert not missing, (f"{missing} is called by the committed caller but not defined by the committed "
                         "helper -- they were committed apart")

    assert "def copy_verified" not in caller, \
        "the caller grew its own copy rule; there must be one rule, defined once"
    assert callable(_load("version_campaign_checkpoints").copy_verified)


def test_the_campaign_manifest_matches_the_weights() -> None:
    """Every ablation checkpoint is versioned, hashed, and trained at the capacity the manifest claims.

    docs/08_results.md#ablations says each arm differs from the study by one configuration key, two of them
    keys that change the network's width. A `.pt` here stores only `{"paradigm", "actor"}`, so nothing in the
    weight file records that; this reads the width back out of the first layer of each actor, compares it
    with the arm's declared override, and re-derives the bytes and SHA-256 the manifest pinned.
    """
    root = PROJECT_ROOT / "data" / "campaign_replicates"
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest["arms"]) == {"no_dr", "narrow_64", "wide_256"}
    derive = _load("version_campaign_checkpoints").actor_shape
    files = 0
    for arm, payload in manifest["arms"].items():
        declared = int(payload["overrides"].get("RL_HIDDEN_DIM", 128))
        assert len(payload["sets"]) == 6, f"{arm}: three seeds x two experiments"
        for name, entry in payload["sets"].items():
            assert entry["hidden_dim"] == declared, (arm, name)
            assert entry["completed_steps"] == entry["target_steps"] == 500000, (arm, name)
            for file, meta in entry["files"].items():
                path = root / arm / name / file
                assert path.is_file(), path
                assert path.stat().st_size == meta["bytes"], file
                assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["sha256"], file
                if file == sorted(entry["files"])[0]:
                    assert derive(path)[0] == meta["hidden_dim"], (arm, name, file)
                files += 1
    assert files == 54, files


def test_the_capacity_guard_refuses_an_arm_the_configuration_cannot_load() -> None:
    """Playing `wide_256` at the committed width must fail loudly, before a single match is played."""
    harness = _load("run_cross_arena")
    from marl_arena.config import CONFIG

    assert CONFIG.rl_hidden_dim == 128, "the study's width is what the no_dr arm was trained at"
    harness.check_campaign_capacity("no_dr")
    for arm in ("narrow_64", "wide_256"):
        with pytest.raises(SystemExit, match="export RL_HIDDEN_DIM="):
            harness.check_campaign_capacity(arm)
    with pytest.raises(SystemExit, match="no campaign"):
        harness.check_campaign_capacity("not_an_arm")


def test_report_only_rederives_the_summary_from_the_recorded_matches(tmp_path: Path) -> None:
    """The published summary must be recomputable from matches.csv, so a reader can check it cheaply."""
    absolute = _load("run_absolute_eval")
    rows = [_abs_row("CTE", "Team 1", "Team 1"), _abs_row("CTE", "Team 2", "Team 3"),
            _abs_row("CTE", "Team 3", "draw")]
    counts = absolute.summarise(rows, _StubHarness)["CTE"]
    assert (counts["matches"], counts["wins"], counts["losses"], counts["draws"]) == (3, 1, 1, 1)
    assert counts["median_duration_seconds"] == pytest.approx(10.0)
