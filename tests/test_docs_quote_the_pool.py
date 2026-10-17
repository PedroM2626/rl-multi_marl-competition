"""The documentation's round-robin tables must be the report's numbers, not a transcription of them.

Every figure in § 8.10 was copied out of `results/cross_arena/blocks10.json` by hand, and this repository has
already been caught twice with a table that no longer matched its own evidence -- once where a summary named
the checkpoints it had not played (A-36), and once where a p-value was stored rounded to zero (A-41). The
gate below removes the copying step from the claim: it parses the published tables and recomputes them from
the pooled report, so a number that drifts between a re-pool and a documentation edit fails a test rather than
shipping.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "08_results.md"
POOL = ROOT / "results" / "cross_arena" / "blocks10.json"

POOL_CSV = ROOT / "results" / "cross_arena" / "blocks10.csv"


def section(text: str, heading: str) -> str:
    start = text.index(heading)
    rest = text[start + len(heading):]
    stop = re.search(r"^(#{2,3} |\*\*Closed)", rest, re.M)
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


def nearly(document: float, recorded: float, places: int) -> bool:
    return round(recorded, places) == round(document, places)


def test_pooled_table_is_the_report() -> None:
    text = RESULTS.read_text(encoding="utf-8")
    report = json.loads(POOL.read_text(encoding="utf-8"))
    table = rows(section(text, "### Pooled result"))
    body = [r for r in table if len(r) == 10 and r[0].isdigit()]
    assert len(body) == 6, f"expected six ranked rows, parsed {len(body)} from {table[:2]}"
    for row in body:
        rank, name, win, ci, versus, elim, survival, shots, rate, accuracy = row
        architecture = name.strip("*")
        recorded = report["per_architecture"][architecture]
        assert int(rank) == sorted(report["per_architecture"],
                                   key=lambda a: -report["per_architecture"][a]["win_rate"]).index(architecture) + 1, \
            f"{architecture} is ranked {rank} in the document, not by the report"
        assert nearly(float(win.strip("*")), recorded["win_rate"], 3), architecture
        assert ci == f"[{recorded['wilson_95_ci'][0]:.3f}, {recorded['wilson_95_ci'][1]:.3f}]", (ci, architecture)
        assert float(versus.strip("+−")) == round(abs(recorded["win_rate"] - 1 / 3), 3), architecture
        assert nearly(float(elim), recorded["eliminations_per_match"], 2), architecture
        assert nearly(float(survival.removesuffix(" s")), recorded["mean_survival_seconds"], 1), architecture
        assert nearly(float(shots), recorded["shots_per_match"], 1), architecture
        assert nearly(float(rate), recorded["shots_per_alive_second"], 2), architecture
        assert nearly(float(accuracy.removesuffix(" %")), 100 * recorded["shot_accuracy"], 2), architecture


def test_pairwise_table_is_the_report() -> None:
    text = RESULTS.read_text(encoding="utf-8")
    report = json.loads(POOL.read_text(encoding="utf-8"))
    table = rows(section(text, "### Four of the fifteen pairs separate"))
    body = [r for r in table if len(r) == 7 and r[1].isdigit()]
    assert len(body) == 15, f"the pairwise table has {len(body)} rows, not fifteen"
    assert len(report["pairwise_mcnemar"]) == 15
    significant_in_doc = [r for r in body if r[6].startswith("**")]
    significant_in_report = [p for p, v in report["pairwise_mcnemar"].items()
                             if v["verdict"].startswith("**")]
    assert len(significant_in_doc) == len(significant_in_report) == 4, (
        [r[0] for r in significant_in_doc], sorted(significant_in_report))
    for row in body:
        pair, met, a_wins, b_wins, discordant, p, verdict = row
        recorded = report["pairwise_mcnemar"][pair]
        assert int(met) == recorded["met"], pair
        assert (int(a_wins), int(b_wins)) == (recorded["a_wins"], recorded["b_wins"]), pair
        assert discordant == "{}/{}".format(*recorded["discordant"]), pair
        assert abs(float(p) - recorded["p"]) <= 0.01 * max(recorded["p"], 1e-12), (pair, p, recorded["p"])
        assert (verdict.startswith("**")) == recorded["verdict"].startswith("**"), pair


def test_leave_one_block_out_table_is_the_report() -> None:
    text = RESULTS.read_text(encoding="utf-8")
    report = json.loads(POOL.read_text(encoding="utf-8"))
    table = rows(section(text, "### Three of the four survive leaving a block out"))
    body = [r for r in table if len(r) == 4 and "|" not in r[0] and r[0] in report["block_sensitivity"]]
    assert len(body) == len(report["block_sensitivity"]), (
        f"the document shows {len(body)} of {len(report['block_sensitivity'])} corrected pairs")
    for row in body:
        pair, pooled_p, worst, verdict = row
        recorded = report["block_sensitivity"][pair]
        assert abs(float(pooled_p) - recorded["pooled_p"]) <= 0.01 * recorded["pooled_p"], pair
        assert abs(float(worst) - max(recorded["without_block_p"].values())) \
            <= 0.01 * max(recorded["without_block_p"].values()), pair
        assert (verdict == "**robust**") == recorded["robust_to_block"], pair


def test_the_pooled_csv_row_count_matches_what_the_document_claims() -> None:
    """Section 8.10's headline counts are quoted in three chapters; the CSV is the only authority."""
    lines = POOL_CSV.read_text(encoding="utf-8").splitlines()
    team_matches = len(lines) - 1
    report = json.loads(POOL.read_text(encoding="utf-8"))
    assert team_matches == 3 * report["matches"] == 90_000, team_matches
    document = RESULTS.read_text(encoding="utf-8")
    for claim in ("**30,000 matches / 90,000 team-matches, 15,000 per architecture, "
                  "6,000 head-to-head meetings per pair.**",):
        assert claim in document, claim
    assert report["matches"] == 30_000
    assert all(v["matches"] == 15_000 for v in report["per_architecture"].values())
    assert all(v["met"] == 6_000 for v in report["pairwise_mcnemar"].values())
