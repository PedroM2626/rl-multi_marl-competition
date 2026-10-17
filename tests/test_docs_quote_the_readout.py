"""Section 8.1's sampled-readout table must be the merged artefacts, not a transcription.

The claim this gates is the strongest one in the repository -- that experiment 2's best architecture is an
artefact of argmax, and that the between-replicate spread survives changing the action rule. Both were
written into the documentation from `report_study.py` output, so the numbers now have to be recomputable from
`results/readout/T*/per_seed_metrics.csv` and `results/study/per_seed_metrics.csv` or the section is prose.
"""

from __future__ import annotations

import csv
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "08_results.md"


def arm_rows(directory: str) -> list[dict[str, str]]:
    with (ROOT / directory / "per_seed_metrics.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def grouped(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        out.setdefault((row["experiment"], row["paradigm"]), []).append(float(row["eval_win_rate"]))
    return out


def means(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    return {key: sum(v) / len(v) for key, v in grouped(rows).items()}


def sds(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    return {key: statistics.stdev(v) for key, v in grouped(rows).items()}


def band(values: list[float]) -> str:
    return f"{min(values):.3f}–{max(values):.3f}"


def published_table() -> list[tuple[str, str, dict[str, float]]]:
    """(experiment, paradigm, {column -> win rate}) for § 8.1's four-readout table."""
    text = RESULTS.read_text(encoding="utf-8")
    start = text.index("### What the same networks do when the readout is not argmax")
    rows = []
    for line in text[start: start + 4000].splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[0].startswith("exp"):
            continue
        experiment, _, architecture = cells[0].partition(" ")
        values = dict(zip(("argmax", "T0", "T0.5", "T1"),
                          (float(c.replace("**", "")) for c in cells[1:])))
        rows.append((experiment, architecture, values))
    assert len(rows) == 6, f"the readout table parsed {len(rows)} rows, expected six"
    return rows


def test_every_cell_of_the_readout_table_is_an_artefact_mean() -> None:
    arms = {
        "argmax": means(arm_rows("results/study")),
        "T0": means(arm_rows("results/readout/T0")),
        "T0.5": means(arm_rows("results/readout/T0.5")),
        "T1": means(arm_rows("results/readout/T1")),
    }
    for experiment, architecture, values in published_table():
        key = (experiment, architecture)
        for column, quoted in values.items():
            recorded = arms[column][key]
            assert abs(recorded - quoted) < 0.0005, (
                f"{architecture} ({experiment}) under {column}: the document says {quoted}, the "
                f"artefact mean is {recorded:.4f}")


def test_the_control_arm_reproduces_the_published_study_on_every_cell() -> None:
    """The section's central warrant: re-measurement noise is zero, so the sampled arms differ by the rule."""
    study = {(r["experiment"], r["seed"], r["paradigm"]): r for r in arm_rows("results/study")}
    control = {(r["experiment"], r["seed"], r["paradigm"]): r for r in arm_rows("results/readout/T0")}
    assert study.keys() == control.keys(), "the control does not cover the same sixty cells as the study"
    for key in study:
        assert float(control[key]["eval_win_rate"]) == float(study[key]["eval_win_rate"]), key
    assert "**every cell to the last decimal**" in RESULTS.read_text(encoding="utf-8"), \
        "the document must keep claiming what this proves"


def test_the_quoted_spreads_are_the_arms_spreads() -> None:
    """The 'sampling does not buy precision' claim is a range over six architectures, per arm.

    The bands are parsed out of the sentence that makes the claim rather than searched for in the file: a
    substring check passes as long as the right figure appears *somewhere*, which is how a stale number in
    one chapter could hide behind a correct one in another.
    """
    text = RESULTS.read_text(encoding="utf-8")
    greedy = sds(arm_rows("results/study"))
    half = sds(arm_rows("results/readout/T0.5"))
    one = sds(arm_rows("results/readout/T1"))
    sentence = re.search(
        r"Between-seed SD over the six architectures is (\d\.\d+)–(\d\.\d+) at\s*\n\s*"
        r"T = 0\.5 and (\d\.\d+)–(\d\.\d+) at T = 1, against \*\*(\d\.\d+)–(\d\.\d+)\*\* under argmax",
        text)
    assert sentence, "the spread claim was reworded; this test reads its numbers out of that sentence"
    half_low, half_high, one_low, one_high, greedy_low, greedy_high = sentence.groups()
    assert band(list(half.values())) == f"{half_low}–{half_high}"
    assert band(list(one.values())) == f"{one_low}–{one_high}"
    assert band(list(greedy.values())) == f"{greedy_low}–{greedy_high}"
    exp2_greedy = band([v for (experiment, _), v in greedy.items() if experiment == "exp2"])
    assert exp2_greedy in text, f"the experiment-2 band {exp2_greedy} is no longer quoted"
    for other in (ROOT / "README.md", ROOT / "docs" / "10_threats_to_validity.md"):
        prose = other.read_text(encoding="utf-8")
        assert f"{greedy_low}–{greedy_high}" in prose, f"{other.name} lost the greedy spread"
        assert band(list(half.values()) + list(one.values())) in prose, \
            f"{other.name} lost the sampled spread"
