"""Tests for the artefact pipeline: MetricsStore and the dashboard generator.

These produce every number in docs/, and were previously at 0 % coverage in both trees, so a silent
change to a column, a denominator or the plotting filter would not have been caught by anything.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

# plotting.py imports pyplot at module scope, which selects a GUI backend; force the headless one
# before that import happens.
os.environ["MPLBACKEND"] = "Agg"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.models import MatchResult, TeamMetrics  # noqa: E402
from marl_arena.systems.metrics import MetricsStore  # noqa: E402
from marl_arena.systems.plotting import _load_team_metric_series, export_metric_dashboard  # noqa: E402

TEAM_ROWS = [
    {"match_index": 1, "variant_id": 1, "team_name": "Team 1", "paradigm": "CTE", "winner": 1,
     "eliminations": 2, "mean_survival_time": 9.5, "shots_hit": 2, "shots_missed": 8,
     "shot_accuracy": 0.2, "remaining_agents": 3, "match_duration_seconds": 30.0},
    {"match_index": 1, "variant_id": 1, "team_name": "Team 2", "paradigm": "DTE", "winner": 0,
     "eliminations": 1, "mean_survival_time": 4.0, "shots_hit": 1, "shots_missed": 9,
     "shot_accuracy": 0.1, "remaining_agents": 0, "match_duration_seconds": 30.0},
]


def _result(rows: list[dict]) -> MatchResult:
    return MatchResult(winner_team="Team 1", duration_seconds=30.0, team_rows=rows, agent_rows=[])


def _cumulative() -> dict[str, TeamMetrics]:
    return {
        "Team 1": TeamMetrics(team_name="Team 1", paradigm="CTE", wins=1, eliminations=2,
                              shots_hit=2, shots_missed=8, survival_time_sum=28.5, matches_played=3),
        "Team 2": TeamMetrics(team_name="Team 2", paradigm="DTE", wins=2, eliminations=5,
                              shots_hit=4, shots_missed=6, survival_time_sum=18.0, matches_played=3),
    }


def test_record_match_writes_csv_and_summary(tmp_path: Path) -> None:
    store = MetricsStore(metrics_dir=tmp_path, exports_dir=tmp_path / "exports")
    store.record_match(_result(TEAM_ROWS), _cumulative())

    text = store.team_metrics_csv.read_text(encoding="utf-8")
    assert text.splitlines()[0].split(",")[0] == "match_index"
    assert len(text.strip().splitlines()) == 3, "header plus two team rows"

    summary = store.summary_json.read_text(encoding="utf-8")
    assert '"win_rate"' in summary and '"Team 1"' in summary


def test_summary_denominator_uses_matches_played(tmp_path: Path) -> None:
    """win_rate is wins / matches_played, not wins / recorded rows - the distinction behind the
    46-recorded-vs-460-cumulative gap documented in docs/07."""
    import json

    store = MetricsStore(metrics_dir=tmp_path, exports_dir=tmp_path / "exports")
    store.record_match(_result(TEAM_ROWS), _cumulative())
    teams = {t["team_name"]: t for t in json.loads(store.summary_json.read_text(encoding="utf-8"))["teams"]}

    assert teams["Team 1"]["win_rate"] == pytest.approx(1 / 3)
    assert teams["Team 1"]["eliminations_per_match"] == pytest.approx(2 / 3)
    assert teams["Team 1"]["mean_survival_time"] == pytest.approx(28.5 / (3 * 3))
    assert teams["Team 1"]["shot_accuracy"] == pytest.approx(2 / 10)


def test_schema_change_rotates_the_old_file(tmp_path: Path) -> None:
    store = MetricsStore(metrics_dir=tmp_path, exports_dir=tmp_path / "exports")
    store.record_match(_result(TEAM_ROWS), _cumulative())

    changed = [dict(TEAM_ROWS[0])]
    changed[0].pop("variant_id")
    with pytest.warns(UserWarning, match="schema changed"):
        store.record_match(_result(changed), _cumulative())

    assert (tmp_path / "team_match_metrics.legacy.csv").exists()
    header = store.team_metrics_csv.read_text(encoding="utf-8").splitlines()[0]
    assert "variant_id" not in header


def test_dashboard_series_only_keeps_known_team_prefix(tmp_path: Path) -> None:
    """A-19: rows whose team_name does not start with 'Team ' are dropped without warning."""
    store = MetricsStore(metrics_dir=tmp_path, exports_dir=tmp_path / "exports")
    store.record_match(_result(TEAM_ROWS), _cumulative())

    series = _load_team_metric_series(store.team_metrics_csv)
    assert set(series) == {"Team 1", "Team 2"}
    assert series["Team 1"][0]["win_rate"] == pytest.approx(1.0)

    rows = [dict(r) for r in TEAM_ROWS]
    rows[0]["team_name"] = "Squad 1"
    store2 = MetricsStore(metrics_dir=tmp_path / "other", exports_dir=tmp_path / "other_exports")
    store2._append_rows(store2.team_metrics_csv, rows)
    assert set(_load_team_metric_series(store2.team_metrics_csv)) == {"Team 2"}


def test_dashboard_png_is_written(tmp_path: Path) -> None:
    store = MetricsStore(metrics_dir=tmp_path, exports_dir=tmp_path / "exports")
    store.record_match(_result(TEAM_ROWS), _cumulative())
    outputs = export_metric_dashboard(store.team_metrics_csv, tmp_path / "exports")
    assert outputs and outputs[0].exists() and outputs[0].stat().st_size > 1000
