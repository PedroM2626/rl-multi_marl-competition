"""The study runner must leave an optimiser trace behind, and leaving it must not change the training.

Why this file exists: `train_rl.py` has written `training_log.json` since A-15, but the study is produced by
`run_experiment.py`, which never read `simulation.last_ppo_stats` at all — so none of the forty committed
replicates can say whether a run that ended at 0.000 held-out win rate looked different from the inside than
one that ended at 0.607. That is open work item 4 of the audit. Adding the capture is only safe if it is
passive, and the only proof of passivity that matters here is that the weights come out identical, so this
file pins both halves: the artefact exists with the schema the analysis needs, and the capture reads nothing
that could advance the simulation.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "data" / "pytest_trace_run"
TRACE_KEYS = {"policy_loss", "value_loss", "entropy", "approx_kl"}


def test_a_training_run_leaves_a_ppo_trace() -> None:
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    environment = dict(os.environ)
    environment.update({"MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"})
    completed = subprocess.run(
        [sys.executable, "scripts/run_experiment.py", "--seed", "7", "--steps", "4000",
         "--out", str(RUN_DIR.relative_to(ROOT)), "--curve-at", "1.0", "--curve-matches", "2",
         "--eval-matches", "2", "--metrics-every", "2"],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=1800)
    try:
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
        log = RUN_DIR / "checkpoints" / "training_log.json"
        assert log.is_file(), f"no training log; run dir holds {sorted(p.name for p in RUN_DIR.iterdir())}"
        payload = json.loads(log.read_text(encoding="utf-8"))
        assert payload["completed_steps"] >= 4000
        assert payload["every_matches"] == 2
        entries = payload["entries"]
        assert len(entries) >= 2, "both the cadence record and the final record are missing"
        assert {"env_steps", "matches", "ppo_stats"} <= set(entries[0])
        teams = entries[-1]["ppo_stats"]
        assert set(teams) == {"Team 1", "Team 2", "Team 3"}, teams.keys()
        for team, stats in teams.items():
            assert set(stats) == TRACE_KEYS, (team, stats)
            assert all(isinstance(value, float) for value in stats.values()), (team, stats)
        # A-14 was a hard-coded zero; a trace of zeros would be the same defect wearing an artefact.
        assert any(stats["approx_kl"] > 0.0 for stats in teams.values()), teams
        assert any(stats["entropy"] > 0.0 for stats in teams.values()), teams
    finally:
        shutil.rmtree(RUN_DIR, ignore_errors=True)


def test_the_trace_capture_is_passive_by_construction() -> None:
    """The two trees must not drift, and the capture must not touch anything that advances the run."""
    root_source = (ROOT / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    ctde_source = (ROOT / "ctde_arena" / "scripts" / "run_experiment.py").read_text(encoding="utf-8")
    assert root_source == ctde_source
    body = root_source[root_source.index("def record_trace"):root_source.index("while sim.total_env_steps")]
    assert "sim.last_ppo_stats" in body
    for forbidden in ("step(", "reset", "controllers", "random", "seed", "torch."):
        assert forbidden not in body, f"the trace capture touches {forbidden!r}, which is not passive"
