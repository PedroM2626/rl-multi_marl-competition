"""Regression tests for the waypoint-to-motion control law.

`angle_to_target` originally read atan2(dx, dy) instead of atan2(dx, dz). Since agents move on the XZ
plane at a constant height, dy is ~0 and the returned bearing collapsed to +/-90 degrees for every
off-axis target, so `turn` saturated and agents spun toward a fixed absolute heading. These tests pin
the corrected behaviour.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.controllers.base import angle_to_target, normalize  # noqa: E402
from marl_arena.controllers.rl_controller import RLTeamController  # noqa: E402
from marl_arena.models import AgentSnapshot  # noqa: E402


def analytic_bearing_degrees(delta: np.ndarray) -> float:
    """Heading that `_forward_from_heading` would point along `delta`, in the XZ plane."""
    return math.degrees(math.atan2(delta[0], delta[2]))


@pytest.mark.parametrize("tx", [-10.0, -3.0, 0.0, 3.0, 10.0])
@pytest.mark.parametrize("tz", [5.0, 12.0, 20.0])
def test_angle_to_target_matches_the_xz_bearing(tx: float, tz: float) -> None:
    origin = np.array([0.0, 1.0, 0.0])
    target = np.array([tx, 1.0, tz])
    heading = 0.0
    expected = (analytic_bearing_degrees(target - origin) - heading + 180.0) % 360.0 - 180.0
    assert angle_to_target(origin, heading, target) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("heading", [0.0, 45.0, 90.0, 180.0, 359.0])
def test_angle_to_target_tracks_a_rotating_heading(heading: float) -> None:
    origin = np.array([1.0, 1.0, -2.0])
    target = np.array([7.0, 1.0, 4.0])
    expected = (analytic_bearing_degrees(target - origin) - heading + 180.0) % 360.0 - 180.0
    assert angle_to_target(origin, heading, target) == pytest.approx(expected, abs=1e-9)


def test_turn_command_shrinks_the_heading_error() -> None:
    """Closed-loop check: applying the commanded turn repeatedly must converge on the target."""
    controller = RLTeamController.__new__(RLTeamController)
    controller.team_name = "Team 1"
    controller.rng = random.Random(0)

    agent = AgentSnapshot(
        agent_id="1-1",
        team_name="Team 1",
        paradigm="CTE",
        position=np.array([0.0, 1.0, 0.0]),
        heading_deg=0.0,
        alive=True,
    )
    target = np.array([8.0, 1.0, 6.0])
    bearing = analytic_bearing_degrees(target - agent.position)

    for _ in range(200):
        decision = controller.make_decision_from_target(agent, target, shoot=False, role="engage")
        turn_delta = angle_to_target(agent.position, agent.heading_deg, target)
        # The command must act in the direction that reduces the error, not saturate uselessly.
        assert decision.turn == pytest.approx(max(-1.0, min(1.0, turn_delta / 35.0)), abs=1e-9)
        agent.heading_deg = (agent.heading_deg + decision.turn * 110.0 * 0.1) % 360.0
        error = abs((agent.heading_deg - bearing + 180.0) % 360.0 - 180.0)
        if error < 1.0:
            break
    assert error < 1.0, f"heading did not converge on the target bearing (ended at {error:.1f} deg)"


def test_aim_direction_is_independent_of_heading() -> None:
    """Aiming is not affected by the turn law, which is why agents could still hit things while it was
    broken. Pinned so a future change to the control law is not mistaken for a change in aim."""
    controller = RLTeamController.__new__(RLTeamController)
    controller.team_name = "Team 1"
    controller.rng = random.Random(0)
    target = np.array([3.0, 1.0, 4.0])
    agent = AgentSnapshot("1-1", "Team 1", "CTE", np.array([0.0, 1.0, 0.0]), 275.0, True)
    decision = controller.make_decision_from_target(agent, target, shoot=True, role="engage")
    assert decision.aim_direction == pytest.approx(normalize(target - agent.position), abs=1e-9)
