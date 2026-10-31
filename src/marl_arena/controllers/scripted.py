"""A fixed, hand-written opponent: the reference point that is neither random nor another trained policy.

Every win rate the study reports is either relative to co-trained peers or relative to uniform-random play
(docs/10 § 10.4, item 13). This controller is the third kind of opponent: it does not learn, does not share
a run with anyone, and is not degenerate. It plays inside the same action space as the policies -- move to
one of the four waypoints, fire or hold -- and always picks "engage, and fire when in range", which is the
action an untrained observer would call the obvious one. Because aim follows the movement waypoint in this
arena, that is also the only waypoint from which a shot points at the enemy.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from marl_arena.controllers.base import BaseTeamController, ControllerContext
from marl_arena.models import AgentSnapshot, StepDecision, TransitionRecord

SCRIPTED_PARADIGM = "SCRIPTED"


class ScriptedTeamController(BaseTeamController):
    paradigm_name = SCRIPTED_PARADIGM

    def __init__(self, team_name: str, rng_seed: int) -> None:
        super().__init__(team_name, rng_seed)
        self.paradigm = SCRIPTED_PARADIGM
        self.training_enabled = False
        self.frozen = False
        self.hidden_dim = 0

    def decide(
        self, agent: AgentSnapshot, all_agents: Iterable[AgentSnapshot], context: ControllerContext,
    ) -> StepDecision:
        agents = list(all_agents)
        enemy = self.nearest_enemy(agent, agents)
        if enemy is None:
            return self.make_decision_from_target(agent, agent.position.copy(), False, "engage")
        in_range = float(np.linalg.norm(enemy.position - agent.position)) <= context.shoot_range
        return self.make_decision_from_target(agent, enemy.position.copy(), in_range, "engage")

    def update(self, transitions: List[TransitionRecord]) -> None:
        return None

    # The RL controllers' lifecycle hooks, so the simulation and the evaluation harnesses can hold this
    # controller in any slot without special-casing it.
    def set_training(self, enabled: bool) -> None:
        return None

    def finish_episode(self) -> None:
        return None
