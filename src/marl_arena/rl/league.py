"""Past-self opponents for the three concurrently training teams.

All three teams train against each other inside one run, so what a team learns is a response to two
co-adapting opponents at that moment (docs/10 § 10.1, "Non-stationary opponents"). The pool keeps frozen
snapshots of each team's actor; in a fraction of matches one team plays a snapshot instead of its current
self, so the other two train against a mixture of current and past opponents -- the fictitious-play
direction, at the cost of one team sitting out an update.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class OpponentPool:
    def __init__(self, controllers: dict[str, Any], seed: int, prob: float, every: int, size: int) -> None:
        if not 0.0 < prob <= 1.0:
            raise ValueError(f"RL_OPPONENT_POOL_PROB must be in (0, 1], got {prob}")
        self.controllers = controllers
        self.prob, self.every = prob, max(1, every)
        # A dedicated stream: the pool's draws never shift the arena's or the controllers' random streams.
        self.rng = np.random.default_rng(seed + 7919)
        self.snapshots: dict[str, deque] = {team: deque(maxlen=max(1, size)) for team in controllers}
        self.active: tuple[str, Any] | None = None
        self.frozen_matches = 0

    def maybe_snapshot(self, match_index: int) -> None:
        if match_index % self.every:
            return
        for team, controller in self.controllers.items():
            # JAX arrays are immutable and the PPO update rebinds `params`, so keeping the reference is a copy.
            self.snapshots[team].append(controller.actor.params)

    def begin_match(self) -> str | None:
        """Freeze one team to a past snapshot for this match, or none; returns the frozen team."""
        assert self.active is None, "begin_match called twice without end_match"
        if not all(controller.training_enabled for controller in self.controllers.values()):
            return None
        if self.rng.random() >= self.prob:
            return None
        candidates = [team for team, pool in self.snapshots.items() if pool]
        if not candidates:
            return None
        team = candidates[int(self.rng.integers(len(candidates)))]
        pool = self.snapshots[team]
        controller = self.controllers[team]
        self.active = (team, controller.actor.params)
        controller.actor.params = pool[int(self.rng.integers(len(pool)))]
        controller.frozen = True
        self.frozen_matches += 1
        return team

    def end_match(self) -> None:
        if self.active is None:
            return
        team, params = self.active
        controller = self.controllers[team]
        controller.actor.params = params
        controller.frozen = False
        self.active = None
