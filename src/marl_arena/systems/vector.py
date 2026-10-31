"""Many arenas, one policy pass per team: a vectorised evaluator.

A training or evaluation step is nine tiny matmuls wrapped in Python and device dispatch (A-26, A-46), so
throughput is bounded by dispatch count, not arithmetic. Running N independent arenas in lock-step and
stacking each team's rows across them turns N dispatches per team per step into one. This is the
environment-side half of what A-26 said a GPU would need; on CPU it already amortises the dispatch.

Scope: evaluation only. The arenas share one set of controllers, and a training controller keys its
pending rollout rows by agent id, which every arena reuses -- so the batch refuses to run while any
controller is training. With one arena it is exactly `ArenaSimulation.step`.
"""

from __future__ import annotations

from typing import Dict, List

from marl_arena.controllers.batching import decide_many_across
from marl_arena.models import MatchResult
from marl_arena.systems.simulation import ArenaSimulation


class VecArena:
    def __init__(self, sims: List[ArenaSimulation]) -> None:
        if not sims:
            raise ValueError("a VecArena needs at least one arena")
        self.sims = sims
        self.controllers = sims[0].controllers
        for sim in sims[1:]:
            # Every arena plays the same policies; the controllers' own networks in the others are dropped.
            sim.controllers = self.controllers

    def _check_frozen(self) -> None:
        training = [team for team, c in self.controllers.items() if getattr(c, "training_enabled", False)]
        if training:
            raise RuntimeError(f"VecArena evaluates frozen policies only; still training: {training}")

    def step(self, dt: float, active: List[int] | None = None) -> Dict[int, bool]:
        """Advance the given arenas (all by default) one step; returns arena index -> match decided."""
        self._check_frozen()
        indices = list(range(len(self.sims))) if active is None else active
        prepared = {index: self.sims[index].begin_step(dt) for index in indices}
        decisions: Dict[int, Dict[str, object]] = {index: {} for index in indices}
        for team_name, controller in self.controllers.items():
            calls, owners = [], []
            for index in indices:
                snapshots, context = prepared[index]
                team_agents = [s for s in snapshots if s.alive and s.team_name == team_name]
                if not team_agents:
                    continue
                if not hasattr(controller, "decide_many"):
                    # A scripted controller has no policy pass to pool.
                    for snapshot in team_agents:
                        decisions[index][snapshot.agent_id] = controller.decide(snapshot, snapshots, context)
                    continue
                calls.append((team_agents, snapshots, context))
                owners.append(index)
            if calls:
                for index, result in zip(owners, decide_many_across(controller, calls), strict=True):
                    decisions[index].update(result)
        return {
            index: self.sims[index].finish_step(dt, prepared[index][0], decisions[index]) for index in indices
        }

    def play(self, matches: int, dt: float) -> List[MatchResult]:
        """Play `matches` complete matches spread over the arenas, in arena-then-match order."""
        self._check_frozen()
        results: List[MatchResult] = []
        remaining = {index: 0 for index in range(len(self.sims))}
        quota = [matches // len(self.sims) + (1 if index < matches % len(self.sims) else 0)
                 for index in range(len(self.sims))]
        active = [index for index in range(len(self.sims)) if quota[index] > 0]
        per_arena: Dict[int, List[MatchResult]] = {index: [] for index in active}
        while active:
            finished = self.step(dt, active)
            for index, done in finished.items():
                if not done:
                    continue
                per_arena[index].append(self.sims[index].finish_match())
                self.sims[index].reset_match()
                remaining[index] += 1
            active = [index for index in active if remaining[index] < quota[index]]
        for index in sorted(per_arena):
            results.extend(per_arena[index])
        return results
