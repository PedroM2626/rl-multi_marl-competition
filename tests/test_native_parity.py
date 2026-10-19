from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.systems import simulation as simulation_module
from marl_arena.systems.simulation import ArenaSimulation


@pytest.mark.skipif(simulation_module._arena_native is None, reason="Rust extension is not built")
def test_rust_and_python_simulation_steps_match(monkeypatch) -> None:
    native_core = simulation_module._arena_native
    rust_sim = ArenaSimulation(seed=22, domain_randomization=False, load_checkpoints=False)
    python_sim = ArenaSimulation(seed=22, domain_randomization=False, load_checkpoints=False)

    for _ in range(32):
        monkeypatch.setattr(simulation_module, "_arena_native", native_core)
        rust_finished = rust_sim.step(0.1)
        monkeypatch.setattr(simulation_module, "_arena_native", None)
        python_finished = python_sim.step(0.1)
        assert rust_finished == python_finished

        for rust_agent, python_agent in zip(rust_sim.agents, python_sim.agents, strict=True):
            np.testing.assert_allclose(rust_agent.position, python_agent.position, rtol=0, atol=1e-12)
            assert rust_agent.heading_deg == pytest.approx(python_agent.heading_deg, abs=1e-12)
            assert rust_agent.vertical_velocity == pytest.approx(python_agent.vertical_velocity, abs=1e-12)
            assert rust_agent.alive == python_agent.alive
            assert (rust_agent.kills, rust_agent.hits, rust_agent.misses) == (
                python_agent.kills, python_agent.hits, python_agent.misses
            )
            assert rust_agent.survival_time == pytest.approx(python_agent.survival_time, abs=1e-12)

        assert len(rust_sim.projectiles) == len(python_sim.projectiles)
        for rust_projectile, python_projectile in zip(
            rust_sim.projectiles, python_sim.projectiles, strict=True
        ):
            assert rust_projectile.projectile_id == python_projectile.projectile_id
            np.testing.assert_allclose(rust_projectile.position, python_projectile.position, rtol=0, atol=1e-12)
            np.testing.assert_allclose(rust_projectile.hit_position, python_projectile.hit_position, rtol=0, atol=1e-12)
            assert rust_projectile.distance_travelled == pytest.approx(
                python_projectile.distance_travelled, abs=1e-12
            )
            assert rust_projectile.age == pytest.approx(python_projectile.age, abs=1e-12)

        for team_name in rust_sim.cumulative_metrics:
            rust_metrics = rust_sim.cumulative_metrics[team_name]
            python_metrics = python_sim.cumulative_metrics[team_name]
            assert vars(rust_metrics) == vars(python_metrics)

    monkeypatch.setattr(simulation_module, "_arena_native", native_core)
