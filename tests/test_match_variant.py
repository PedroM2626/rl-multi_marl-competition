from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from marl_arena.config import CONFIG
from marl_arena.systems.match_variant import create_default_variant, sample_training_variant


def test_default_variant_matches_config() -> None:
    variant = create_default_variant(CONFIG)
    assert variant.arena_size == CONFIG.arena_size
    assert len(variant.team_spawns) == 3
    assert len(variant.obstacles) >= 5


def test_training_variant_within_bounds() -> None:
    rng = random.Random(7)
    variant = sample_training_variant(rng, CONFIG, 1)
    assert CONFIG.dr_arena_size_min <= variant.arena_size <= CONFIG.dr_arena_size_max
    assert CONFIG.dr_shoot_range_min <= variant.shoot_range <= CONFIG.dr_shoot_range_max
    assert CONFIG.dr_obstacle_count_min <= len(variant.obstacles) <= CONFIG.dr_obstacle_count_max


def test_paradigm_rotation_covers_every_slot_assignment() -> None:
    """A slot carries a spawn corner, a seed offset and a construction-order position, so a fixed
    paradigm-to-slot mapping makes a slot effect indistinguishable from an architecture effect."""
    from marl_arena.systems.match_variant import PARADIGM_CYCLE, TEAM_NAMES, paradigm_assignment

    seen = []
    for rotation in range(len(TEAM_NAMES)):
        assignment = paradigm_assignment(rotation)
        assert sorted(assignment.values()) == sorted(PARADIGM_CYCLE)
        assert list(assignment) == list(TEAM_NAMES)
        seen.append(tuple(assignment.values()))
    assert len(set(seen)) == len(TEAM_NAMES), "rotations must produce distinct assignments"
    # Every paradigm must land in every slot exactly once across the rotations.
    for index, name in enumerate(TEAM_NAMES):
        assert {s[index] for s in seen} == set(PARADIGM_CYCLE)


def test_simulation_and_controllers_agree_on_the_rotation() -> None:
    from marl_arena.systems.simulation import ArenaSimulation

    for seed in (1, 2, 3, 4):
        simulation = ArenaSimulation(seed=seed, domain_randomization=False, load_checkpoints=False)
        from_spawns = {spawn.team_name: spawn.paradigm for spawn in simulation.match_variant.team_spawns}
        from_controllers = {name: c.paradigm for name, c in simulation.controllers.items()}
        from_metrics = {name: m.paradigm for name, m in simulation.cumulative_metrics.items()}
        assert from_spawns == from_controllers == from_metrics, f"seed {seed} disagrees on the mapping"
