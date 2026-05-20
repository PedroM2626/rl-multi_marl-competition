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
