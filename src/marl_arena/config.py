from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
METRICS_DIR = DATA_DIR / "metrics"
EXPORTS_DIR = DATA_DIR / "exports"

load_dotenv(ROOT_DIR / ".env")


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ArenaConfig:
    arena_size: float = float(os.getenv("ARENA_SIZE", "32"))
    match_duration_seconds: float = float(os.getenv("MATCH_DURATION_SECONDS", "90"))
    respawn_enabled: bool = _read_bool("RESPAWN_ENABLED", False)
    headless_matches: int = int(os.getenv("HEADLESS_MATCHES", "20"))
    agent_move_speed: float = float(os.getenv("AGENT_MOVE_SPEED", "4.5"))
    agent_turn_speed: float = float(os.getenv("AGENT_TURN_SPEED", "110"))
    jump_speed: float = float(os.getenv("JUMP_SPEED", "6.3"))
    gravity: float = float(os.getenv("GRAVITY", "14.0"))
    shoot_range: float = float(os.getenv("SHOOT_RANGE", "20.0"))
    shoot_cooldown: float = float(os.getenv("SHOOT_COOLDOWN", "0.45"))
    plot_update_interval: float = float(os.getenv("PLOT_UPDATE_INTERVAL", "1.0"))
    metrics_flush_interval: float = float(os.getenv("METRICS_FLUSH_INTERVAL", "1.5"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "7"))


CONFIG = ArenaConfig()

for directory in (DATA_DIR, METRICS_DIR, EXPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
