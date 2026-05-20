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
    agent_move_speed: float = float(os.getenv("AGENT_MOVE_SPEED", "4.5"))
    agent_turn_speed: float = float(os.getenv("AGENT_TURN_SPEED", "110"))
    jump_speed: float = float(os.getenv("JUMP_SPEED", "6.3"))
    gravity: float = float(os.getenv("GRAVITY", "14.0"))
    shoot_range: float = float(os.getenv("SHOOT_RANGE", "20.0"))
    shoot_cooldown: float = float(os.getenv("SHOOT_COOLDOWN", "0.45"))
    plot_update_interval: float = float(os.getenv("PLOT_UPDATE_INTERVAL", "1.0"))
    metrics_flush_interval: float = float(os.getenv("METRICS_FLUSH_INTERVAL", "1.5"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "7"))
    sim_step_dt: float = float(os.getenv("SIM_STEP_DT", "0.1"))
    domain_randomization: bool = _read_bool("DOMAIN_RANDOMIZATION", True)
    rl_train_total_steps: int = int(os.getenv("RL_TRAIN_TOTAL_STEPS", "3000000"))
    rl_save_every_steps: int = int(os.getenv("RL_SAVE_EVERY_STEPS", "100000"))
    rl_log_every_steps: int = int(os.getenv("RL_LOG_EVERY_STEPS", "50000"))
    rl_metrics_every_matches: int = int(os.getenv("RL_METRICS_EVERY_MATCHES", "10"))
    dr_arena_size_min: float = float(os.getenv("DR_ARENA_SIZE_MIN", "28"))
    dr_arena_size_max: float = float(os.getenv("DR_ARENA_SIZE_MAX", "36"))
    dr_match_duration_min: float = float(os.getenv("DR_MATCH_DURATION_MIN", "60"))
    dr_match_duration_max: float = float(os.getenv("DR_MATCH_DURATION_MAX", "120"))
    dr_move_speed_min: float = float(os.getenv("DR_MOVE_SPEED_MIN", "3.5"))
    dr_move_speed_max: float = float(os.getenv("DR_MOVE_SPEED_MAX", "5.5"))
    dr_turn_speed_min: float = float(os.getenv("DR_TURN_SPEED_MIN", "90"))
    dr_turn_speed_max: float = float(os.getenv("DR_TURN_SPEED_MAX", "130"))
    dr_shoot_range_min: float = float(os.getenv("DR_SHOOT_RANGE_MIN", "16"))
    dr_shoot_range_max: float = float(os.getenv("DR_SHOOT_RANGE_MAX", "24"))
    dr_shoot_cooldown_min: float = float(os.getenv("DR_SHOOT_COOLDOWN_MIN", "0.35"))
    dr_shoot_cooldown_max: float = float(os.getenv("DR_SHOOT_COOLDOWN_MAX", "0.6"))
    dr_obstacle_count_min: int = int(os.getenv("DR_OBSTACLE_COUNT_MIN", "5"))
    dr_obstacle_count_max: int = int(os.getenv("DR_OBSTACLE_COUNT_MAX", "10"))
    rl_learning_rate: float = float(os.getenv("RL_LEARNING_RATE", "0.0003"))
    rl_gamma: float = float(os.getenv("RL_GAMMA", "0.99"))
    rl_gae_lambda: float = float(os.getenv("RL_GAE_LAMBDA", "0.95"))
    rl_clip_eps: float = float(os.getenv("RL_CLIP_EPS", "0.2"))
    rl_value_coef: float = float(os.getenv("RL_VALUE_COEF", "0.5"))
    rl_entropy_coef: float = float(os.getenv("RL_ENTROPY_COEF", "0.01"))
    rl_max_grad_norm: float = float(os.getenv("RL_MAX_GRAD_NORM", "0.5"))
    rl_ppo_epochs: int = int(os.getenv("RL_PPO_EPOCHS", "4"))
    rl_batch_size: int = int(os.getenv("RL_BATCH_SIZE", "256"))
    rl_hidden_dim: int = int(os.getenv("RL_HIDDEN_DIM", "128"))
    rl_device: str = os.getenv("RL_DEVICE", "cpu")
    rl_checkpoint_dir: Path = ROOT_DIR / "data" / "checkpoints"


CONFIG = ArenaConfig()

for directory in (DATA_DIR, METRICS_DIR, EXPORTS_DIR, CONFIG.rl_checkpoint_dir):
    directory.mkdir(parents=True, exist_ok=True)
