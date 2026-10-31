from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Configure common CPU thread pools before a training entry point imports JAX/XLA.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

from dotenv import load_dotenv

from marl_arena import PROJECT_DIR


# The experiment tree this process belongs to, not the directory holding this file: experiment 2 runs
# the same engine through its shim and must read its own .env and data/.
ROOT_DIR = PROJECT_DIR
# Load the committed configuration before deriving any path from the environment: an ARENA_DATA_DIR
# written into .env must re-root the data tree just as one exported by a campaign does.
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = Path(os.getenv("ARENA_DATA_DIR", str(ROOT_DIR / "data")))
METRICS_DIR = DATA_DIR / "metrics"
EXPORTS_DIR = DATA_DIR / "exports"
# A demo run must not append to the versioned metrics tree: `python main.py` is the documented
# "just look at it" command, and its matches are not part of the study's evidence base.
DEMO_DATA_DIR = DATA_DIR / "demo"


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _read_list(name: str, default: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


@dataclass(frozen=True)
class ArenaConfig:
    # Which three paradigms the experiment compares, in rotation order. This is the only thing the two
    # experiment trees used to differ in at the engine level; each tree's .env now states it.
    paradigm_cycle: tuple[str, ...] = _read_list("PARADIGM_CYCLE", "CTE,DTE,CTDE")
    arena_size: float = float(os.getenv("ARENA_SIZE", "32"))
    match_duration_seconds: float = float(os.getenv("MATCH_DURATION_SECONDS", "90"))
    agent_move_speed: float = float(os.getenv("AGENT_MOVE_SPEED", "4.5"))
    agent_turn_speed: float = float(os.getenv("AGENT_TURN_SPEED", "110"))
    jump_speed: float = float(os.getenv("JUMP_SPEED", "6.3"))
    gravity: float = float(os.getenv("GRAVITY", "14.0"))
    shoot_range: float = float(os.getenv("SHOOT_RANGE", "20.0"))
    shoot_cooldown: float = float(os.getenv("SHOOT_COOLDOWN", "0.45"))
    random_seed: int = int(os.getenv("RANDOM_SEED", "7"))
    sim_step_dt: float = float(os.getenv("SIM_STEP_DT", "0.1"))
    domain_randomization: bool = _read_bool("DOMAIN_RANDOMIZATION", True)
    # Reward and termination shape. The defaults reproduce exactly the behaviour every committed number
    # was measured under; "arena v2" (docs/08_results.md#arena-v2) is a set of overrides, not a rewrite,
    # so the two are comparable and the old results stay reproducible.
    reward_alive_bonus: float = float(os.getenv("REWARD_ALIVE_BONUS", "0.015"))
    reward_alive_decay_half_life: float = float(os.getenv("REWARD_ALIVE_DECAY_HALF_LIFE_S", "0"))
    reward_kill: float = float(os.getenv("REWARD_KILL", "1.2"))
    reward_death: float = float(os.getenv("REWARD_DEATH", "-0.5"))
    reward_hit: float = float(os.getenv("REWARD_HIT", "-1.0"))
    termination_cap_awards_win: bool = _read_bool("TERMINATION_CAP_AWARDS_WIN", True)
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
    # Evaluation readout. 0 keeps the argmax policy — the readout every published number was measured with.
    # A positive value makes a held-out evaluation sample from the policy at that softmax temperature instead,
    # which is the other thing the trained network does; docs/08_results.md#readout-divergence shows the two
    # readouts of the same weights can rank the arms differently.
    eval_temperature: float = float(os.getenv("EVAL_TEMPERATURE", "0"))
    rl_device: str = os.getenv("RL_DEVICE", "cpu")
    # Remedies for the threats in docs/10_threats_to_validity.md. Every default reproduces the instrument
    # every committed number was measured with (byte-identical checkpoints, verified the A-34 way); the
    # "v3" campaign (scripts/run_v3_campaign.sh) turns them on together.
    # GAE recursion per agent instead of over the team's interleaved rows (§ 10.1, "GAE over an
    # interleaved buffer").
    rl_gae_per_agent: bool = _read_bool("RL_GAE_PER_AGENT", False)
    # >0: every PPO update takes exactly this many minibatches, sampled with replacement, so a team's
    # gradient volume no longer grows with how long its agents survive (§ 10.1, "Confounded training
    # signal volume"). 0 keeps RL_PPO_EPOCHS passes over the whole buffer.
    rl_update_minibatches: int = int(os.getenv("RL_UPDATE_MINIBATCHES", "0"))
    # >0: stop an update once a minibatch's approx KL exceeds 1.5x this value (A-14 made KL observable,
    # this makes it enforced). 0 disables the guard.
    rl_target_kl: float = float(os.getenv("RL_TARGET_KL", "0"))
    # Trainable parameters per team. 0 gives every arm RL_HIDDEN_DIM; a positive value picks each arm's
    # hidden width so its total lands closest to it; -1 targets the largest of the six arms at
    # RL_HIDDEN_DIM (§ 10.1, "Unmatched model capacity").
    rl_param_target: int = int(os.getenv("RL_PARAM_TARGET", "0"))
    # CTDE-Comm: communication rounds, and whether dead teammates are excluded from the mean instead of
    # contributing msg_net(0) (§ 10.2).
    comm_rounds: int = int(os.getenv("COMM_ROUNDS", "1"))
    comm_mask_dead: bool = _read_bool("COMM_MASK_DEAD", False)
    # CTDE-VD: sum per-agent values of each agent's *local* observation (VDN proper) instead of slices of
    # the global state (§ 10.2, "CTDE-VD is not value decomposition over local observations").
    vd_local_critic: bool = _read_bool("VD_LOCAL_CRITIC", False)
    # Ray-cast distances to the nearest obstacle or wall, appended to every observation; 0 keeps cover
    # invisible to the policy (A-17).
    obs_obstacle_rays: int = int(os.getenv("OBS_OBSTACLE_RAYS", "0"))
    # "symmetric" replaces the fixed evaluation arena with a three-fold layout of square pillars and
    # spawns on a circle (A-32); "legacy" is the measured arena.
    arena_layout: str = os.getenv("ARENA_LAYOUT", "legacy")
    # Rotate the spawn positions among the teams every match, so each team spends a third of its matches
    # at each position whatever the geometry (A-32).
    spawn_cycle_per_match: bool = _read_bool("SPAWN_CYCLE_PER_MATCH", False)
    # Opponent pool: with this probability per match, one team plays a frozen past snapshot of itself, so
    # the others train against more than their current co-adapting opponents (§ 10.1, "Non-stationary
    # opponents"). Snapshots are taken every N matches and the pool keeps the most recent M.
    rl_opponent_pool_prob: float = float(os.getenv("RL_OPPONENT_POOL_PROB", "0"))
    rl_opponent_pool_every: int = int(os.getenv("RL_OPPONENT_POOL_EVERY_MATCHES", "50"))
    rl_opponent_pool_size: int = int(os.getenv("RL_OPPONENT_POOL_SIZE", "20"))
    rl_checkpoint_dir: Path = DATA_DIR / "checkpoints"


def seed_all(seed: int) -> None:
    """Seed every RNG the training loop consumes.

    ArenaSimulation seeds its own random.Random and each controller gets an offset seed, but weight
    initialisation and action sampling use explicit JAX keys; PPO shuffling and advantage
    normalisation use NumPy, so a run is only reproducible if those and Python's RNG are pinned.
    """
    import random as _random

    import numpy as _np

    _random.seed(seed)
    _np.random.seed(seed)


CONFIG = ArenaConfig()

for directory in (DATA_DIR, METRICS_DIR, EXPORTS_DIR, CONFIG.rl_checkpoint_dir):
    directory.mkdir(parents=True, exist_ok=True)
