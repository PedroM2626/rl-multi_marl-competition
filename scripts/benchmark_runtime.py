"""Compare warmed JAX training throughput with the Rust core and Python fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1500, help="measured environment steps per run")
    parser.add_argument("--warmup-steps", type=int, default=1500, help="steps to run before timing")
    parser.add_argument("--repeats", type=int, default=2, help="paired measurements per runtime")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _digest(controller) -> str:
    checksum = hashlib.sha256()
    for prefix, model in ((b"actor", controller.actor), (b"critic", controller.critic)):
        if model is None:
            continue
        for name, value in sorted(model.state_dict().items()):
            checksum.update(prefix + name.encode("utf-8"))
            checksum.update(value.tobytes())
    return checksum.hexdigest()


def _run(enabled: bool, steps: int, seed: int, native, simulation_module, ArenaSimulation, set_rl_training):
    import jax

    simulation_module._arena_native = native if enabled else None
    simulation = ArenaSimulation(seed=seed, domain_randomization=False, load_checkpoints=False)
    set_rl_training(simulation.controllers, True)
    boundaries: list[int] = []
    winners: list[str] = []
    started = time.perf_counter()
    for _ in range(steps):
        if simulation.step(0.1):
            boundaries.append(simulation.total_env_steps)
            winners.append(simulation.finish_match().winner_team)
            simulation.reset_match()
    for controller in simulation.controllers.values():
        jax.tree_util.tree_map(jax.block_until_ready, controller.actor.params)
        if controller.critic is not None:
            jax.tree_util.tree_map(jax.block_until_ready, controller.critic.params)
    elapsed = time.perf_counter() - started
    return {
        "steps": steps,
        "seconds": elapsed,
        "steps_per_second": steps / elapsed,
        "boundaries": boundaries,
        "winners": winners,
        "policy_sha256": {
            name: _digest(controller) for name, controller in simulation.controllers.items()
        },
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.warmup_steps <= 0 or args.repeats <= 0:
        raise SystemExit("steps, warmup-steps and repeats must be positive")

    import jax
    from marl_arena.controllers.rl_controller import set_rl_training
    from marl_arena.systems import simulation as simulation_module
    from marl_arena.systems.simulation import ArenaSimulation

    native = simulation_module._arena_native
    if native is None:
        raise SystemExit("Rust extension is not installed; build it with maturin develop --release first")

    warmups = {
        "rust": _run(True, args.warmup_steps, args.seed, native, simulation_module, ArenaSimulation, set_rl_training),
        "python": _run(False, args.warmup_steps, args.seed, native, simulation_module, ArenaSimulation, set_rl_training),
    }
    runs = {"rust": [], "python": []}
    for _ in range(args.repeats):
        for name, enabled in (("rust", True), ("python", False)):
            runs[name].append(
                _run(enabled, args.steps, args.seed, native, simulation_module, ArenaSimulation, set_rl_training)
            )
    parity = all(
        rust["boundaries"] == python["boundaries"]
        and rust["winners"] == python["winners"]
        and rust["policy_sha256"] == python["policy_sha256"]
        for rust, python in zip(runs["rust"], runs["python"], strict=True)
    )
    print(json.dumps({
        "jax_backend": jax.default_backend(),
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "warmups": warmups,
        "measurements": runs,
        "exact_outcome_and_weight_parity": parity,
    }, indent=2))
    if not parity:
        raise SystemExit("Rust and Python runs diverged; see the JSON outcome and weight summaries")


if __name__ == "__main__":
    main()
