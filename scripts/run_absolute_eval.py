"""Absolute strength: each trained policy against a fixed pool of uniform-random opponents.

Every win rate in this repository is relative -- an arm beating two other arms that trained against it.
That is why no number here is comparable to the literature, and it is the gap chapter 10 lists as open.
This script closes the cheap half of it: put one trained policy in one team, put two uniform-random
controllers in the others, and measure. Random opponents do not adapt, co-evolve, or share a run's
initialisation, so the result is a floor rather than a ranking.

It reuses the round-robin harness rather than reimplementing the arena wiring, so the variant, the slot
rotation and the checkpoint loading are the same code paths that produced results/cross_arena.

Usage:
    cd ctde_arena && python ../scripts/run_absolute_eval.py --matches 60
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_harness():
    """Import scripts/run_cross_arena.py, which owns the CrossArena wiring."""
    path = Path(__file__).resolve().parent / "run_cross_arena.py"
    spec = importlib.util.spec_from_file_location("cross_arena_harness", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=60, help="matches per architecture")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/absolute"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    harness = load_harness()
    from marl_arena.config import CONFIG
    from marl_arena.rl.actions import NUM_ACTIONS
    from marl_arena.systems.match_variant import TeamSpawnSpec
    import numpy as np

    arena = harness.CrossArena(args.seed)
    rng = np.random.default_rng(args.seed)

    def make_random_controller(team_name: str, slot: int):
        """A real controller -- so the simulation is untouched -- forced to sample uniformly.

        The greedy argmax of an untrained network is a near-constant policy (A-30), which is why the
        opponent here is drawn from the action space rather than read off an untrained actor.
        """
        controller = arena.ctde_controller(team_name, "RANDOM", 1000 + slot, arena.device, False)
        controller._greedy_action = lambda logits, _r=rng: int(_r.integers(NUM_ACTIONS))
        return controller

    results = {}
    rows = []
    for paradigm in sorted(harness.POLICIES):
        wins = draws = losses = 0
        durations: list[float] = []
        for index in range(args.matches):
            # Rotate which slot the trained policy occupies, so a corner advantage cannot land on it.
            slot = index % 3
            team_names = [spec.team_name for spec in arena.base_variant.team_spawns]
            spawns = tuple(
                TeamSpawnSpec(name, paradigm if position == slot else "RANDOM",
                              spec.spawn_center, spec.color_rgb)
                for position, (name, spec) in enumerate(zip(team_names, arena.base_variant.team_spawns))
            )
            arena.sim.pending = harness.replace(arena.base_variant, team_spawns=spawns)
            controllers = {}
            for position, spec in enumerate(spawns):
                if position == slot:
                    controllers[spec.team_name] = arena.policy_controller(spec.team_name, paradigm, position)
                else:
                    controllers[spec.team_name] = make_random_controller(spec.team_name, position)
            arena.sim.controllers = controllers
            arena.set_training(arena.sim.controllers, False)
            arena.sim.reset_match()

            dt = CONFIG.sim_step_dt
            while not arena.sim.step(dt):
                pass
            result = arena.sim.finish_match()
            durations.append(result.duration_seconds)
            if result.winner_team == team_names[slot]:
                wins += 1
            elif result.winner_team == harness.DRAW_TEAM:
                draws += 1
            else:
                losses += 1
            rows.append({"paradigm": paradigm, "match": index, "slot": team_names[slot],
                         "winner": result.winner_team, "duration_seconds": round(result.duration_seconds, 2)})

        n = args.matches
        lo, hi = harness.wilson(wins, n)
        results[paradigm] = {
            "matches": n, "wins": wins, "draws": draws, "losses": losses,
            "win_rate_vs_random": round(wins / n, 4),
            "wilson_95_ci": [round(lo, 4), round(hi, 4)],
            "draw_rate": round(draws / n, 4),
            "loss_rate": round(losses / n, 4),
            "median_duration_seconds": float(np.median(durations)),
        }
        print(f"{paradigm:11s} win {wins/n:.3f} [{lo:.3f}, {hi:.3f}]  draw {draws/n:.3f}  "
              f"loss {losses/n:.3f}  median {float(np.median(durations)):5.1f}s", flush=True)

    out_dir = PROJECT_ROOT / args.out if not args.out.is_absolute() else args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps({
        "design": "one trained policy versus two uniform-random controllers, trained slot rotated",
        "seed": args.seed,
        "matches_per_architecture": args.matches,
        "chance_level": round(1 / 3, 4),
        "per_architecture": results,
    }, indent=2), encoding="utf-8")
    import csv
    with (out_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
