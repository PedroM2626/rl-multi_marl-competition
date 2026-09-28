"""Untrained reference point: how does the arena behave when nobody has learned anything?

Runs headless matches with freshly initialised networks (no checkpoints loaded). Two action policies are
available and they answer different questions:

  greedy   argmax of the random logits — the same selection rule the held-out evaluation uses, so this is
           the direct "what does an untrained network do under the measurement harness" reference. It is a
           degenerate policy: a random network's argmax is usually one fixed action, so a team may never
           fire at all.
  uniform  actions drawn uniformly from the real action space, ignoring the weights — the actual "random
           play" baseline.

The variant is held fixed and domain randomisation is off so the distribution is not smoothed by arena
variation.

Usage:
    python scripts/random_baseline.py --matches 300 --policy uniform
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DRAW_TEAM = "draw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/baseline"))
    parser.add_argument(
        "--policy", choices=("greedy", "uniform"), default="greedy",
        help="greedy = argmax of the random weights, uniform = actions drawn from the action space"
    )
    parser.add_argument(
        "--rotation", type=int, default=0, help="slot rotation applied to the paradigm assignment"
    )
    return parser.parse_args()


def run(matches: int, seed: int, rotation: int, policy: str) -> dict[str, object]:
    from marl_arena.config import CONFIG, seed_all
    from marl_arena.controllers.rl_controller import set_rl_training
    from marl_arena.rl.actions import NUM_ACTIONS
    from marl_arena.systems.simulation import ArenaSimulation

    seed_all(seed)
    sim = ArenaSimulation(
        seed=seed, domain_randomization=False, load_checkpoints=False, paradigm_rotation=rotation
    )
    set_rl_training(sim.controllers, False)
    if policy == "uniform":
        # Replace the argmax the controllers would use with a draw from the action space, so the reference
        # point does not depend on which action the random initialisation happens to favour.
        action_rng = np.random.default_rng(seed)
        for controller in sim.controllers.values():
            controller._greedy_action = lambda logits, _r=action_rng: int(_r.integers(NUM_ACTIONS))

    per_team: dict[str, dict[str, float]] = {
        name: {"wins": 0, "eliminations": 0, "survival_sum": 0.0, "hits": 0, "misses": 0, "matches": 0}
        for name in sim.cumulative_metrics
    }
    durations: list[float] = []
    outcomes = {"wipeout": 0, "timeout": 0, "draw": 0}
    dt = CONFIG.sim_step_dt
    cap = sim.match_variant.match_duration_seconds

    for _ in range(matches):
        while not sim.step(dt):
            pass
        result = sim.finish_match()
        durations.append(result.duration_seconds)
        if result.winner_team == DRAW_TEAM:
            outcomes["draw"] += 1
        elif result.duration_seconds >= cap - dt:
            outcomes["timeout"] += 1
        else:
            outcomes["wipeout"] += 1
        for row in result.team_rows:
            stats = per_team[row["team_name"]]
            stats["wins"] += row["winner"]
            stats["eliminations"] += row["eliminations"]
            stats["survival_sum"] += row["mean_survival_time"]
            stats["hits"] += row["shots_hit"]
            stats["misses"] += row["shots_missed"]
            stats["matches"] += 1
        sim.reset_match()

    paradigms = {spec.team_name: spec.paradigm for spec in sim.match_variant.team_spawns}
    summary = {}
    for name, stats in per_team.items():
        n = max(stats["matches"], 1)
        total_shots = stats["hits"] + stats["misses"]
        summary[name] = {
            "paradigm": paradigms.get(name, ""),
            "win_rate": stats["wins"] / n,
            "eliminations_per_match": stats["eliminations"] / n,
            "mean_survival_time": stats["survival_sum"] / n,
            "shots_per_match": total_shots / n,
            "shot_accuracy": stats["hits"] / max(total_shots, 1),
        }
    return {
        "matches": matches,
        "seed": seed,
        "action_policy": policy,
        "paradigm_rotation": rotation,
        "match_duration_cap_seconds": cap,
        "outcomes": outcomes,
        "duration_seconds": {
            "min": min(durations),
            "median": statistics.median(durations),
            "max": max(durations),
            "mean": statistics.fmean(durations),
        },
        "per_team": summary,
    }


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = run(args.matches, args.seed, args.rotation, args.policy)
    out_dir = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"random_baseline_{args.policy}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    outcomes = result["outcomes"]
    n = result["matches"]
    print(f"{n} headless matches ({args.policy}-policy init, no checkpoints): {outcomes}")
    for key, count in outcomes.items():
        print(f"  {key:8s} {count:4d}  {count / n:6.1%}")
    dur = result["duration_seconds"]
    print(
        f"duration min={dur['min']:.1f}s median={dur['median']:.1f}s "
        f"max={dur['max']:.1f}s mean={dur['mean']:.1f}s"
    )
    for name, stats in sorted(result["per_team"].items(), key=lambda kv: kv[1]["paradigm"]):
        print(
            f"  {stats['paradigm']:12s} win {stats['win_rate']:.3f}  "
            f"elim/match {stats['eliminations_per_match']:.2f}  "
            f"survival {stats['mean_survival_time']:.1f}s  "
            f"shots/match {stats['shots_per_match']:.1f}  "
            f"accuracy {stats['shot_accuracy']:.3f}"
        )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
