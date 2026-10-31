"""Measure the vectorised evaluator: throughput against the one-arena path, and how often it agrees.

For each arena count N, the same N arena seeds play the same number of greedy held-out matches twice: once
arena by arena through `ArenaSimulation.step` (the published path), once together through `VecArena`. The
first comparison is wall clock; the second is how many matches come out identical row for row, which is the
float32 rounding cost of a wider batch (A-44's ~6e-8, surfacing only when it flips a near-tied argmax).

Usage:
    python scripts/benchmark_vec.py --arenas 1,4,16 --matches 48 --out results/vec_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arenas", default="1,4,16")
    parser.add_argument("--matches", type=int, default=48, help="matches per arena count, spread over the arenas")
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def make_sims(count: int, base_seed: int):
    from marl_arena.controllers.rl_controller import set_rl_training
    from marl_arena.systems.simulation import ArenaSimulation

    sims = []
    for index in range(count):
        # Every arena loads the same checkpoints at the same rotation, so the policies are one set.
        sim = ArenaSimulation(seed=base_seed + 3 * index, domain_randomization=False, load_checkpoints=True,
                              paradigm_rotation=base_seed % 3)
        set_rl_training(sim.controllers, False)
        sims.append(sim)
    return sims


def rows_of(result) -> list[tuple]:
    return [tuple(sorted(row.items())) for row in result.team_rows]


def main() -> None:
    args = parse_args()
    from marl_arena.config import CONFIG
    from marl_arena.runtime import runtime_provenance
    from marl_arena.systems.vector import VecArena

    dt = CONFIG.sim_step_dt
    report = {"matches": args.matches, "seed": args.seed, "runtime": runtime_provenance(), "cells": []}
    for count in (int(item) for item in args.arenas.split(",")):
        quota = [args.matches // count + (1 if i < args.matches % count else 0) for i in range(count)]

        sequential = make_sims(count, args.seed)
        started = time.perf_counter()
        reference = []
        for sim, matches in zip(sequential, quota, strict=True):
            for _ in range(matches):
                while not sim.step(dt):
                    pass
                reference.append(sim.finish_match())
                sim.reset_match()
        sequential_seconds = time.perf_counter() - started
        steps = sum(sim.total_env_steps for sim in sequential)

        vec = VecArena(make_sims(count, args.seed))
        started = time.perf_counter()
        vectorised = vec.play(args.matches, dt)
        vec_seconds = time.perf_counter() - started

        identical = sum(rows_of(a) == rows_of(b) for a, b in zip(reference, vectorised, strict=True))
        cell = {
            "arenas": count,
            "env_steps": steps,
            "sequential_steps_per_second": round(steps / sequential_seconds, 1),
            "vectorised_steps_per_second": round(sum(s.total_env_steps for s in vec.sims) / vec_seconds, 1),
            "speedup": round(sequential_seconds / vec_seconds, 3),
            "identical_matches": identical,
            "compared_matches": len(reference),
        }
        report["cells"].append(cell)
        print(json.dumps(cell), flush=True)
    if args.out is not None:
        path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    main()
