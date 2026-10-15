"""Cross-experiment round-robin: all six architectures against a common pool of opponents.

The two experiments never meet during the study: experiment 1's CTE/DTE/CTDE only ever fight each other,
and experiment 2's VD/CAC/Comm only ever fight each other. That makes their win rates comparable to one
another only by assumption, because each is measured against opponents co-adapted to it. This script drops
that assumption: it takes the six *already trained and already versioned* policies and plays every
three-architecture combination against them, so all six end up measured on the same footing.

No training happens here, and no architecture is re-implemented. Each policy is driven by the controller
class from the tree that produced it: experiment 2's package is a strict superset of experiment 1's at the
network and PPO level, so it is the host package, and experiment 1's `rl_controller.py` is loaded from
source under an alias for the CTE/DTE/CTDE arms. Evaluating a policy under a re-implemented actuator would
silently invalidate the whole comparison, so the exact training code is reused instead.

Slot position is rotated within every triple, because the study already showed a slot effect can masquerade
as an architecture effect.

Run from experiment 2's directory (its package is the superset):

    cd ctde_arena && python ../scripts/run_cross_arena.py --matches-per-triple 30
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path

CTDE_ROOT = Path(__file__).resolve().parents[1] / "ctde_arena"
ROOT_ROOT = CTDE_ROOT.parent

# The host package must be experiment 2's: its networks.py and ppo.py contain every class experiment 1
# uses plus the VD and Comm ones, so both trees' controllers resolve their imports against it.
SRC_DIR = CTDE_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Default policies: the promoted replicate of each experiment, versioned in the two data/ trees.
POLICIES = {
    "CTE": CTDE_ROOT.parent / "data" / "checkpoints" / "team_1_cte.pt",
    "DTE": CTDE_ROOT.parent / "data" / "checkpoints" / "team_2_dte.pt",
    "CTDE": CTDE_ROOT.parent / "data" / "checkpoints" / "team_3_ctde.pt",
    "CTDE-VD": CTDE_ROOT / "data" / "checkpoints" / "team_2_ctde-vd.pt",
    "CTDE-CAC": CTDE_ROOT / "data" / "checkpoints" / "team_3_ctde-cac.pt",
    "CTDE-Comm": CTDE_ROOT / "data" / "checkpoints" / "team_1_ctde-comm.pt",
}
# Which architectures experiment 1's controller class was written for.
ROOT_PARADIGMS = {"CTE", "DTE", "CTDE"}
DRAW_TEAM = "draw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches-per-triple", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1, help="harness seed: the match stream, not the policy")
    parser.add_argument(
        "--exp1-replicate", type=int, default=None,
        help="training seed of the experiment 1 policies, loaded from data/runs/exp1_seedN"
    )
    parser.add_argument(
        "--exp2-replicate", type=int, default=None,
        help="training seed of the experiment 2 policies, loaded from data/runs/exp2_seedN"
    )
    parser.add_argument("--out", type=Path, default=Path("results/cross_arena"))
    parser.add_argument(
        "--fixed-rotation", action="store_true",
        help="do not cycle slot permutations; keep the paradigms in the order given"
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="replay experiment 1's own triple and compare against its recorded held-out win rates"
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def wilson(wins: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Same estimator the study uses, loaded from scripts/report_study.py when available."""
    if trials == 0:
        return (0.0, 0.0)
    p = wins / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return (centre - half, centre + half)


def replicate_policies(exp1_seed: int, exp2_seed: int) -> dict[str, Path]:
    """The six policies taken from named *training* replicates instead of the promoted ones.

    Without this the round-robin measures two particular checkpoints per architecture and cannot say
    anything about the architectures. With it, a block is one training run per experiment, and the spread
    across blocks is the training variance the study has not been able to resolve.
    """
    runs = ROOT_ROOT / "data" / "runs"
    versioned = ROOT_ROOT / "data" / "training_replicates"
    out: dict[str, Path] = {}
    for experiment, seed, paradigms in (
        ("exp1", exp1_seed, ("CTE", "DTE", "CTDE")),
        ("exp2", exp2_seed, ("CTDE-VD", "CTDE-CAC", "CTDE-Comm")),
    ):
        # Prefer the versioned copy so a reader who clones the repository runs the same policies; the
        # git-ignored study scratch is only present on the machine that produced them.
        directory = versioned / f"{experiment}_seed{seed}"
        if not directory.is_dir():
            directory = runs / f"{experiment}_seed{seed}" / "checkpoints"
        if not directory.is_dir():
            raise SystemExit(f"no such training replicate: {experiment}_seed{seed}")
        for paradigm in paradigms:
            found = sorted(directory.glob(f"*_{paradigm.lower()}.pt"))
            if len(found) != 1:
                raise SystemExit(
                    f"expected one {paradigm} checkpoint in {directory}, found {[m.name for m in found]}"
                )
            out[paradigm] = found[0]
    return out


class CrossArena:
    """Owns one simulation whose match variant is supplied per match rather than sampled."""

    def __init__(self, seed: int, policies: dict[str, Path] | None = None):
        self.policies = policies or POLICIES
        from marl_arena.config import CONFIG
        from marl_arena.controllers.rl_controller import resolve_device, set_rl_training
        from marl_arena.systems.match_variant import create_default_variant
        from marl_arena.systems.simulation import ArenaSimulation

        self.config = CONFIG
        self.device = resolve_device()
        self.set_training = set_rl_training
        self.base_variant = create_default_variant(CONFIG, rotation=0)
        self.root_controller = load_module(
            "exp1_rl_controller", ROOT_ROOT / "src" / "marl_arena" / "controllers" / "rl_controller.py"
        ).RLTeamController
        self.ctde_controller = __import__(
            "marl_arena.controllers.rl_controller", fromlist=["RLTeamController"]
        ).RLTeamController
        self.load_checkpoint = __import__(
            "marl_arena.rl.actions", fromlist=["load_checkpoint"]
        ).load_checkpoint

        class _Sim(ArenaSimulation):
            def __init__(inner, pending):
                # Set before super().__init__(): the base constructor plays reset_match() itself, and
                # that is what calls _sample_next_variant below.
                inner.pending = pending
                super().__init__(
                    seed=seed, domain_randomization=False, load_checkpoints=False, paradigm_rotation=0
                )

            def _sample_next_variant(inner):
                inner.variant_counter += 1
                return inner.pending

        self._sim_cls = _Sim
        self.sim = _Sim(self.base_variant)

    def _variant_for(self, paradigms: tuple[str, str, str]) -> object:
        spawns = tuple(
            replace(spec, paradigm=paradigm)
            for spec, paradigm in zip(self.base_variant.team_spawns, paradigms, strict=True)
        )
        return replace(self.base_variant, team_spawns=spawns)

    def _controllers_for(self, paradigms: tuple[str, str, str]) -> dict:
        controllers = {}
        for slot, (spec, paradigm) in enumerate(zip(self.base_variant.team_spawns, paradigms, strict=True)):
            cls = self.root_controller if paradigm in ROOT_PARADIGMS else self.ctde_controller
            controller = cls(spec.team_name, paradigm, self.config.random_seed + 7 * slot, self.device, False)
            path = self.policies[paradigm]
            payload = self.load_checkpoint(path, self.device)
            if payload.get("paradigm") != paradigm:
                raise SystemExit(f"{path.name} records paradigm {payload.get('paradigm')!r}, expected {paradigm!r}")
            controller.actor.load_state_dict(payload["actor"])
            if controller.critic is not None and "critic" in payload:
                controller.critic.load_state_dict(payload["critic"])
            controllers[spec.team_name] = controller
        return controllers

    def play(self, paradigms: tuple[str, str, str], matches: int,
             fixed_rotation: bool = False) -> list[dict[str, object]]:
        """Play `matches` matches of this triple, cycling the six slot permutations.

        `fixed_rotation` pins the paradigms to the slots they were given, which is what the study's own
        held-out evaluation did; it exists so the harness can be checked against a recorded number rather
        than only against its own output.
        """
        permutations = [tuple(paradigms)] if fixed_rotation else list(itertools.permutations(paradigms))
        rows: list[dict[str, object]] = []
        for index in range(matches):
            order = permutations[index % len(permutations)]
            # The variant has to be staged before reset_match(), because that is what reads the spawn
            # centres and hands each team its paradigm label.
            self.sim.pending = self._variant_for(order)
            self.sim.controllers = self._controllers_for(order)
            self.set_training(self.sim.controllers, False)
            self.sim.reset_match()

            dt = self.config.sim_step_dt
            while not self.sim.step(dt):
                pass
            result = self.sim.finish_match()
            # team_rows carries the paradigm straight from the variant this match was built with, so the
            # labels below are the engine's own account of who played, not this script's bookkeeping.
            winner_paradigm = next(
                (row["paradigm"] for row in result.team_rows if row["winner"]), DRAW_TEAM
            )
            for row in result.team_rows:
                rows.append({
                    "triple": "|".join(sorted(paradigms)),
                    "slot": row["team_name"],
                    "paradigm": row["paradigm"],
                    "match": index,
                    "permutation": index % len(permutations),
                    "winner": winner_paradigm,
                    "eliminations": row["eliminations"],
                    "remaining_agents": row["remaining_agents"],
                    "survival_seconds": round(row["mean_survival_time"], 3),
                    "shots": row["shots_hit"] + row["shots_missed"],
                    "shots_hit": row["shots_hit"],
                    "duration_seconds": round(result.duration_seconds, 2),
                })
        return rows


def summarise(rows: list[dict[str, object]]) -> dict[str, object]:
    by_paradigm: dict[str, dict[str, float]] = {}
    for row in rows:
        stats = by_paradigm.setdefault(row["paradigm"], {
            "wins": 0, "matches": 0, "draws": 0, "eliminations": 0, "survival": 0.0,
            "shots": 0, "hits": 0, "timeouts": 0,
        })
        stats["matches"] += 1
        stats["eliminations"] += row["eliminations"]
        stats["survival"] += row["survival_seconds"]
        stats["shots"] += row["shots"]
        stats["hits"] += row["shots_hit"]
        if row["winner"] == DRAW_TEAM:
            stats["draws"] += 1
        elif row["winner"] == row["paradigm"]:
            stats["wins"] += 1
        if row["duration_seconds"] >= 89.0:
            stats["timeouts"] += 1

    table = {}
    for paradigm, stats in sorted(by_paradigm.items()):
        matches = stats["matches"]
        lo, hi = wilson(int(stats["wins"]), int(matches))
        table[paradigm] = {
            "matches": int(matches),
            "wins": int(stats["wins"]),
            "win_rate": round(stats["wins"] / matches, 4),
            "wilson_95_ci": [round(lo, 4), round(hi, 4)],
            "draw_rate": round(stats["draws"] / matches, 4),
            "timeout_rate": round(stats["timeouts"] / matches, 4),
            "eliminations_per_match": round(stats["eliminations"] / matches, 3),
            "mean_survival_seconds": round(stats["survival"] / matches, 2),
            "shots_per_match": round(stats["shots"] / matches, 2),
            "shot_accuracy": round(stats["hits"] / max(stats["shots"], 1), 4),
        }
    return table


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if args.self_check:
        # Two checks, one per controller class: replaying experiment 1's triple exercises the aliased
        # experiment 1 controller, and replaying experiment 2's exercises the host one. Validating only
        # the first would leave the second silently mis-driving three of the six policies.
        #
        # With --fixed-rotation the replay uses the same seed and the same slot order as the recorded
        # evaluation. Greedy action selection is deterministic given the weights and the observation
        # stream, so that run should reproduce the recorded win rates closely; a gap that survives it is
        # an instrument difference, not sampling noise.
        homes = {"experiment 1": (ROOT_ROOT, 9), "experiment 2": (CTDE_ROOT, 2)}
        matches = args.matches_per_triple or 150
        for label, (tree, home_seed) in homes.items():
            provenance = json.loads(
                (tree / "data" / "PROVENANCE.json").read_text(encoding="utf-8")
            )["summary"]
            recorded = provenance["held_out_greedy"]["per_team"]
            expected = {provenance["paradigms"][team]: stats["win_rate"] for team, stats in recorded.items()}
            # The slot order has to come from the record itself. Hard-coding the paradigms in alphabetical
            # order silently puts every arm in a different spawn corner than the one it was measured in,
            # and slot position is exactly the confound the study rotates to control.
            home = tuple(provenance["paradigms"][f"Team {n}"] for n in (1, 2, 3))
            seed = home_seed if args.fixed_rotation else args.seed
            rows = CrossArena(seed).play(home, matches, fixed_rotation=args.fixed_rotation)
            table = summarise(rows)
            mode = ("same seed and same slot order as the record" if args.fixed_rotation
                    else f"seed {args.seed} and all six slot permutations cycled")
            print(f"\nself-check {label}: {matches} matches replayed here, {mode}; "
                  "recorded column is its 150 held-out matches")
            print("| paradigm | this harness | recorded held-out | difference |")
            print("|---|---:|---:|---:|")
            for paradigm in home:
                here = table[paradigm]["win_rate"]
                print(f"| {paradigm} | {here:.3f} | {expected[paradigm]:.3f} | {here - expected[paradigm]:+.3f} |")
        return

    policies = POLICIES
    block = "promoted"
    if args.exp1_replicate is not None or args.exp2_replicate is not None:
        if args.exp1_replicate is None or args.exp2_replicate is None:
            raise SystemExit("--exp1-replicate and --exp2-replicate must be given together")
        policies = replicate_policies(args.exp1_replicate, args.exp2_replicate)
        block = f"exp1_seed{args.exp1_replicate}+exp2_seed{args.exp2_replicate}"
    arena = CrossArena(args.seed, policies)

    triples = list(itertools.combinations(sorted(POLICIES), 3))
    all_rows: list[dict[str, object]] = []
    for position, triple in enumerate(triples, 1):
        rows = arena.play(triple, args.matches_per_triple)
        all_rows.extend(rows)
        print(f"[{position:2d}/{len(triples)}] {'|'.join(triple)} done "
              f"({position * len(triple) * args.matches_per_triple} team-matches)", flush=True)

    out_dir = args.out if args.out.is_absolute() else ROOT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    table = summarise(all_rows)
    matches_played = len(all_rows) // 3
    summary = {
        "design": "all C(6,3) architecture triples, each played to completion with slot rotation",
        "policies": {p: str(path.relative_to(ROOT_ROOT)) for p, path in POLICIES.items()},
        "triples": len(triples),
        "matches_per_triple": args.matches_per_triple,
        "matches_played": matches_played,
        "team_matches": len(all_rows),
        "seed": args.seed,
        "training_block": block,
        "arena": {
            "arena_size": arena.base_variant.arena_size,
            "match_duration_seconds": arena.base_variant.match_duration_seconds,
            "domain_randomization": False,
        },
        "per_architecture": table,
        "chance_level": round(1 / 3, 4),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{matches_played} matches over {len(triples)} triples\n")
    print("| architecture | win rate | 95 % CI | vs 1/3 | draws | timeouts | elim./match | survival | shots | accuracy |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for paradigm, stats in sorted(table.items(), key=lambda kv: -kv[1]["win_rate"]):
        print(
            f"| {paradigm} | **{stats['win_rate']:.3f}** | [{stats['wilson_95_ci'][0]:.3f}, "
            f"{stats['wilson_95_ci'][1]:.3f}] | {stats['win_rate'] - 1/3:+.3f} "
            f"| {stats['draw_rate']:.3f} | {stats['timeout_rate']:.3f} "
            f"| {stats['eliminations_per_match']:.2f} | {stats['mean_survival_seconds']:.1f}s "
            f"| {stats['shots_per_match']:.1f} | {stats['shot_accuracy']:.4f} |"
        )
    print(f"\nwrote {out_dir / 'matches.csv'} and {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
