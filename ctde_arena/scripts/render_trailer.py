"""Render a short gameplay trailer from the versioned policies and write it as a GIF.

The arena is rendered for real here: Ursina in offscreen window mode, the same visual classes main.py
uses, and the same fixed SIM_STEP_DT the training loop steps on, so the clip is a match of the arena the
study measured rather than a re-drawn approximation of it.

Nothing in the versioned tree is written. ARENA_DATA_DIR is redirected to a scratch directory that is
seeded with a copy of data/checkpoints, so the promoted policies load but MetricsStore cannot touch the
committed artefacts.

Each frame carries a burn-in caption (experiment, replicate, paradigms, match clock, alive count) because
a clip of an RL arena that does not say which policy it is running invites exactly the misattribution this
repository has already been bitten by.

Usage:
    python scripts/render_trailer.py --frames 90 --out docs/assets/trailer_exp1.gif
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
# The clip must reuse the tree's own renderer, and main.py is only importable with the tree root on the
# path. Copying this script into ctde_arena/scripts/ is what renders experiment 2 instead.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402


def configure_panda() -> None:
    """Set the Panda3D config keys, and only then may a window be created.

    This is a function rather than module-level imports because Panda3D fixes the directory that relative
    screenshot names resolve against at *import* time. The scratch frame directory has to be current
    before panda3d is first imported, or every frame silently lands in the repository instead.
    """
    from panda3d.core import NodePath, loadPrcFileData

    loadPrcFileData("", "window-type offscreen")
    loadPrcFileData("", "audio-library-name null")
    loadPrcFileData("", "notify-level error")

FRAME_WIDTH = 640


def experiment_label() -> str:
    """Which experiment this tree is, read from the tree itself.

    Derived from PROJECT_ROOT rather than from --out: the output path is user-supplied and often relative,
    and a caption reading "experiment 1" over experiment 2 footage is precisely the mislabelling this
    repository keeps having to undo.
    """
    return "experiment 2 (ctde_arena)" if PROJECT_ROOT.name == "ctde_arena" else "experiment 1"


def _display(path: Path) -> str:
    """Path relative to the repository root when it is inside it, absolute otherwise.

    Experiment 2 renders from inside ctde_arena/ but writes its clip into the shared docs/assets/, so the
    two are not in a relative relationship and relative_to() would raise. Resolved, and with forward slashes
    in both branches: the sidecar is a committed artefact, so `ctde_arena/../docs/...` would be a path that
    only reads correctly after normalisation and a Windows spelling of it is unreadable on a POSIX host
    (A-42).
    """
    try:
        base = (PROJECT_ROOT.parent if PROJECT_ROOT.name == "ctde_arena" else PROJECT_ROOT).resolve()
        return path.resolve().relative_to(base).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=90, help="captured frames; one frame = one sim step")
    parser.add_argument("--seed", type=int, default=None, help="overrides RANDOM_SEED, hence the slot rotation")
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--fov", type=float, default=62.0, help="horizontal-scaled lens fov; widens the view without moving it"
    )
    parser.add_argument("--untrained", action="store_true", help="render random initialisation instead")
    parser.add_argument(
        "--sidecar", type=Path, default=None,
        help="JSON describing what the clip shows; defaults to the clip's own path with a .json suffix"
    )
    return parser.parse_args()


def scratch_data_dir(tree: Path) -> Path:
    """A private data dir seeded with the tree's versioned checkpoints.

    Redirecting ARENA_DATA_DIR is what keeps MetricsStore's writes out of the committed artefacts; copying
    the checkpoints in is what lets them still load, since the checkpoint path is derived from that same dir.
    """
    scratch = Path(tempfile.mkdtemp(prefix="trailer_")) / "data"
    (scratch / "checkpoints").mkdir(parents=True)
    source = tree / "data" / "checkpoints"
    if source.is_dir():
        for item in list(source.glob("*.npz")) or list(source.glob("*.pt")):
            shutil.copy2(item, scratch / "checkpoints" / item.name)
    return scratch


def alive_counts(simulation) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent in simulation.agents:
        counts[agent.team_name] = counts.get(agent.team_name, 0) + int(agent.alive)
    return counts


def draw_caption(image: Image.Image, lines: list[str]) -> Image.Image:
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    band = 18 + 15 * len(lines)
    draw.rectangle((0, canvas.size[1] - band, canvas.size[0], canvas.size[1]), fill=(12, 14, 18))
    try:
        font = ImageFont.load_default(size=15)
    except TypeError:  # Pillow < 10 has no size argument
        font = ImageFont.load_default()
    for index, line in enumerate(lines):
        draw.text((10, canvas.size[1] - band + 6 + index * 15), line, fill=(235, 235, 235), font=font)
    return canvas


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    scratch = scratch_data_dir(PROJECT_ROOT)
    os.environ["ARENA_DATA_DIR"] = str(scratch)
    if args.seed is not None:
        os.environ["RANDOM_SEED"] = str(args.seed)
    frames_dir = scratch / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # The screenshot name is resolved against the working directory the window saw when it was created,
    # so the move has to happen before Ursina() exists, and the prefix has to stay relative. Absolute
    # prefixes silently write nothing at all.
    os.chdir(frames_dir)
    configure_panda()
    from panda3d.core import NodePath

    from main import ArenaApp
    from marl_arena.config import CONFIG
    from ursina import Ursina, time

    app = Ursina(size=(args.width, int(args.width * 9 / 16)), fullscreen=False, dev_mode=False)
    arena = ArenaApp()
    # The legend panel, the per-agent overlay and the restart button are screen-space entities, and the
    # legend sits on top of a third of the arena. Everything they say is repeated in the burn-in caption
    # below, so the 2D layers are hidden for the clip.
    # The screen layer is a separate scene root ("ui" under "ui_render"), not a child of render, so it is
    # reached through one of the entities known to live there.
    screen_layer = NodePath(arena.overlay.node()).get_parent()
    screen_layer.hide()

    # The renderer installs an EditorCamera, which reparents the scene camera to itself, so writing
    # camera.position from here sets an offset against a node that also moves and the view drifts. Widen
    # the lens instead: same transform, more of the arena in frame.
    lens = app.camNode.get_lens()
    lens.set_fov(args.fov)
    app.camNode.set_lens(lens)
    # One step before any capture: a screenshot taken before the window has rendered a single frame
    # queues nothing at all and no file appears.
    app.taskMgr.step()
    # Replace the window-loop end-of-match handler: it records metrics and schedules an auto-restart,
    # neither of which belongs in a single deterministic clip.
    arena.finished = False
    result_box: dict[str, object] = {}

    def stop_at_match_end() -> None:
        if result_box:
            return
        result_box["result"] = arena.simulation.finish_match()
        arena.finished = True
        screen_layer.show()
        Text(
            text=f"Winner: {result_box['result'].winner_team}",
            scale=1.6,
            y=0.34,
            background=True,
        )

    arena._handle_match_end = stop_at_match_end  # type: ignore[method-assign]

    paradigms = {spec.team_name: spec.paradigm for spec in arena.simulation.match_variant.team_spawns}
    seed = int(os.environ.get("RANDOM_SEED", CONFIG.random_seed))
    label = "random init" if args.untrained else "promoted replicate"
    frames: list[Image.Image] = []
    timeline: list[dict[str, object]] = []

    for index in range(args.frames):
        if arena.finished:
            break
        # One captured frame = one fixed training step, so the clip advances at exactly the rate the
        # optimiser sees, not at whatever rate this machine manages to render.
        time.dt = CONFIG.sim_step_dt
        arena.update()
        name = f"f{index:04d}.png"
        app.screenshot(namePrefix=name, defaultFilename=False)
        # The write is queued as a task, so one more step is needed before the file is on disk.
        app.taskMgr.step()
        path = frames_dir / name
        if not path.exists():
            raise SystemExit(f"screenshot did not land: {path}")
        image = Image.open(path)
        image.load()  # PIL keeps the file handle open until the pixels are read, which blocks the delete
        frames.append(image)
        path.unlink()
        alive = alive_counts(arena.simulation)
        timeline.append(
            {
                "frame": index,
                "match_time": round(arena.simulation.match_time, 3),
                "alive": alive,
                "projectiles": len(arena.simulation.projectiles),
                "shots": int(sum(m.shots_hit + m.shots_missed for m in arena.simulation.cumulative_metrics.values())),
            }
        )

    if not frames:
        raise SystemExit("no frames captured")

    captions = [
        f"{experiment_label()} | "
        f"{label} | seed {seed} | slot rotation {arena.simulation.paradigm_rotation}",
        " | ".join(f"{team} = {paradigm}" for team, paradigm in sorted(paradigms.items()))
        + f" | arena {arena.simulation.match_variant.arena_size:.1f} m",
    ]
    scaled = [f.resize((args.width, int(args.width * 9 / 16)), Image.LANCZOS) for f in frames]
    annotated = [
        draw_caption(
            image,
            captions
            + [
                f"t = {entry['match_time']:6.2f} s   alive "
                + " / ".join(str(entry["alive"].get(t, 0)) for t in sorted(paradigms))
                + f"   shots {entry['shots']:4d}"
            ],
        )
        for image, entry in zip(scaled, timeline)
    ]

    out_path = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 / args.fps)
    annotated[0].save(
        out_path,
        save_all=True,
        append_images=annotated[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )

    finished = result_box.get("result")
    summary = {
        "clip": _display(out_path),
        "frames": len(annotated),
        "fps": args.fps,
        "sim_step_dt": CONFIG.sim_step_dt,
        "seed": seed,
        "paradigm_rotation": arena.simulation.paradigm_rotation,
        "paradigms": paradigms,
        "trained": not args.untrained,
        "match_ended_in_clip": bool(result_box),
        "winner": getattr(finished, "winner_team", None),
        "clip_duration_sim_seconds": round(arena.simulation.match_time, 3),
        "final_alive": alive_counts(arena.simulation),
        "total_shots": int(sum(m.shots_hit + m.shots_missed for m in arena.simulation.cumulative_metrics.values())),
    }
    # Written for every render, not only when asked: the sidecar is what makes the caption checkable, and the
    # documented regenerate command names only --out, so an opt-in sidecar silently decays into a clip whose
    # recorded evidence no longer matches the clip.
    sidecar = args.sidecar if args.sidecar is not None else out_path.with_suffix(".json")
    sidecar = sidecar if sidecar.is_absolute() else PROJECT_ROOT / sidecar
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps({"clip": summary, "timeline": timeline}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")
    shutil.rmtree(scratch.parent, ignore_errors=True)


if __name__ == "__main__":
    main()
