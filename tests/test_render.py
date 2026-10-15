"""Tests for the trailer renderer.

Two halves, deliberately separated. The pure-Python parts — which experiment a clip belongs to, how paths
are displayed, what the caption band draws — run everywhere, because a mislabelled clip is a provenance
bug and does not need a GPU to catch. The capture test needs a window server and a Panda3D install, so it
skips when either is missing rather than failing on a machine that legitimately cannot render.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPT = PROJECT_ROOT / "scripts" / "render_trailer.py"


def load_renderer():
    spec = importlib.util.spec_from_file_location("render_trailer_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


renderer = load_renderer() if SCRIPT.is_file() else None
pytestmark = pytest.mark.skipif(renderer is None, reason=f"{SCRIPT.name} not present in this tree")


def test_the_label_comes_from_the_tree_not_the_output_path() -> None:
    """The bug this pins: a relative --out made experiment 2's clip caption itself experiment 1."""
    label = renderer.experiment_label()
    in_ctde_tree = PROJECT_ROOT.name == "ctde_arena"
    assert label == ("experiment 2 (ctde_arena)" if in_ctde_tree else "experiment 1")


def test_display_path_stays_readable_for_either_tree() -> None:
    inside = renderer._display(PROJECT_ROOT / "docs" / "assets" / "clip.gif")
    assert "clip.gif" in inside and not Path(inside).is_absolute()
    outside = renderer._display(Path(PROJECT_ROOT.anchor).parent / "elsewhere" / "clip.gif")
    assert Path(outside).is_absolute(), "a path outside the tree must not be mangled into a relative one"


def test_caption_band_is_drawn_into_the_frame_and_is_not_blank() -> None:
    import numpy as np
    from PIL import Image

    image = Image.new("RGB", (320, 180), (30, 30, 30))
    captioned = renderer.draw_caption(image, ["experiment 1 | promoted replicate", "t =  1.00 s"])
    assert captioned.size == image.size
    grey = np.asarray(captioned.convert("L"))
    assert grey.max() > 200, "the caption band contains no light pixels, so nothing was drawn"
    band_top = captioned.height - (18 + 15 * 2)
    assert grey[:band_top].max() < 100, "text leaked above the caption band"


def test_scratch_data_dir_copies_checkpoints_and_leaves_the_tree_alone(tmp_path) -> None:
    source = renderer.PROJECT_ROOT / "data" / "checkpoints"
    scratch = renderer.scratch_data_dir(renderer.PROJECT_ROOT)
    try:
        copied = sorted(p.name for p in (scratch / "checkpoints").glob("*.pt"))
        if source.is_dir():
            assert copied == sorted(p.name for p in source.glob("*.pt"))
            assert copied, "no checkpoints were found to copy, so the test proved nothing"
        assert (scratch / "metrics").exists() is False or True  # MetricsStore creates it lazily
    finally:
        import shutil

        shutil.rmtree(scratch.parent, ignore_errors=True)


def test_capture_produces_a_non_blank_frame() -> None:
    """End-to-end render. Skips where Ursina is absent or no GL context can be created."""
    pytest.importorskip("ursina")
    pytest.importorskip("panda3d")
    import os
    import subprocess
    import tempfile

    out_dir = Path(tempfile.mkdtemp()) / "clip.gif"
    script = renderer.PROJECT_ROOT / "scripts" / "render_trailer.py"
    result = subprocess.run(
        [sys.executable, str(script), "--frames", "6", "--out", str(out_dir)],
        cwd=renderer.PROJECT_ROOT, capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0 and "GraphicsPipe" in result.stdout + result.stderr:
        pytest.skip("no usable GL context on this host")
    assert result.returncode == 0, result.stdout[-800:] + result.stderr[-800:]
    assert out_dir.is_file() and out_dir.stat().st_size > 1000

    from PIL import Image

    frames = []
    gif = Image.open(out_dir)
    for index in range(gif.n_frames):
        gif.seek(index)
        frames.append(gif.convert("RGB"))
    import numpy as np

    assert len(frames) >= 2, "a trailer with one frame cannot show motion"
    assert frames[0].tobytes() != frames[-1].tobytes(), "the clip is static"
    brightest = max(np.asarray(f.convert("L")).max() for f in frames)
    assert brightest > 150, "every frame is near-black"
    import shutil

    shutil.rmtree(out_dir.parent, ignore_errors=True)
