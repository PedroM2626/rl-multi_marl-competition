"""Every committed checkpoint must exist in the format the runtime can actually load.

A-44 replaced PyTorch with JAX, and A-45 found the consequence: the harness loads `.npz`, while the entire
published evidence base -- the promoted policies, the twenty round-robin blocks and the thirty-two ablation
sets -- was stored as `.pt`. `run_cross_arena.py` then failed on its own archive with
`KeyError: '__metadata__ is not a file in the archive'`, which is the sound of § 8.10 to § 8.13 becoming
irreproducible: the tables stayed, the policies behind them no longer opened.

The conversion was verified where it happened: `scripts/convert_torch_checkpoints.py` re-reads each `.npz`
and compares every array against the tensor it came from, so the weights are byte-identical to the archive
(a float32 matmul reassociates sums, so the *forward pass* is not bit-identical -- that difference is
measured and stated in A-45, not hidden here). What this file can check without PyTorch installed, which CI
no longer has, is the structural invariant: the two formats come in pairs, both are tracked, and every
`.npz` holds what a policy file has to hold.
"""

from __future__ import annotations

import importlib.util
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PREFIXES = (
    "data/checkpoints/",
    "data/training_replicates/",
    "data/campaign_replicates/",
    "ctde_arena/data/checkpoints/",
    "ctde_arena/data/historical_100k/checkpoints/",
)


def tracked(prefix: str, suffix: str) -> list[str]:
    listing = subprocess.run(["git", "ls-files", "--", f"{prefix}*{suffix}"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    return sorted(listing)


def archive_files(suffix: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for prefix in ARCHIVE_PREFIXES:
        grouped[prefix].extend(tracked(prefix, suffix))
    return grouped


def test_every_legacy_checkpoint_has_a_loadable_twin() -> None:
    legacy, loadable = archive_files(".pt"), archive_files(".npz")
    for prefix in ARCHIVE_PREFIXES:
        old = {Path(name).stem for name in legacy[prefix]}
        new = {Path(name).stem for name in loadable[prefix]}
        assert old, f"{prefix}: this gate is meant to cover a real archive, found no .pt"
        assert old == new, (f"{prefix}: .pt and .npz must come in pairs; missing .npz for "
                            f"{sorted(old - new)[:3]}, orphan .npz {sorted(new - old)[:3]}")
    total = sum(len(v) for v in legacy.values())
    print(f"{total} committed checkpoints, each present as .pt and .npz")


def test_a_loadable_checkpoint_carries_a_policy_and_its_dtype() -> None:
    """An .npz that loads as nothing, or as float64, is not a policy file a reader can reproduce results with."""
    names = [name for group in archive_files(".npz").values() for name in group]
    assert names
    for name in names:
        with np.load(ROOT / name) as stored:
            keys = list(stored.files)
            tensors = [key for key in keys if "::" in key]
            assert tensors, f"{name}: no parameter arrays"
            assert all(key.startswith(("actor::", "critic::")) for key in tensors), (name, keys)
            assert "__metadata__" in keys, f"{name}: the file does not say which architecture it is"
            import json

            assert "paradigm" in json.loads(str(stored["__metadata__"].item())), name
            for key in tensors:
                assert stored[key].dtype == np.float32, (name, key, stored[key].dtype)
            weights = [key for key in tensors if key.endswith(".0.weight")]
            assert weights, f"{name}: no first layer to read a width out of"
            widths = {stored[key].shape[0] for key in weights}
            assert len(widths) == 1, (name, widths)


def test_the_capacity_guard_reads_the_width_without_pytorch(tmp_path: Path) -> None:
    """The ablation arms are distinguished by width, and the manifest is checked against the loadable file.

    Before A-45 the width came out of a `torch.load`; in CI, which now installs JAX and not Torch, that made
    the test of the capacity guard the one thing that could not run.
    """
    spec = importlib.util.spec_from_file_location("vcc_capacity", ROOT / "scripts" / "version_campaign_checkpoints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = ROOT / "data" / "campaign_replicates" / "MANIFEST.json"
    import json

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    for arm, record in payload["arms"].items():
        declared = int(record["overrides"].get("RL_HIDDEN_DIM", 128))
        for name, entry in record["sets"].items():
            assert entry["hidden_dim"] == declared, (arm, name)
            for file in entry["files"]:
                path = ROOT / "data" / "campaign_replicates" / arm / name / file
                assert path.is_file(), path
                assert path.suffix == ".npz", f"{path}: the manifest must describe the loadable format"
                assert module.actor_shape(path)[0] == declared, (arm, name, file)
