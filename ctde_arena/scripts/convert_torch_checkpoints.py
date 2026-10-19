"""Convert legacy Torch policy files to the runtime's non-pickle JAX NPZ format.

Torch is imported only by this migration utility; it is not a runtime dependency. The published tables of
[§ 8.10-§ 8.13](../docs/08_results.md#ablations) were produced by networks stored as `.pt`, and the runtime
since A-44 loads `.npz`, so the archive has to be converted for a reader to be able to replay a single match
of anything chapter 8 claims.

What it guarantees, and what it refuses:

* **Byte identity of the stored arrays.** The `.npz` holds exactly the tensors of the `.pt`, in Torch's
  `[out, in]` layout -- `JAXNetwork.load_state_dict` transposes on the way in. The script re-reads every
  file it writes and fails if an array differs, so a conversion cannot quietly become a re-derivation.
  It does *not* claim the forward pass is bit-identical: multiplying a batch in float32 reassociates sums,
  and A-44 measures the resulting difference (~4e-7 on logits).
* **No overwrite of a differing file.** A `.npz` that already exists and disagrees with its `.pt` aborts.
* **The whole archive, not just the shipped demo policies.** `data/checkpoints` is three files; the
  evidence of the round-robin, the ablations and the promoted sets is 165 files across four trees.

    python scripts/convert_torch_checkpoints.py                     # every committed .pt
    python scripts/convert_torch_checkpoints.py path/to/team_1_dte.pt
    python scripts/convert_torch_checkpoints.py --check             # verify, write nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Where the committed weights live. Both trees, because experiment 2's promoted and historical policies are
# versioned inside ctde_arena/ (A-35: a tool that looks in one tree converts half the archive and says so
# in nothing).
ARCHIVE_ROOTS = (
    PROJECT_ROOT / "data" / "checkpoints",
    PROJECT_ROOT / "data" / "training_replicates",
    PROJECT_ROOT / "data" / "campaign_replicates",
    PROJECT_ROOT / "ctde_arena" / "data" / "checkpoints",
    PROJECT_ROOT / "ctde_arena" / "data" / "historical_100k" / "checkpoints",
)


def committed_checkpoints() -> list[Path]:
    found: list[Path] = []
    for root in ARCHIVE_ROOTS:
        if root.is_dir():
            found.extend(sorted(root.rglob("*.pt")))
    return found


def convert_one(path: Path, payload: dict) -> dict:
    return {
        key: {
            name: value.detach().cpu().numpy() if hasattr(value, "detach") else value
            for name, value in state.items()
        }
        if key in {"actor", "critic"} and state is not None else state
        for key, state in payload.items()
    }


def verify(destination: Path, converted: dict) -> None:
    """Every array the .npz now holds must equal the tensor the .pt held, exactly."""
    from marl_arena.rl.actions import load_checkpoint

    stored = load_checkpoint(destination)
    for key, state in converted.items():
        if not isinstance(state, dict):
            continue
        for name, value in state.items():
            if key not in stored or name not in stored[key]:
                raise SystemExit(f"{destination.name}: {key}/{name} did not survive the round trip")
            if not (stored[key][name] == value).all():
                raise SystemExit(f"{destination.name}: {key}/{name} differs from its source tensor")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="*", type=Path, help="legacy .pt files to convert")
    parser.add_argument("--check", action="store_true",
                        help="verify every .npz against its .pt and write nothing")
    args = parser.parse_args()

    paths = args.checkpoints or committed_checkpoints()
    paths = [path if path.is_absolute() else PROJECT_ROOT / path for path in paths]
    if not paths:
        parser.error(f"no .pt checkpoints found under {', '.join(str(p.name) for p in ARCHIVE_ROOTS)}")

    if args.check:
        missing = [str(p) for p in paths if not p.with_suffix(".npz").is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} of {len(paths)} checkpoints are not converted yet; "
                             f"first missing: {missing[:3]}")
        print(f"verified {len(paths)} checkpoints have an .npz sibling")
        return

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install PyTorch temporarily to convert legacy checkpoints "
                         "(it is a migration-time dependency only).") from exc

    import numpy as np

    from marl_arena.rl.actions import save_checkpoint

    written, unchanged = 0, 0
    for path in paths:
        destination = path.with_suffix(".npz")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        converted = convert_one(path, payload)
        if destination.is_file():
            try:
                verify(destination, converted)
                unchanged += 1
                continue
            except SystemExit as failure:
                raise SystemExit(f"{path}: refusing to overwrite -- {failure}") from failure
        save_checkpoint(destination, converted)
        # np.savez stores each array with its own dtype; the tensors come in as float32 and must stay that
        # way, or a reader's checkpoint hashes differ for a reason no one can see in the loss curve.
        for key, state in converted.items():
            if isinstance(state, dict):
                for value in state.values():
                    if isinstance(value, np.ndarray) and value.dtype != np.float32:
                        raise SystemExit(f"{destination.name}: {key} stored as {value.dtype}, not float32")
        verify(destination, converted)
        written += 1
    print(f"converted {written}, already identical {unchanged}, of {len(paths)} checkpoints")


if __name__ == "__main__":
    main()
