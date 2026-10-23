"""Fail unless this interpreter is the runtime the measurements describe; print its provenance.

    python scripts/check_runtime.py

Campaign scripts call this through scripts/_runtime.sh before launching anything, so a stale
.venv-check (a different JAX build without the Rust extension) cannot silently become the runtime
behind a new table.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marl_arena.runtime import require_measured_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="experiment tree to read requirements.txt from")
    parser.add_argument("--json", action="store_true", help="also print the provenance as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        info = require_measured_runtime(args.root)
    except Exception as exc:  # ImportError, RuntimeError, missing pin: all mean "do not measure here"
        print(f"runtime check failed: {exc}", file=sys.stderr)
        return 1
    print(
        "runtime: python {python} | jax {jax} | numpy {numpy} | rust_extension {rust_extension}".format(**info)
    )
    if args.json:
        print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
