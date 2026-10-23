#!/usr/bin/env bash
# Resolve and verify the one interpreter a campaign may measure on.
#
# Every campaign script sources this file after setting ROOT. It prefers the repository runtime
# (.venv), which holds the JAX build pinned in requirements.txt and the Rust extension built from
# native/arena_core. It deliberately does not fall back to .venv-check: that environment was created
# to verify the CI install and has held a different JAX build without the extension (A-45, A-48).
#
# Usage:
#   ROOT="$(cd "$(dirname "$0")/.." && pwd)"
#   source "$ROOT/scripts/_runtime.sh"
#   "$PY" scripts/...
set -u

if [ -z "${ROOT:-}" ]; then
  echo "_runtime.sh: set ROOT before sourcing this file" >&2
  exit 1
fi

PY=""
for candidate in "$ROOT/.venv/Scripts/python.exe" "$ROOT/.venv/bin/python"; do
  if [ -x "$candidate" ]; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "_runtime.sh: the repository runtime ($ROOT/.venv) is not installed;" >&2
  echo "            a measurement must not run on any other interpreter" >&2
  exit 1
fi

if ! "$PY" "$ROOT/scripts/check_runtime.py" --root "$ROOT"; then
  echo "_runtime.sh: refusing to run a measurement on this interpreter" >&2
  exit 1
fi

export PY
