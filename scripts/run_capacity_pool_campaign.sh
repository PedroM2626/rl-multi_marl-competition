#!/usr/bin/env bash
# Play the two capacity arms on the common opponent pool, which is the one confound that instrument has
# never varied.
#
# docs/10 § 10.1 says the study cannot rule out that CTDE-Comm lost because its critic is smaller, and the
# ablation campaign answered that under its *own* protocol (three arms co-adapting inside one run). The
# round-robin is the instrument that balances opponents across architectures, and the capacity axes have
# never been played on it: `narrow_64` and `wide_256` are versioned under `data/campaign_replicates/` with
# their hidden width read out of the weights, so this is evaluation time on checkpoints that already exist.
#
# The capacity guard in `run_cross_arena.py` is what makes the run safe to script: it refuses to play a set
# whose width the running configuration cannot hold, so the export below is derived from the manifest rather
# than typed. Three blocks per arm because that is how many seeds each arm trained -- the same limit that
# keeps § 8.13's no_dr pool at three, and stated as a limit rather than smoothed over.
#
# Chained behind the v2 extension: overlapping campaigns do not change an answer, they make every
# wall-clock figure in docs/09 § 9.5 wrong for both.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
TMP="${TEMP:-/tmp}"

PAIRS=("1 1" "2 2" "3 3")
OUTDIR="results/cross_arena_capacity"

while [ ! -f /tmp/v2_ten.flag ]; do sleep 180; done
echo "v2 extension complete; starting the capacity-arm pool at $(date -Is)"

for arm in narrow_64 wide_256; do
  width=$("$PY" - "$arm" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path("data/campaign_replicates/MANIFEST.json").read_text(encoding="utf-8"))
arm = manifest["arms"].get(sys.argv[1])
if arm is None:
    raise SystemExit(f"no arm {sys.argv[1]!r} in the manifest")
widths = {entry["hidden_dim"] for entry in arm["sets"].values()}
if len(widths) != 1:
    raise SystemExit(f"arm {sys.argv[1]} is not one width: {sorted(widths, key=lambda w: (w is None, w))}")
print(next(iter(widths)))
PY
)
  echo "=== $arm at RL_HIDDEN_DIM=$width, $(date -Is)"
  export RL_HIDDEN_DIM="$width"
  # Pre-flight: one match per triple would be 20 runs; one run at one match per triple is enough to prove
  # the guard passes and the arm's weights load at this width before 15 runs are launched.
  "$PY" "$ROOT/scripts/run_cross_arena.py" --campaign "$arm" --exp1-replicate 1 --exp2-replicate 1 \
    --seed 99 --matches-per-triple 1 --out "$TMP/preflight_cap_${arm}" > "$TMP/cap_${arm}_preflight.log" 2>&1
  if [ ! -s "$TMP/preflight_cap_${arm}/matches.csv" ]; then
    echo "$arm pre-flight played no matches; see $TMP/cap_${arm}_preflight.log" >&2
    tail -5 "$TMP/cap_${arm}_preflight.log" >&2
    exit 1
  fi
  for pair in "${PAIRS[@]}"; do
    set -- $pair; S=$1; M=$2
    for H in 1 2 3 4 5; do
      dir="${arm}__e1s${S}_e2s${M}_h${H}"
      if [ -f "$ROOT/$OUTDIR/$dir/matches.csv" ]; then
        echo "  $dir already played, skipping (delete it to re-run)"
        continue
      fi
      ( "$PY" "$ROOT/scripts/run_cross_arena.py" --campaign "$arm" \
          --exp1-replicate "$S" --exp2-replicate "$M" --seed "$H" \
          --out "$OUTDIR/$dir" > "$TMP/cap_${arm}_${S}_${M}_${H}.log" 2>&1 ) &
    done
    wait
    echo "  $arm block exp1_seed${S}+exp2_seed${M} done at $(date -Is)"
  done
  unset RL_HIDDEN_DIM

  merged=$("$PY" scripts/combine_cross_arena.py "$ROOT"/$OUTDIR/${arm}__* \
    --out "results/cross_arena/${arm}_blocks.csv")
  echo "$merged"
  case "$merged" in
    *"merged 3 training blocks x 5 harness seeds = 15 runs"*) ;;
    *) echo "$arm did not merge as three blocks over five streams; see the line above" >&2; exit 1 ;;
  esac
  "$PY" scripts/report_cross_arena.py --in "results/cross_arena/${arm}_blocks.csv" \
    --json-out "results/cross_arena/${arm}_blocks.json" > "$TMP/cap_${arm}_report.txt" 2>&1
done
echo CAP_POOL_DONE > /tmp/cap_pool.flag
