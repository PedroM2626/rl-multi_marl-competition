#!/usr/bin/env bash
# Extend the absolute-strength instrument from six training blocks to ten, so both scales of this
# repository measure the same population of trained networks.
#
# Section 8.11's headline agreement (+0.77) is deliberately six-against-six, because that is the replication
# count the committed artefact was produced at. The round-robin has since gone to ten blocks -- every
# replicate either experiment trained -- and the recomputation against that pool (+0.66) is a stress test
# rather than the headline, precisely because the two sides no longer hold the same population. This closes
# the gap: ten blocks on both sides, at the same cell size the six-set artefact used (25 matches per
# architecture per block per harness seed, three streams).
#
# Chained behind the capacity-arm pool so each campaign's wall clock stays its own measurement.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="${TEMP:-/tmp}"

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"

BLOCKS="1:7,10:5,2:1,3:3,4:4,5:6,6:8,7:9,8:10,9:2"

while [ ! -f /tmp/cap_pool.flag ]; do sleep 180; done
echo "capacity pool complete; starting the ten-block absolute evaluation at $(date -Is)"

if [ -s "$ROOT/results/absolute_ten/matches.csv" ]; then
  echo "  results/absolute_ten already has matches.csv; delete it to re-play"
else
  # Run from ctde_arena: that tree's package is the superset holding all six architectures, and the script
  # path is absolute because the venv interpreter is a native launcher that resolves relative paths against
  # the wrong root once a subshell has moved.
  ( cd "$ROOT/ctde_arena" && "$PY" "$ROOT/scripts/run_absolute_eval.py" --blocks "$BLOCKS" \
      --matches 25 --seeds 1,2,3 --out results/absolute_ten \
      --null-json results/baseline/random_baseline_uniform.json \
      --compare-pool results/cross_arena/blocks10.json ) > "$TMP/abs_ten.log" 2>&1
fi

if [ ! -s "$ROOT/results/absolute_ten/matches.csv" ] || [ ! -s "$ROOT/results/absolute_ten/summary.json" ]; then
  echo "the ten-block absolute artefact is incomplete; last lines of its log:" >&2
  tail -8 "$TMP/abs_ten.log" >&2
  exit 1
fi
rows=$(( $(wc -l < "$ROOT/results/absolute_ten/matches.csv") - 1 ))
echo "  wrote $rows team-match rows"
if [ "$rows" -ne 27000 ]; then
  # 10 blocks x 3 streams x 6 architectures x 25 matches = 4,500 matches, three rows each.
  echo "expected 27,000 rows (4,500 matches), got $rows -- refusing to call this the ten-block pool" >&2
  exit 1
fi
echo ABS_TEN_DONE > /tmp/abs_ten.flag
