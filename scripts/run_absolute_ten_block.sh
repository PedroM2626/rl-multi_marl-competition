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
echo "  wrote $rows rows"
# One row per match, not per team-match: the absolute instrument plays one trained team against two random
# controllers, so a match contributes a single row naming the trained paradigm. The shape the campaign must
# have is therefore 10 blocks x 3 streams x 6 architectures x 25 matches = 4,500 rows, 750 per architecture.
"$PY" - "$ROOT/results/absolute_ten/matches.csv" "$ROOT/results/absolute_ten/summary.json" <<'PY'
import csv, json, sys
from collections import Counter
from pathlib import Path
csv_path, summary_path = Path(sys.argv[1]), Path(sys.argv[2])
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
per_architecture = Counter(row["paradigm"] for row in rows)
blocks = {row["training_block"] for row in rows}
streams = {row["harness_seed"] for row in rows}
summary = json.loads(summary_path.read_text(encoding="utf-8"))
problems = []
if len(rows) != 4_500:
    problems.append(f"{len(rows)} rows, expected 4,500 (10 blocks x 3 streams x 6 arms x 25 matches)")
if set(per_architecture.values()) != {750}:
    problems.append(f"unequal per-architecture counts: {dict(per_architecture)}")
if len(blocks) != 10 or len(streams) != 3:
    problems.append(f"{len(blocks)} blocks x {len(streams)} streams, expected 10 x 3")
if summary.get("blocks") and len(summary["blocks"]) != 10:
    problems.append("the summary lists fewer than ten blocks, so this is not the ten-block pool")
if summary.get("inputs", {}).get("compare_pool") != "results/cross_arena/blocks10.json":
    problems.append(f"the agreement was computed against {summary.get('inputs', {}).get('compare_pool')!r}, "
                    "not the ten-block peer pool")
if problems:
    raise SystemExit("; ".join(problems) + " -- refusing to call this the ten-block absolute pool")
print(f"  shape ok: {len(rows)} rows, {len(blocks)} blocks x {len(streams)} streams, "
      f"750 matches per architecture, peer pool blocks10.json")
PY
if [ "$?" -ne 0 ]; then exit 1; fi
echo ABS_TEN_DONE > /tmp/abs_ten.flag
