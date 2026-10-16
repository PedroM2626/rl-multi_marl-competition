#!/usr/bin/env bash
# Re-measure the round-robin under arena-v2 rules, then over the no_dr ablation checkpoints.
#
# Two questions the study cannot answer from its own tables, and both are evaluation-only -- no training:
#
#   1. Does the architecture ordering depend on the *buzzer rule*? Section 8.12 answered that by retraining
#      20 replicates under v2, which changes the objective and the policies at the same time. Here the 36
#      versioned checkpoints are fixed and only the rule moves, so a reversal can be attributed to the rule
#      alone. A-34 made the knobs configuration, and run_cross_arena.py snapshots them into every summary.
#   2. Does it depend on the *training environment*? The no_dr arm is the one key that moved held-out win
#      rates by up to 0.253 and inverted both experiments' rankings (section 8.13). Its checkpoints are now
#      versioned, so they can be played against the committed pool for the first time.
#
# Cost basis, measured rather than assumed: 40 v2 matches in 129 s single-process (3.2 s/match), because a
# match that no longer has a buzzer winner runs to the 90 s cap far more often -- the committed round-robin
# is faster per match. 18,000 matches over five concurrent processes is on the order of five hours; the
# no_dr campaign is half that.
#
# Nothing here touches a committed file: the run directories and the pooled tables are new paths, and the
# script refuses to overwrite a run that already exists.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-check/Scripts/python.exe"
[ -x "$PY" ] || PY="$ROOT/.venv/Scripts/python.exe"
TMP="${TEMP:-/tmp}"

BLOCKS=("1 7" "10 5" "9 2" "2 1" "3 3" "4 4")   # the six versioned study blocks
NO_DR_BLOCKS=("1 1" "2 2" "3 3")                # three blocks, one per no_dr seed pair

# Pre-flight, about a minute: play 20 matches into a scratch directory and insist they exist before the
# five-hour campaign is launched. The six-block campaign once "completed" having played nothing at all,
# because `cd` plus a relative script path resolves against the wrong drive root when the interpreter is a
# native Windows launcher; every invocation below therefore passes an absolute script path, and this check
# is what turns that class of failure into a message instead of an empty table.
"$PY" "$ROOT/scripts/run_cross_arena.py" --exp1-replicate 1 --exp2-replicate 7 --seed 99 \
  --matches-per-triple 1 --out "$TMP/preflight_v2_pool" > "$TMP/v2rr_preflight.log" 2>&1
if [ ! -s "$TMP/preflight_v2_pool/matches.csv" ]; then
  echo "pre-flight played no matches; see $TMP/v2rr_preflight.log" >&2
  exit 1
fi
echo "pre-flight ok: $(( $(wc -l < "$TMP/preflight_v2_pool/matches.csv") - 1 )) team-match rows played, starting the campaign at $(date -Is)"

play_wave() {  # play one set of five harness seeds in parallel and wait for all of them
  local prefix="$1"; shift
  local s="$1" m="$2" outdir="$3"; shift 3
  local h
  for h in 1 2 3 4 5; do
    local dir="${prefix}__e1s${s}_e2s${m}_h${h}"
    if [ -f "$ROOT/$outdir/$dir/matches.csv" ]; then
      echo "  $dir already played, skipping (delete it to re-run)"
      continue
    fi
    ( "$PY" "$ROOT/scripts/run_cross_arena.py" "$@" \
        --exp1-replicate "$s" --exp2-replicate "$m" --seed "$h" \
        --out "$outdir/$dir" > "$TMP/v2rr_${prefix}_${s}_${m}_${h}.log" 2>&1 ) &
  done
  wait
  echo "  wave $outdir/$prefix__e1s${s}_e2s${m} done at $(date -Is)"
}

echo "=== v2-rules round-robin, six blocks x five harness seeds, at $(date -Is)"
# The three keys arena v2 changed, exported rather than written to disk: dotenv loads .env with
# override=False, so a real environment variable wins and the committed .env stays the study's instrument.
export REWARD_ALIVE_BONUS=0.0015 REWARD_ALIVE_DECAY_HALF_LIFE_S=30 TERMINATION_CAP_AWARDS_WIN=false
for pair in "${BLOCKS[@]}"; do
  set -- $pair
  play_wave v2 "$1" "$2" results/cross_arena_v2_blocks
done
unset REWARD_ALIVE_BONUS REWARD_ALIVE_DECAY_HALF_LIFE_S TERMINATION_CAP_AWARDS_WIN

"$PY" scripts/combine_cross_arena.py results/cross_arena_v2_blocks/* \
  --out results/cross_arena/v2_blocks.csv
"$PY" scripts/report_cross_arena.py --in results/cross_arena/v2_blocks.csv \
  --json-out results/cross_arena/v2_blocks.json > "$TMP/v2rr_report.txt" 2>&1
echo "v2 pool report in $TMP/v2rr_report.txt"
echo V2_POOL_DONE > /tmp/v2_pool.flag

echo "=== no_dr-arm round-robin, three blocks x five harness seeds, at $(date -Is)"
for pair in "${NO_DR_BLOCKS[@]}"; do
  set -- $pair
  play_wave no_dr "$1" "$2" results/cross_arena_campaign --campaign no_dr
done

"$PY" scripts/combine_cross_arena.py results/cross_arena_campaign/* \
  --out results/cross_arena/no_dr_blocks.csv
"$PY" scripts/report_cross_arena.py --in results/cross_arena/no_dr_blocks.csv \
  --json-out results/cross_arena/no_dr_blocks.json > "$TMP/no_dr_report.txt" 2>&1
echo NO_DR_POOL_DONE > /tmp/no_dr_pool.flag
