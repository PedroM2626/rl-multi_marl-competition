#!/usr/bin/env bash
# docs/11 open item 16, as one resumable queue: the v3 checkpoints on every opponent scale, then the two
# single-key arms that could explain v3's halved between-seed spread.
#
#   G  v3 round-robin       the six v3 arms on a common pool, the headline's ten block pairings x five harness
#                           seeds (30,000 matches), so v3 gets the instrument § 8.10 gave the study
#   H  v3 random pool       the same ten blocks against two uniform-random teams, against the slot-matched
#                           null of the *symmetric* arena v3 was trained in (results/baseline_symmetric/)
#   I  v3 scripted pool     the same ten blocks against two scripted engage-and-fire teams
#   J  spawn_cycle          the JAX study with SPAWN_CYCLE_PER_MATCH=true and nothing else, seeds 1-10
#   K  symmetric            the JAX study with ARENA_LAYOUT=symmetric and nothing else, seeds 1-10
#
# J and K are paired against results/study_jax (the same runtime, every other key at its default), which is
# what makes each a single-key ablation; v3 itself is the bundle. G-I replay with the keys that shape a v3
# network or its arena exported and the two training-only remedies off: the harness assigns slots and
# policies itself, and refuses to run with spawn cycling or the past-self pool on.
#
# Every phase skips itself when its output is complete. Phases run one after another, never overlapped, so
# each one's wall clock is its own.
#
# Usage:  bash scripts/run_item16_campaign.sh            (from the repository root)
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_runtime.sh"
TMP="${TEMP:-/tmp}"
JOBS="${ITEM16_JOBS:-4}"
POOL_PAR="${ITEM16_POOL_PARALLEL:-5}"
SEEDS=1,2,3,4,5,6,7,8,9,10
BLOCKS="1:7,10:5,2:1,3:3,4:4,5:6,6:8,7:9,8:10,9:2"
V3_PLAY=(RL_PARAM_TARGET=-1 COMM_ROUNDS=2 COMM_MASK_DEAD=true VD_LOCAL_CRITIC=true OBS_OBSTACLE_RAYS=8
         ARENA_LAYOUT=symmetric)

stamp() { echo "[$(date -Is)] $*"; }

# --- G ---------------------------------------------------------------------------------------------------
OUTDIR="results/cross_arena_v3_blocks"
if [ -s "$ROOT/results/cross_arena/v3_blocks.json" ]; then
  stamp "G: v3 round-robin already played"
else
  stamp "G: v3 round-robin, ten blocks x five harness seeds, $POOL_PAR at a time"
  running=0
  for pair in ${BLOCKS//,/ }; do
    S=${pair%%:*}; M=${pair##*:}
    for H in 1 2 3 4 5; do
      dir="v3__e1s${S}_e2s${M}_h${H}"
      [ -f "$ROOT/$OUTDIR/$dir/matches.csv" ] && continue
      env "${V3_PLAY[@]}" "$PY" "$ROOT/scripts/run_cross_arena.py" --campaign v3 --exp1-replicate "$S" \
        --exp2-replicate "$M" --seed "$H" --out "$OUTDIR/$dir" > "$TMP/item16_G_${S}_${M}_${H}.log" 2>&1 &
      running=$((running+1))
      if [ "$running" -ge "$POOL_PAR" ]; then wait -n; running=$((running-1)); fi
    done
  done
  wait
  for dir in "$ROOT"/$OUTDIR/v3__*; do
    [ -f "$dir/matches.csv" ] || { stamp "G: $dir has no matches.csv" >&2; exit 1; }
  done
  merged=$("$PY" scripts/combine_cross_arena.py "$ROOT"/$OUTDIR/* --out results/cross_arena/v3_blocks.csv)
  echo "$merged"
  case "$merged" in
    *"merged 10 training blocks x 5 harness seeds = 50 runs"*) ;;
    *) stamp "G: the merge did not report ten blocks over five seeds" >&2; exit 1 ;;
  esac
  "$PY" scripts/report_cross_arena.py --in results/cross_arena/v3_blocks.csv \
    --json-out results/cross_arena/v3_blocks.json > "$TMP/item16_G_report.txt" 2>&1 \
    || { tail "$TMP/item16_G_report.txt"; exit 1; }
fi

# --- H, I ------------------------------------------------------------------------------------------------
absolute() {  # <phase> <opponent> <out> [extra args...]
  local phase=$1 opponent=$2 out=$3; shift 3
  if [ -s "$ROOT/$out/matches.csv" ]; then stamp "$phase: $out already played"; return 0; fi
  stamp "$phase: v3's ten blocks against $opponent opponents"
  ( cd "$ROOT/ctde_arena" && env "${V3_PLAY[@]}" "$PY" "$ROOT/scripts/run_absolute_eval.py" --campaign v3 \
      --opponent "$opponent" --blocks "$BLOCKS" --matches 25 --seeds 1,2,3 --out "$out" "$@" \
      --compare-pool results/cross_arena/v3_blocks.json ) > "$TMP/item16_$phase.log" 2>&1 \
    || { tail "$TMP/item16_$phase.log"; exit 1; }
}
# The slot-matched null is a uniform-random team's share, so it belongs to the random scale only; the scripted
# scale carries none, as scripted_ten does not.
absolute H random results/absolute_v3 --null-json results/baseline_symmetric/random_baseline_uniform.json
absolute I scripted results/scripted_v3

# --- J, K ------------------------------------------------------------------------------------------------
missing_experiments() {  # <out dir relative to each tree>
  local exp1="" exp2=""
  for s in ${SEEDS//,/ }; do
    [ -f "$ROOT/$1/exp1_seed$s/evaluation/run_summary.json" ] || exp1=exp1
    [ -f "$ROOT/ctde_arena/$1/exp2_seed$s/evaluation/run_summary.json" ] || exp2=exp2
  done
  local both="$exp1,$exp2"; both="${both#,}"; echo "${both%,}"
}

arm() {  # <name> <key=value>
  local name=$1 override=$2 out="data/runs_jax_$1" results="results/ablation_jax_$1"
  local missing; missing="$(missing_experiments "$out")"
  if [ -z "$missing" ] && [ -s "$ROOT/$results/per_seed_metrics.csv" ]; then
    stamp "$name: already complete, skipping"; return 0
  fi
  local scope=(--experiments "$missing")
  [ -z "$missing" ] && scope=(--collect-only)
  stamp "$name: training ${missing:-nothing, re-collecting} with $override at seeds $SEEDS, $JOBS at a time"
  env "$override" "$PY" scripts/run_study.py --steps 500000 --seeds "$SEEDS" --jobs "$JOBS" --eval-matches 150 \
    "${scope[@]}" --out "$out" --results-dir "$results" > "$TMP/item16_$name.log" 2>&1
  tail -4 "$TMP/item16_$name.log"
  grep -q "failures: none" "$TMP/item16_$name.log" || { stamp "$name did not finish cleanly" >&2; exit 1; }
  "$PY" scripts/report_study.py --results-dir "$results" --label "$override on the JAX runtime" \
    --compare-to results/study_jax --json-out "$results/coupled_tests.json" > "$results/report.md"
}
arm spawn_cycle SPAWN_CYCLE_PER_MATCH=true
arm symmetric ARENA_LAYOUT=symmetric

"$PY" scripts/version_campaign_checkpoints.py --arms spawn_cycle,symmetric > "$TMP/item16_versioning.log" 2>&1 \
  || { tail "$TMP/item16_versioning.log"; exit 1; }
tail -3 "$TMP/item16_versioning.log"
stamp "item 16 campaign complete"
echo ITEM16_DONE > "$TMP/item16_campaign.flag"
