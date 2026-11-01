#!/usr/bin/env bash
# The campaign that docs/10's remaining threats were waiting for, as one resumable queue.
#
#   A  symmetric-arena residual   uniform-random baseline in ARENA_LAYOUT=symmetric: how much slot bias the
#                                 three-fold layout leaves (A-32), against the legacy arena's 0.257/0.270/0.463
#   B  JAX replication            the published study's 20 replicates re-trained on the JAX/Rust runtime with
#                                 every default unchanged -- the run docs/09 and A-45 say no number has had
#   C  v3                         the same 20 replicates with every docs/10 remedy on together (below)
#   D  scripted opponents         each of the study's committed ten blocks against the scripted
#                                 engage-and-fire controller: a non-random, non-learned reference (§ 10.4/13)
#   E  capacity arms at ten seeds narrow_64 and wide_256 re-trained on the JAX runtime at seeds 1-10, paired
#                                 against B (same runtime) rather than mixed with their PyTorch-era seeds 1-3
#   F  random pool for no_dr      the absolute instrument over the no_dr arm's ten versioned blocks, which
#                                 § 8.13 says the random-pool scale had never measured
#
# Every phase skips itself when its output is already complete, so the queue can be relaunched after an
# interruption without retraining anything. Width defaults to four concurrent replicates: that is where
# docs/09 § 9.5 measured each JAX process still training at solo speed.
#
# Usage:  bash scripts/run_v3_campaign.sh            (from the repository root)
#         V3_JOBS=6 bash scripts/run_v3_campaign.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_runtime.sh"
TMP="${TEMP:-/tmp}"
JOBS="${V3_JOBS:-4}"
SEEDS=1,2,3,4,5,6,7,8,9,10
BLOCKS="1:7,10:5,2:1,3:3,4:4,5:6,6:8,7:9,8:10,9:2"

# The v3 profile: every remedy of docs/06 § 6.8 at once. A bundle, deliberately -- the question it answers
# is "does a corrected instrument rank the arms", not which remedy moves what; the single-key ablations of
# § 8.13 are the instrument for the second question.
V3_ENV=(
  RL_GAE_PER_AGENT=true RL_UPDATE_MINIBATCHES=16 RL_TARGET_KL=0.02 RL_PARAM_TARGET=-1
  COMM_ROUNDS=2 COMM_MASK_DEAD=true VD_LOCAL_CRITIC=true OBS_OBSTACLE_RAYS=8
  ARENA_LAYOUT=symmetric SPAWN_CYCLE_PER_MATCH=true
  RL_OPPONENT_POOL_PROB=0.2 RL_OPPONENT_POOL_EVERY_MATCHES=50 RL_OPPONENT_POOL_SIZE=20
)

stamp() { echo "[$(date -Is)] $*"; }

missing_experiments() {  # <out dir relative to each tree>: the experiments with a seed still to train
  local exp1="" exp2=""
  for s in ${SEEDS//,/ }; do
    [ -f "$ROOT/$1/exp1_seed$s/evaluation/run_summary.json" ] || exp1=exp1
    [ -f "$ROOT/ctde_arena/$1/exp2_seed$s/evaluation/run_summary.json" ] || exp2=exp2
  done
  local both="$exp1,$exp2"; both="${both#,}"; echo "${both%,}"
}

study() {  # <name> <out> <results-dir> [env...]
  local name=$1 out=$2 results=$3; shift 3
  local missing; missing="$(missing_experiments "$out")"
  if [ -z "$missing" ] && [ -s "$ROOT/$results/per_seed_metrics.csv" ]; then
    stamp "$name: already complete, skipping"; return 0
  fi
  # Only an experiment with seeds still to train is launched (A-53 left one tree to re-train); collection
  # reads both trees either way, so the aggregate still holds all 20 replicates.
  local scope=(--experiments "$missing")
  [ -z "$missing" ] && scope=(--collect-only)
  stamp "$name: training ${missing:-nothing, re-collecting} at seeds $SEEDS, $JOBS at a time"
  env "$@" "$PY" scripts/run_study.py --steps 500000 --seeds "$SEEDS" --jobs "$JOBS" --eval-matches 150 \
    "${scope[@]}" --out "$out" --results-dir "$results" > "$TMP/v3_$name.log" 2>&1
  tail -4 "$TMP/v3_$name.log"
  grep -q "failures: none" "$TMP/v3_$name.log" || { stamp "$name did not finish cleanly" >&2; exit 1; }
}

# --- A ---------------------------------------------------------------------------------------------------
if [ -s "$ROOT/results/baseline_symmetric/random_baseline_uniform.json" ]; then
  stamp "A: symmetric baseline already measured"
else
  stamp "A: uniform baseline in the symmetric arena"
  ARENA_LAYOUT=symmetric "$PY" scripts/random_baseline.py --matches 900 --policy uniform \
    --out results/baseline_symmetric > "$TMP/v3_A.log" 2>&1 || { tail "$TMP/v3_A.log"; exit 1; }
fi

# --- B, C ----------------------------------------------------------------------------------------------------
study jax data/runs_jax results/study_jax
study v3 data/runs_v3 results/study_v3 "${V3_ENV[@]}"

# --- D ---------------------------------------------------------------------------------------------------
if [ -s "$ROOT/results/scripted_ten/matches.csv" ]; then
  stamp "D: scripted-opponent instrument already played"
else
  stamp "D: the study's ten blocks against scripted opponents"
  ( cd "$ROOT/ctde_arena" && "$PY" "$ROOT/scripts/run_absolute_eval.py" --opponent scripted \
      --blocks "$BLOCKS" --matches 25 --seeds 1,2,3 --out results/scripted_ten \
      --compare-pool results/cross_arena/blocks10.json ) > "$TMP/v3_D.log" 2>&1 \
    || { tail "$TMP/v3_D.log"; exit 1; }
fi

# --- E ---------------------------------------------------------------------------------------------------
study narrow_64_jax data/runs_jax_narrow_64 results/ablation_jax_narrow_64 RL_HIDDEN_DIM=64
study wide_256_jax data/runs_jax_wide_256 results/ablation_jax_wide_256 RL_HIDDEN_DIM=256

# --- F ---------------------------------------------------------------------------------------------------
if [ -s "$ROOT/results/absolute_no_dr/matches.csv" ]; then
  stamp "F: no_dr random pool already played"
else
  stamp "F: the no_dr arm's ten blocks against the random pool"
  # The no_dr pool paired seed s with seed s, so the absolute scale uses the same ten blocks.
  ( cd "$ROOT/ctde_arena" && "$PY" "$ROOT/scripts/run_absolute_eval.py" --campaign no_dr \
      --blocks "1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10" --matches 25 --seeds 1,2,3 --out results/absolute_no_dr \
      --null-json results/baseline/random_baseline_uniform.json \
      --compare-pool results/cross_arena/no_dr_blocks.json ) > "$TMP/v3_F.log" 2>&1 \
    || { tail "$TMP/v3_F.log"; exit 1; }
fi

# --- reports -------------------------------------------------------------------------------------------------
"$PY" scripts/report_study.py --results-dir results/study_jax --label "JAX replication" \
  --compare-to results/study --json-out results/study_jax/coupled_tests.json > results/study_jax/report.md
"$PY" scripts/report_study.py --results-dir results/study_v3 --label "v3 (every docs/10 remedy)" \
  --compare-to results/study_jax --json-out results/study_v3/coupled_tests.json > results/study_v3/report.md
for arm in narrow_64 wide_256; do
  "$PY" scripts/report_study.py --results-dir "results/ablation_jax_$arm" --label "$arm on the JAX runtime" \
    --compare-to results/study_jax > "results/ablation_jax_$arm/report.md"
done
stamp "v3 campaign complete"
echo V3_DONE > "$TMP/v3_campaign.flag"
