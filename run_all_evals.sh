#!/usr/bin/env bash
# run_all_evals.sh
# Evaluate all saved weights under models_all/<DIR>/*.pt on:
#   (1) PURE-only subset and (2) MIXTURE-only subset
# in one go, writing logs + per-ingredient CSVs.
#
# Defaults:
#   - evaluate DIRS: no1 (no-standardize) then yes1 (standardized)
#   - calibrate/temp on ALL validation to stabilize tiny PURE set
#   - per-ingredient CSVs enabled
#
# Env overrides (optional):
#   MODELS_ROOT=models_all         # root containing subdirs (no1, yes1, ...)
#   PYTHON_BIN=python
#   EVAL_PATH=models/eval_saved_models.py
#   DIRS="no1 yes1"                # space-separated list
#   NOSTD_DIRS=no1,nostd           # CSV: dirs to pass --no-standardize
#   BATCH=32 MAX_LEN=600 PURE_EPS=1e-6
#   LAG_FROM_NAME=1 LAG_DEFAULT=0  # infer lag from filename '..._lag25_...'
#   THR_ACC=0.2                    # within-x absolute error for per-class
#   CALIBRATE_ON=all THRESH_ON=all # where to fit temp / sweep threshold
#   NO_TEMP=1                      # disable temperature scaling
#   EVAL_PERCLASS=1                # 0 to skip per-ingredient metrics
#   SKIP_GUARD=1                   # bypass capability check on evaluator
#   CLASS_NAMES=/path/classes.txt  # 12 lines or JSON list
#
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TRAIN_DIR TEST_DIR [DIR ...]"
  echo "Example (defaults to 'no1 yes1'):"
  echo "  bash $0 /path/to/train /path/to/test"
  exit 1
fi

TRAIN_DIR="$1"; shift
TEST_DIR="$1"; shift

# Directories to scan for weights
if [[ $# -ge 1 ]]; then
  DIRS=("$@")
else
  if [[ -n "${DIRS:-}" ]]; then
    # shellcheck disable=SC2206
    DIRS=(${DIRS})
  else
    DIRS=(no1 yes1)
  fi
fi

MODELS_ROOT="${MODELS_ROOT:-models_all}"
PY="${PYTHON_BIN:-python}"
EVAL="${EVAL_PATH:-models/eval_saved_models.py}"

BATCH="${BATCH:-32}"
MAX_LEN="${MAX_LEN:-600}"
PURE_EPS="${PURE_EPS:-1e-6}"
LAG_FROM_NAME="${LAG_FROM_NAME:-1}"
LAG_DEFAULT="${LAG_DEFAULT:-0}"
THR_ACC="${THR_ACC:-0.2}"

CALIBRATE_ON="${CALIBRATE_ON:-all}"   # subset|all
THRESH_ON="${THRESH_ON:-all}"         # subset|all
NO_TEMP="${NO_TEMP:-}"

EVAL_PERCLASS="${EVAL_PERCLASS:-1}"
SKIP_GUARD="${SKIP_GUARD:-}"

IFS=',' read -r -a FORCED_NOSTD <<< "${NOSTD_DIRS:-no1}"

CLASS_NAMES_PATH="${CLASS_NAMES:-/home/dewei/workspace/SmellNet/classes.txt}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_LOG="logs_eval/${STAMP}"; mkdir -p "$BASE_LOG"

# Guard: ensure evaluator supports per-class flags (unless turned off or skipped)
if [[ "$EVAL_PERCLASS" == "1" && -z "${SKIP_GUARD}" ]]; then
  if ! $PY "$EVAL" --help | grep -q -- "--per-class-save" ; then
    echo "[ERROR] $EVAL does not support --per-class-save."
    echo "Update the evaluator (models/eval_saved_models.py), or run with EVAL_PERCLASS=0 or SKIP_GUARD=1."
    exit 1
  fi
fi

infer_arch() {
  local f="$(basename "$1")"
  if [[ "$f" == *"transformer"* || "$f" == transformer* ]]; then echo "transformer"
  elif [[ "$f" == *"lstm"* || "$f" == lstm* ]]; then echo "lstm"
  else echo "tcn"
  fi
}

infer_lag() {
  local f="$1"
  if [[ "$LAG_FROM_NAME" == "1" ]]; then
    if [[ "$f" =~ lag([0-9]+) ]]; then echo "${BASH_REMATCH[1]}"; else echo "$LAG_DEFAULT"; fi
  else
    echo "$LAG_DEFAULT"
  fi
}

is_forced_nostd() {
  local target="$1"
  for d in "${FORCED_NOSTD[@]:-}"; do [[ "$d" == "$target" ]] && return 0; done
  return 1
}

run_one() {
  local label="$1"; local w="$2"
  local arch; arch=$(infer_arch "$w")
  local lag; lag=$(infer_lag "$w")
  local stdflag=""; if is_forced_nostd "$label"; then stdflag="--no-standardize"; fi
  local name="$(basename "${w%.pt}")"
  local outdir="${BASE_LOG}/${label}"; mkdir -p "$outdir"

  echo "[RUN] DIR=${label}  WEIGHT=${w}  ARCH=${arch}  LAG=${lag}  STD_FLAG='${stdflag}'"

  extra=( --calibrate-on "$CALIBRATE_ON" --thresh-on "$THRESH_ON" )
  if [[ -n "$NO_TEMP" ]]; then extra+=( --no-temp ); fi

  pcflags=()
  if [[ "$EVAL_PERCLASS" == "1" ]]; then
    if [[ -n "$CLASS_NAMES_PATH" && -f "$CLASS_NAMES_PATH" ]]; then pcflags+=( --class-names "$CLASS_NAMES_PATH" ); fi
  fi

  # PURE
  cmd=( "$PY" "$EVAL" --weights "$w" --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR"
        --arch "$arch" --batch-size "$BATCH" --max-len "$MAX_LEN" --lag "$lag" $stdflag
        --eval-pure-only --pure-eps "$PURE_EPS"
        "${extra[@]}"
      )
  if [[ "$EVAL_PERCLASS" == "1" ]]; then
    p_csv="${outdir}/${name}_PURE_perclass"
    cmd+=( --per-class --thr-acc "$THR_ACC" --per-class-save "$p_csv" "${pcflags[@]}" )
  fi
  printf '%q ' "${cmd[@]}" | tee "${outdir}/${name}_PURE.cmd"; echo
  "${cmd[@]}" 2>&1 | tee "${outdir}/${name}_PURE.log"

  # MIXTURE
  cmd=( "$PY" "$EVAL" --weights "$w" --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR"
        --arch "$arch" --batch-size "$BATCH" --max-len "$MAX_LEN" --lag "$lag" $stdflag
        --eval-mixture-only --pure-eps "$PURE_EPS"
        "${extra[@]}"
      )
  if [[ "$EVAL_PERCLASS" == "1" ]]; then
    m_csv="${outdir}/${name}_MIX_perclass"
    cmd+=( --per-class --thr-acc "$THR_ACC" --per-class-save "$m_csv" "${pcflags[@]}" )
  fi
  printf '%q ' "${cmd[@]}" | tee "${outdir}/${name}_MIX.cmd"; echo
  "${cmd[@]}" 2>&1 | tee "${outdir}/${name}_MIX.log"
}

do_bucket() {
  local label="$1"; local dir="${MODELS_ROOT}/${label}"
  [[ -d "$dir" ]] || { echo "[WARN] $dir not found; skipping"; return; }
  shopt -s nullglob; local any=0
  for w in "$dir"/*.pt; do any=1; run_one "$label" "$w"; done
  shopt -u nullglob
  [[ $any -eq 1 ]] || echo "[WARN] No .pt files in $dir"
}

echo "[INFO] Evaluating DIRS in order: ${DIRS[*]}"
for d in "${DIRS[@]}"; do do_bucket "$d"; done
echo "Done. Logs under: ${BASE_LOG}"
