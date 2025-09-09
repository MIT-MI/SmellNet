#!/usr/bin/env bash
# Evaluate saved models under models_all/<DIR>/*.pt for PURE-only and MIXTURE-only subsets.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TRAIN_DIR TEST_DIR [DIR ...]"
  exit 1
fi

TRAIN_DIR="$1"; shift
TEST_DIR="$1"; shift
if [[ $# -ge 1 ]]; then DIRS=("$@"); else DIRS=(std nostd); fi

MODELS_ROOT="${MODELS_ROOT:-models_all}"
PY=python
EVAL=models/eval_saved_models.py

BATCH="${BATCH:-32}"
MAX_LEN="${MAX_LEN:-600}"
PURE_EPS="${PURE_EPS:-1e-6}"
LAG_FROM_NAME="${LAG_FROM_NAME:-1}"
LAG_DEFAULT="${LAG_DEFAULT:-0}"
# Force certain dirs to --no-standardize, CSV list (optional)
IFS=',' read -r -a FORCED_NOSTD <<< "${NOSTD_DIRS:-}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_LOG="logs_eval/${STAMP}"; mkdir -p "$BASE_LOG"

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
  local stdflag=""
  if is_forced_nostd "$label"; then stdflag="--no-standardize"; fi
  local name="$(basename "${w%.pt}")"
  local outdir="${BASE_LOG}/${label}"; mkdir -p "$outdir"

  cmd=( "$PY" "$EVAL" --weights "$w" --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR"
        --arch "$arch" --batch-size "$BATCH" --max-len "$MAX_LEN" --lag "$lag" $stdflag
        --eval-pure-only --pure-eps "$PURE_EPS" )
  printf '%q ' "${cmd[@]}" | tee "${outdir}/${name}_PURE.cmd"; echo
  "${cmd[@]}" 2>&1 | tee "${outdir}/${name}_PURE.log"

  cmd=( "$PY" "$EVAL" --weights "$w" --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR"
        --arch "$arch" --batch-size "$BATCH" --max-len "$MAX_LEN" --lag "$lag" $stdflag
        --eval-mixture-only --pure-eps "$PURE_EPS" )
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
for d in "${DIRS[@]}"; do do_bucket "$d"; done
echo "Done. Logs under: ${BASE_LOG}"
