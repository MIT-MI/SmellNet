#!/usr/bin/env bash
# Evaluate ALL saved models under models_all/{std,nostd} for PURE-only and MIXTURE-only subsets.
set -euo pipefail

TRAIN_DIR="${1:-/home/dewei/workspace/SmellNet/chi_paper_data/training_new}"
TEST_DIR="${2:-/home/dewei/workspace/SmellNet/chi_paper_data/test_seen}"
SCOPE="${3:-both}"  # std|nostd|both

PY=python
EVAL=models/eval_saved_models.py

BATCH="${BATCH:-32}"
MAX_LEN="${MAX_LEN:-600}"
PURE_EPS="${PURE_EPS:-1e-6}"
LAG_FROM_NAME="${LAG_FROM_NAME:-1}"
LAG_DEFAULT="${LAG_DEFAULT:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_LOG="logs_eval/${STAMP}"
mkdir -p "$BASE_LOG"

infer_arch() {
  local f="$(basename "$1")"
  if [[ "$f" == *"transformer"* ]]; then echo "transformer"
  elif [[ "$f" == *"lstm"* ]]; then echo "lstm"
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

run_one() {
  local label="$1"  # std|nostd
  local w="$2"
  local arch; arch=$(infer_arch "$w")
  local lag; lag=$(infer_lag "$w")
  local stdflag=""; [[ "$label" == "nostd" ]] && stdflag="--no-standardize"

  local name="$(basename "${w%.pt}")"
  local outdir="${BASE_LOG}/${label}"; mkdir -p "$outdir"

  # PURE
  cmd=( "$PY" "$EVAL" --weights "$w" --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR"
        --arch "$arch" --batch-size "$BATCH" --max-len "$MAX_LEN" --lag "$lag" $stdflag
        --eval-pure-only --pure-eps "$PURE_EPS" )
  printf '%q ' "${cmd[@]}" | tee "${outdir}/${name}_PURE.cmd"; echo
  "${cmd[@]}" 2>&1 | tee "${outdir}/${name}_PURE.log"

  # MIX
  cmd=( "$PY" "$EVAL" --weights "$w" --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR"
        --arch "$arch" --batch-size "$BATCH" --max-len "$MAX_LEN" --lag "$lag" $stdflag
        --eval-mixture-only --pure-eps "$PURE_EPS" )
  printf '%q ' "${cmd[@]}" | tee "${outdir}/${name}_MIX.cmd"; echo
  "${cmd[@]}" 2>&1 | tee "${outdir}/${name}_MIX.log"
}

do_bucket() {
  local label="$1"; local dir="models_all/${label}"
  [[ -d "$dir" ]] || { echo "[WARN] $dir not found; skipping"; return; }
  shopt -s nullglob; for w in "$dir"/*.pt; do run_one "$label" "$w"; done; shopt -u nullglob
}

[[ "$SCOPE" == "both" || "$SCOPE" == "std" ]] && do_bucket "std"
[[ "$SCOPE" == "both" || "$SCOPE" == "nostd" ]] && do_bucket "nostd"

echo "Done. Logs under: ${BASE_LOG}"
