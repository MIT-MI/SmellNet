#!/usr/bin/env bash
# run_experiments.sh
# Comprehensive experiment runner for standardized and no-standardized pipelines.
# Usage:
#   bash run_experiments.sh [TRAIN_DIR] [TEST_DIR] [MODE] [SCOPE]
#   MODE: both | std | nostd   (default: both)
#   SCOPE: fast | full         (default: fast)
#
# This script logs to logs_all/<timestamp>/<std|nostd>/
# and saves models to models_all/<std|nostd>/

set -euo pipefail

TRAIN_DIR="${1:-/home/dewei/workspace/SmellNet/chi_paper_data/training_new}"
TEST_DIR="${2:-/home/dewei/workspace/SmellNet/chi_paper_data/test_seen}"
MODE="${3:-both}"
SCOPE="${4:-fast}"

PY=python
SCRIPT=models/train_dist_model.py

EPOCHS=60
BATCH=32
VAL_SPLIT=0.15
MAX_LEN=600
SEEDS="${SEEDS:-42}"  # override by env var: SEEDS="42 1337"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE_LOG="logs_all/${STAMP}"
BASE_MODEL="models_all"
mkdir -p "$BASE_LOG" "$BASE_MODEL"

# ----------------------
# Helpers
# ----------------------
run() {
  local stdflag="$1" arch="$2" lr="$3" wd="$4" alpha="$5" beta="$6" synth_p="$7" synth_k="$8" lag="$9" seed="${10}"
  local label="$11"  # "std" or "nostd"
  local logdir="${BASE_LOG}/${label}"
  local modeldir="${BASE_MODEL}/${label}"
  mkdir -p "$logdir" "$modeldir"

  local name="${arch}_lr${lr}_wd${wd}_a${alpha}_b${beta}_sp${synth_p}_k${synth_k}_lag${lag}_s${seed}"
  echo "==> [$label][$arch] $name"

  CMD=( "$PY" "$SCRIPT"
    --train-dir "$TRAIN_DIR"
    --test-dir "$TEST_DIR"
    --epochs "$EPOCHS" --batch-size "$BATCH"
    --val-split "$VAL_SPLIT" --max-len "$MAX_LEN"
    --lr "$lr" --weight-decay "$wd"
    --alpha "$alpha" --beta "$beta"
    --synth-p "$synth_p" --synth-max-k "$synth_k" --lag "$lag"
    --arch "$arch" --seed "$seed"
    --save "$modeldir/$name.pt"
  )

  if [[ "$stdflag" == "no" ]]; then
    CMD+=( --no-standardize )
  fi

  # Save exact command and stream logs
  printf '%q ' "${CMD[@]}" | tee "$logdir/$name.cmd"
  echo
  "${CMD[@]}" 2>&1 | tee "$logdir/$name.log"
}

# ----------------------
# Search spaces
# ----------------------
# Common knobs
ALPHAS=(0.5 1.0)
BETAS=(0.5 0.8)
SYNTH_P=(0.5)
SYNTH_K=(4)
LAGS_STD=(0 25)
LAGS_NOSTD=(0 25)  # emphasize transients without scaling

# Arch-specific LR/WD
declare -A LRS_STD WDS_STD LRS_NOSTD WDS_NOSTD

if [[ "$SCOPE" == "fast" ]]; then
  # Standardized
  LRS_STD[lstm]="3e-4";        WDS_STD[lstm]="3e-4"
  LRS_STD[transformer]="3e-4"; WDS_STD[transformer]="3e-4"
  LRS_STD[tcn]="3e-4";         WDS_STD[tcn]="3e-4"
  ALPHAS_STD=(1.0)  # simplify for fast
  BETAS_STD=(0.5 0.8)

  # No-standardize
  LRS_NOSTD[lstm]="1e-4";        WDS_NOSTD[lstm]="1e-4"
  LRS_NOSTD[transformer]="1e-4"; WDS_NOSTD[transformer]="1e-4"
  LRS_NOSTD[tcn]="2e-4";         WDS_NOSTD[tcn]="2e-4"
  ALPHAS_NOSTD=(1.0)
  BETAS_NOSTD=(0.5 0.8)
else
  # Standardized (broader)
  LRS_STD[lstm]="3e-4 2e-4";        WDS_STD[lstm]="3e-4"
  LRS_STD[transformer]="3e-4 2e-4"; WDS_STD[transformer]="3e-4"
  LRS_STD[tcn]="3e-4 2e-4";         WDS_STD[tcn]="3e-4 2e-4"
  ALPHAS_STD=("${ALPHAS[@]}")
  BETAS_STD=("${BETAS[@]}")

  # No-standardize (broader)
  LRS_NOSTD[lstm]="1e-4 5e-5";        WDS_NOSTD[lstm]="1e-4"
  LRS_NOSTD[transformer]="1e-4 5e-5"; WDS_NOSTD[transformer]="1e-4"
  LRS_NOSTD[tcn]="2e-4 1e-4";         WDS_NOSTD[tcn]="2e-4 1e-4"
  ALPHAS_NOSTD=("${ALPHAS[@]}")
  BETAS_NOSTD=("${BETAS[@]}")
fi

# ----------------------
# Launch
# ----------------------
for seed in $SEEDS; do
  if [[ "$MODE" == "both" || "$MODE" == "std" ]]; then
    for arch in lstm transformer tcn; do
      for lr in ${LRS_STD[$arch]}; do
        for wd in ${WDS_STD[$arch]}; do
          for alpha in "${ALPHAS_STD[@]}"; do
            for beta in "${BETAS_STD[@]}"; do
              for sp in "${SYNTH_P[@]}"; do
                for k in "${SYNTH_K[@]}"; do
                  for lag in "${LAGS_STD[@]}"; do
                    run "yes" "$arch" "$lr" "$wd" "$alpha" "$beta" "$sp" "$k" "$lag" "$seed" "std"
                  done
                done
              done
            done
          done
        done
      done
    done
  fi

  if [[ "$MODE" == "both" || "$MODE" == "nostd" ]]; then
    for arch in lstm transformer tcn; do
      for lr in ${LRS_NOSTD[$arch]}; do
        for wd in ${WDS_NOSTD[$arch]}; do
          for alpha in "${ALPHAS_NOSTD[@]}"; do
            for beta in "${BETAS_NOSTD[@]}"; do
              for sp in "${SYNTH_P[@]}"; do
                for k in "${SYNTH_K[@]}"; do
                  for lag in "${LAGS_NOSTD[@]}"; do
                    run "no" "$arch" "$lr" "$wd" "$alpha" "$beta" "$sp" "$k" "$lag" "$seed" "nostd"
                  done
                done
              done
            done
          done
        done
      done
    done
  fi
done

echo "Done. Logs under: ${BASE_LOG}"
