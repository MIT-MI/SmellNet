#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ts_classification/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"              # SmellNet/
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
DEFAULT_CLASSES="all"
DEFAULT_FEATURES="all"  # "all" = auto-detect all numeric columns

WANDB_PROJECT="smell-net-pretrain"
WANDB_RUN_NAME=""  # leave empty to let WandB auto-generate a name

# different models to choose from:
# timesnet | transformer | tscmamba | tslanet

PRETRAIN_SAVE_DIR="/scratch/keane/smellnet/pretrain_ckpts"

# --window-stride can be none or an integer
# --temporal-diff: positive integer = lag (enable), none/0/false = disable

python "${SCRIPT_DIR}/pretrain_run.py" \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES}" \
  --features "${DEFAULT_FEATURES}" \
  --model "tslanet" \
  --seq-len 512 \
  --batch-size 32 \
  --pretrain-epochs 50 \
  --pretrain-lr 1e-3 \
  --pretrain-weight-decay 1e-4 \
  --temperature 0.07 \
  --proj-dim 128 \
  --proj-hidden 256 \
  --aug-list jitter scale time_shift magnitude_warp \
  --normalization "zscore" \
  --window-stride "none" \
  --temporal-diff "none" \
  --pretrain-save-dir "${PRETRAIN_SAVE_DIR}" \
  --wandb-project "${WANDB_PROJECT}" \
  ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
  "$@"
