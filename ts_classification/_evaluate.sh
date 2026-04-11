#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ts_classification/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"              # SmellNet/
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
DEFAULT_CLASSES="all"
DEFAULT_FEATURES="all"
CHECKPOINT_PATH="/scratch/keane/smellnet/smell_model_ckpts_tslanet/classifier_epoch_10_best.pt"

DEVICE="cuda:0"  # e.g. cuda:0, cuda:1, cpu, auto

WANDB_PROJECT="smell-net"
WANDB_RUN_NAME=""   # leave empty to let WandB auto-generate a name

# different models to choose from:
# timesnet | transformer | tscmamba | tslanet
MODEL="tslanet"

# --- Evaluation split ---
EVAL_SPLIT="test"   # which split to evaluate on: val | test

python "${SCRIPT_DIR}/_finetune_run.py" \
  --mode eval \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES[@]}" \
  --features "${DEFAULT_FEATURES[@]}" \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --seq-len 512 \
  --checkpoint "${CHECKPOINT_PATH}" \
  --eval-split "${EVAL_SPLIT}" \
  --use-wandb \
  --wandb-project "${WANDB_PROJECT}" \
  ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
  "$@"
