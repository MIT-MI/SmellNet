#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ts_classification/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"              # SmellNet/
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"
# DEFAULT_CLASSES=("apple" "banana" "asparagus" "avocado")
DEFAULT_CLASSES="all"
DEFAULT_FEATURES="all"   # "all" = auto-detect all numeric columns, or e.g. ("NO2" "C2H5OH" "VOC" "CO" "Alcohol" "LPG" "Benzene")

DEVICE="cuda:0"  # e.g. cuda:0, cuda:1, cpu, auto

WANDB_PROJECT="smell-net"
WANDB_RUN_NAME="tslanet_test"   # leave empty to let WandB auto-generate a name

# different models to choose from:
# timesnet | transformer | tscmamba | tslanet

# Set these to load a pretrained encoder backbone; leave empty to train from scratch
PRETRAIN_ENCODER=""  # e.g. "/scratch/keane/smellnet/pretrain_ckpts/pretrained_encoder.pt"
PRETRAIN_METADATA="" # e.g. "/scratch/keane/smellnet/pretrain_ckpts/pretrain_metadata.json"

# --window-stride can be none or an integer;

python "${SCRIPT_DIR}/_finetune_run.py" \
  --mode train \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES[@]}" \
  --features "${DEFAULT_FEATURES[@]}" \
  --model "tslanet" \
  --device "${DEVICE}" \
  --seq-len 512 \
  --batch-size 32 \
  --epochs 200 \
  --learning-rate 1e-3 \
  --val-frequency 1 \
  --eval-split "test" \
  --eval-at-end \
  --window-stride "100" \
  --temporal-diff "none" \
  --save-frequency 5 \
  --save-dir "/scratch/keane/smellnet/smell_model_ckpts_tslanet" \
  --wandb-project "${WANDB_PROJECT}" \
  ${WANDB_RUN_NAME:+--wandb-run-name "${WANDB_RUN_NAME}"} \
  ${PRETRAIN_ENCODER:+--load-pretrained-checkpoint "${PRETRAIN_ENCODER}"} \
  ${PRETRAIN_METADATA:+--load-pretrained-metadata "${PRETRAIN_METADATA}"} \
  "$@"
