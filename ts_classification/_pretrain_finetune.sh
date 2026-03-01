#!/usr/bin/env bash
# Runs contrastive pretraining then immediately fine-tunes from the saved encoder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # ts_classification/
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"              # SmellNet/
DATA_ROOT="${PROJECT_ROOT}/smell_ts_dataset/SmellNet"

# ── Shared ────────────────────────────────────────────────────────────────────
DEFAULT_CLASSES="all"
DEFAULT_FEATURES="all"
MODEL="tslanet"
SEQ_LEN=512
NORMALIZATION="zscore"
WINDOW_STRIDE="200"      # none or integer
TEMPORAL_DIFF="none"      # none/0/false = off; positive integer = lag

# ── Pretrain ──────────────────────────────────────────────────────────────────
PRETRAIN_EPOCHS=10
PRETRAIN_LR=1e-3
PRETRAIN_WEIGHT_DECAY=1e-4
TEMPERATURE=0.07
PROJ_DIM=128
PROJ_HIDDEN=256
# AUG_LIST="jitter scale time_shift magnitude_warp"
AUG_LIST="none"
PRETRAIN_BATCH_SIZE=36
PRETRAIN_SAVE_DIR="/scratch/keane/smellnet/pretrain_ckpts"

WANDB_PROJECT_PRETRAIN="smell-net"
WANDB_RUN_NAME_PRETRAIN="pretrain_tsla"   # leave empty to auto-generate

# ── Fine-tune ─────────────────────────────────────────────────────────────────
FINETUNE_EPOCHS=300
FINETUNE_LR=1e-3
FINETUNE_BATCH_SIZE=50
FINETUNE_SAVE_DIR="/scratch/keane/smellnet/smell_model_ckpts"
SAVE_FREQUENCY=5
VAL_FREQUENCY=1

WANDB_PROJECT_FINETUNE="smell-net"
WANDB_RUN_NAME_FINETUNE="finetune_pretrained_tsla"   # leave empty to auto-generate

# ── Step 1: Pretrain ──────────────────────────────────────────────────────────
echo "================================================================"
echo " STEP 1 — Contrastive pretraining"
echo "================================================================"

python "${SCRIPT_DIR}/pretrain_run.py" \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES}" \
  --features "${DEFAULT_FEATURES}" \
  --model "${MODEL}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${PRETRAIN_BATCH_SIZE}" \
  --pretrain-epochs "${PRETRAIN_EPOCHS}" \
  --pretrain-lr "${PRETRAIN_LR}" \
  --pretrain-weight-decay "${PRETRAIN_WEIGHT_DECAY}" \
  --temperature "${TEMPERATURE}" \
  --proj-dim "${PROJ_DIM}" \
  --proj-hidden "${PROJ_HIDDEN}" \
  --aug-list ${AUG_LIST} \
  --normalization "${NORMALIZATION}" \
  --window-stride "${WINDOW_STRIDE}" \
  --temporal-diff "${TEMPORAL_DIFF}" \
  --pretrain-save-dir "${PRETRAIN_SAVE_DIR}" \
  --wandb-project "${WANDB_PROJECT_PRETRAIN}" \
  ${WANDB_RUN_NAME_PRETRAIN:+--wandb-run-name "${WANDB_RUN_NAME_PRETRAIN}"}

ENCODER_CKPT="${PRETRAIN_SAVE_DIR}/pretrained_encoder.pt"
ENCODER_META="${PRETRAIN_SAVE_DIR}/pretrain_metadata.json"

echo ""
echo "Pretraining complete."
echo "  Encoder  : ${ENCODER_CKPT}"
echo "  Metadata : ${ENCODER_META}"

# ── Step 2: Fine-tune ─────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo " STEP 2 — Fine-tuning from pretrained encoder"
echo "================================================================"

python "${SCRIPT_DIR}/_finetune_run.py" \
  --mode train \
  --data-root "${DATA_ROOT}" \
  --classes "${DEFAULT_CLASSES}" \
  --features "${DEFAULT_FEATURES}" \
  --model "${MODEL}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${FINETUNE_BATCH_SIZE}" \
  --epochs "${FINETUNE_EPOCHS}" \
  --learning-rate "${FINETUNE_LR}" \
  --normalization "${NORMALIZATION}" \
  --window-stride "${WINDOW_STRIDE}" \
  --temporal-diff "${TEMPORAL_DIFF}" \
  --val-frequency "${VAL_FREQUENCY}" \
  --eval-split "test" \
  --eval-at-end \
  --save-frequency "${SAVE_FREQUENCY}" \
  --save-dir "${FINETUNE_SAVE_DIR}" \
  --load-pretrained-checkpoint "${ENCODER_CKPT}" \
  --load-pretrained-metadata "${ENCODER_META}" \
  --wandb-project "${WANDB_PROJECT_FINETUNE}" \
  ${WANDB_RUN_NAME_FINETUNE:+--wandb-run-name "${WANDB_RUN_NAME_FINETUNE}"}

echo ""
echo "================================================================"
echo " Done. Fine-tuned checkpoints saved to: ${FINETUNE_SAVE_DIR}"
echo "================================================================"
