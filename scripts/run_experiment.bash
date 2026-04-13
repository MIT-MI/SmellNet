#!/usr/bin/env bash
# Example contrastive sweep. Set seed=42 (default) or export seed before running.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/models"
seed="${seed:-42}"

for lr in 0.0003 0.001 0.003; do
  for m in mlp cnn lstm transformer; do
    for g in 0 25; do
      for w in 50 100; do
        python run.py \
          --train-dir "$REPO/ICLR_data/training" \
          --test-dir "$REPO/ICLR_data/testing" \
          --real-test-dir "$REPO/ICLR_data/testing" \
          --gcms-csv "$REPO/gcms_processed/gcms_food_vectors.csv" \
          --models "$m" --contrastive on --gradients "$g" --window-sizes "$w" \
          --seed "${seed}" \
          --epochs 90 --batch-size 32 --lr "$lr" \
          --run-name-prefix "SEL_grad${g}_w100" \
          --log-dir "./contrastive_runs_w${w}_seed${seed}"
      done
    done
  done
done

# --- Optional templates (uncomment and set paths) ---
# Mixture runs (expects train/test trees on disk, e.g. under data/ or a paper split):
#   TRAIN_DIR="$REPO/data/..."
#   TEST_DIR="$REPO/data/..."
#   UNSEEN_DIR="$REPO/data/..."
#   for w in 100; do
#     LOG_DIR="./mixture_runs_w${w}"
#     SAVE_DIR="./mixture_checkpoints_w${w}"
#     mkdir -p "${LOG_DIR}" "${SAVE_DIR}"
#     python run_mixture.py --train-dir "$TRAIN_DIR" --test-dir "$TEST_DIR" \
#       --unseen-test-dir "$UNSEEN_DIR" --models transformer --log-dir "$LOG_DIR" --save-dir "$SAVE_DIR" ...
#   done
