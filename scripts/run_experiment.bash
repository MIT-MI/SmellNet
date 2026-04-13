#!/usr/bin/env bash
# Example contrastive sweep. Set seed=42 (default) or export seed before running.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/models"
seed="${seed:-42}"

# for lr in 0.0003 0.001 0.003; do
#   for m in mlp cnn lstm transformer; do
#     for g in 0 25; do
#       for w in 50 100; do
#         python run.py \
#           --train-dir "$REPO/base_data/training" \
#           --test-dir "$REPO/base_data/testing" \
#           --real-test-dir "$REPO/base_data/testing" \
#           --gcms-csv "$REPO/gcms_processed/gcms_food_vectors.csv" \
#           --models "$m" --contrastive on --gradients "$g" --window-sizes "$w" \
#           --seed "${seed}" \
#           --epochs 90 --batch-size 32 --lr "$lr" \
#           --run-name-prefix "SEL_grad${g}_w100" \
#           --log-dir "./contrastive_runs_w${w}_seed${seed}"
#       done
#     done
#   done
# done

# ---------------------------------------------------------------------------
# SmellNet-Mixture (run_mixture.py) — commented template, same style as above
#
# Expects a folder tree of mixture sessions (see README / dataset card). Point
# the three dirs at your mixture_data split: seen train, seen test, unseen test.
# Gradients: run_mixture.py allows 0, 100, 250, 500 (temporal differencing periods).
# ---------------------------------------------------------------------------
MIXTURE_TRAIN="$REPO/mixture_data/training_seen"
MIXTURE_TEST="$REPO/mixture_data/test_seen"
MIXTURE_UNSEEN="$REPO/mixture_data/test_unseen"

for lr in 0.0003 0.001 0.003; do
  for m in mlp cnn lstm transformer; do
    for g in 0; do
      for w in 50 100; do
        mkdir -p "./mixture_runs_w${w}_seed${seed}" "./mixture_checkpoints_w${w}_seed${seed}"
        python run_mixture.py \
          --train-dir "$MIXTURE_TRAIN" \
          --test-dir "$MIXTURE_TEST" \
          --unseen-test-dir "$MIXTURE_UNSEEN" \
          --models "$m" --gradients "$g" --window-sizes "$w" \
          --seed "${seed}" \
          --epochs 60 --batch-size 64 --lr "$lr" \
          --run-name-prefix "MIX_grad${g}" \
          --log-dir "./mixture_runs_w${w}_seed${seed}" \
          --save-dir "./mixture_checkpoints_w${w}_seed${seed}"
      done
    done
  done
done
