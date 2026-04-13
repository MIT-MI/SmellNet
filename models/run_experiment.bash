# Choose gradient for each model at window=100 (fast, fair)

set -e

for lr in 0.0003 0.001 0.003; do
  for m in mlp cnn lstm transformer; do
    for g in 0 25; do
      for w in 50 100; do
        python run.py \
          --train-dir /home/dewei/workspace/SmellNet/ICLR_data/training \
          --test-dir  /home/dewei/workspace/SmellNet/ICLR_data/testing \
          --real-test-dir /home/dewei/workspace/SmellNet/ICLR_data/testing \
          --gcms-csv /home/dewei/workspace/SmellNet/gcms_analysis/gcms_food_vectors.csv \
          --models "$m" --contrastive on --gradients "$g" --window-sizes "$w" \
          --seed ${seed} \
          --epochs 90 --batch-size 32 --lr "$lr" \
          --run-name-prefix SEL_grad${g}_w100 \
          --log-dir ./contrastive_runs_w${w}_seed${seed}
      done
    done
  done
done

# for day in 1 2 3 4 5 6; do
#     for lr in 0.0003 0.001 0.003; do
#         for m in transformer; do
#             for g in 0 25; do
#                 python run.py \
#                 --train-dir /home/dewei/workspace/SmellNet/data/offline_training \
#                 --test-dir  /home/dewei/workspace/SmellNet/data/offline_testing \
#                 --real-test-dir /home/dewei/workspace/SmellNet/data/online_nuts \
#                 --gcms-csv /home/dewei/workspace/SmellNet/gcms_analysis/gcms_food_vectors.csv \
#                 --models $m --contrastive off --gradients $g --window-sizes 50 \
#                 --epochs 90 --batch-size 32 --lr $lr \
#                 --held-out-day $day \
#                 --run-name-prefix SEL_heldoutday${day}_grad${g}_w100 --log-dir ./held_runs_w50_heldout_days_${day}
#             done
#         done
#     done
# done

# for lr in 0.0003 0.001 0.003; do
#     for m in mlp cnn lstm transformer; do
#         for g in 0 25; do
#             python run.py \
#             --train-dir /home/dewei/workspace/SmellNet/data/offline_training \
#             --test-dir  /home/dewei/workspace/SmellNet/data/offline_testing \
#             --real-test-dir /home/dewei/workspace/SmellNet/data/online_nuts \
#             --gcms-csv /home/dewei/workspace/SmellNet/gcms_analysis/gcms_food_vectors.csv \
#             --models $m --contrastive on --gradients $g --window-sizes 100 \
#             --epochs 90 --batch-size 32 --lr $lr \
#             --run-name-prefix SEL_grad_w100 --log-dir ./new_gcms_runs_w100_all
#         done
#     done
# done

#!/usr/bin/env bash
# set -e

# TRAIN_DIR="/home/dewei/workspace/SmellNet/chi_paper_data/training_new"
# TEST_DIR="/home/dewei/workspace/SmellNet/chi_paper_data/test_seen"

# for w in 100; do
#   LOG_DIR="./mixture_runs_w${w}"
#   SAVE_DIR="./mixture_checkpoints_w${w}"
#   mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

#   for g in 0; do
#     for lr in 0.0003 0.001 0.003; do
#       for m in mlp cnn lstm transformer; do
#         echo "=== mixture: model=${m}, grad=${g}, w=${w}, lr=${lr} ==="
#         python run_mixture.py \
#           --train-dir "${TRAIN_DIR}" \
#           --test-dir "${TEST_DIR}" \
#           --unseen-test-dir "/home/dewei/workspace/SmellNet/chi_paper_data/test_unseen" \
#           --models "${m}" \
#           --gradients "${g}" \
#           --window-sizes "${w}" \
#           --epochs 60 \
#           --batch-size 64 \
#           --lr "${lr}" \
#           --fft off \
#           --sampling-rate 1.0 \
#           --run-name-prefix "mix-${m}-g${g}-w${w}" \
#           --log-dir "${LOG_DIR}" \
#           --save-dir "${SAVE_DIR}"
#       done
#     done
#   done
# done