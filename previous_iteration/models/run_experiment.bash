# Choose gradient for each model at window=100 (fast, fair)
for lr in 0.0003 0.001 0.003; do
    for m in mlp cnn lstm transformer; do
        for g in 0 25; do
            python run.py \
            --train-dir /home/dewei/workspace/SmellNet/data/offline_training \
            --test-dir  /home/dewei/workspace/SmellNet/data/offline_testing \
            --real-test-dir /home/dewei/workspace/SmellNet/data/online_nuts \
            --gcms-csv /home/dewei/workspace/SmellNet/data/gcms_dataframe.csv \
            --models $m --contrastive on --gradients $g --window-sizes 50 \
            --epochs 90 --batch-size 32 --lr $lr \
            --run-name-prefix cSEL_grad_w50 --log-dir ./runs_cw50
        done
    done
done


# # Choose gradient for each model at window=100 (fast, fair)
# for lr in 0.0003 0.001 0.003; do
#     for m in mlp cnn lstm transformer; do
#         for g in 0 25; do
#             python run.py \
#             --train-dir /home/dewei/workspace/SmellNet/data/offline_training \
#             --test-dir  /home/dewei/workspace/SmellNet/data/offline_testing \
#             --real-test-dir /home/dewei/workspace/SmellNet/data/online_nuts \
#             --gcms-csv /home/dewei/workspace/SmellNet/data/gcms_dataframe.csv \
#             --models $m --contrastive on --gradients $g --window-sizes 500 \
#             --epochs 90 --batch-size 32 --lr $lr \
#             --run-name-prefix cSEL_grad_w500 --log-dir ./runs_cw500
#         done
#     done
# done


# Choose gradient for each model at window=100 (fast, fair)
# for lr in 0.0003 0.001 0.003; do
#     for m in mlp cnn lstm transformer; do
#         for g in 0 25; do
#             python run.py \
#             --train-dir /home/dewei/workspace/SmellNet/data/offline_training \
#             --test-dir  /home/dewei/workspace/SmellNet/data/offline_testing \
#             --real-test-dir /home/dewei/workspace/SmellNet/data/online_nuts \
#             --gcms-csv /home/dewei/workspace/SmellNet/data/gcms_dataframe.csv \
#             --models $m --contrastive off --gradients $g --window-sizes 50 \
#             --epochs 90 --batch-size 32 --lr $lr \
#             --run-name-prefix SEL_grad_w50 --log-dir ./runs_w50
#         done
#     done
# done


# # Choose gradient for each model at window=100 (fast, fair)
# for lr in 0.0003 0.001 0.003; do
#     for m in mlp cnn lstm transformer; do
#         for g in 0 25; do
#             python run.py \
#             --train-dir /home/dewei/workspace/SmellNet/data/offline_training \
#             --test-dir  /home/dewei/workspace/SmellNet/data/offline_testing \
#             --real-test-dir /home/dewei/workspace/SmellNet/data/online_nuts \
#             --gcms-csv /home/dewei/workspace/SmellNet/data/gcms_dataframe.csv \
#             --models $m --contrastive off --gradients $g --window-sizes 500 \
#             --epochs 90 --batch-size 32 --lr $lr \
#             --run-name-prefix SEL_grad_w500 --log-dir ./runs_w500
#         done
#     done
# done


for lr in 0.0003 0.001 0.003; do
    python run_mixture.py \
        --train-dir /home/dewei/workspace/SmellNet/chi_paper_data/training_new \
        --test-dir /home/dewei/workspace/SmellNet/chi_paper_data/test_seen \
        --unseen-test-dir /home/dewei/workspace/SmellNet/chi_paper_data/test_unseen\
        --models mlp cnn lstm transformer \
        --gradients 0 \
        --window-sizes 50 \
        --epochs 60 \
        --batch-size 64 \
        --lr $lr \
        --fft off \
        --sampling-rate 1.0 \
        --run-name-prefix mix \
        --log-dir ./mixture_runs_w50 \
        --save-dir ./checkpoints
done

for lr in 0.0003 0.001 0.003; do
    python run_mixture.py \
        --train-dir /home/dewei/workspace/SmellNet/chi_paper_data/training_new \
        --test-dir /home/dewei/workspace/SmellNet/chi_paper_data/test_seen \
        --unseen-test-dir /home/dewei/workspace/SmellNet/chi_paper_data/test_unseen\
        --models mlp cnn lstm transformer \
        --gradients 0 \
        --window-sizes 100 \
        --epochs 60 \
        --batch-size 64 \
        --lr $lr \
        --fft off \
        --sampling-rate 1.0 \
        --run-name-prefix mix \
        --log-dir ./mixture_runs_w100 \
        --save-dir ./checkpoints
done
