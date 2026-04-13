for seed in 42; do
    for w in 100; do
        python analyze_runs.py \
            --log-dir /home/dewei/workspace/SmellNet/models/contrastive_runs_w100_seed${seed} \
            --out /home/dewei/workspace/SmellNet/models/analyze_contrastive_runs_w100_seed${seed} \
            --select-metric acc1 --contrastive on
    done
done