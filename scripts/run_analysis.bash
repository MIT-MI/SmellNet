#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/models"

for seed in 42; do
  for w in 100; do
    python analyze_runs.py \
      --log-dir "$REPO/models/contrastive_runs_w${w}_seed${seed}" \
      --out "$REPO/models/analyze_contrastive_runs_w${w}_seed${seed}" \
      --select-metric acc1 --contrastive on
  done
done
