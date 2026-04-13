#!/usr/bin/env bash
# Pass-through to training; run from repo root context.
# Example: ./scripts/run_experiments.sh --train-dir ../ICLR_data/training --help
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO/models"
exec python run.py "$@"
