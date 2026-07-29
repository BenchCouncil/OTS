#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$PROJECT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

DEFAULT_PYTHON="python3"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" main.py --model revin_ols --task grid --data ETTh1,ETTh2,ETTm1,ETTm2 \
  --cases NN,RN,NR,RR --pred-lens 96,192,336,720 \
  --output-dir "$WORKSPACE_DIR/results/revin_ols" --model-dir "$WORKSPACE_DIR/checkpoints/revin_ols" "$@"
"$PYTHON_BIN" main.py --model revin_ols --task grid --data electricity,exchange_rate,traffic,weather \
  --cases NN,RN,NR,RR --pred-lens 96,192,336,720 \
  --output-dir "$WORKSPACE_DIR/results/revin_ols" --model-dir "$WORKSPACE_DIR/checkpoints/revin_ols" "$@"
"$PYTHON_BIN" main.py --model revin_ols --task grid --data illness \
  --cases NN,RN,NR,RR --pred-lens 24,36,48,60 \
  --output-dir "$WORKSPACE_DIR/results/revin_ols" --model-dir "$WORKSPACE_DIR/checkpoints/revin_ols" "$@"
