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

"$PYTHON_BIN" main.py --task grid --data ETTh1,ETTh2,ETTm1,ETTm2 --cases RN,NR,RR \
  --pred-lens 96,192,336,720 --output-dir "$WORKSPACE_DIR/results/reversal" --model-dir "$WORKSPACE_DIR/checkpoints/reversal" "$@"
"$PYTHON_BIN" main.py --task grid --data electricity,exchange_rate,traffic,weather --cases RN,NR,RR \
  --pred-lens 96,192,336,720 --output-dir "$WORKSPACE_DIR/results/reversal" --model-dir "$WORKSPACE_DIR/checkpoints/reversal" "$@"
"$PYTHON_BIN" main.py --task grid --data illness --cases RN,NR,RR \
  --pred-lens 24,36,48,60 --output-dir "$WORKSPACE_DIR/results/reversal" --model-dir "$WORKSPACE_DIR/checkpoints/reversal" "$@"
