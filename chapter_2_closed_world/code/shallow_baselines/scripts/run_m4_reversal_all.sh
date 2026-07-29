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

OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/results/m4_reversal}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE_DIR/checkpoints/m4_reversal}"

for model in ols revin_ols; do
  for case in NN NR RN RR; do
    for pattern in Yearly Quarterly Monthly Weekly Daily Hourly; do
      "$PYTHON_BIN" main.py \
        --task m4 \
        --model "$model" \
        --case "$case" \
        --seasonal-pattern "$pattern" \
        --output-dir "$OUTPUT_DIR" \
        --model-dir "$MODEL_DIR" \
        "$@"
    done
  done
done
