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

OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/results/long_input}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE_DIR/checkpoints/long_input}"

for model in ols revin_ols; do
  "$PYTHON_BIN" main.py \
    --task grid \
    --model "$model" \
    --data ETTh1,ETTh2,ETTm1,ETTm2 \
    --seq-len 336 \
    --cases NN,NR,RN,RR \
    --pred-lens 96,192,336,720 \
    --output-dir "$OUTPUT_DIR" \
    --model-dir "$MODEL_DIR" \
    "$@"

  "$PYTHON_BIN" main.py \
    --task grid \
    --model "$model" \
    --data electricity,exchange_rate,traffic,weather \
    --seq-len 336 \
    --cases NN,NR,RN,RR \
    --pred-lens 96,192,336,720 \
    --output-dir "$OUTPUT_DIR" \
    --model-dir "$MODEL_DIR" \
    "$@"

  "$PYTHON_BIN" main.py \
    --task grid \
    --model "$model" \
    --data illness \
    --seq-len 336 \
    --cases NN,NR,RN,RR \
    --pred-lens 24,36,48,60 \
    --output-dir "$OUTPUT_DIR" \
    --model-dir "$MODEL_DIR" \
    "$@"
done
