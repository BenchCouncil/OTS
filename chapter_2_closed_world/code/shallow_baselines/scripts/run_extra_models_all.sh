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

if "$PYTHON_BIN" -c "import xgboost" >/dev/null 2>&1; then
  DEFAULT_MODELS="ridge knn xgboost"
else
  DEFAULT_MODELS="ridge knn"
  echo "xgboost is not installed; skipping --model xgboost."
fi

MODELS="${MODELS:-$DEFAULT_MODELS}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE_DIR/results/extra_models}"
MODEL_DIR="${MODEL_DIR:-$WORKSPACE_DIR/checkpoints/extra_models}"

for model in $MODELS; do
  "$PYTHON_BIN" main.py --task grid --model "$model" \
    --data ETTh1,ETTh2,ETTm1,ETTm2 --cases NN,NR,RN,RR \
    --pred-lens 96,192,336,720 --output-dir "$OUTPUT_DIR" --model-dir "$MODEL_DIR" "$@"
  "$PYTHON_BIN" main.py --task grid --model "$model" \
    --data electricity,exchange_rate,traffic,weather --cases NN,NR,RN,RR \
    --pred-lens 96,192,336,720 --output-dir "$OUTPUT_DIR" --model-dir "$MODEL_DIR" "$@"
  "$PYTHON_BIN" main.py --task grid --model "$model" \
    --data illness --cases NN,NR,RN,RR \
    --pred-lens 24,36,48,60 --output-dir "$OUTPUT_DIR" --model-dir "$MODEL_DIR" "$@"

  for case in NN NR RN RR; do
    for pattern in Yearly Quarterly Monthly Weekly Daily Hourly; do
      "$PYTHON_BIN" main.py --task m4 --model "$model" \
        --case "$case" --seasonal-pattern "$pattern" \
        --output-dir "$OUTPUT_DIR" --model-dir "$MODEL_DIR" "$@"
    done
  done
done
