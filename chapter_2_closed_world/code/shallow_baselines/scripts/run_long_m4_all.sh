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

seq_len_for_pattern() {
  case "$1" in
    Yearly) echo 27 ;;
    Quarterly) echo 36 ;;
    Monthly) echo 81 ;;
    Weekly) echo 390 ;;
    Daily) echo 420 ;;
    Hourly) echo 480 ;;
    *) echo "Unknown M4 seasonal pattern: $1" >&2; exit 1 ;;
  esac
}

for model in ols revin_ols; do
  for case_name in NN NR RN RR; do
    for pattern in Yearly Quarterly Monthly Weekly Daily Hourly; do
      seq_len="$(seq_len_for_pattern "$pattern")"
      "$PYTHON_BIN" main.py \
        --task m4 \
        --model "$model" \
        --case "$case_name" \
        --seasonal-pattern "$pattern" \
        --seq-len "$seq_len" \
        --output-dir "$OUTPUT_DIR" \
        --model-dir "$MODEL_DIR" \
        "$@"
    done
  done
done
