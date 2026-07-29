#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="TimeMixer"
MODEL_LR="${MODEL_LR:-0.01}"
MODEL_LABEL_LEN=0
MODEL_EXTRA_ARGS=(
  --e_layers 2
  --d_model 16
  --d_ff 32
  --down_sampling_layers 3
  --down_sampling_method avg
  --down_sampling_window 2
)

source "$SCRIPT_DIR/run_model_grid.sh"
