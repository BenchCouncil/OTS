#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="TimeFilter"
MODEL_LR="${MODEL_LR:-0.0001}"
MODEL_EXTRA_ARGS=(
  --e_layers 2
  --d_model 64
  --d_ff 128
  --n_heads 8
  --patch_len 16
  --alpha 0.1
  --top_p 0.5
  --pos 1
)

source "$SCRIPT_DIR/run_model_grid.sh"
