#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="TimesNet"
MODEL_LR="${MODEL_LR:-0.0001}"
MODEL_EXTRA_ARGS=(--e_layers 2 --d_layers 1 --factor 3 --d_model 16 --d_ff 32 --top_k 5)

source "$SCRIPT_DIR/run_model_grid.sh"
