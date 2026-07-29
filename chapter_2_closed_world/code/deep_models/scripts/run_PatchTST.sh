#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_NAME="PatchTST"
MODEL_LR="${MODEL_LR:-0.0001}"
MODEL_EXTRA_ARGS=(--e_layers 1 --d_layers 1 --factor 3 --n_heads 8)

source "$SCRIPT_DIR/run_model_grid.sh"
