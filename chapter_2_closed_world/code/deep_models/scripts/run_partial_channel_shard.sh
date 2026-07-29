#!/usr/bin/env bash
set -euo pipefail

SHARD_INDEX="${1:?usage: run_partial_channel_shard.sh SHARD_INDEX SHARD_COUNT GPU}"
SHARD_COUNT="${2:?usage: run_partial_channel_shard.sh SHARD_INDEX SHARD_COUNT GPU}"
GPU="${3:?usage: run_partial_channel_shard.sh SHARD_INDEX SHARD_COUNT GPU}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$(cd "$REPO_DIR/.." && pwd)/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/partial_channel_results}"
MASK_ROOT="${MASK_ROOT:-$OUTPUT_ROOT/masks}"
CHECKPOINT_BASE="${CHECKPOINT_BASE:-/root/autodl-tmp/tsl_outputs/checkpoints_partial_half}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-partial_half_seed2021}"
MODELS="${MODELS:-DLinear iTransformer PatchTST TimesNet GRU}"

export DATA_ROOT
export DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate weather illness}"
export CASES="${CASES:-NN RN NR RR}"
export SEQ_LENS="${SEQ_LENS:-96 336}"
export PRED_LENS="${PRED_LENS:-96 336}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
export PATIENCE="${PATIENCE:-3}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export USE_GPU=1
export LOCAL_GPU=0
export GPU
export SHARD_INDEX
export SHARD_COUNT
export SKIP_EXISTING="${SKIP_EXISTING:-1}"
export REVERSAL_MASK_ROOT="$MASK_ROOT"
export CHANNEL_RESULTS_ROOT="$OUTPUT_ROOT"
export EXPERIMENT_TAG
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-12}"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"

mkdir -p "$OUTPUT_ROOT/logs" "$CHECKPOINT_BASE"

for model in ${MODELS//,/ }; do
  export CHECKPOINT_ROOT="$CHECKPOINT_BASE/long_term/$model"
  bash "$SCRIPT_DIR/run_${model}.sh"
done
