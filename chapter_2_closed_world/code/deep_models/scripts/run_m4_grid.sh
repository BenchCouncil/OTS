#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_DATA_ROOT="$(cd "$REPO_DIR/.." && pwd)/dataset"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$HOME/miniconda3/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda3/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/m4_reversal_experiments}"
MODELS="${MODELS:-DLinear GRU PatchTST TimeFilter TimeMixer TimesNet iTransformer}"
SEASONAL_PATTERNS="${SEASONAL_PATTERNS:-Yearly Quarterly Monthly Weekly Daily Hourly}"
CASES="${CASES:-NN RN NR RR}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
USE_GPU="${USE_GPU:-0}"
GPU_TYPE="${GPU_TYPE:-cuda}"
GPU="${GPU:-0}"
LOCAL_GPU="${LOCAL_GPU:-0}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
REVERSAL_MASK_ROOT="${REVERSAL_MASK_ROOT:-}"
CHANNEL_RESULTS_ROOT="${CHANNEL_RESULTS_ROOT:-}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/tsl_matplotlib}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

RUN_COUNT=0
CANDIDATE_COUNT=0

model_lr() {
  case "$1" in
    DLinear) echo "${DLINEAR_LR:-0.005}" ;;
    GRU) echo "${GRU_LR:-0.001}" ;;
    TimeMixer) echo "${TIMEMIXER_LR:-0.01}" ;;
    *) echo "${MODEL_LR:-0.0001}" ;;
  esac
}

model_extra_args() {
  case "$1" in
    DLinear) echo "--e_layers 2 --d_layers 1 --factor 3" ;;
    GRU) echo "--e_layers 2 --d_model 128 --d_ff 128 --dropout 0.1" ;;
    PatchTST) echo "--e_layers 1 --d_layers 1 --factor 3 --n_heads 8" ;;
    TimeFilter) echo "--e_layers 2 --d_model 64 --d_ff 128 --n_heads 8 --patch_len 4 --alpha 0.1 --top_p 0.5 --pos 1" ;;
    TimeMixer) echo "--e_layers 2 --d_model 16 --d_ff 32 --down_sampling_layers 3 --down_sampling_method avg --down_sampling_window 2" ;;
    TimesNet) echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 16 --d_ff 32 --top_k 5" ;;
    iTransformer) echo "--e_layers 2 --d_layers 1 --factor 3 --d_model 128 --d_ff 128" ;;
    *) echo "" ;;
  esac
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_one() {
  local model="$1"
  local pattern="$2"
  local case_name="$3"
  local lr
  local lr_tag
  local model_id
  local des
  local result_key
  local result_file
  lr="$(model_lr "$model")"
  lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/m}"
  model_id="${model}_M4_${pattern}_${case_name}_lr${lr_tag}"
  des="m4_${pattern}_${case_name}_lr_${lr_tag}"
  result_key="${model}_${case_name}_lr${lr_tag}"
  if [ -n "$EXPERIMENT_TAG" ]; then
    result_key="${result_key}_${EXPERIMENT_TAG}"
    model_id="${model_id}_${EXPERIMENT_TAG}"
    des="${des}_${EXPERIMENT_TAG}"
  fi
  result_file="$REPO_DIR/m4_results/$result_key/${pattern}_forecast.csv"

  if [ "$SKIP_EXISTING" = "1" ] && [ -s "$result_file" ]; then
    echo "Skipping existing M4 forecast: $result_key/$pattern"
    return
  fi

  local cmd=(
    "$PYTHON_BIN" -u run.py
    --task_name short_term_forecast
    --is_training 1
    --root_path "$DATA_ROOT/m4"
    --seasonal_patterns "$pattern"
    --model_id "$model_id"
    --model "$model"
    --data m4
    --features M
    --seq_len 1
    --label_len 1
    --pred_len 1
    --enc_in 1
    --dec_in 1
    --c_out 1
    --learning_rate "$lr"
    --train_epochs "$TRAIN_EPOCHS"
    --patience "$PATIENCE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --checkpoints "$CHECKPOINT_ROOT"
    --reversal_case "$case_name"
    --des "$des"
    --m4_result_dir "$result_key"
    --loss SMAPE
  )

  if [ -n "$REVERSAL_MASK_ROOT" ]; then
    cmd+=(--reversal_mask_path "$REVERSAL_MASK_ROOT/m4.json")
  fi
  if [ -n "$CHANNEL_RESULTS_ROOT" ]; then
    cmd+=(--channel_results_root "$CHANNEL_RESULTS_ROOT" --skip_visualization)
  fi

  if [ "$USE_GPU" = "1" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU}"
    cmd+=(--use_gpu --gpu_type "$GPU_TYPE" --gpu "$LOCAL_GPU")
  else
    cmd+=(--no_use_gpu)
  fi

  # shellcheck disable=SC2206
  local extra_args=($(model_extra_args "$model"))
  cmd+=("${extra_args[@]}")

  RUN_COUNT=$((RUN_COUNT + 1))
  echo "[$RUN_COUNT] $model M4 pattern=$pattern case=$case_name lr=$lr"
  if [ "$DRY_RUN" = "1" ]; then
    print_command "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

main() {
  cd "$REPO_DIR"
  local model
  local pattern
  local case_name
  local shard_mod
  for model in ${MODELS//,/ }; do
    for pattern in ${SEASONAL_PATTERNS//,/ }; do
      for case_name in ${CASES//,/ }; do
        shard_mod=$((CANDIDATE_COUNT % SHARD_COUNT))
        CANDIDATE_COUNT=$((CANDIDATE_COUNT + 1))
        if [ "$shard_mod" != "$SHARD_INDEX" ]; then
          continue
        fi
        if [ "$MAX_RUNS" != "0" ] && [ "$RUN_COUNT" -ge "$MAX_RUNS" ]; then
          echo "Reached MAX_RUNS=$MAX_RUNS; stopping."
          return
        fi
        run_one "$model" "$pattern" "$case_name"
      done
    done
  done
}

main "$@"
