#!/usr/bin/env bash
set -euo pipefail

if [ -z "${MODEL_NAME:-}" ]; then
  echo "MODEL_NAME must be set by a model wrapper script." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [ "$(basename "$(dirname "$REPO_DIR")")" = "code" ] && [ "$(basename "$(dirname "$(dirname "$REPO_DIR")")")" = "chapter_2_closed_world" ]; then
  DEFAULT_DATA_ROOT="$(cd "$REPO_DIR/../../.." && pwd)/shared/datasets"
else
  DEFAULT_DATA_ROOT="$(cd "$REPO_DIR/.." && pwd)/dataset"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "$HOME/miniconda3/bin/python" ]; then
    PYTHON_BIN="$HOME/miniconda3/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi
DATA_ROOT="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-./checkpoints/reversal_experiments}"
DATASETS="${DATASETS:-ETTh1 ETTh2 ETTm1 ETTm2 electricity exchange_rate weather illness}"
CASES="${CASES:-NN RN NR RR}"
LEARNING_RATES="${LEARNING_RATES:-${MODEL_LR:-0.0001}}"
SEQ_LENS="${SEQ_LENS:-96 336}"
PRED_LENS="${PRED_LENS:-96 336}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ITR="${ITR:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNS="${MAX_RUNS:-0}"
SHARD_INDEX="${SHARD_INDEX:-0}"
SHARD_COUNT="${SHARD_COUNT:-1}"
USE_GPU="${USE_GPU:-0}"
GPU_TYPE="${GPU_TYPE:-cuda}"
GPU="${GPU:-0}"
LOCAL_GPU="${LOCAL_GPU:-0}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
REVERSAL_MASK_ROOT="${REVERSAL_MASK_ROOT:-}"
CHANNEL_RESULTS_ROOT="${CHANNEL_RESULTS_ROOT:-}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/tsl_matplotlib}"
export MPLCONFIGDIR
mkdir -p "$MPLCONFIGDIR"

RUN_COUNT=0
CANDIDATE_COUNT=0
MODEL_LABEL_LEN="${MODEL_LABEL_LEN:-}"
MODEL_EXTRA_ARGS=("${MODEL_EXTRA_ARGS[@]}")

dataset_config() {
  local dataset="$1"
  case "$dataset" in
    ETTh1)
      ROOT_PATH="$DATA_ROOT/ETT-small"
      DATA_PATH="ETTh1.csv"
      TSL_DATA="ETTh1"
      CHANNELS=7
      FREQ="h"
      ;;
    ETTh2)
      ROOT_PATH="$DATA_ROOT/ETT-small"
      DATA_PATH="ETTh2.csv"
      TSL_DATA="ETTh2"
      CHANNELS=7
      FREQ="h"
      ;;
    ETTm1)
      ROOT_PATH="$DATA_ROOT/ETT-small"
      DATA_PATH="ETTm1.csv"
      TSL_DATA="ETTm1"
      CHANNELS=7
      FREQ="t"
      ;;
    ETTm2)
      ROOT_PATH="$DATA_ROOT/ETT-small"
      DATA_PATH="ETTm2.csv"
      TSL_DATA="ETTm2"
      CHANNELS=7
      FREQ="t"
      ;;
    electricity)
      ROOT_PATH="$DATA_ROOT/electricity"
      DATA_PATH="electricity.csv"
      TSL_DATA="custom"
      CHANNELS=321
      FREQ="h"
      ;;
    exchange_rate)
      ROOT_PATH="$DATA_ROOT/exchange_rate"
      DATA_PATH="exchange_rate.csv"
      TSL_DATA="custom"
      CHANNELS=8
      FREQ="d"
      ;;
    weather)
      ROOT_PATH="$DATA_ROOT/weather"
      DATA_PATH="weather.csv"
      TSL_DATA="custom"
      CHANNELS=21
      FREQ="t"
      ;;
    illness)
      ROOT_PATH="$DATA_ROOT/illness"
      DATA_PATH="national_illness.csv"
      TSL_DATA="custom"
      CHANNELS=7
      FREQ="w"
      ;;
    *)
      echo "Unknown dataset: $dataset" >&2
      exit 2
      ;;
  esac
}

print_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_one() {
  local dataset="$1"
  local case_name="$2"
  local lr="$3"
  local seq_len="$4"
  local pred_len="$5"
  local label_len="$((seq_len / 2))"
  local lr_tag="${lr//./p}"
  lr_tag="${lr_tag//-/m}"
  shift 5

  if [ -n "$MODEL_LABEL_LEN" ]; then
    label_len="$MODEL_LABEL_LEN"
  fi

  local tag_suffix=""
  if [ -n "$EXPERIMENT_TAG" ]; then
    tag_suffix="_${EXPERIMENT_TAG}"
  fi
  local model_id="${MODEL_NAME}_${dataset}_${case_name}_lr${lr_tag}_s${seq_len}_p${pred_len}${tag_suffix}"
  local des="rev_${case_name}_lr_${lr_tag}${tag_suffix}"
  local channel_result=""
  local lock_fd=""
  if [ -n "$CHANNEL_RESULTS_ROOT" ]; then
    channel_result="$CHANNEL_RESULTS_ROOT/long_term/$MODEL_NAME/$dataset/$case_name/seq${seq_len}_pred${pred_len}/channel_metrics.csv"
    local lock_file="$CHANNEL_RESULTS_ROOT/.locks/long_term/$MODEL_NAME/$dataset/$case_name/seq${seq_len}_pred${pred_len}.lock"
    mkdir -p "$(dirname "$lock_file")"
    exec {lock_fd}>"$lock_file"
    if ! flock -n "$lock_fd"; then
      echo "Skipping locked result: $MODEL_NAME/$dataset/$case_name/seq${seq_len}_pred${pred_len}"
      exec {lock_fd}>&-
      return
    fi
  fi

  if [ "$SKIP_EXISTING" = "1" ]; then
    if [ -n "$channel_result" ] && [ -s "$channel_result" ]; then
      echo "Skipping existing channel result: $channel_result"
      if [ -n "$lock_fd" ]; then exec {lock_fd}>&-; fi
      return
    fi
    shopt -s nullglob
    local existing_metrics=("$REPO_DIR"/results/*"${model_id}"*"${des}"_0/metrics.npy)
    shopt -u nullglob
    if [ "${#existing_metrics[@]}" -gt 0 ]; then
      echo "Skipping existing result: $model_id"
      if [ -n "$lock_fd" ]; then exec {lock_fd}>&-; fi
      return
    fi
  fi

  local cmd=(
    "$PYTHON_BIN" -u run.py
    --task_name long_term_forecast
    --is_training 1
    --root_path "$ROOT_PATH"
    --data_path "$DATA_PATH"
    --model_id "$model_id"
    --model "$MODEL_NAME"
    --data "$TSL_DATA"
    --features M
    --target OT
    --freq "$FREQ"
    --seq_len "$seq_len"
    --label_len "$label_len"
    --pred_len "$pred_len"
    --enc_in "$CHANNELS"
    --dec_in "$CHANNELS"
    --c_out "$CHANNELS"
    --learning_rate "$lr"
    --train_epochs "$TRAIN_EPOCHS"
    --patience "$PATIENCE"
    --batch_size "$BATCH_SIZE"
    --num_workers "$NUM_WORKERS"
    --checkpoints "$CHECKPOINT_ROOT"
    --reversal_case "$case_name"
    --des "$des"
    --itr "$ITR"
  )

  if [ -n "$REVERSAL_MASK_ROOT" ]; then
    cmd+=(--reversal_mask_path "$REVERSAL_MASK_ROOT/$dataset.json")
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

  cmd+=("${MODEL_EXTRA_ARGS[@]}")
  cmd+=("$@")

  RUN_COUNT=$((RUN_COUNT + 1))
  echo "[$RUN_COUNT] $MODEL_NAME dataset=$dataset case=$case_name lr=$lr seq_len=$seq_len pred_len=$pred_len"
  if [ "$DRY_RUN" = "1" ]; then
    print_command "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
  if [ -n "$lock_fd" ]; then exec {lock_fd}>&-; fi
}

main() {
  cd "$REPO_DIR"

  local dataset
  local case_name
  local lr
  local seq_len
  local pred_len

  for dataset in ${DATASETS//,/ }; do
    dataset_config "$dataset"
    for case_name in ${CASES//,/ }; do
      for lr in ${LEARNING_RATES//,/ }; do
        for seq_len in ${SEQ_LENS//,/ }; do
          for pred_len in ${PRED_LENS//,/ }; do
            if [ "$dataset" = "illness" ] && [ "$pred_len" = "336" ]; then
              echo "Skipping invalid illness combo: seq_len=$seq_len pred_len=$pred_len"
              continue
            fi
            shard_mod=$((CANDIDATE_COUNT % SHARD_COUNT))
            CANDIDATE_COUNT=$((CANDIDATE_COUNT + 1))
            if [ "$shard_mod" != "$SHARD_INDEX" ]; then
              continue
            fi
            if [ "$MAX_RUNS" != "0" ] && [ "$RUN_COUNT" -ge "$MAX_RUNS" ]; then
              echo "Reached MAX_RUNS=$MAX_RUNS; stopping."
              return
            fi
            run_one "$dataset" "$case_name" "$lr" "$seq_len" "$pred_len" "$@"
          done
        done
      done
    done
  done
}

main "$@"
