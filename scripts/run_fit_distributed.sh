#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <h5ad_path> <output_prefix> <world_size> <gpu_list_csv> [-- <extra prism fit args...>]"
  exit 1
fi

H5AD_PATH="$1"
OUTPUT_PREFIX="$2"
WORLD_SIZE="$3"
GPU_LIST_CSV="$4"
shift 4

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  EXTRA_ARGS=("$@")
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_LIST_CSV"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "gpu_list_csv 不能为空"
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_PREFIX}.dist_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
SHARD_DIR="${RUN_DIR}/shards"
MERGED_CKPT="${OUTPUT_PREFIX}.merged.ckpt"
mkdir -p "$LOG_DIR" "$SHARD_DIR"

echo "[prism-dist] run dir: $RUN_DIR"
echo "[prism-dist] merged output: $MERGED_CKPT"

PIDS=()
SHARDS=()
COMPLETED=0

cleanup() {
  if [[ $COMPLETED -eq 1 ]]; then
    return
  fi

  if [[ ${#PIDS[@]} -eq 0 ]]; then
    return
  fi

  echo "[prism-dist] cleanup: terminating child processes"
  for PID in "${PIDS[@]}"; do
    if kill -0 "$PID" 2>/dev/null; then
      kill -TERM "$PID" 2>/dev/null || true
    fi
  done

  sleep 2

  for PID in "${PIDS[@]}"; do
    if kill -0 "$PID" 2>/dev/null; then
      kill -KILL "$PID" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

for (( RANK=0; RANK<WORLD_SIZE; RANK++ )); do
  GPU_INDEX=$(( RANK % ${#GPU_LIST[@]} ))
  GPU_ID="${GPU_LIST[$GPU_INDEX]}"
  SHARD_CKPT="${SHARD_DIR}/rank${RANK}.ckpt"
  LOG_PATH="${LOG_DIR}/rank${RANK}.log"
  SHARDS+=("$SHARD_CKPT")
  CMD=(
    env
    CUDA_VISIBLE_DEVICES="$GPU_ID"
    TERM="xterm-256color"
    TTY_COMPATIBLE="1"
    TTY_INTERACTIVE="0"
    prism
    fit
    "$H5AD_PATH"
    "$SHARD_CKPT"
    --rank
    "$RANK"
    --world-size
    "$WORLD_SIZE"
    "${EXTRA_ARGS[@]}"
  )
  printf -v CMD_STR '%q ' "${CMD[@]}"

  echo "[prism-dist] launch rank=${RANK} gpu=${GPU_ID} log=${LOG_PATH}"
  nohup script -qefc "$CMD_STR" "$LOG_PATH" >/dev/null 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    STATUS=1
  fi
done

if [[ $STATUS -ne 0 ]]; then
  echo "[prism-dist] at least one rank failed; skip merge"
  exit $STATUS
fi

echo "[prism-dist] all ranks finished; merging checkpoints"
prism merge-ckpt "${SHARDS[@]}" --output "$MERGED_CKPT"
COMPLETED=1
echo "[prism-dist] done"
