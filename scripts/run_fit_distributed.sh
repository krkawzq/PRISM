#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <h5ad_path> <output_prefix> <world_size> <gpu_list_csv> <reference_genes.txt> <fit_genes.txt> [-- <extra prism fit priors args...>]"
  exit 1
fi

H5AD_PATH="$1"
OUTPUT_PREFIX="$2"
WORLD_SIZE="$3"
GPU_LIST_CSV="$4"
REFERENCE_GENES="$5"
FIT_GENES="$6"
shift 6

EXTRA_ARGS=()
if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  EXTRA_ARGS=("$@")
fi

IFS=',' read -r -a GPU_LIST <<< "$GPU_LIST_CSV"
if [[ ${#GPU_LIST[@]} -eq 0 ]]; then
  echo "gpu_list_csv cannot be empty"
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTPUT_PREFIX}.dist_${TIMESTAMP}"
LOG_DIR="${RUN_DIR}/logs"
SHARD_DIR="${RUN_DIR}/shards"
MERGED_CKPT="${OUTPUT_PREFIX}.merged.pkl"
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
  for PID in "${PIDS[@]:-}"; do
    if kill -0 "$PID" 2>/dev/null; then
      kill -TERM "$PID" 2>/dev/null || true
    fi
  done
}

trap cleanup EXIT INT TERM

for (( RANK=0; RANK<WORLD_SIZE; RANK++ )); do
  GPU_INDEX=$(( RANK % ${#GPU_LIST[@]} ))
  GPU_ID="${GPU_LIST[$GPU_INDEX]}"
  SHARD_CKPT="${SHARD_DIR}/rank${RANK}.pkl"
  LOG_PATH="${LOG_DIR}/rank${RANK}.log"
  SHARDS+=("$SHARD_CKPT")
  CMD=(
    env
    CUDA_VISIBLE_DEVICES="$GPU_ID"
    prism
    fit
    priors
    "$H5AD_PATH"
    --output "$SHARD_CKPT"
    --reference-genes "$REFERENCE_GENES"
    --fit-genes "$FIT_GENES"
    --shard "${RANK}/${WORLD_SIZE}"
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
prism checkpoint merge "${SHARDS[@]}" --output "$MERGED_CKPT"
COMPLETED=1
echo "[prism-dist] done"
