#!/usr/bin/env bash
set -uo pipefail

ROOT="${CEMA_MLP_ROOT:-$PWD}"
PY="${CEMA_PYTHON:-/home/user/anaconda3/envs/torch_env/bin/python}"
GPU="${CEMA_GPU:-0}"
SEEDS="${CEMA_SEEDS:-0,1,2}"
EPOCHS="${CEMA_EPOCHS:-200}"
BATCH_SIZE="${CEMA_BATCH_SIZE:-2048}"

export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$ROOT/Scripts/run_ocvstart_full280_fast.py" \
  --base-dir "$ROOT" \
  --raw-root nmc_ocvstart_lopo_clean \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE"
