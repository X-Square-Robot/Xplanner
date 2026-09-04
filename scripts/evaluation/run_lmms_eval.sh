#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluate an X-Planner checkpoint with lmms-eval.
# Usage: CKPT=/path/to/checkpoint bash scripts/evaluation/run_lmms_eval.sh [tasks] [gpus] [limit]

PYTHON_BIN=${PYTHON_BIN:-python}
CKPT=${CKPT:-}
TASKS=${1:-mmstar}
GPUS=${2:-0}
LIMIT=${3:-}
OUT=${OUT:-eval_results}
MIN_PIXELS=${MIN_PIXELS:-1024}
MAX_PIXELS=${MAX_PIXELS:-589824}

if [[ -z "${CKPT}" ]]; then
    echo "CKPT is required; export CKPT=/path/to/checkpoint" >&2
    exit 64
fi

export CUDA_VISIBLE_DEVICES=${GPUS}
export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}

# Avoid inheriting stale distributed-launch state from a scheduler shell.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
    GROUP_RANK ROLE_RANK ROLE_NAME NODE_RANK NPROC_PER_NODE GROUP_WORLD_SIZE \
    TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS \
    2>/dev/null || true

model_args="pretrained=${CKPT},attn_implementation=sdpa"
model_args="${model_args},max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS}"
common=(
    --model qwen3_5
    --model_args "${model_args}"
    --tasks "${TASKS}"
    --batch_size 1
    --log_samples
    --output_path "${OUT}"
)
extra=()
if [[ -n "${LIMIT}" ]]; then
    extra+=(--limit "${LIMIT}")
fi

gpu_count=$(awk -F, '{print NF}' <<<"${GPUS}")
if (( gpu_count > 1 )); then
    "${PYTHON_BIN}" -m accelerate.commands.launch --num_processes "${gpu_count}" \
        -m lmms_eval eval "${common[@]}" "${extra[@]}"
else
    "${PYTHON_BIN}" -m lmms_eval eval "${common[@]}" "${extra[@]}"
fi
