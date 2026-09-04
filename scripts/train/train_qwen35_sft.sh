#!/usr/bin/env bash
# Qwen3.5-VL supervised fine-tuning for X-Planner event-state data.

set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_PATH=${MODEL_PATH:-}
DATA_CONFIG=${DATA_CONFIG:-${REPO_ROOT}/workspace/example/data/planner_sft.yml}
OUTPUT_DIR=${OUTPUT_DIR:-${REPO_ROOT}/work_dirs/x_planner_sft}
RUN_NAME=${RUN_NAME:-x_planner_qwen35_sft}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${REPO_ROOT}/workspace/example/training/deepspeed_zero1.json}
REPORT_TO=${REPORT_TO:-none}

if [[ -z "${MODEL_PATH}" || ! -e "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH must point to a local Qwen3.5-VL model or checkpoint." >&2
  exit 64
fi
if [[ ! -f "${DATA_CONFIG}" ]]; then
  echo "DATA_CONFIG does not exist: ${DATA_CONFIG}" >&2
  exit 66
fi

NNODES=${WORLD_SIZE:-${1:-1}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${2:-8}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-16667}
NODE_RANK=${RANK:-0}

GLOBAL_ROWS_PER_STEP=${GLOBAL_ROWS_PER_STEP:-256}
LOCAL_BATCH_SIZE=1
GRAD_ACCUM=$((GLOBAL_ROWS_PER_STEP / (NNODES * NPROC_PER_NODE * LOCAL_BATCH_SIZE)))
if (( GRAD_ACCUM < 1 )); then
  echo "GLOBAL_ROWS_PER_STEP is smaller than the distributed world size." >&2
  exit 64
fi

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=${X2ROBOT_AV_SEQUENTIAL_SPAN_MAX:-128}

python -m torch.distributed.run \
  --nnodes "${NNODES}" --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" --master_port "${MASTER_PORT}" \
  --node_rank "${NODE_RANK}" \
  -m x_planner.trainer.launcher \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --data_config "${DATA_CONFIG}" \
  --attn_implementation flash_attention_2 \
  --model_max_length 8192 \
  --image_min_pixels 1024 --image_max_pixels 589824 \
  --bf16 true --tf32 true \
  --per_device_train_batch_size "${LOCAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --num_train_epochs 1 \
  --learning_rate 6e-6 --llm_lr 6e-6 --vision_lr 2e-6 --projector_lr 6e-6 \
  --weight_decay 0.0 --warmup_ratio 0.03 \
  --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
  --gradient_checkpointing true \
  --average_tokens_across_devices true \
  --ignore_data_skip true \
  --save_strategy steps --save_steps 1000 --save_total_limit 2 \
  --logging_steps 1 \
  --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
  --output_dir "${OUTPUT_DIR}/${RUN_NAME}" \
  --report_to "${REPORT_TO}" --run_name "${RUN_NAME}"
