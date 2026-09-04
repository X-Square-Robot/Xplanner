#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall_wm_B300
WALL_REPO=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/wall-vlm_B300
DATASET_REPO=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/x2robot_dataset_v2_B300
DATA_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/final_vqa_dedup_train_v4
MODEL_PATH=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/models/Qwen3.5-9B
PYTHON_BIN=${ENV_ROOT}/bin/python
DRIVER=${WALL_REPO}/scripts/train/final_vqa_smoke.py

RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
WORK_DIR=${WORK_DIR:-${WALL_REPO}/work_dirs/final_vqa_smoke/${RUN_STAMP}}
MAX_STEPS=${MAX_STEPS:-30}
EXTENDED_STEPS=${EXTENDED_STEPS:-60}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
MASTER_PORT=${MASTER_PORT:-$((20000 + $$ % 20000))}

source "${ENV_ROOT}/bin/activate"
export PYTHONPATH="${WALL_REPO}:${DATASET_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128
export FINAL_VQA_SMOKE_SKIP_MODEL_SAVE=1

cd "${WALL_REPO}"
mkdir -p "${WORK_DIR}"

echo "[preflight] work_dir=${WORK_DIR}"
echo "[preflight] python=${PYTHON_BIN}"
"${PYTHON_BIN}" -c '
import pathlib
import qwenvl
import x2robot_dataset_v2

expected_wall = pathlib.Path("/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/wall-vlm_B300")
expected_data = pathlib.Path("/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/x2robot_dataset_v2_B300")
actual_wall = pathlib.Path(qwenvl.__file__).resolve()
actual_data = pathlib.Path(x2robot_dataset_v2.__file__).resolve()
print("qwenvl", actual_wall)
print("x2robot_dataset_v2", actual_data)
if expected_wall not in actual_wall.parents:
    raise SystemExit(f"wrong qwenvl source: {actual_wall}")
if expected_data not in actual_data.parents:
    raise SystemExit(f"wrong x2robot_dataset_v2 source: {actual_data}")
'

test -f "${MODEL_PATH}/model.safetensors.index.json"
test -f "${DATA_ROOT}/manifest.json"

echo "[self-test] validating runtime video_frame processor"
"${PYTHON_BIN}" "${DRIVER}" self-test | tee "${WORK_DIR}/self_test.log"

echo "[prepare] selecting and decoding real samples"
"${PYTHON_BIN}" "${DRIVER}" prepare \
    --data-root "${DATA_ROOT}" \
    --model-path "${MODEL_PATH}" \
    --work-dir "${WORK_DIR}" \
    --seed 20260804 \
    --per-scenario 2 \
    --max-scan 512 \
    --max-length "${MODEL_MAX_LENGTH}" \
    | tee "${WORK_DIR}/prepare.log"

choose_gpu() {
    if [[ -n "${GPU_ID:-}" ]]; then
        echo "${GPU_ID}"
        return
    fi
    nvidia-smi \
        --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits \
        | awk -F, '
            {
                for (i = 1; i <= NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
                if (($2 + 0) <= 1024 && ($3 + 0) >= 225280 && ($4 + 0) <= 5) {
                    print $1
                    exit
                }
            }
        '
}

run_training() {
    local steps=$1
    local suffix=$2
    local train_dir="${WORK_DIR}/train_${suffix}"
    local train_log="${WORK_DIR}/train_${suffix}.log"
    local selected_gpu
    selected_gpu=$(choose_gpu)
    if [[ -z "${selected_gpu}" ]]; then
        echo "No idle GPU satisfies memory.used<=1GiB, memory.free>=220GiB, util<=5%." >&2
        return 3
    fi

    echo "[gpu] selected physical GPU ${selected_gpu} for ${steps} steps"
    nvidia-smi \
        --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader \
        | tee "${WORK_DIR}/gpu_before_${suffix}.txt"

    CUDA_VISIBLE_DEVICES="${selected_gpu}" "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes 1 \
        --nproc_per_node 1 \
        --master_addr 127.0.0.1 \
        --master_port "${MASTER_PORT}" \
        "${DRIVER}" train \
        --model_path "${MODEL_PATH}" \
        --data_config "${WORK_DIR}/data.yml" \
        --attn_implementation flash_attention_2 \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --image_min_pixels 1024 \
        --image_max_pixels 589824 \
        --bf16 true \
        --tf32 true \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --max_steps "${steps}" \
        --num_train_epochs 100 \
        --learning_rate 1e-5 \
        --llm_lr 1e-5 \
        --vision_lr 1e-6 \
        --projector_lr 1e-5 \
        --weight_decay 0.0 \
        --warmup_steps 3 \
        --lr_scheduler_type cosine \
        --gradient_checkpointing true \
        --loss_reduction_scope sample \
        --save_strategy no \
        --logging_steps 1 \
        --dataloader_num_workers 4 \
        --dataloader_prefetch_factor 2 \
        --ignore_data_skip true \
        --seed 42 \
        --data_seed 42 \
        --disable_tqdm true \
        --output_dir "${train_dir}" \
        --report_to none \
        --run_name "final_vqa_smoke_${suffix}" \
        2>&1 | tee "${train_log}"

    "${PYTHON_BIN}" "${DRIVER}" verify \
        --output-dir "${train_dir}" \
        --log-file "${train_log}" \
        --expected-steps "${steps}" \
        --min-drop-fraction 0.5
}

set +e
run_training "${MAX_STEPS}" "${MAX_STEPS}step"
first_status=$?
set -e

if [[ ${first_status} -ne 0 ]]; then
    if [[ ${first_status} -ne 2 ]]; then
        echo "Primary smoke failed before convergence verification (status=${first_status})." >&2
        exit "${first_status}"
    fi
    echo "[verify] ${MAX_STEPS}-step loss drop was insufficient; retrying a clean ${EXTENDED_STEPS}-step run"
    MASTER_PORT=$((MASTER_PORT + 1))
    run_training "${EXTENDED_STEPS}" "${EXTENDED_STEPS}step"
    FINAL_TRAIN_DIR="${WORK_DIR}/train_${EXTENDED_STEPS}step"
    FINAL_TRAIN_LOG="${WORK_DIR}/train_${EXTENDED_STEPS}step.log"
else
    FINAL_TRAIN_DIR="${WORK_DIR}/train_${MAX_STEPS}step"
    FINAL_TRAIN_LOG="${WORK_DIR}/train_${MAX_STEPS}step.log"
fi

echo "FINAL_VQA_SMOKE_PASSED"
echo "WORK_DIR=${WORK_DIR}"
echo "TRAIN_DIR=${FINAL_TRAIN_DIR}"
echo "TRAIN_LOG=${FINAL_TRAIN_LOG}"
