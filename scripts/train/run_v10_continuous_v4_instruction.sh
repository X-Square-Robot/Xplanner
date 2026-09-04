#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall_wm_B300}
WALL_REPO=${WALL_REPO:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/wall-vlm_B300}
DATASET_REPO=${DATASET_REPO:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/x2robot_dataset_v2_B300}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous_v4_instruction}
SOURCE_SNAPSHOT=${SOURCE_SNAPSHOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous_v3_memory/snapshots/bff555ded4007466a5a35a5b}
MODEL_PATH=${MODEL_PATH:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous_v3_memory/runs/memory_v3_schedulerreset_20260811T1100PDT/train_generations/bff555ded4007466a5a35a5b-refresh-schedulerreset_finalrefresh_20260811T1110PDT/checkpoint-370010}
PYTHON_BIN=${PYTHON_BIN:-${ENV_ROOT}/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-6}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}
SMOKE_STEPS=${SMOKE_STEPS:-50}
SMOKE_MAX_BUDGET=${SMOKE_MAX_BUDGET:-100}
BUILD_WORKERS=${BUILD_WORKERS:-144}

source "${ENV_ROOT}/bin/activate"
export PYTHONPATH="${WALL_REPO}:${DATASET_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128
export V10_SKIP_FINAL_MODEL_SAVE=1
cd "${WALL_REPO}"

module() {
    "${PYTHON_BIN}" -m "scripts.train.v10_continuous_v4_instruction.$1" "${@:2}"
}

current_snapshot() {
    if [[ -n "${SNAPSHOT:-}" ]]; then
        echo "${SNAPSHOT}"
        return
    fi
    "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["snapshot"])' \
        "${OUTPUT_ROOT}/current_snapshot.json"
}

choose_gpu() {
    if [[ -n "${GPU_ID:-}" ]]; then
        echo "${GPU_ID}"
        return
    fi
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits | awk -F, '
        {
            for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
            if (($2+0)<=50000 && ($3+0)>=170000 && ($4+0)<=5) {print $1; exit}
        }'
}

prepare_data() {
    local snapshot=$1
    local work_dir=$2
    local tasks=$3
    module dataset_v4 \
        --snapshot "${snapshot}" --work-dir "${work_dir}" \
        --model-path "${MODEL_PATH}" --max-length "${MODEL_MAX_LENGTH}" \
        --tasks "${tasks}" --resize-mode B_auto_near_640 \
        --max-budget "${SMOKE_MAX_BUDGET}" --allow-small
}

train_smoke() {
    local snapshot=$1
    local name=$2
    local tasks=$3
    local work_dir="${SMOKE_ROOT}/${name}"
    local train_dir="${work_dir}/train"
    local report="${work_dir}/gpu_smoke_report.json"
    local gpu
    if [[ -f "${report}" ]] && "${PYTHON_BIN}" -c \
        'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed") else 1)' \
        "${report}"; then
        echo "MEMORY_V4_GPU_SMOKE_REUSE name=${name} report=${report}"
        return
    fi
    gpu=$(choose_gpu)
    if [[ -z "${gpu}" ]]; then
        echo "No safe idle GPU satisfies used<=2GiB, free>=170GB, util<=5%." >&2
        return 3
    fi
    mkdir -p "${train_dir}"
    prepare_data "${snapshot}" "${work_dir}" "${tasks}"
    module data_smoke_v4 \
        --data-config "${work_dir}/data.yml" --model-path "${MODEL_PATH}" \
        --output "${work_dir}/data_smoke_batch${PER_DEVICE_BATCH_SIZE}.json" \
        --batch-size "${PER_DEVICE_BATCH_SIZE}"
    nvidia-smi --id="${gpu}" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits --loop-ms=1000 >"${work_dir}/gpu_monitor.csv" &
    local monitor_pid=$!
    local master_port=$((22000 + $$ % 18000))
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes 1 --nproc_per_node 1 --master_addr 127.0.0.1 --master_port "${master_port}" \
        -m scripts.train.v10_continuous_v4_instruction.train_v4 \
        --snapshot "${snapshot}" --step-state "${work_dir}/v10_step_state.json" \
        --resume-mode exact --model_path "${MODEL_PATH}" \
        --data_config "${work_dir}/data.yml" --attn_implementation sdpa \
        --model_max_length "${MODEL_MAX_LENGTH}" --image_min_pixels 1024 \
        --image_max_pixels 589824 --bf16 true --tf32 true \
        --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient_accumulation_steps 1 --max_steps "${SMOKE_STEPS}" --num_train_epochs 100 \
        --learning_rate 1e-5 --llm_lr 1e-5 --weight_decay 0 --warmup_steps 1 \
        --lr_scheduler_type cosine --gradient_checkpointing true \
        --ddp_find_unused_parameters false --loss_reduction_scope sample \
        --lm_head_loss_only_on_labels false --save_strategy no --logging_steps 1 \
        --dataloader_num_workers "${DATALOADER_WORKERS}" --dataloader_prefetch_factor 2 \
        --ignore_data_skip true --seed 42 --data_seed 42 --disable_tqdm true \
        --output_dir "${train_dir}" --report_to none --run_name "memory_v4_${name}_${RUN_STAMP}" \
        2>&1 | tee "${train_dir}/launch.log"
    local train_status=${PIPESTATUS[0]}
    set -e
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    module train_smoke_report_v4 \
        --log "${train_dir}/launch.log" --monitor "${work_dir}/gpu_monitor.csv" \
        --output "${report}" --task "${tasks}" --resize-mode B_auto_near_640 \
        --required-steps "${SMOKE_STEPS}" --exit-code "${train_status}" \
        --snapshot "${snapshot}" --checkpoint "${MODEL_PATH}" --gpu-id "${gpu}" \
        --batch-size "${PER_DEVICE_BATCH_SIZE}"
}

run_smokes() {
    local snapshot
    snapshot=$(current_snapshot)
    SMOKE_ROOT=${SMOKE_ROOT:-${OUTPUT_ROOT}/smoke/${RUN_STAMP}}
    export SMOKE_ROOT
    mkdir -p "${SMOKE_ROOT}"
    module convert_snapshot_v4 validate --snapshot "${snapshot}" --samples-per-dataset 100
    train_smoke "${snapshot}" continuous continuous
    train_smoke "${snapshot}" initial_plan initial_plan
    train_smoke "${snapshot}" terminal terminal
    train_smoke "${snapshot}" mixed continuous,initial_plan,terminal
    echo "MEMORY_V4_SMOKE_PASSED root=${SMOKE_ROOT}"
}

command_name=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "${command_name}" in
    preflight)
        module convert_snapshot_v4 preflight --source-snapshot "${SOURCE_SNAPSHOT}"
        ;;
    build-json)
        module convert_snapshot_v4 build --source-snapshot "${SOURCE_SNAPSHOT}" \
            --output-root "${OUTPUT_ROOT}" --workers "${BUILD_WORKERS}" "$@"
        ;;
    publish)
        module convert_snapshot_v4 publish --source-snapshot "${SOURCE_SNAPSHOT}" \
            --output-root "${OUTPUT_ROOT}" "$@"
        ;;
    finalize)
        module convert_snapshot_v4 publish --source-snapshot "${SOURCE_SNAPSHOT}" \
            --output-root "${OUTPUT_ROOT}"
        module convert_snapshot_v4 validate --snapshot "$(current_snapshot)" \
            --samples-per-dataset 100
        ;;
    gallery)
        module gallery_v4 --input-root "$(current_snapshot)" \
            --output-root /mnt/cpfs/zbl-cpfs-new/USERS/luhao/APlan/0811/memory_v4/examples_prompt_v2 \
            --samples-per-cell 3 "$@"
        ;;
    smoke)
        run_smokes
        ;;
    smoke-task)
        name=${1:?missing smoke name}
        tasks=${2:?missing comma-separated tasks}
        SMOKE_ROOT=${SMOKE_ROOT:-${OUTPUT_ROOT}/smoke/${RUN_STAMP}}
        export SMOKE_ROOT
        mkdir -p "${SMOKE_ROOT}"
        train_smoke "$(current_snapshot)" "${name}" "${tasks}"
        ;;
    train)
        echo "Formal V4 training is intentionally disabled in this implementation cycle; only bounded smoke is authorized." >&2
        exit 4
        ;;
    help|-h|--help)
        echo "Usage: $0 {preflight|build-json|publish|finalize|gallery|smoke}"
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        exit 2
        ;;
esac
