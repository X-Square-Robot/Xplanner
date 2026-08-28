#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall_wm_B300
WALL_REPO=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/wall-vlm_B300
DATASET_REPO=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/x2robot_dataset_v2_B300
MODEL_PATH=${MODEL_PATH:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/models/Qwen3.5-9B}
PYTHON_BIN=${ENV_ROOT}/bin/python
SCAN_OUTPUT=${SCAN_OUTPUT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/scans}
RUNS_ROOT=${RUNS_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/runs}
SMOKE_SCAN_ROOT=${SMOKE_SCAN_ROOT:-/tmp/v10_continuous_scan_smoke/scan-143a590a0c1f}
SMOKE_VERSION=${SMOKE_VERSION:-smoke-v2}
RUN_STAMP=${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
MASTER_PORT=${MASTER_PORT:-$((20000 + $$ % 20000))}
# FlashAttention is unstable on this B30Z/CC10.3 stack for the V10 multi-image
# sequence (NaN with GC, illegal memory access without GC). SDPA is finite.
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
BF16=${BF16:-true}
TF32=${TF32:-true}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}
LM_HEAD_LOSS_ONLY_ON_LABELS=${LM_HEAD_LOSS_ONLY_ON_LABELS:-false}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
LLM_LR=${LLM_LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
WARMUP_STEPS=${WARMUP_STEPS:-3}
LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-100}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}
DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-false}

source "${ENV_ROOT}/bin/activate"
export PYTHONPATH="${WALL_REPO}:${DATASET_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128
export V10_SKIP_FINAL_MODEL_SAVE=1
cd "${WALL_REPO}"

latest_scan_root() {
    find "${SCAN_OUTPUT}" -maxdepth 1 -type d -name 'scan-*' | sort | tail -1
}

latest_early_snapshot() {
    local scan_root
    scan_root=$(latest_scan_root)
    if [[ -z "${scan_root}" ]]; then
        return 1
    fi
    find "${scan_root}/training_snapshots" -maxdepth 1 -type d -name 'early-*' | sort | tail -1
}

latest_catalog_snapshot() {
    local scan_root=${1:-$(latest_scan_root)}
    if [[ -z "${scan_root}" ]]; then
        return 1
    fi
    find "${scan_root}/catalog_snapshots" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' | sort | tail -1
}

choose_gpu() {
    if [[ -n "${GPU_IDS:-}" ]]; then
        echo "${GPU_IDS}"
        return
    fi
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
                if (($2 + 0) <= 2048 && ($3 + 0) >= 220000 && ($4 + 0) <= 10) {
                    print $1
                    exit
                }
            }
        '
}

prepare_snapshot() {
    local snapshot=$1
    local work_dir=$2
    local small=${3:-false}
    mkdir -p "${work_dir}"
    local extra=()
    if [[ "${small}" == true ]]; then
        extra+=(--allow-small)
    fi
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.prepare_training \
        --snapshot "${snapshot}" \
        --work-dir "${work_dir}" \
        --model-path "${MODEL_PATH}" \
        --max-length "${MODEL_MAX_LENGTH}" \
        "${extra[@]}"
}

launch_train() {
    local snapshot=$1
    local work_dir=$2
    local output_dir=$3
    local max_steps=$4
    local save_steps=$5
    local run_name=$6
    local resume_mode=${7:-exact}
    local selected_gpus
    selected_gpus=$(choose_gpu)
    if [[ -z "${selected_gpus}" ]]; then
        echo "No idle GPU satisfies memory.used<=2GiB, memory.free>=220GB, util<=10%." >&2
        return 3
    fi
    local nproc_per_node
    nproc_per_node=$(awk -F, '{print NF}' <<<"${selected_gpus}")
    mkdir -p "${output_dir}"
    local log_file="${output_dir}/launch.log"
    echo "[v10] gpus=${selected_gpus} nproc=${nproc_per_node} batch_per_gpu=${PER_DEVICE_BATCH_SIZE} grad_accum=${GRADIENT_ACCUMULATION_STEPS} snapshot=${snapshot} max_steps=${max_steps} output=${output_dir}"
    CUDA_VISIBLE_DEVICES="${selected_gpus}" "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes 1 \
        --nproc_per_node "${nproc_per_node}" \
        --master_addr 127.0.0.1 \
        --master_port "${MASTER_PORT}" \
        -m scripts.train.v10_continuous.train_v10 \
        --snapshot "${snapshot}" \
        --step-state "${work_dir}/v10_step_state.json" \
        --resume-mode "${resume_mode}" \
        --model_path "${MODEL_PATH}" \
        --data_config "${work_dir}/data.yml" \
        --attn_implementation "${ATTN_IMPLEMENTATION}" \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --image_min_pixels 1024 \
        --image_max_pixels 589824 \
        --bf16 "${BF16}" \
        --tf32 "${TF32}" \
        --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --max_steps "${max_steps}" \
        --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
        --learning_rate "${LEARNING_RATE}" \
        --llm_lr "${LLM_LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --warmup_steps "${WARMUP_STEPS}" \
        --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
        --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
        --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS}" \
        --loss_reduction_scope sample \
        --lm_head_loss_only_on_labels "${LM_HEAD_LOSS_ONLY_ON_LABELS}" \
        --save_strategy steps \
        --save_steps "${save_steps}" \
        --save_total_limit 3 \
        --logging_steps 1 \
        --dataloader_num_workers "${DATALOADER_WORKERS}" \
        --dataloader_prefetch_factor 2 \
        --ignore_data_skip true \
        --seed 42 \
        --data_seed 42 \
        --disable_tqdm true \
        --output_dir "${output_dir}" \
        --report_to none \
        --run_name "${run_name}" \
        2>&1 | tee -a "${log_file}"
}

publish_smoke() {
    local snapshot="${SMOKE_SCAN_ROOT}/training_snapshots/${SMOKE_VERSION}"
    if [[ ! -d "${snapshot}" ]]; then
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.build_v10_snapshot \
            --run-root "${SMOKE_SCAN_ROOT}" \
            --version "${SMOKE_VERSION}"
    fi
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.validate_schema --snapshot "${snapshot}"
    echo "${snapshot}"
}

smoke() {
    local snapshot
    snapshot=$(publish_smoke | tail -1)
    local work_dir=${WORK_DIR:-${RUNS_ROOT}/smoke_${RUN_STAMP}}
    local train_dir="${work_dir}/train"
    prepare_snapshot "${snapshot}" "${work_dir}" true
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.smoke_data \
        --data-config "${work_dir}/data.yml" \
        --model-path "${MODEL_PATH}" \
        --output "${work_dir}/data_smoke.json" \
        2>&1 | tee "${work_dir}/data_smoke.log"
    launch_train "${snapshot}" "${work_dir}" "${train_dir}" 3 3 v10_smoke_3
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.verify_checkpoint \
        --output-dir "${train_dir}" --min-step 3 --log-file "${train_dir}/launch.log"
    MASTER_PORT=$((MASTER_PORT + 1))
    launch_train "${snapshot}" "${work_dir}" "${train_dir}" 5 5 v10_smoke_resume_5
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.verify_checkpoint \
        --output-dir "${train_dir}" --min-step 5 --log-file "${train_dir}/launch.log"
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.validate_teacher \
        --snapshot "${snapshot}" --split train --limit 12 --oracle \
        --output "${work_dir}/teacher_oracle.jsonl"
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.validate_rollout \
        --snapshot "${snapshot}" --split train --limit 12 --oracle \
        --output "${work_dir}/rollout_oracle.jsonl"
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.compare_validation \
        --teacher "${work_dir}/teacher_oracle.metrics.json" \
        --rollout "${work_dir}/rollout_oracle.metrics.json" \
        --output "${work_dir}/consistency_oracle.json"
    echo "V10_SMOKE_AND_RESUME_PASSED work_dir=${work_dir}"
}

experiment() {
    local snapshot
    snapshot=$(publish_smoke | tail -1)
    local work_dir=${WORK_DIR:-${RUNS_ROOT}/experiment_${RUN_STAMP}}
    local train_dir="${work_dir}/train"
    local steps=${EXPERIMENT_STEPS:-2}
    prepare_snapshot "${snapshot}" "${work_dir}" true
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.smoke_data \
        --data-config "${work_dir}/data.yml" \
        --model-path "${MODEL_PATH}" \
        --output "${work_dir}/data_smoke.json" \
        2>&1 | tee "${work_dir}/data_smoke.log"
    launch_train "${snapshot}" "${work_dir}" "${train_dir}" \
        "${steps}" "${steps}" "v10_experiment_${RUN_STAMP}"
}

formal() {
    local snapshot=${SNAPSHOT:-$(latest_early_snapshot)}
    if [[ -z "${snapshot}" || ! -d "${snapshot}" ]]; then
        echo "No early snapshot with >=100 train, >=10 validation and >=2 Profiles yet." >&2
        return 4
    fi
    local work_dir=${WORK_DIR:-${RUNS_ROOT}/formal_${RUN_STAMP}}
    local train_dir="${work_dir}/train"
    local max_steps=${MAX_STEPS:-200}
    local save_steps=${SAVE_STEPS:-20}
    local run_name=${RUN_NAME:-v10_formal_${max_steps}}
    prepare_snapshot "${snapshot}" "${work_dir}" false
    launch_train "${snapshot}" "${work_dir}" "${train_dir}" "${max_steps}" "${save_steps}" "${run_name}"
    "${PYTHON_BIN}" -m scripts.train.v10_continuous.verify_checkpoint \
        --output-dir "${train_dir}" --min-step "${max_steps}" --log-file "${train_dir}/launch.log"
}

resume_existing() {
    local snapshot=${SNAPSHOT:?SNAPSHOT is required for resume-existing}
    local work_dir=${WORK_DIR:?WORK_DIR is required for resume-existing}
    local train_dir=${TRAIN_DIR:-${work_dir}/train}
    local max_steps=${MAX_STEPS:-10000}
    local save_steps=${SAVE_STEPS:-500}
    local run_name=${RUN_NAME:-v10_resume_${max_steps}}
    if [[ ! -f "${snapshot}/manifest.json" ]]; then
        echo "Snapshot manifest missing: ${snapshot}/manifest.json" >&2
        return 4
    fi
    if [[ ! -f "${work_dir}/data.yml" ]]; then
        echo "Existing V10 data config missing: ${work_dir}/data.yml" >&2
        return 4
    fi
    if ! find "${train_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
        echo "No checkpoint exists under ${train_dir}." >&2
        return 4
    fi
    launch_train \
        "${snapshot}" "${work_dir}" "${train_dir}" \
        "${max_steps}" "${save_steps}" "${run_name}" exact
}

resume_refreshed() {
    local snapshot=${SNAPSHOT:?SNAPSHOT is required for resume-refreshed}
    local work_dir=${WORK_DIR:?WORK_DIR is required for resume-refreshed}
    local train_dir=${TRAIN_DIR:-${work_dir}/train}
    local max_steps=${MAX_STEPS:-20000}
    local save_steps=${SAVE_STEPS:-500}
    local run_name=${RUN_NAME:-v10_refresh_data_${max_steps}}
    if [[ ! -f "${snapshot}/manifest.json" ]]; then
        echo "Snapshot manifest missing: ${snapshot}/manifest.json" >&2
        return 4
    fi
    if [[ ! -f "${work_dir}/data.yml" ]]; then
        echo "Refreshed V10 data config missing: ${work_dir}/data.yml" >&2
        return 4
    fi
    if [[ ! -f "${train_dir}/branch_manifest.json" ]]; then
        echo "Refresh-data branch manifest missing: ${train_dir}/branch_manifest.json" >&2
        return 4
    fi
    if ! find "${train_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -print -quit | grep -q .; then
        echo "No branched checkpoint exists under ${train_dir}." >&2
        return 4
    fi
    launch_train \
        "${snapshot}" "${work_dir}" "${train_dir}" \
        "${max_steps}" "${save_steps}" "${run_name}" refresh-data
}

command=${1:-help}
shift || true
case "${command}" in
    unit)
        "${PYTHON_BIN}" -m unittest discover -v scripts/train/v10_continuous/tests
        ;;
    compile)
        "${PYTHON_BIN}" -m compileall -q scripts/train/v10_continuous
        ;;
    publish-smoke)
        publish_smoke
        ;;
    smoke)
        smoke
        ;;
    experiment)
        experiment
        ;;
    scan)
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.scan_v10 \
            --output-root "${SCAN_OUTPUT}" \
            --workers "${SCAN_WORKERS:-16}" \
            --episode-timeout "${EPISODE_TIMEOUT:-900}" \
            --validation-ratio "${VALIDATION_RATIO:-0.10}" \
            --publish-every "${PUBLISH_EVERY:-25}" \
            --publish-seconds "${PUBLISH_SECONDS:-60}" \
            --resume "$@"
        ;;
    start-scan)
        mkdir -p "${SCAN_OUTPUT}"
        log_file="${SCAN_OUTPUT}/scan_${RUN_STAMP}.log"
        nohup bash "$0" scan "$@" >"${log_file}" 2>&1 &
        pid=$!
        echo "${pid}" >"${SCAN_OUTPUT}/scan.pid"
        echo "V10_SCAN_STARTED pid=${pid} log=${log_file}"
        ;;
    status)
        scan_root=${SCAN_RUN_ROOT:-$(latest_scan_root)}
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.scan_status --run-root "${scan_root}"
        ;;
    snapshot)
        scan_root=${SCAN_RUN_ROOT:-$(latest_scan_root)}
        catalog_snapshot=${CATALOG_SNAPSHOT:-$(latest_catalog_snapshot "${scan_root}")}
        if [[ -z "${catalog_snapshot}" || ! -d "${catalog_snapshot}" ]]; then
            echo "No immutable catalog snapshot exists under ${scan_root}." >&2
            exit 4
        fi
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.build_v10_snapshot \
            --run-root "${scan_root}" \
            --catalog-snapshot "${catalog_snapshot}" \
            --version "${SNAPSHOT_VERSION:-manual-${RUN_STAMP}}"
        ;;
    formal)
        formal
        ;;
    resume-existing)
        resume_existing
        ;;
    resume-refreshed)
        resume_refreshed
        ;;
    teacher)
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.validate_teacher "$@"
        ;;
    rollout)
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.validate_rollout "$@"
        ;;
    infer)
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.stream_infer "$@"
        ;;
    episode-demo)
        selected_gpu=$(choose_gpu)
        if [[ -z "${selected_gpu}" ]]; then
            echo "No idle GPU satisfies memory.used<=2GiB, memory.free>=220GB, util<=10%." >&2
            exit 3
        fi
        selected_gpu=${selected_gpu%%,*}
        echo "[v10-episode-demo] gpu=${selected_gpu} args=$*"
        CUDA_VISIBLE_DEVICES="${selected_gpu}" "${PYTHON_BIN}" \
            -m scripts.train.v10_continuous.episode_demo "$@"
        ;;
    branch-checkpoint)
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.branch_checkpoint "$@"
        ;;
    verify)
        "${PYTHON_BIN}" -m scripts.train.v10_continuous.verify_checkpoint "$@"
        ;;
    *)
        echo "Usage: $0 {unit|compile|publish-smoke|smoke|experiment|scan|start-scan|status|snapshot|formal|resume-existing|resume-refreshed|teacher|rollout|infer|episode-demo|branch-checkpoint|verify}"
        ;;
esac
