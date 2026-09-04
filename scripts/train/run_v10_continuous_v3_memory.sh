#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT=${ENV_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall_wm_B300}
WALL_REPO=${WALL_REPO:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/wall-vlm_B300}
DATASET_REPO=${DATASET_REPO:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/x2robot_dataset_v2_B300}
MODEL_PATH=${MODEL_PATH:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/models/Qwen3.5-9B}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous_v3_memory}
SOURCE_V2_RUN=${SOURCE_V2_RUN:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/runs/aihc/v10-maxdata-refresh-8g-b6-0807_job-0u11ct3i43ni_maxdata2h_20260807T0400Z_full2ep}
CONFIG=${CONFIG:-${WALL_REPO}/scripts/train/v10_continuous_v3_memory/configs/v3_memory.yaml}
PYTHON_BIN=${PYTHON_BIN:-${ENV_ROOT}/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-4096}
MASTER_PORT=${MASTER_PORT:-$((22000 + $$ % 18000))}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-6}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-4}
SMOKE_STEPS=${SMOKE_STEPS:-5}
RESOLUTION_SMOKE_STEPS=${RESOLUTION_SMOKE_STEPS:-2}

source "${ENV_ROOT}/bin/activate"
export PYTHONPATH="${WALL_REPO}:${DATASET_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128
export V10_SKIP_FINAL_MODEL_SAVE=1
cd "${WALL_REPO}"

module() {
    "${PYTHON_BIN}" -m "scripts.train.v10_continuous_v3_memory.$1" "${@:2}"
}

current_snapshot() {
    if [[ -n "${SNAPSHOT:-}" ]]; then
        echo "${SNAPSHOT}"
        return
    fi
    "${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["root"])' \
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
            if (($2+0)<=2048 && ($3+0)>=220000 && ($4+0)<=10) {print $1; exit}
        }'
}

prepare_data() {
    local snapshot=$1
    local work_dir=$2
    local tasks=$3
    local resize_mode=$4
    local small_flag=()
    local budget_flag=()
    if [[ "${ALLOW_SMALL:-1}" == 1 ]]; then
        small_flag+=(--allow-small)
    elif [[ "${ALLOW_PARTIAL:-0}" == 1 ]]; then
        small_flag+=(--allow-partial)
    fi
    if [[ -n "${MAX_BUDGET:-}" ]]; then
        budget_flag+=(--max-budget "${MAX_BUDGET}")
    fi
    module dataset_v3 \
        --snapshot "${snapshot}" --work-dir "${work_dir}" \
        --model-path "${MODEL_PATH}" --max-length "${MODEL_MAX_LENGTH}" \
        --tasks "${tasks}" --resize-mode "${resize_mode}" \
        "${budget_flag[@]}" "${small_flag[@]}"
}

train_args() {
    local snapshot=$1
    local work_dir=$2
    local output_dir=$3
    local steps=$4
    local resume_mode=$5
    local save_strategy=$6
    local save_steps=$7
    local nproc_per_node=${8:-1}
    "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes 1 --nproc_per_node "${nproc_per_node}" \
        --master_addr 127.0.0.1 --master_port "${MASTER_PORT}" \
        -m scripts.train.v10_continuous_v3_memory.train_v3 \
        --snapshot "${snapshot}" --step-state "${work_dir}/v10_step_state.json" \
        --resume-mode "${resume_mode}" --model_path "${MODEL_PATH}" \
        --data_config "${work_dir}/data.yml" --attn_implementation sdpa \
        --model_max_length "${MODEL_MAX_LENGTH}" --image_min_pixels 1024 \
        --image_max_pixels 589824 --bf16 true --tf32 true \
        --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
        --gradient_accumulation_steps 1 --max_steps "${steps}" --num_train_epochs 100 \
        --learning_rate "${LEARNING_RATE:-1e-5}" --llm_lr "${LLM_LR:-1e-5}" \
        --weight_decay 0 --warmup_steps "${WARMUP_STEPS:-1}" --lr_scheduler_type cosine \
        --gradient_checkpointing true --ddp_find_unused_parameters false \
        --loss_reduction_scope sample --lm_head_loss_only_on_labels false \
        --save_strategy "${save_strategy}" --save_steps "${save_steps}" \
        --save_total_limit 3 --logging_steps 1 \
        --dataloader_num_workers "${DATALOADER_WORKERS}" --dataloader_prefetch_factor 2 \
        --ignore_data_skip true --seed 42 --data_seed 42 --disable_tqdm true \
        --output_dir "${output_dir}" --report_to none --run_name "memory_v3_${RUN_STAMP}"
}

gpu_smoke() {
    local snapshot=$1
    local name=$2
    local tasks=$3
    local resize_mode=$4
    local steps=$5
    local work_dir="${SMOKE_ROOT}/${name}"
    local train_dir="${work_dir}/train"
    local report="${work_dir}/gpu_smoke_report.json"
    local gpu
    if [[ -f "${report}" ]] && "${PYTHON_BIN}" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed") else 1)' "${report}"; then
        echo "MEMORY_V3_GPU_SMOKE_REUSE name=${name} report=${report}"
        return
    fi
    gpu=$(choose_gpu)
    if [[ -z "${gpu}" ]]; then
        echo "No idle GPU satisfies used<=2GiB, free>=220GB, util<=10%." >&2
        return 3
    fi
    mkdir -p "${train_dir}"
    ALLOW_SMALL=1 MAX_BUDGET=${SMOKE_MAX_BUDGET:-100} prepare_data \
        "${snapshot}" "${work_dir}" "${tasks}" "${resize_mode}"
    module data_smoke_v3 --data-config "${work_dir}/data.yml" \
        --model-path "${MODEL_PATH}" --output "${work_dir}/data_smoke_batch6.json" \
        --batch-size 6
    nvidia-smi --id="${gpu}" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits --loop-ms=1000 >"${work_dir}/gpu_monitor.csv" &
    local monitor_pid=$!
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" train_args \
        "${snapshot}" "${work_dir}" "${train_dir}" "${steps}" exact no "${steps}" \
        2>&1 | tee "${train_dir}/launch.log"
    local train_status=${PIPESTATUS[0]}
    set -e
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    module train_smoke_report_v3 \
        --log "${train_dir}/launch.log" --monitor "${work_dir}/gpu_monitor.csv" \
        --output "${work_dir}/gpu_smoke_report.json" --task "${tasks}" \
        --resize-mode "${resize_mode}" --required-steps "${steps}" --exit-code "${train_status}"
}

run_all_smokes() {
    local snapshot
    snapshot=$(current_snapshot)
    SMOKE_ROOT=${SMOKE_ROOT:-${OUTPUT_ROOT}/smoke/${RUN_STAMP}}
    export SMOKE_ROOT
    mkdir -p "${SMOKE_ROOT}"
    "${PYTHON_BIN}" -m unittest scripts.train.v10_continuous_v3_memory.tests.test_v3_contract -v
    module smoke_v3 --snapshot "${snapshot}" --output "${SMOKE_ROOT}/semantic_smoke.json"
    gpu_smoke "${snapshot}" continuous_B continuous B_auto_near_640 "${SMOKE_STEPS}"
    gpu_smoke "${snapshot}" initial_plan_B initial_plan B_auto_near_640 "${SMOKE_STEPS}"
    gpu_smoke "${snapshot}" terminal_B terminal B_auto_near_640 "${SMOKE_STEPS}"
    gpu_smoke "${snapshot}" mixed_B continuous,initial_plan,terminal B_auto_near_640 "${SMOKE_STEPS}"
    gpu_smoke "${snapshot}" mixed_A continuous,initial_plan,terminal A_current "${RESOLUTION_SMOKE_STEPS}"
    gpu_smoke "${snapshot}" mixed_C continuous,initial_plan,terminal C_original_capped "${RESOLUTION_SMOKE_STEPS}"
    module resolution_report_v3 \
        --probe "${PROBE_OUTPUT:-${OUTPUT_ROOT}/probes/resolution_1000.json}" \
        --output "${SMOKE_ROOT}/resolution_gpu_acceptance.json" \
        --report "${SMOKE_ROOT}/mixed_A/gpu_smoke_report.json" \
        --report "${SMOKE_ROOT}/mixed_B/gpu_smoke_report.json" \
        --report "${SMOKE_ROOT}/mixed_C/gpu_smoke_report.json"
    echo "MEMORY_V3_SMOKE_PASSED root=${SMOKE_ROOT}"
}

launch_formal_train() {
    local resume_mode=$1
    local trainer_resume_mode=refresh-data
    local snapshot
    snapshot=$(current_snapshot)
    local work_dir
    if [[ "${resume_mode}" =~ ^(refresh-data|exact)$ && -z "${WORK_DIR:-}" ]]; then
        work_dir=$("${PYTHON_BIN}" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["work_dir"])' \
            "${OUTPUT_ROOT}/current_training_run.json")
    else
        work_dir=${WORK_DIR:-${OUTPUT_ROOT}/runs/memory_v3_${RUN_STAMP}}
    fi
    local run_state=${work_dir}/current_training_generation.json
    local snapshot_id
    snapshot_id=$(basename "${snapshot}")
    local train_dir
    ALLOW_SMALL=0 ALLOW_PARTIAL=${ALLOW_PARTIAL_TRAIN:-1} \
        MAX_BUDGET=${FORMAL_MAX_BUDGET:-} prepare_data \
        "${snapshot}" "${work_dir}" continuous,initial_plan,terminal B_auto_near_640
    if [[ "${resume_mode}" == branch ]]; then
        # checkpoint_v3 has already rewritten provenance to this immutable V3
        # snapshot; the underlying trainer therefore performs an exact resume.
        trainer_resume_mode=exact
        train_dir=${TRAIN_DIR:-${work_dir}/train_generations/${snapshot_id}-branch}
        module checkpoint_v3 branch --source-run "${SOURCE_V2_RUN}" \
            --target-output "${train_dir}" --snapshot "${snapshot}" \
            --data-config "${work_dir}/data.yml" \
            --reset-scheduler-stage \
            --audit-output "${work_dir}/source_checkpoint_audit.json" \
            --run-state "${run_state}" \
            --global-run-state "${OUTPUT_ROOT}/current_training_run.json"
    elif [[ "${resume_mode}" == refresh-data ]]; then
        [[ -f "${run_state}" ]] || {
            echo "Missing V3 current_training_generation.json under ${work_dir}" >&2; return 4;
        }
        local previous_train
        previous_train=$("${PYTHON_BIN}" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["current_train_dir"])' \
            "${run_state}")
        train_dir=${TRAIN_DIR:-${work_dir}/train_generations/${snapshot_id}-refresh-${RUN_STAMP}}
        module checkpoint_v3 branch --source-run "${previous_train}" \
            --target-output "${train_dir}" --snapshot "${snapshot}" \
            --data-config "${work_dir}/data.yml" \
            --audit-output "${work_dir}/refresh_source_checkpoint_audit_${RUN_STAMP}.json" \
            --run-state "${run_state}" \
            --global-run-state "${OUTPUT_ROOT}/current_training_run.json"
    elif [[ "${resume_mode}" == exact ]]; then
        [[ -f "${run_state}" ]] || {
            echo "Missing V3 current_training_generation.json under ${work_dir}" >&2; return 4;
        }
        train_dir=$("${PYTHON_BIN}" -c \
            'import json,sys; print(json.load(open(sys.argv[1]))["current_train_dir"])' \
            "${run_state}")
        trainer_resume_mode=exact
    else
        echo "resume mode must be branch, refresh-data, or exact" >&2
        return 2
    fi
    local selected_gpus=${GPU_IDS:-}
    if [[ -z "${selected_gpus}" ]]; then selected_gpus=$(choose_gpu); fi
    [[ -n "${selected_gpus}" ]] || { echo "No idle GPU for V3 train." >&2; return 3; }
    local nproc_per_node
    nproc_per_node=$(awk -F, '{print NF}' <<<"${selected_gpus}")
    mkdir -p "${train_dir}"
    local target_steps
    if [[ -n "${MAX_STEPS:-}" ]]; then
        target_steps=${MAX_STEPS}
    else
        target_steps=$("${PYTHON_BIN}" -c \
            'import json,pathlib,sys; p=max(pathlib.Path(sys.argv[1]).glob("checkpoint-*"), key=lambda x:int(x.name.rsplit("-",1)[1])); print(int(json.load(open(p/"trainer_state.json"))["global_step"])+int(sys.argv[2]))' \
            "${train_dir}" "${ADDITIONAL_STEPS:-10000}")
    fi
    CUDA_VISIBLE_DEVICES="${selected_gpus}" train_args \
        "${snapshot}" "${work_dir}" "${train_dir}" "${target_steps}" \
        "${trainer_resume_mode}" steps "${SAVE_STEPS:-500}" "${nproc_per_node}" \
        2>&1 | tee -a "${train_dir}/launch.log"
    module checkpoint_v3 complete-generation \
        --train-dir "${train_dir}" --run-state "${run_state}" \
        --global-run-state "${OUTPUT_ROOT}/current_training_run.json"
}

command_name=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "${command_name}" in
    preflight)
        module build_json_v3 --config "${CONFIG}" preflight
        module checkpoint_v3 audit --source-run "${SOURCE_V2_RUN}" \
            --output "${OUTPUT_ROOT}/source_checkpoint_audit.json"
        ;;
    probe-resolution)
        snapshot=$(current_snapshot)
        module resolution_probe --snapshot "${snapshot}" \
            --output "${PROBE_OUTPUT:-${OUTPUT_ROOT}/probes/resolution_1000.json}" "$@"
        ;;
    build-json)
        module build_json_v3 --config "${CONFIG}" build "$@"
        ;;
    publish)
        allow_partial=0
        if [[ "${1:-}" == --allow-partial ]]; then allow_partial=1; shift; fi
        merge_args=()
        if [[ ${allow_partial} -eq 1 ]]; then merge_args+=(--allow-partial); fi
        module merge_lists_v3 --config "${CONFIG}" "${merge_args[@]}" "$@"
        module build_snapshot_v3 --config "${CONFIG}"
        ;;
    smoke)
        run_all_smokes
        ;;
    train)
        resume_mode=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --resume-mode) resume_mode=${2:?missing resume mode}; shift 2 ;;
                *) echo "Unknown train argument: $1" >&2; exit 2 ;;
            esac
        done
        launch_formal_train "${resume_mode}"
        ;;
    finalize)
        module merge_lists_v3 --config "${CONFIG}"
        module build_snapshot_v3 --config "${CONFIG}" --require-complete
        ;;
    evaluate-terminal)
        module metrics_v3 "$@"
        ;;
    evaluate-terminal-model)
        module terminal_model_eval_v3 "$@"
        ;;
    help|-h|--help)
        echo "Usage: $0 {preflight|probe-resolution|build-json|publish|smoke|train|finalize|evaluate-terminal|evaluate-terminal-model}"
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        exit 2
        ;;
esac
