#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ENV_ROOT=${ENV_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall_wm_B300}
WALL_REPO=${WALL_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}
DATASET_REPO=${DATASET_REPO:-$(cd -- "${WALL_REPO}/../x2robot_dataset_v2_B300" && pwd)}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous_v5}
MODEL_PATH=${MODEL_PATH:-/mnt/cpfs/zbl-cpfs-new/SHARE/lumen/20260818_w_planning_qwen35_9b_sft_main_full/qwen3_5_9b_sft_lumen_0818_w_planning_20260818_130704_vqa_mix_qwen3_5_lumen_0818_w_planning/checkpoint-12000}
BENCHMARK3_MANIFEST=${BENCHMARK3_MANIFEST:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/APlan/Benchmark3/benchmark3_1500_manifest.jsonl}
BENCHMARK3_SHA256=${BENCHMARK3_SHA256:-1a5df7df3af1f43e33bee8121e324d27d31a1d0d9e91aa0a9acd5ec8b5cd5c76}
PYTHON_BIN=${PYTHON_BIN:-${ENV_ROOT}/bin/python}
RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-65536}
SMOKE_STEPS=${SMOKE_STEPS:-1}
SMOKE_MAX_BUDGET=${SMOKE_MAX_BUDGET:-200}
DATALOADER_WORKERS=${DATALOADER_WORKERS:-2}
SAVE_SMOKE_CHECKPOINT=${SAVE_SMOKE_CHECKPOINT:-0}
REQUIRE_LOSS_DECREASE=${REQUIRE_LOSS_DECREASE:-0}
SMOKE_LEARNING_RATE=${SMOKE_LEARNING_RATE:-1e-5}
SMOKE_MAX_GRAD_NORM=${SMOKE_MAX_GRAD_NORM:-1.0}
SMOKE_WARMUP_STEPS=${SMOKE_WARMUP_STEPS:-0}
SMOKE_OPTIM=${SMOKE_OPTIM:-adamw_torch}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-70000}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-5}
GPU_STABILITY_SECONDS=${GPU_STABILITY_SECONDS:-10}
V5_GPU_LOCK_FILE=${V5_GPU_LOCK_FILE:-/tmp/luhao-v5-dev13-gpu.lock}
V5_GPU_LOCK_WAIT_SECONDS=${V5_GPU_LOCK_WAIT_SECONDS:-5}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-${WALL_REPO}/scripts/zero1.json}

for required in \
    "${ENV_ROOT}/bin/activate" \
    "${PYTHON_BIN}" \
    "${WALL_REPO}" \
    "${DATASET_REPO}" \
    "${MODEL_PATH}/model.safetensors" \
    "${BENCHMARK3_MANIFEST}"; do
    if [[ ! -e "${required}" ]]; then
        echo "missing required V5 path: ${required}" >&2
        exit 66
    fi
done
printf '%s  %s\n' "${BENCHMARK3_SHA256}" "${BENCHMARK3_MANIFEST}" | sha256sum -c -

source "${ENV_ROOT}/bin/activate"
export PYTHONPATH="${WALL_REPO}:${DATASET_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-/tmp/v5-dev13-pycache-${USER:-luhao}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128
export V10_SKIP_FINAL_MODEL_SAVE=${V10_SKIP_FINAL_MODEL_SAVE:-1}
cd "${WALL_REPO}"

module() {
    "${PYTHON_BIN}" -m "scripts.train.v10_continuous_v5.$1" "${@:2}"
}

safe_gpus() {
    nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader,nounits | awk -F, \
        -v min_free="${MIN_GPU_FREE_MIB}" -v max_util="${MAX_GPU_UTIL}" '
        {
            for (i=1; i<=NF; i++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
            free=($2+0)-($3+0)
            if (free>=min_free && ($4+0)<=max_util) print $1
        }'
}

choose_gpus() {
    local count=${1:?missing GPU count}
    if [[ -n "${GPU_IDS:-}" ]]; then
        local supplied
        supplied=$(tr ',' '\n' <<<"${GPU_IDS}" | sed '/^$/d' | head -n "${count}" | paste -sd, -)
        if [[ $(tr ',' '\n' <<<"${supplied}" | wc -l) -ne ${count} ]]; then
            echo "GPU_IDS does not provide ${count} devices: ${GPU_IDS}" >&2
            return 3
        fi
        printf '%s\n' "${supplied}"
        return
    fi
    local selected selected_second
    selected=$(safe_gpus | head -n "${count}" | paste -sd, -)
    if [[ $(tr ',' '\n' <<<"${selected}" | sed '/^$/d' | wc -l) -ne ${count} ]]; then
        echo "Need ${count} idle GPUs with free>=${MIN_GPU_FREE_MIB}MiB and util<=${MAX_GPU_UTIL}%." >&2
        return 3
    fi
    sleep "${GPU_STABILITY_SECONDS}"
    selected_second=$(safe_gpus | head -n "${count}" | paste -sd, -)
    if [[ "${selected_second}" != "${selected}" ]]; then
        echo "GPU safety set changed during ${GPU_STABILITY_SECONDS}s stability check: ${selected} -> ${selected_second:-none}." >&2
        return 3
    fi
    printf '%s\n' "${selected}"
}

choose_master_port() {
    if [[ -n "${V5_MASTER_PORT:-}" ]]; then
        if [[ ! "${V5_MASTER_PORT}" =~ ^[0-9]+$ ]] || \
                (( V5_MASTER_PORT < 1024 || V5_MASTER_PORT > 65535 )); then
            echo "invalid V5_MASTER_PORT: ${V5_MASTER_PORT}" >&2
            return 64
        fi
        printf '%s\n' "${V5_MASTER_PORT}"
        return
    fi
    "${PYTHON_BIN}" -c '
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
'
}

prepare_data() {
    local dataset_root=$1
    local work_dir=$2
    local -a generation_args=()
    local digest
    digest=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_digest"])' "${dataset_root}/manifest.json")
    if "${PYTHON_BIN}" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("partial") is True else 1)' "${dataset_root}/manifest.json"; then
        generation_args=(
            --allow-partial-generation
            --expected-content-digest "${digest}"
        )
    fi
    module dataset_v5 \
        --dataset-root "${dataset_root}" \
        --work-dir "${work_dir}" \
        --model-path "${MODEL_PATH}" \
        --max-length "${MODEL_MAX_LENGTH}" \
        --max-budget "${SMOKE_MAX_BUDGET}" \
        --benchmark3-manifest "${BENCHMARK3_MANIFEST}" \
        --benchmark3-expected-sha256 "${BENCHMARK3_SHA256}" \
        "${generation_args[@]}"
}

run_train_smoke() {
    local dataset_root=${1:?missing dataset root}
    local gpu_count=${2:?missing GPU count}
    local label=${3:?missing smoke label}
    local work_dir=${SMOKE_ROOT:-${OUTPUT_ROOT}/smoke/${RUN_STAMP}}/${label}
    local train_dir=${work_dir}/train
    local data_dir=${PREPARED_DATA_DIR:-${work_dir}}
    if [[ -e "${work_dir}" ]]; then
        echo "refusing to reuse V5 smoke directory: ${work_dir}" >&2
        return 73
    fi
    local v5_gpu_lock_fd
    exec {v5_gpu_lock_fd}>"${V5_GPU_LOCK_FILE}"
    if ! flock -w "${V5_GPU_LOCK_WAIT_SECONDS}" "${v5_gpu_lock_fd}"; then
        echo "another V5 GPU launcher holds ${V5_GPU_LOCK_FILE}" >&2
        return 75
    fi
    local gpus
    gpus=$(choose_gpus "${gpu_count}")
    mkdir -p "${train_dir}"
    if [[ -n "${PREPARED_DATA_DIR:-}" ]]; then
        for prepared_file in data.yml prepare_summary.json exposure_plan.json; do
            if [[ ! -f "${data_dir}/${prepared_file}" ]]; then
                echo "missing precomputed V5 data file: ${data_dir}/${prepared_file}" >&2
                return 66
            fi
        done
        "${PYTHON_BIN}" -c '
import json, pathlib, sys
snapshot = pathlib.Path(sys.argv[1]).resolve(strict=True)
prepared = json.load(open(sys.argv[2]))
manifest = json.load(open(snapshot / "manifest.json"))
if pathlib.Path(prepared.get("dataset_root", "")).resolve() != snapshot:
    raise SystemExit("precomputed V5 data belongs to another snapshot")
if prepared.get("content_digest") != manifest.get("content_digest"):
    raise SystemExit("precomputed V5 data content digest mismatch")
' "${dataset_root}" "${data_dir}/prepare_summary.json"
    else
        prepare_data "${dataset_root}" "${data_dir}"
    fi
    module data_smoke_v5 \
        --data-config "${data_dir}/data.yml" \
        --model-path "${MODEL_PATH}" \
        --output "${work_dir}/data_smoke.json" \
        --batch-size 1
    nvidia-smi --id="${gpus}" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits --loop-ms=1000 >"${work_dir}/gpu_monitor.csv" &
    local monitor_pid=$!
    local master_port
    master_port=$(choose_master_port)
    local train_status
    local distributed_optimizer_args=()
    local generation_args=()
    local save_args=(--save_strategy no)
    local report_args=()
    local digest
    digest=$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["content_digest"])' "${dataset_root}/manifest.json")
    if "${PYTHON_BIN}" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("partial") is True else 1)' "${dataset_root}/manifest.json"; then
        generation_args=(
            --v5_allow_partial_generation
            --v5_expected_content_digest "${digest}"
        )
    fi
    if [[ "${SAVE_SMOKE_CHECKPOINT}" == 1 ]]; then
        save_args=(
            --save_strategy steps --save_steps "${SMOKE_STEPS}"
            --save_total_limit 1
        )
        report_args=(--trained-checkpoint "${train_dir}/checkpoint-${SMOKE_STEPS}")
    fi
    if [[ "${REQUIRE_LOSS_DECREASE}" == 1 ]]; then
        report_args+=(--require-loss-decrease)
    fi
    if (( gpu_count > 1 )); then
        if [[ ! -f "${DEEPSPEED_CONFIG}" ]]; then
            echo "missing DeepSpeed config for multi-GPU smoke: ${DEEPSPEED_CONFIG}" >&2
            return 66
        fi
        distributed_optimizer_args=(--deepspeed "${DEEPSPEED_CONFIG}")
    fi
    set +e
    CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes 1 --nproc_per_node "${gpu_count}" \
        --master_addr 127.0.0.1 --master_port "${master_port}" \
        -m scripts.train.v10_continuous_v5.train_v5 \
        --snapshot "${dataset_root}" \
        "${generation_args[@]}" \
        --step-state "${work_dir}/v10_step_state.json" \
        --resume-mode weights-only \
        --model_path "${MODEL_PATH}" \
        --data_config "${data_dir}/data.yml" \
        --attn_implementation sdpa \
        --model_max_length "${MODEL_MAX_LENGTH}" \
        --image_min_pixels 1024 --image_max_pixels 589824 \
        --bf16 true --tf32 true \
        --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
        --max_steps "${SMOKE_STEPS}" --num_train_epochs 100 \
        --learning_rate "${SMOKE_LEARNING_RATE}" \
        --llm_lr "${SMOKE_LEARNING_RATE}" --weight_decay 0 \
        --optim "${SMOKE_OPTIM}" \
        --max_grad_norm "${SMOKE_MAX_GRAD_NORM}" \
        --warmup_steps "${SMOKE_WARMUP_STEPS}" --lr_scheduler_type constant \
        --gradient_checkpointing true --ddp_find_unused_parameters false \
        --loss_reduction_scope sample --lm_head_loss_only_on_labels false \
        "${save_args[@]}" --logging_steps 1 \
        --dataloader_num_workers "${DATALOADER_WORKERS}" \
        --ignore_data_skip true --seed 42 --data_seed 42 --disable_tqdm true \
        --output_dir "${train_dir}" --report_to none \
        --run_name "v5_${label}_${RUN_STAMP}" \
        "${distributed_optimizer_args[@]}" \
        2>&1 | tee "${train_dir}/launch.log"
    train_status=${PIPESTATUS[0]}
    set -e
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
    module smoke_report_v5 \
        --log "${train_dir}/launch.log" \
        --monitor "${work_dir}/gpu_monitor.csv" \
        --output "${work_dir}/gpu_smoke_report.json" \
        --required-steps "${SMOKE_STEPS}" --exit-code "${train_status}" \
        --dataset-root "${dataset_root}" --checkpoint "${MODEL_PATH}" \
        --gpu-id "${gpus}" --max-length "${MODEL_MAX_LENGTH}" \
        "${report_args[@]}"
    if [[ "${SAVE_SMOKE_CHECKPOINT}" == 1 ]]; then
        printf '%s\n' "${train_dir}/checkpoint-${SMOKE_STEPS}" > \
            "${work_dir}/trained_checkpoint.txt"
    fi
    flock -u "${v5_gpu_lock_fd}"
    exec {v5_gpu_lock_fd}>&-
}

run_infer_smoke() {
    local dataset_root=${1:?missing dataset root}
    local checkpoint=${2:?missing inference checkpoint}
    local local_output=${3:?missing inference output directory}
    local v5_gpu_lock_fd
    exec {v5_gpu_lock_fd}>"${V5_GPU_LOCK_FILE}"
    if ! flock -w "${V5_GPU_LOCK_WAIT_SECONDS}" "${v5_gpu_lock_fd}"; then
        echo "another V5 GPU launcher holds ${V5_GPU_LOCK_FILE}" >&2
        return 75
    fi
    local selected_gpu
    selected_gpu=$(choose_gpus 1)
    CUDA_VISIBLE_DEVICES="${selected_gpu}" "${PYTHON_BIN}" -m \
        scripts.train.v10_continuous_v5.infer_v5 \
        --checkpoint "${checkpoint}" \
        --processor "${MODEL_PATH}" \
        --snapshot "${dataset_root}" \
        --output-dir "${local_output}" --device cuda:0 \
        --max-new-tokens "${MAX_NEW_TOKENS:-128}"
    flock -u "${v5_gpu_lock_fd}"
    exec {v5_gpu_lock_fd}>&-
}

command_name=${1:-help}
if [[ $# -gt 0 ]]; then shift; fi
case "${command_name}" in
    unit)
        "${PYTHON_BIN}" -m unittest discover \
            -s scripts/train/v10_continuous_v5 -t . -p 'test_*.py' -v
        ;;
    prepare)
        prepare_data "${1:?missing dataset root}" "${2:?missing work dir}"
        ;;
    data-smoke)
        prepare_data "${1:?missing dataset root}" "${2:?missing work dir}"
        module data_smoke_v5 \
            --data-config "${2}/data.yml" --model-path "${MODEL_PATH}" \
            --output "${2}/data_smoke.json" --batch-size 1
        ;;
    examples)
        if [[ -n "${TAKEOVER_SOURCE:-}" ]]; then
            module export_confirmation_samples_v5 \
                --snapshot "${1:?missing dataset root}" \
                --takeover-source "${TAKEOVER_SOURCE}" \
                --output "${2:?missing output directory}"
        else
            module export_confirmation_samples_v5 \
                --snapshot "${1:?missing dataset root}" \
                --output "${2:?missing output directory}"
        fi
        ;;
    smoke|smoke-single)
        run_train_smoke "${1:?missing dataset root}" 1 single_gpu_1step
        ;;
    smoke-ddp4)
        run_train_smoke "${1:?missing dataset root}" 4 four_gpu_zero1_1step
        ;;
    smoke-ddp3)
        run_train_smoke "${1:?missing dataset root}" 3 three_gpu_zero1_1step
        ;;
    overfit-ddp3)
        SMOKE_STEPS=${OVERFIT_STEPS:-30}
        SAVE_SMOKE_CHECKPOINT=1
        REQUIRE_LOSS_DECREASE=1
        run_train_smoke "${1:?missing dataset root}" 3 three_gpu_zero1_overfit
        ;;
    smoke-all)
        run_train_smoke "${1:?missing dataset root}" 1 single_gpu_1step
        run_train_smoke "${1:?missing dataset root}" 4 four_gpu_ddp_1step
        ;;
    infer)
        run_infer_smoke \
            "${1:?missing dataset root}" "${MODEL_PATH}" \
            "${2:?missing inference output directory}"
        ;;
    infer-checkpoint)
        run_infer_smoke \
            "${1:?missing dataset root}" \
            "${2:?missing trained checkpoint}" \
            "${3:?missing inference output directory}"
        ;;
    train)
        echo "Formal long-running V5 training is disabled; bounded smoke only." >&2
        exit 4
        ;;
    help|-h|--help)
        echo "Usage: $0 {unit|prepare DATA ROOT|data-smoke DATA ROOT|examples DATA OUT|smoke-single DATA|smoke-ddp3 DATA|smoke-ddp4 DATA|overfit-ddp3 DATA|smoke-all DATA|infer DATA OUT|infer-checkpoint DATA CHECKPOINT OUT}"
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        exit 2
        ;;
esac
