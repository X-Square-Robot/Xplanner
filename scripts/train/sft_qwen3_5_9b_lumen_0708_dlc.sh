#!/bin/bash
#
# Qwen3.5-VL 9B 全参数 SFT（lumen_0708 数据配置）— 本地提交 DLC PyTorchJob。
# 训练超参与 scripts/train/sft_qwen3_5_9b_lumen_0708.sh 一致；集群内由 torchrun 启动分布式训练。
# 集群变量由平台注入：MLP_WORKER_GPU、MLP_WORKER_NUM、MLP_ROLE_INDEX、
# MLP_WORKER_0_HOST、MLP_WORKER_0_PORT（以及 RANK / WORLD_SIZE / NPROC_PER_NODE）。
#
# 用法（在已配置 dlc CLI 的本机）：
#   bash scripts/train/sft_qwen3_5_9b_lumen_0708_dlc.sh

# =============================================================================
# 可调参数（一般只改这里）
# =============================================================================

# --- DLC 作业元数据 ---
DLC_JOB_NAME="wall_vlm_qwen35_9b_sft_lumen_0708"
DLC_WORKERS=8
DLC_QUEUE_NAME='multimodal'
# DLC_QUEUE_NAME='quota_a800'
# DLC_QUEUE_NAME='pretrain2'
# DLC_QUEUE_NAME='pretrain'

# 根据队列名称自动设置 DLC_RESOURCE_ID（用 POSIX `[`，避免 `sh` 下 `[[` 报错）
if [ "$DLC_QUEUE_NAME" = "multimodal" ]; then
    DLC_RESOURCE_ID="quotaohtl3ep8kyd"
elif [ "$DLC_QUEUE_NAME" = "quota_a800" ]; then
    DLC_RESOURCE_ID="quotaewyznuc7b9l"
elif [ "$DLC_QUEUE_NAME" = "pretrain2" ]; then
    DLC_RESOURCE_ID="quota1igelu2ln4b"
elif [ "$DLC_QUEUE_NAME" = "pretrain" ]; then
    DLC_RESOURCE_ID="quota1uvnuvdoucg"
else
    echo "Invalid DLC_QUEUE_NAME: $DLC_QUEUE_NAME"
    exit 1
fi

# ---- 路径（与 sft_qwen3_5_9b_lumen_0708.sh 一致）----
MODEL_PATH=/mnt/cpfs/zbl-cpfs-new/Models/Qwen3.5-9B # Qwen3.5-VL 预训练权重目录（HF 格式）
DATA_CONFIG=configs/vqa_mix_qwen3_5_lumen_0708.yml # 数据 YAML：vision/text/epilogue 流水线、采样器、打包
OUTPUT_DIR=/mnt/checkpoint/lumen/wall-vlm/20260708_qwen35_9b_sft_main_full

# --- 训练日志 / 实验名 ---
# RUN_SLUG：提交时间 + 数据配置名（无路径与 .yml 后缀）
RUN_SLUG="$(date +%Y%m%d_%H%M%S)_$(basename -- "$DATA_CONFIG" .yml)"
RUN_NAME="qwen3_5_9b_sft_lumen_0708_${RUN_SLUG}" # 实验名；输出在 $OUTPUT_DIR/$RUN_NAME，也是 wandb run 名
TRAIN_LOG="work_dirs/logs/${RUN_SLUG}.log"

# ---- 训练超参（改这里）----
MODEL_MAX_LENGTH=8192 # 每行最大 token；须与 YAML epilogue.max_seq_length / sampler.cutoff 一致
IMAGE_MIN_PIXELS=1024 # 图像最小像素预算（32²）；须与 YAML vision.min_pixels 一致
IMAGE_MAX_PIXELS=589824 # 图像最大像素预算（≈768²）；须与 YAML vision.max_pixels 一致
NUM_TRAIN_EPOCHS=2 # 训练 epoch 数（数据集遍历轮数）

LEARNING_RATE=7e-6 # 基础 LR：给 scheduler 打底；未单独设 knob 的组件兜底
LLM_LR=7e-6 # 语言模型 LR（含 embed / lm_head）；设了才训练 LM，未设=冻结
# 设了数值才训练对应组件；留空则不传 CLI 参数（默认 None=冻结）。勿写 None。
VISION_LR= # 例: 1e-6
PROJECTOR_LR= # 例: 7e-6
WEIGHT_DECAY=0.0 # 权重衰减（0=关闭；Norm/bias 仍不衰减）
WARMUP_RATIO=0.02 # 前 3% 步线性 warmup
LOSS_REDUCTION_SCOPE=sample # batch=token 均值（默认）；sample=样本均值（长回答不主导）

# 按需拼 LR 参数（空值不传，避免 --vision_lr None 解析失败）
LR_ARGS="--learning_rate ${LEARNING_RATE} --llm_lr ${LLM_LR}"
[ -n "${VISION_LR}" ] && LR_ARGS="${LR_ARGS} --vision_lr ${VISION_LR}"
[ -n "${PROJECTOR_LR}" ] && LR_ARGS="${LR_ARGS} --projector_lr ${PROJECTOR_LR}"

# ---- 批大小（knapsack_packed：一个 bin=一行=一步，LOCAL 必须为 1）----
GLOBAL_ROWS_PER_STEP=256 # 全局每 optimizer step 的打包行数（×8192 ≈ 2M tok/step）
LOCAL_BATCH_SIZE=1 # 每卡 micro-batch；knapsack 模式必须为 1

# ---- 保存 / 日志 ----
SAVE_STEPS=1000 # 每 N 个 optimizer step 存一次 checkpoint
SAVE_TOTAL_LIMIT=8 # 最多保留 N 个 checkpoint，超出自动删旧的
LOGGING_STEPS=1 # 每 N step 打 loss 等到 console / wandb
DATALOADER_NUM_WORKERS=16 # 每卡 DataLoader 子进程数（解码图像/视频、collate）
DATALOADER_PREFETCH_FACTOR=4 # 每 worker 预取 batch 数；workers×prefetch=预加载深度

# ---- wandb（USE_WANDB=false 则 --report_to none，不依赖 API key）----
USE_WANDB=true # true=开启 wandb；false=只打 console / 本地 log
WANDB_PROJECT=0715_VLM_Pretraining
WANDB_ENTITY=x2robot
WANDB_BASE_URL=http://wandb.plat.x2robot.com
# API key 优先放仓库根 .env（WANDB_API_KEY=...）；也可在此直接赋值
WANDB_API_KEY=local-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

# ---- 运行时环境 ----
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 缓解 CUDA 显存碎片（长序列+GC 时更稳）
X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128 # 视频解码走 seek+短窗口；须配合 GOP-30 转码

# wandb / 其它密钥（可选）：仓库根 .env 可覆盖上面的 WANDB_* / USE_WANDB
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "${_REPO_ROOT}/.env" ]; then
  set -a
  source "${_REPO_ROOT}/.env"
  set +a
fi

if [ "${USE_WANDB}" = "true" ]; then
  REPORT_TO=wandb
  if [ -z "${WANDB_API_KEY}" ]; then
    echo "USE_WANDB=true but WANDB_API_KEY is empty. Set it in ${_REPO_ROOT}/.env or in this script."
    exit 1
  fi
else
  REPORT_TO=none
fi

# --- DLC 数据挂载（dataset_id:version:挂载路径，逗号分隔）---
DLC_DATA_SOURCES="d-2doj6vwe6jms01spyd:v1:/mnt/checkpoint/lumen/,d-nai1ftnnp0k5mvgnxy:v1:/mnt/cpfs/lumen/,d-u06av4om42e7pwrl49:v1:/mnt/cpfs/share/,d-cvrfspoc8aj4vl9hji:v1:/mnt/cpfs/zbl-cpfs-new/,d-bass3dq57qv4sjxh24:v1:/mnt/cpfs/open_data/"

# --- Worker 镜像（较长，单独变量便于替换版本）---
DLC_WORKER_IMAGE="dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/pytorch:2.8.0-gpu-py311-cu128-ubuntu24.04-accl-b244fc94-1764407151"

# =============================================================================
# 提交 PyTorchJob（--command 在集群容器内执行）
# 注意：DLC 可能对 --command 做 sh 包装；嵌入脚本里若含未转义的单引号（如 conda/awk 的 '...'），
# 会截断整段命令，表现为 torchrun 后只剩「$」或「... 2>&1' \」等错位。
# =============================================================================

echo "=========================================="
echo "Submitting DLC job: ${DLC_JOB_NAME}"
echo "  MODEL_PATH:       ${MODEL_PATH}"
echo "  DATA_CONFIG:      ${DATA_CONFIG}"
echo "  OUTPUT_DIR:       ${OUTPUT_DIR}/${RUN_NAME}"
echo "  RUN_NAME:         ${RUN_NAME}"
echo "  TRAIN_LOG:        ${TRAIN_LOG}"
echo "  DLC_WORKERS:      ${DLC_WORKERS}"
echo "  DLC_QUEUE_NAME:   ${DLC_QUEUE_NAME}"
echo "  REPORT_TO:        ${REPORT_TO}"
echo "  WANDB_PROJECT:    ${WANDB_PROJECT}"
echo "  LR_ARGS:          ${LR_ARGS}"
echo "=========================================="

dlc submit pytorchjob \
    --name="${DLC_JOB_NAME}" \
    --command="$(
        cat <<'DLC_JOB_CMD'
# ---------- CPFS / 目录：与 DLC_DATA_SOURCES 中的挂载点一致 ----------
mkdir -p /x2robot_v2
mkdir -p /x2robot_data
mkdir -p /mnt/data

mkdir -p /mnt/data/x2robot_v2
ln -sfn /mnt/cpfs/lumen /mnt/data/x2robot_v2/lumen

ln -s /mnt/cpfs/lumen /x2robot_v2/lumen
ln -s /mnt/cpfs/share/ /x2robot_v2/share
ln -s /mnt/cpfs/zbl-cpfs-new/x2robot_data/zhengwei /x2robot_data/zhengwei
ln -s /mnt/cpfs/zbl-cpfs-new/x2robot_data/collection /x2robot_data/collection
ln -s /mnt/cpfs/open_data /mnt/data/open_data

cd /x2robot_v2/lumen/code/wall-vlm

set -e
mkdir -p work_dirs/logs

# ---------- Conda（路径与镜像/挂载约定一致）----------
# >>> conda initialize >>>
# Contents within this block are managed by conda init (embedded script must not contain ASCII apostrophe)
__conda_setup=$(/x2robot_v2/lumen/miniforge3/bin/conda shell.bash hook 2> /dev/null)
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/x2robot_v2/lumen/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/x2robot_v2/lumen/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/x2robot_v2/lumen/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate llama-factory

# ---------- 分布式通信（DLC 会注入 RANK / WORLD_SIZE / NPROC_PER_NODE 等）----------
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-16667}
RESOLVED_MASTER=$(getent hosts "$MASTER_ADDR" | head -n1 | cut -d " " -f1)
if [ -n "$RESOLVED_MASTER" ]; then
    MASTER_ADDR="$RESOLVED_MASTER"
fi
export COORDINATOR_ADDRESS="${MASTER_ADDR}:${MASTER_PORT}"

export WORLD_SIZE=${WORLD_SIZE:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export RANK=${RANK:-0}

echo "WORLD_SIZE=$WORLD_SIZE  NPROC_PER_NODE=$NPROC_PER_NODE  RANK=$RANK"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Starting distributed training on ${NPROC_PER_NODE} GPUs per node..."

DLC_JOB_CMD
        printf '%s\n' \
            "export WANDB_PROJECT=${WANDB_PROJECT}" \
            "export WANDB_ENTITY=${WANDB_ENTITY}" \
            "export WANDB_BASE_URL=${WANDB_BASE_URL}" \
            "export WANDB_API_KEY=${WANDB_API_KEY}" \
            "export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}" \
            "export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=${X2ROBOT_AV_SEQUENTIAL_SPAN_MAX}" \
            "export GLOBAL_ROWS_PER_STEP=${GLOBAL_ROWS_PER_STEP}" \
            "export LOCAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}" \
            'GRAD_ACCUM=$(( GLOBAL_ROWS_PER_STEP / (WORLD_SIZE * NPROC_PER_NODE * LOCAL_BATCH_SIZE) ))' \
            'echo "grad_accum=$GRAD_ACCUM"' \
            'torchrun \' \
            '    --nnodes=${WORLD_SIZE} \' \
            '    --nproc_per_node=${NPROC_PER_NODE} \' \
            '    --node_rank=${RANK} \' \
            '    --master_addr=${MASTER_ADDR} \' \
            '    --master_port=${MASTER_PORT} \' \
            '    -m qwenvl.train.launcher \' \
            "    --deepspeed scripts/zero1.json \\" \
            "    --model_path ${MODEL_PATH} \\" \
            "    --data_config ${DATA_CONFIG} \\" \
            '    --attn_implementation flash_attention_2 \' \
            "    --model_max_length ${MODEL_MAX_LENGTH} \\" \
            "    --image_min_pixels ${IMAGE_MIN_PIXELS} --image_max_pixels ${IMAGE_MAX_PIXELS} \\" \
            '    --bf16 True --tf32 True \' \
            "    --per_device_train_batch_size ${LOCAL_BATCH_SIZE} \\" \
            '    --gradient_accumulation_steps ${GRAD_ACCUM} \' \
            "    --num_train_epochs ${NUM_TRAIN_EPOCHS} \\" \
            "    ${LR_ARGS} \\" \
            "    --weight_decay ${WEIGHT_DECAY} --warmup_ratio ${WARMUP_RATIO} \\" \
            '    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '"'"'{"min_lr_rate": 0.1}'"'"' \' \
            '    --gradient_checkpointing True \' \
            '    --average_tokens_across_devices True \' \
            "    --loss_reduction_scope ${LOSS_REDUCTION_SCOPE} \\" \
            '    --ignore_data_skip True \' \
            "    --save_strategy steps --save_steps ${SAVE_STEPS} --save_total_limit ${SAVE_TOTAL_LIMIT} \\" \
            "    --logging_steps ${LOGGING_STEPS} \\" \
            "    --dataloader_num_workers ${DATALOADER_NUM_WORKERS} --dataloader_prefetch_factor ${DATALOADER_PREFETCH_FACTOR} \\" \
            "    --output_dir ${OUTPUT_DIR}/${RUN_NAME} \\" \
            "    --report_to ${REPORT_TO} --run_name ${RUN_NAME} > ${TRAIN_LOG} 2>&1"
    )" \
    --data_sources="${DLC_DATA_SOURCES}" \
    --resource_id="${DLC_RESOURCE_ID}" \
    --workspace_id=179169 \
    --vpc_id=vpc-0jl4fvc8iyfr3mwlfqssj \
    --switch_id=vsw-0jldmhv2n24pofrzyu8ev,vsw-0jln9swvd0eqcrd9f1o2q \
    --security_group_id=sg-0jlflpwmnysurjbu2u7c \
    --priority=9 \
    --extended_cidrs="192.168.251.0/24,192.168.1.32/27,192.168.1.8/29,192.168.252.0/24,192.168.253.0/24,192.168.254.0/24,192.168.1.0/29" \
    --workers="${DLC_WORKERS}" \
    --worker_image="${DLC_WORKER_IMAGE}" \
    --worker_cpu=46 \
    --worker_memory=800Gi \
    --worker_shared_memory=800Gi \
    --worker_gpu=8

echo ""
echo "DLC job submitted: ${DLC_JOB_NAME}"
