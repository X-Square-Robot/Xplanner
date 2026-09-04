#!/bin/bash
#
# 长度均衡 packing 离线预热 — 本地提交 DLC PyTorchJob。
# 与 scripts/packing/123.sh 等价，在集群容器内跑 precompute_lengths（不加载模型）。
#
# 用法（在已配置 dlc CLI 的本机）：
#   bash scripts/packing/precompute_lengths_lumen_0708_dlc.sh

# =============================================================================
# 可调参数（一般只改这里）
# =============================================================================

# --- DLC 作业元数据 ---
DLC_JOB_NAME="wall_vlm_precompute_lengths_lumen"
DLC_WORKERS=1
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

# ---- 路径（与 123.sh / 训练脚本一致）----
DATA_CONFIG=configs/vqa_mix_qwen3_5_lumen_0708.yml
CUTOFF=8192
WORKERS=96

# --- 运行日志 ---
RUN_SLUG="$(date +%Y%m%d_%H%M%S)_$(basename -- "$DATA_CONFIG" .yml)_precompute"
PRECOMPUTE_LOG="work_dirs/logs/${RUN_SLUG}.log"

# ---- 运行时环境 ----
X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128

# wandb / 其它密钥（可选）：仓库根 .env
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "${_REPO_ROOT}/.env" ]; then
  set -a
  source "${_REPO_ROOT}/.env"
  set +a
fi

# --- DLC 数据挂载（dataset_id:version:挂载路径，逗号分隔）---
DLC_DATA_SOURCES="d-2doj6vwe6jms01spyd:v1:/mnt/checkpoint/lumen/,d-nai1ftnnp0k5mvgnxy:v1:/mnt/cpfs/lumen/,d-u06av4om42e7pwrl49:v1:/mnt/cpfs/share/,d-cvrfspoc8aj4vl9hji:v1:/mnt/cpfs/zbl-cpfs-new/,d-bass3dq57qv4sjxh24:v1:/mnt/cpfs/open_data/"

# --- Worker 镜像（较长，单独变量便于替换版本）---
DLC_WORKER_IMAGE="dsw-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/pai/pytorch:2.8.0-gpu-py311-cu128-ubuntu24.04-accl-b244fc94-1764407151"

# =============================================================================
# 提交 PyTorchJob（--command 在集群容器内执行）
# 注意：DLC 可能对 --command 做 sh 包装；嵌入脚本里若含未转义的单引号（如 conda/awk 的 '...'），
# 会截断整段命令，表现为命令后只剩错位片段。
# =============================================================================

echo "=========================================="
echo "Submitting DLC job: ${DLC_JOB_NAME}"
echo "  DATA_CONFIG:      ${DATA_CONFIG}"
echo "  CUTOFF:           ${CUTOFF}"
echo "  WORKERS:          ${WORKERS}"
echo "  PRECOMPUTE_LOG:   ${PRECOMPUTE_LOG}"
echo "  DLC_WORKERS:      ${DLC_WORKERS}"
echo "  DLC_QUEUE_NAME:   ${DLC_QUEUE_NAME}"
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
ln -s /mnt/cpfs/open_data /mnt/data/open_data

ln -s /mnt/cpfs/zbl-cpfs-new/x2robot_data/zhengwei /x2robot_data/zhengwei
ln -s /mnt/cpfs/zbl-cpfs-new/x2robot_data/collection /x2robot_data/collection

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

echo "Starting length precompute (CPU-only, no model load)..."

DLC_JOB_CMD
        printf '%s\n' \
            "export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=${X2ROBOT_AV_SEQUENTIAL_SPAN_MAX}" \
            'python -m qwenvl.tools.precompute_lengths \' \
            "    --config ${DATA_CONFIG} \\" \
            "    --cutoff ${CUTOFF} \\" \
            "    --workers ${WORKERS} > ${PRECOMPUTE_LOG} 2>&1"
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
    --worker_cpu=96 \
    --worker_memory=800Gi \
    --worker_shared_memory=800Gi \
    --worker_gpu=8
echo ""
echo "DLC job submitted: ${DLC_JOB_NAME}"
echo "Log (on cluster): work_dirs/logs/${RUN_SLUG}.log"
