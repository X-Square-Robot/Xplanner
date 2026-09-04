#!/bin/bash
# Qwen3.5-VL 9B 全参数 SFT 默认脚本（qwen 原生视觉塔，LM/vision/projector 三组件全训）。
#   单机:  bash scripts/train/sft_qwen3_5_9b.sh [NNODES] [NGPU_PER_NODE]
#   多机:  由启动器设好 WORLD_SIZE / NPROC_PER_NODE / MASTER_ADDR / MASTER_PORT / RANK 后直接跑。

# ---- 路径（改这里）----------------------------------------------------------
MODEL_PATH=/mnt/data/x2robot_v2/Models/Qwen3.5-9B
DATA_CONFIG=configs/vqa_mix_qwen3_5.yml         # 仓库内示例配置；改成你自己的
OUTPUT_DIR=work_dirs/qwen3_5_vl_sft
RUN_NAME=qwen3_5_9b_sft

export WANDB_PROJECT=qwen3_5_vl_sft

# ---- 分布式（WORLD_SIZE=节点数；单机用位置参数）-----------------------------
NNODES=${WORLD_SIZE:-${1:-1}}
NPROC_PER_NODE=${NPROC_PER_NODE:-${2:-8}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-16667}
NODE_RANK=${RANK:-0}
echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  NODE_RANK=$NODE_RANK"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128    # 视频走关键帧 seek 解码（配合 GOP-30，见 docs）

# 多机 RoCE/NCCL：按你的网卡取消注释并调整（单机忽略这一段）。
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0

# ---- 批大小（knapsack_packed：一个 bin=一行=一步，LOCAL 必须为 1）-----------
GLOBAL_ROWS_PER_STEP=256      # 每个优化步的全局打包行数（× cutoff 8192 ≈ 2M tok/step）
LOCAL_BATCH_SIZE=1
GRAD_ACCUM=$(( GLOBAL_ROWS_PER_STEP / (NNODES * NPROC_PER_NODE * LOCAL_BATCH_SIZE) ))
echo "grad_accum=$GRAD_ACCUM"

python -m torch.distributed.run \
    --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK \
    -m qwenvl.train.launcher \
    --deepspeed scripts/zero1.json \
    --model_path $MODEL_PATH \
    --data_config $DATA_CONFIG \
    --attn_implementation flash_attention_2 \
    --model_max_length 8192 \
    --image_min_pixels 1024 --image_max_pixels 589824 \
    --bf16 True --tf32 True \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs 1 \
    --learning_rate 6e-6 --llm_lr 6e-6 --vision_lr 2e-6 --projector_lr 6e-6 \
    --weight_decay 0.0 --warmup_ratio 0.03 \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --gradient_checkpointing True \
    --average_tokens_across_devices True \
    --ignore_data_skip True \
    --save_strategy steps --save_steps 1000 --save_total_limit 2 \
    --logging_steps 1 \
    --dataloader_num_workers 16 --dataloader_prefetch_factor 8 \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --report_to wandb --run_name $RUN_NAME
