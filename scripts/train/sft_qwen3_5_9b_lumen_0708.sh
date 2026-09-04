#!/bin/bash
# Qwen3.5-VL 9B 全参数 SFT 默认脚本（qwen 原生视觉塔，LM/vision/projector 三组件全训）。
#   单机:  bash scripts/train/sft_qwen3_5_9b_lumen_0708.sh [NNODES] [NGPU_PER_NODE]
#   多机:  由启动器设好 WORLD_SIZE / NPROC_PER_NODE / MASTER_ADDR / MASTER_PORT / RANK 后直接跑。

# ---- 路径 ----
MODEL_PATH=/mnt/cpfs/zbl-cpfs-new/Models/Qwen3.5-9B # Qwen3.5-VL 预训练权重目录（HF 格式）
DATA_CONFIG=configs/vqa_mix_qwen3_5_lumen_0708.yml # 数据 YAML：vision/text/epilogue 流水线、采样器、打包
OUTPUT_DIR=work_dirs/qwen3_5_vl_sft # checkpoint / 日志根目录
RUN_NAME=qwen3_5_9b_sft_lumen_0708 # 实验名；输出在 $OUTPUT_DIR/$RUN_NAME，也是 wandb run 名

# ---- 训练超参（改这里）----
MODEL_MAX_LENGTH=8192 # 每行最大 token；须与 YAML epilogue.max_seq_length / sampler.cutoff 一致
IMAGE_MIN_PIXELS=1024 # 图像最小像素预算（32²）；须与 YAML vision.min_pixels 一致
IMAGE_MAX_PIXELS=589824 # 图像最大像素预算（≈768²）；须与 YAML vision.max_pixels 一致
NUM_TRAIN_EPOCHS=1 # 训练 epoch 数（数据集遍历轮数）

LEARNING_RATE=8e-6 # 基础 LR：给 scheduler 打底；未单独设 knob 的组件兜底
LLM_LR=8e-6 # 语言模型 LR（含 embed / lm_head）；设了才训练 LM，未设=冻结
VISION_LR=1e-6 # 视觉编码器（ViT）LR；设了才训练 vision，未设=冻结
PROJECTOR_LR=8e-6 # patch merger LR；设了才训练 merger，未设=冻结
WEIGHT_DECAY=0.0 # 权重衰减（0=关闭；Norm/bias 仍不衰减）
WARMUP_RATIO=0.03 # 前 3% 步线性 warmup

# ---- 批大小（knapsack_packed：一个 bin=一行=一步，LOCAL 必须为 1）----
GLOBAL_ROWS_PER_STEP=256 # 全局每 optimizer step 的打包行数（×8192 ≈ 2M tok/step）
LOCAL_BATCH_SIZE=1 # 每卡 micro-batch；knapsack 模式必须为 1

# ---- 保存 / 日志 ----
SAVE_STEPS=1000 # 每 N 个 optimizer step 存一次 checkpoint
SAVE_TOTAL_LIMIT=8 # 最多保留 N 个 checkpoint，超出自动删旧的
LOGGING_STEPS=1 # 每 N step 打 loss 等到 console / wandb
DATALOADER_NUM_WORKERS=16 # 每卡 DataLoader 子进程数（解码图像/视频、collate）
DATALOADER_PREFETCH_FACTOR=4 # 每 worker 预取 batch 数；workers×prefetch=预加载深度

export WANDB_PROJECT=qwen3_5_vl_sft_lumen_0708 # wandb 项目名（网页分组）；配合 --report_to wandb

# ---- 分布式（WORLD_SIZE=节点数；单机用位置参数）----
NNODES=${WORLD_SIZE:-${1:-1}} # 节点数；多机用 env WORLD_SIZE，单机用 $1
NPROC_PER_NODE=${NPROC_PER_NODE:-${2:-4}} # 每节点 GPU 数；默认 $2 或 4
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} # 多机 rendezvous 地址
MASTER_PORT=${MASTER_PORT:-16667} # 多机 rendezvous 端口
NODE_RANK=${RANK:-0} # 当前节点 rank（多机用 env RANK）
echo "NNODES=$NNODES  NPROC_PER_NODE=$NPROC_PER_NODE  NODE_RANK=$NODE_RANK"

GRAD_ACCUM=$(( GLOBAL_ROWS_PER_STEP / (NNODES * NPROC_PER_NODE * LOCAL_BATCH_SIZE) )) # 梯度累积=全局行数/(节点×GPU)
echo "grad_accum=$GRAD_ACCUM"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # 缓解 CUDA 显存碎片（长序列+GC 时更稳）
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128 # 视频解码走 seek+短窗口；须配合 GOP-30 转码

# 多机 RoCE/NCCL：按你的网卡取消注释并调整（单机忽略这一段）。
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0

python -m torch.distributed.run \
    --nnodes $NNODES --nproc_per_node $NPROC_PER_NODE \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $NODE_RANK \
    -m qwenvl.train.launcher \
    --deepspeed scripts/zero1.json \
    --model_path $MODEL_PATH \
    --data_config $DATA_CONFIG \
    --attn_implementation flash_attention_2 \
    --model_max_length $MODEL_MAX_LENGTH \
    --image_min_pixels $IMAGE_MIN_PIXELS --image_max_pixels $IMAGE_MAX_PIXELS \
    --bf16 True --tf32 True \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE --llm_lr $LLM_LR --vision_lr $VISION_LR --projector_lr $PROJECTOR_LR \
    --weight_decay $WEIGHT_DECAY --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' \
    --gradient_checkpointing True \
    --average_tokens_across_devices True \
    --ignore_data_skip True \
    --save_strategy steps --save_steps $SAVE_STEPS --save_total_limit $SAVE_TOTAL_LIMIT \
    --logging_steps $LOGGING_STEPS \
    --dataloader_num_workers $DATALOADER_NUM_WORKERS --dataloader_prefetch_factor $DATALOADER_PREFETCH_FACTOR \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --report_to wandb --run_name $RUN_NAME
