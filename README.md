# VLM SFT 训练框架

在 **x2robot_dataset_v2** 数据层之上，对 **Qwen3.5-VL**（`Qwen3_5ForConditionalGeneration`，
图像 **和** 视频）做全参数微调或部分参数微调，使用 DeepSpeed ZeRO。

面向 A800-80GB（单机 ×4 或多机）。训练代码在 **`qwenvl/`** 包里；数据流水线（vision/text/epilogue
处理器、采样器）在独立的 **`x2robot_dataset_v2`** 仓库里，完全由一个 YAML 配置驱动。

---

## 目录

- [框架能力一览](#框架能力一览)
- [1. 环境](#1-环境)
- [2. 仓库结构](#2-仓库结构)
- [3. 快速开始](#3-快速开始)
- [4. 数据层（x2robot_dataset_v2）](#4-数据层x2robot_dataset_v2)
- [5. 训练配置与命令行参数](#5-训练配置与命令行参数)
- [6. 功能详解](#6-功能详解)
- [7. 断点与续训](#7-断点与续训)
- [8. 评测](#8-评测)
- [9. 常见问题](#9-常见问题)

---

## 框架能力一览

| 功能 | 开关 / 配置 | 一句话说明 |
|---|---|---|
| **序列打包**（neat-packing） | `epilogue.params.packing: true` | 一行装多条 doc、块对角注意力，带 GatedDeltaNet 状态重置，让 Qwen3.5 的线性注意力层不跨 doc 泄漏。 |
| **长度均衡采样** | `sampler.type: knapsack_packed` | 每步/每卡 token 数均衡 → 不出现 DDP 掉队；纯元数据长度预估器 + 磁盘缓存。 |
| **Cut Cross-Entropy** | `--use_cce True` | 融合 LM head + loss，不物化 `[T, vocab]` 的 fp32 logits（VLM-SFT 最大的激活）。 |
| **分组件学习率 / 冻结** | `--llm_lr --vision_lr --projector_lr --embedding_lr` | 每个组件仅当设置了对应 LR 才训练，否则冻结。 |
| **梯度累积正确的 loss** | （常开） | 按全局 token 数归一，loss/梯度缩放不随 grad_accum / 卡数漂移。 |
| **实时逐卡 MFU** | `--log_mfu`（默认关） | 打印 `mfu` + `tokens_per_sec_per_gpu`，无集合通信。 |
| **选择性梯度检查点** | `--gc_keep_lm_layers N --gc_checkpoint_vision` | 让部分层常驻，用省下来的显存换吞吐。 |
| **坏样本容忍** | `dataset.bad_sample_tolerance.enabled: true` | 解码期损坏的样本从 batch 里丢掉，而不是让整个任务崩掉。 |
| **O(1) 续训** | `--ignore_data_skip True` | 采样器从 checkpoint 直接快进到流位置，不重放 batch。 |

---

## 1. 环境

用 conda 环境 `wx`（`environment.yml` 是它的清单），激活后直接用 `python`：

```bash
conda activate wx      # environment.yml 里 name: wx
```

核心版本：

| 包 | 版本 | 备注 |
|---|---|---|
| `torch` | 2.6.0+cu124 | |
| `transformers` | 5.2.0 | 原生带 `qwen3_5`（`Qwen3_5ForConditionalGeneration`、`Qwen3VLProcessor`） |
| `accelerate` | ≥1.1.0 | transformers 5.2 Trainer 要求 |
| `deepspeed` | 0.17.1 | ZeRO |
| `flash-attn` | 2.7.4.post1 | 预编译 wheel（本机无 nvcc）；**评测时改用 sdpa**，见 [§9](#9-常见问题) |
| `flash-linear-attention` | 0.4.2 | 打包用到的 GatedDeltaNet kernel |
| `cut-cross-entropy` | 25.1.1 | 可选，`--use_cce` 用 |
| `causal-conv1d` | 1.6.0 | 可选，装了走 GDN 快路径 |

从头重建环境：`conda env create -f environment.yml`，然后按 `environment.yml` 底部装几个 editable 依赖。

`x2robot_dataset_v2`（数据层）以 **editable 包**装进 `wx`，直接 `import` 即可，无需 `PYTHONPATH`：

```bash
pip install -e <path to x2robot_dataset_v2> --no-deps
```

模型权重：`Qwen3.5-9B` 在 `/mnt/data/x2robot_v2/Models/Qwen3.5-9B`。

---

## 2. 仓库结构

```
qwenvl/
├── train/
│   ├── launcher.py     # 入口：-m qwenvl.train.launcher
│   ├── builders.py     # load_model / dataset_v2 拼接 / 打包 / LR 门控
│   └── trainer.py      # QwenVLTrainer（采样器接入、loss、MFU、选择性 GC）
├── data/
│   ├── packing.py            # apply_qwen3_5_packing_patch + 纯配置的 get_rope_index
│   ├── length_estimator.py   # 纯元数据的逐帧 token 长度预估
│   ├── length_samplers.py    # knapsack_packed 采样器（import 即注册）
│   └── bad_sample_fallback.py# 整个 bin 全坏时的 collate 兜底
├── model/
│   └── cce.py          # Cut Cross-Entropy forward 补丁
├── tools/
│   ├── precompute_lengths.py # 离线预热长度缓存
│   ├── smoke_test_packing.py # GPU 上验证打包不跨 doc 泄漏
│   └── gop30/                # 视频 GOP-30 转码脚本
└── constants.py

scripts/
├── train/sft_qwen3_5_9b.sh       # 9B 全参 SFT 默认脚本（改这里的路径 + 旋钮）
├── zero1.json                    # DeepSpeed ZeRO-1
├── eval/{lmms_eval,eval_4bench}.sh
└── convert_data_list.py          # ShareGPT JSONL -> 索引化的 multimodal_jsonl

docs/qwenvl/                      # 设计文档：length_balanced_packing / gop30_video_transcode
tests/
```

数据侧的处理器（`multimodal_jsonl` vision、`multimodal_jsonl_qwen3_5` text、
`multimodal_qwen3_5` epilogue）在 **`x2robot_dataset_v2`** 仓库里，不在本仓库。

---

## 3. 快速开始

所有东西通过一个启动脚本串起来。改脚本顶部的路径段，然后跑：

```bash
cd <本仓库>
bash scripts/train/sft_qwen3_5_9b.sh 1 8    # 参数：NNODES NGPU_PER_NODE
```

多机时脚本从环境变量读 `WORLD_SIZE` / `NPROC_PER_NODE` / `MASTER_ADDR` / `MASTER_PORT` / `RANK`，
单机时退回到那两个位置参数。脚本已设好 DeepSpeed ZeRO、bf16、FA2、打包感知的 batch 计算、
分组件 LR、`--use_cce`、W&B 日志。

脚本里通常只需改这三处：

```bash
RUN_NAME=my_run
MODEL_PATH=/mnt/data/x2robot_v2/Models/Qwen3.5-9B
DATA_CONFIG=configs/vqa_mix_qwen3_5.yml   # 仓库内示例配置；改成你自己的
```

手动调用（脚本展开后大致如此）：

```bash
python -m torch.distributed.run --nnodes 1 --nproc_per_node 4 \
    -m qwenvl.train.launcher \
    --deepspeed scripts/zero1.json \
    --model_path /mnt/data/x2robot_v2/Models/Qwen3.5-9B \
    --data_config configs/vqa_mix_qwen3_5.yml \
    --model_max_length 8192 --bf16 True --tf32 True \
    --image_min_pixels 1024 --image_max_pixels 589824 \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 64 \
    --llm_lr 6e-6 --vision_lr 2e-6 --projector_lr 6e-6 --learning_rate 6e-6 \
    --use_cce True --gradient_checkpointing True \
    --average_tokens_across_devices True --ignore_data_skip True \
    --output_dir work_dirs/my_run --num_train_epochs 1
```

---

## 4. 数据层（x2robot_dataset_v2）

训练数据由 **一个 YAML** 描述，通过 `--data_config` 传入。launcher 调用
`X2RobotDataset.from_config(cfg)` 得到 `(dataset, sampler)`；epilogue 把 tokenization + 视觉预处理
折叠进 collate，所以每个 batch 已是模型可直接吃的：
`input_ids / labels / attention_mask / pixel_values / image_grid_thw`
（视频再加 `pixel_values_videos / video_grid_thw`，打包时再加打包张量）。

### 配置结构

示例配置：`configs/vqa_mix_qwen3_5.yml`（改 source 里的 path 后即可用）。

```yaml
dataset:
  train_test_split: 1.0
  multimodal_chunk_size: 200

  bad_sample_tolerance:            # 解码期坏样本丢弃，而不是让训练崩
    enabled: true
    report_path: ./bad_samples_runtime.jsonl

  sampler:
    length_cache_dir: ./length_cache
    seed: 42
    type: knapsack_packed          # 长度均衡打包（见 §6）
    cutoff: 8192                   # 每行 token 预算

  pipeline: [vision, text, metadata]
  cache: { enabled: true, dir: ./cache_vqa, rebuild: false }

  processors:
    vision:                        # 读 `image` 或 `video` 字段
      type: multimodal_jsonl
      params:
        image_factor: 32           # patch16 * spatial_merge2 -> grid 落在 32 上
        min_pixels: 1024           # 32^2
        max_pixels: 589824         # 768^2（保持长宽比的面积预算）
        video_fps: 1
        video_maxlen: 32
        video_max_pixels: 147456   # 384^2
        video_min_pixels: 256      # 16^2
        decoder_backend: av
    text:   { type: multimodal_jsonl_qwen3_5 }   # 归一化后的对话原样透传
    epilogue:
      type: multimodal_qwen3_5                    # 官方 chat template + label masking
      params:
        processor_path: /mnt/data/x2robot_v2/Models/Qwen3.5-9B
        max_seq_length: 8192
        padding_side: right
        packing: true

  sources:
    - name: robovqa_train
      source_type: multimodal
      paths:
        - { path: /path/to/robovqa_train, episode_type: x2_multimodal, task_name: robovqa_train }
    # … 更多 source …
```

`video` 字段名不变；每条可以是整段路径，或带帧区间的 clip（**闭区间**）：

```json
{
  "text": [
    {"role": "user", "text": "<video>\n请准确描述片段中发生的机器人操作。"},
    {"role": "assistant", "text": "右臂多次调整位姿，为后续马桶清理操作做准备"}
  ],
  "video": [
    {
      "path": "/path/to/faceImg.mp4",
      "start_frame": 20,
      "end_frame": 163
    }
  ]
}
```

- `start_frame` / `end_frame` 为 **0-based 帧序号**，闭区间 `[start_frame, end_frame]`（两端都包含）；越界会 clamp。
- 在窗口内按 `video_fps` / `video_maxlen` 采样。
- 喂给模型的时间戳从 clip 起点算起（第一帧 `0.0 seconds`），不是源视频绝对时间。
- `length_cache` 只读容器元数据（帧数/时长/分辨率），按 clip 窗口估算，不解码像素。

流水线里已落实的关键正确性点：

- **官方 Qwen3.5 chat template**（`apply_chat_template`）；**所有** assistant 轮都被监督，
  prompt / 图像 token 是 `-100`。
- 每个 assistant 轮注入空的 `<think>\n\n</think>\n\n` 块（`per_turn_think`，默认开）。训练是
  **非 thinking** 模式——评测须用 `enable_thinking=False` 对齐（见 [§8](#8-评测)）。
- Grounding 坐标是 `[0,1000)` 归一化（Qwen3.5 约定），原样透传。
- **全链路同一个 pixel 预算**：YAML 的 `vision.min/max_pixels` 必须等于 HF processor 的
  （`--image_min_pixels/--image_max_pixels`），不一致会**直接报错**。

视频 source 建议额外做 **GOP-30 转码**（每 30 帧一个关键帧），避免训练期长视频解码卡死；
见 `docs/qwenvl/gop30_video_transcode.md`。

---

## 5. 训练配置与命令行参数

`qwenvl.train.launcher` 扩展了 `transformers.TrainingArguments`。额外的旋钮：

**数据 / 模型**
- `--data_config <yaml>` — dataset_v2 配置（必填）。
- `--model_path` — HF 模型目录（默认 `Qwen/Qwen3.5-9B`）。
- `--model_max_length` — 每行最大 token 数（与 `epilogue.max_seq_length` 一致；`8192`）。
- `--image_min_pixels / --image_max_pixels` — HF image processor 预算（必须与 YAML 一致）。
- `--attn_implementation flash_attention_2` — 训练默认。

**分组件学习率**（组件仅当设置了对应 LR 才训练，未设置 = 冻结）：
- `--llm_lr` — 语言模型（含 embeddings / lm_head）。
- `--vision_lr` — 视觉编码器。
- `--projector_lr` — patch merger（别名 `--merger_lr`）。
- `--embedding_lr` — 单独给 embed/lm_head 行的 LR（新增 token 时用）。
- `--learning_rate` — 给调度器打底 + 未显式设置的旋钮兜底。至少设一个，否则报错。

**Loss / 显存**
- `--use_cce True` — Cut Cross-Entropy（大loss时是有损的，非显存紧张不要加）。`--cce_impl {cce,torch_compile}`、`--cce_upcast`
  用显存换梯度精度（见 [§6](#6-功能详解)）。
- `--loss_reduction_scope {batch,sample}` — token 均值（默认）vs 样本均值（每条 doc 的 token 均值再对样本取平均；`sequence` 为 `sample` 的别名）。
- `--average_tokens_across_devices True` — 打包下跨卡 loss 缩放正确所必需。

**吞吐 / 日志**
- `--gradient_checkpointing True` — 全量检查点。
- `--gc_keep_lm_layers N` — 让 32 层 LM 中的 N 层常驻（省去重算），盯着峰值显存往上调。
- `--gc_checkpoint_vision {True,False}` — `False` 让 27 层 ViT 常驻。
- `--log_mfu {True,False}` — 逐卡 MFU 日志（默认 False；`--log_mfu True` 开）。`--peak_flops_per_gpu` 是分母（单卡 bf16 峰值 FLOPs，默认 `312e12` = A100/A800）；换卡型要改。

> **`knapsack_packed`**：一个 bin = 一个打包行 = 一个 micro-step → `--per_device_train_batch_size` **必须是 1**。

---

## 6. 功能详解

### 序列打包（neat-packing）
设 `epilogue.params.packing: true`。epilogue 把一个 micro-batch 折成一个 `bsz=1` 的行，每条 doc 的
`position_ids` 各自从 0 重排；launcher 应用 `apply_qwen3_5_packing_patch()`，让全注意力层（显式
`cu_seq_lens`）**和** GatedDeltaNet 线性注意力层（conv state + delta 递归）都在 doc 边界处重置——
少了 GDN 那一半，Qwen3.5 有 3/4 的层会跨 doc 泄漏。升级 `transformers`/`fla` 后可用
`qwenvl/tools/smoke_test_packing.py` 复验无泄漏。

### 长度均衡采样
DDP 需要每步/每卡 token 数大致相等。一个纯元数据的 **长度预估器**（读图像/视频头，不解码）驱动
`knapsack_packed` 采样器：贪心把行填到 token `cutoff`，一个 bin = 一步；bin 补齐到 `world_size`
整数倍，各卡步调一致。

离线预热缓存，训练可秒启动（否则首次运行时现算）：

```bash
python -m qwenvl.tools.precompute_lengths \
    --config configs/vqa_mix_qwen3_5.yml --cutoff 8192 --workers 40
```

无法测量的帧会被丢弃（不编造长度），并写 `lengths.*.report.jsonl` 旁路文件。缓存 key = 分辨率 /
视频参数 / 模型 / `max_seq_length` 的哈希（不含 source 路径），媒体不变时重指向 source 缓存仍命中。
设计细节见 `docs/qwenvl/length_balanced_packing.md`。

### Cut Cross-Entropy（`--use_cce`）（无显存紧缺的情况不要加，大loss时有精度损失，暂时未验证）
把 LM head + 交叉熵融合进一个 kernel，不物化 `[T, vocab]` 的 logits（T=8192、9B 时约 15 GB）。
loss 与 fp32 CE 相符到 ~5e-6。精度/显存档位：

| `--cce_impl` | 9B 额外峰值显存 | d/dhidden 误差（峰化 SFT） |
|---|---|---|
| `cce`（默认） | ~1.1 GB | ~2–4% |
| `torch_compile` | ~3.3 GB | ~0.5% |
| `torch_compile --cce_upcast True` | ~6.5 GB | ~1e-7（与 stock 位级对齐） |

三档都比不用 CCE（~15 GB）省。`cce` 对预训练模型的 SFT 足够（softmax 峰化区间，误差被 bf16 训练
噪声淹没）；想要精度余量就上 `torch_compile`。需要 `cut-cross-entropy` 包。

### 分组件 LR / 冻结
每个组件仅当设了对应 LR 才训练，未设 = 冻结。
- 全量训练：`--llm_lr 6e-6 --vision_lr 2e-6 --projector_lr 6e-6`。
- 只训 LM、冻结视觉塔：省掉 `--vision_lr`（默认脚本即全量训练）。

### 实时 MFU（`--log_mfu`，默认关）
`--log_mfu True` 打开后，每个日志步报 rank 0 的 `mfu` 和 `tokens_per_sec_per_gpu`（用本卡
`input_ids.numel()`，无 gather）。分母是 `--peak_flops_per_gpu`（默认 `312e12` = A100/A800 的 bf16 峰值，换别的卡型要改）。
不要开 `--include_num_input_tokens_seen`——它每步的 all-gather 在不均匀打包下会挂 NCCL。

---

## 7. 断点与续训

- Checkpoint 落在 `--output_dir/checkpoint-<step>`，`--save_steps` / `--save_total_limit` 照常用。
- **O(1) 续训**：每个 checkpoint 旁写 `x2_sampler_state.json`（采样器的 `consumed`/`epoch`）。重启时
  launcher 自动检测最新 checkpoint、恢复流位置继续训。带 **`--ignore_data_skip True`**（否则 HF 会
  再重放一遍 batch）。直接重跑同一条启动命令即可。

---

## 8. 评测

评测用独立的 **lmms-eval** conda 环境（不影响训练环境）：

```bash
bash scripts/eval/lmms_eval.sh  [TASKS] [GPUS] [LIMIT]   # 通用任务，如 mmstar/mmmu_val/ai2d
bash scripts/eval/eval_4bench.sh [TASKS]                 # 具身/空间 4-bench
```

两条铁律（已写进脚本）：
- **用 `attn_implementation=sdpa`，不要 FA2**：flash-attn 2.7.4 在这个架构（`head_dim=256`、线性/全
  注意力混合）生成时会崩。
- **`enable_thinking=False`**：训练是非 thinking 的，评测开 think 会跑偏分布且慢很多，结果不可比。

结果落在 `eval_results/`。

---

## 9. 常见问题

- **长视频解码卡死 → NCCL 超时。** 某个 rank 卡在 DataLoader 解码一个又长又高清的视频。修法：视频
  source 做 GOP-30 转码 **并且** 设 `X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128`（两者都要，脚本已设）。
  见 `docs/qwenvl/gop30_video_transcode.md`。
- **`--per_device_train_batch_size` 必须为 `1`。** `knapsack_packed` 一个 bin 就是一个 batch；
  设成别的值 trainer 会报错。
- **pixel 预算不匹配。** YAML 的 `vision.min/max_pixels` 必须等于 `--image_min_pixels/--image_max_pixels`。
- **开了 `--use_cce` 但报错说 forward 返回了 logits。** CCE 补丁没生效（缺 `cut-cross-entropy` 包，或
  模型类不对）；装上包，或去掉 `--use_cce`。

---

测试在 `wx` python 下、仓库根目录跑（`x2robot_dataset_v2` 已是 editable 包，无需 `PYTHONPATH`）。
