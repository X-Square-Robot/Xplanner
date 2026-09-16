<div align="center">

<h1><strong>X-Planner</strong></h1>

<h3>面向具身智能的事件结构化任务规划</h3>

</div>

<div id="top" align="center">

[![项目主页](https://img.shields.io/badge/Homepage-%F0%9F%8C%90-116466?style=flat)](https://x-square-robot.github.io/Xplanner/)
[![代码](https://img.shields.io/badge/Code-GitHub-181717?style=flat&logo=github)](https://github.com/X-Square-Robot/Xplanner)
[![论文](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat&logo=adobeacrobatreader&logoColor=white)](docs/paper/X_Planner_Event_Structured_Task_Planning_for_Embodied_Intelligence.pdf)
[![模型](https://img.shields.io/badge/Model-X--Planner--9B--0916-ffd21e?style=flat&logo=huggingface)](https://huggingface.co/x-square-robot/X-Planner-9B-0916)
[![评测集](https://img.shields.io/badge/Benchmark-xplanner--benchmark-4c8bf5?style=flat&logo=huggingface)](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)
[![许可证](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br>

<a href="README.md">English</a> | <strong>简体中文</strong>

</div>

**X-Planner** 是一个面向**长时序机器人操作**的任务规划前端。它接收高层任务指令、同步的
多视角观测以及可选的执行历史，将下一步行为表示为动作落地的事件，并将该表示传递给下游
世界—动作模型。

<div align="center">
  <img src="assets/X-Planner.jpg" alt="X-Planner 系统概览" width="90%">
</div>

**核心思路：**
- **事件落地的数据。** 对示范数据进行同步，并按照任务（Task）/子任务（Subtask）/动作
  （Action）/片段（Segment）的嵌套层级组织。
- **结构化规划状态。** 将训练样本确定性地构造成初始计划、进行中的事件状态或回合结束状态
  JSON。
- **两种规划接口。** 事件模式提供可读的事件状态；统一模式通过阶梯式解码（Staircase
  Decoding）使用紧凑的隐式规划状态。

## 最新进展

- 2026-09-16：发布 [X-Planner-9B-0916](https://huggingface.co/x-square-robot/X-Planner-9B-0916)
  推理权重（`checkpoint-10000`，BF16），以及包含 1,500 个 episode、3,490 个视频的
  [XPlanner benchmark](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)，支持在 Dataset Preview 中查看多视角视频。
- 2026-09：仓库结构已与 X-Planner 技术报告对齐，并完成首次开源审查前的整理。

## 仓库结构

| 路径 | 内容 |
| --- | --- |
| `x_planner/modeling/` | Qwen3.5-VL 建模扩展与视觉塔支持 |
| `x_planner/trainer/` | 分布式 SFT 启动器、模型/数据组装与训练器 |
| `x_planner/data/pipeline/` | 回合归一化、层级校验与快照工具 |
| `x_planner/data/discovery/` | 可扩展的数据源发现、校验、采样与不可变快照 |
| `x_planner/data/context/` | 初始计划及历史条件数据构建 |
| `x_planner/data/event_states/` | 当前结构化事件状态的物化、留出、训练与推理 |
| `x_planner/evaluation/rollout/` | 离线预测、rollout 分析与审阅图库 |
| `scripts/` | 面向用户的训练、推理、数据导出与评估入口 |
| `workspace/example/` | 可移植的示例配置；请在本地替换 `/path/to/...` |
| `benchmarks/xplanner_eval/` | 评测数据的格式与 schema |
| `benchmarks/real_robot/` | 报告中的评测套件、任务进度协议与汇总结果 |

公开名称有意描述模块职责，而不暴露内部实验版本。保存的数据保留明确的模式版本，方便检查旧快照。

## 安装

创建 Python 3.10 环境并安装训练依赖：

```bash
conda create -n xplanner python=3.10 -y
conda activate xplanner
pip install -e '.[train]'
```

X-Planner 的数据后端单独安装，不把内部数据源随代码仓库分发。公开的
[`xDataset`](https://github.com/X-Square-Robot/xDataset) 提供了 WALL-WM 使用的通用
事件级视频/动作数据接口：

```bash
git clone https://github.com/X-Square-Robot/xDataset.git ../xDataset
pip install --no-deps -e ../xDataset
```

开发期间使用的精确 CUDA 环境见 `environment.yml`。目标 GPU 支持时，请单独安装 FlashAttention。

需要注意：截至 2026 年 9 月 10 日，公开 `xDataset/main` 尚未包含 X-Planner 事件状态运行时
使用的全部 `multimodal_jsonl`、Qwen3.5 processor 和 whole-episode reader 模块。因此它可以
用于通用视频/动作路径，但还不能直接替代 X-Planner 事件状态训练和完整 episode 推理所需的
兼容后端。启动器会在运行前检查这些模块并明确报出缺失项；待兼容的公开 backend snapshot
发布后，再将其作为默认依赖。

## 数据准备

复制示例配置，并将其中的单一数据源指向本地已建立索引的多模态 JSONL 数据集：

```bash
cp workspace/example/data/planner_sft.yml workspace/local_planner_sft.yml
```

每条事件状态记录包含同步视角、任务指令、可选的计划/历史上下文以及一个紧凑 JSON 目标。当前模式和渲染逻辑位于 `x_planner/data/event_states/schema.py` 与 `x_planner/data/event_states/prompt.py`。

完整物化流程通过稳定的回合标识和清单 SHA-256 强制隔离评估留出集：

```bash
cp .env.example .env
# 填写 XPLANNER_MODEL_PATH、兼容的 XPLANNER_DATASET_REPO、
# XPLANNER_EVALUATION_MANIFEST 和 XPLANNER_EVALUATION_SHA256。

bash scripts/train/train_event_planner.sh prepare \
  /path/to/event_snapshot /path/to/prepared_data
```

## 训练

使用可移植数据配置对 Qwen3.5-VL 进行监督微调：

```bash
MODEL_PATH=/path/to/Qwen3.5-9B \
DATA_CONFIG=workspace/local_planner_sft.yml \
OUTPUT_DIR=work_dirs/x_planner_sft \
bash scripts/train/train_qwen35_sft.sh 1 8
```

候选发布版的事件状态启动器目前提供有边界的验证运行：

```bash
bash scripts/train/train_event_planner.sh unit
bash scripts/train/train_event_planner.sh smoke-single /path/to/event_snapshot
```

其默认拒绝放行的检查会在优化开始前验证冻结快照摘要、评估留出集隔离、损失掩码约定以及断点续训元数据。

## 推理

下载已发布的 [X-Planner-9B-0916](https://huggingface.co/x-square-robot/X-Planner-9B-0916)
推理 checkpoint（约 94.1 亿参数，BF16 权重约 18.82 GB）：

```bash
hf download x-square-robot/X-Planner-9B-0916 \
  --local-dir checkpoints/X-Planner-9B-0916
```

模型卡提供独立的 Transformers 加载示例。生成结构化事件状态时，先按上文安装兼容的数据后端，
准备 event snapshot，再运行：

```bash
python scripts/inference/run_event_planner.py \
  --checkpoint checkpoints/X-Planner-9B-0916 \
  --snapshot /path/to/event_snapshot \
  --output-dir work_dirs/inference
```

预测结果会依据训练时使用的同一套紧凑 JSON 约定进行解析和校验。
benchmark 的视频清单不是 event snapshot，不能直接传给 `--snapshot`。

## 评估

评测与训练相关内容按用途分为三部分：

1. [Hugging Face 上的 xplanner-benchmark](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)
   已提供 1,500 个 episode 的视频、episode 级元数据、校验值和 Dataset Preview。
   已发布目录格式与独立的完整时序标注导出约定见 [benchmarks/xplanner_eval/](benchmarks/xplanner_eval/)。
2. `benchmarks/real_robot/` 记录技术报告中的推理操作与泛化评测套件，以及 Task Progress 评测协议。
3. 训练流程使用独立的本地评估留出清单；它不是训练数据，也不会提交到本仓库。

此外还提供通用多模态评估封装：

```bash
CKPT=checkpoints/X-Planner-9B-0916 bash scripts/evaluation/run_lmms_eval.sh mmstar 0
CKPT=checkpoints/X-Planner-9B-0916 TASKS=erqa,vsibench \
  bash scripts/evaluation/run_embodied_benchmarks.sh
```

下载完整 benchmark：

```bash
hf download x-square-robot/xplanner-benchmark --repo-type dataset \
  --local-dir data/xplanner-benchmark
```

Dataset Preview 中每行对应一个 episode，各相机列可以播放对应的视频。当前发布支持离线分析和任务规划研究；
完整时序评分标注、真机逐次试验记录属于独立发布内容。评测范围见
[benchmarks/README.md](benchmarks/README.md)。论文中的历史结果不代表本次 `checkpoint-10000` 的新评测结果。

## 引用

配套技术报告已随仓库提供：
[`X_Planner_Event_Structured_Task_Planning_for_Embodied_Intelligence.pdf`](docs/paper/X_Planner_Event_Structured_Task_Planning_for_Embodied_Intelligence.pdf)。

```bibtex
@article{xplanner2026event,
  title   = {X-Planner: Event-Structured Task Planning for Embodied Intelligence},
  author  = {{X Square Robot Team}},
  year    = {2026},
  note    = {Technical report}
}
```

## 许可证

本仓库源代码采用 [MIT License](LICENSE)；
[X-Planner-9B-0916 模型权重](https://huggingface.co/x-square-robot/X-Planner-9B-0916)采用 Apache-2.0。
benchmark 媒体和标注保留各上游来源的使用条款，来源及许可证说明见
[数据集卡片](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)。
