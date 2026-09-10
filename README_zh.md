# X-Planner

### 面向具身智能的事件结构化任务规划

<p align="right">
  <a href="README.md">English</a> | <strong>简体中文</strong>
</p>

X-Planner 是一个面向长时序机器人操作的任务规划前端。它接收高层任务指令、同步的多视角观测以及可选的执行历史，将下一步行为表示为动作落地的事件，并将该表示传递给下游世界—动作模型。

<p align="center">
  <img src="assets/X-Planner.jpg" alt="X-Planner 系统概览" width="100%">
</p>

本项目围绕配套技术报告中的三个核心思路展开：

- **事件落地的数据。** 对示范数据进行同步，并按照任务（Task）/子任务（Subtask）/动作（Action）/片段（Segment）的嵌套层级组织。
- **结构化规划状态。** 将训练样本确定性地构造成初始计划、进行中的事件状态或回合结束状态 JSON。
- **两种规划接口。** 事件模式提供可读的事件状态；统一模式通过阶梯式解码（Staircase Decoding）使用紧凑的隐式规划状态。



## 最新进展

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

X-Planner 使用 [`X-Square-Robot/xDataset`](https://github.com/X-Square-Robot/xDataset) 提供的数据后端。请将其安装在本仓库同级目录：

```bash
git clone https://github.com/X-Square-Robot/xDataset.git ../xDataset
pip install --no-deps -e ../xDataset
```

开发期间使用的精确 CUDA 环境见 `environment.yml`。目标 GPU 支持时，请单独安装 FlashAttention。

## 数据准备

复制示例配置，并将其中的单一数据源指向本地已建立索引的多模态 JSONL 数据集：

```bash
cp workspace/example/data/planner_sft.yml workspace/local_planner_sft.yml
```

每条事件状态记录包含同步视角、任务指令、可选的计划/历史上下文以及一个紧凑 JSON 目标。当前模式和渲染逻辑位于 `x_planner/data/event_states/schema.py` 与 `x_planner/data/event_states/prompt.py`。

完整物化流程通过稳定的回合标识和清单 SHA-256 强制隔离评估留出集：

```bash
cp .env.example .env
# 填写 XPLANNER_MODEL_PATH、XPLANNER_DATASET_REPO、
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

从已训练的检查点生成结构化事件状态：

```bash
python scripts/inference/run_event_planner.py \
  --checkpoint /path/to/checkpoint \
  --snapshot /path/to/event_snapshot \
  --output-dir work_dirs/inference
```

预测结果会依据训练时使用的同一套紧凑 JSON 约定进行解析和校验。

## 评估

评测与训练相关内容按用途分为三部分：

1. `benchmarks/xplanner_eval/` 定义确定性的 1,500 条回合评测数据及其可移植目录格式；发布前会先审计媒体和标注完整性。
2. `benchmarks/real_robot/` 记录技术报告中的推理操作与泛化评测套件，以及 Task Progress 评测协议。
3. 训练流程使用独立的本地评估留出清单；它不是训练数据，也不会提交到本仓库。

此外还提供通用多模态评估封装：

```bash
CKPT=/path/to/checkpoint bash scripts/evaluation/run_lmms_eval.sh mmstar 0
CKPT=/path/to/checkpoint TASKS=erqa,vsibench \
  bash scripts/evaluation/run_embodied_benchmarks.sh
```

关于当前可复现的内容，以及仍需发布审批的媒体、逐次试验记录和评分细则，请参阅 [benchmarks/README.md](benchmarks/README.md)。

### 复现快照

仓库中保留的 V5.3 progress/MAE 评测快照对应 `wall-x` 的 `luhao/planner` 分支，提交为
`8151641c`，使用 `checkpoint-80500`。评测入口和 checkpoint 要求见
[`docs/evaluation/evaluation_whole_episode.md`](docs/evaluation/evaluation_whole_episode.md)。
Checkpoint 文件不存放在 Git 中，运行评测时通过 `--checkpoint`（或 `CKPT`）传入本地目录。

可移植评测数据统一命名为 **XPlanner-OpenBenchmark**，包含 1,500 个 episode、3,490 个同步视频引用。
使用时请通过本地路径（例如 `/path/to/XPlanner-OpenBenchmark`）提供数据目录；正式发布时应将
manifest 和校验和作为带版本的外部下载提供，不提交到代码仓库。

## 模型与数据集

- X-Planner 评测 checkpoint：**V5.3 `checkpoint-80500`**（外部文件，不提交到 Git）。
- 事件级训练数据：**不随代码一起开源**，因为其中的标注和数据来源存在再分发限制。
- X-Planner 评测数据：**完整的 1,500 个 episode 已在内部集群整理完成**；公开发布仍需完成来源授权和版本化下载包。

模型权重和完整媒体应在 Git 仓库之外进行版本管理。代码仓库会固定其发布 ID 和校验和。

## 引用

X-Planner 技术报告获得稳定的公开标识后，将在此补充引用信息。

## 许可证

本仓库源代码采用 [MIT License](LICENSE) 发布。模型权重、数据集、媒体及第三方组件仍受各自许可证和使用条款约束。重新分发派生文件前，请阅读 [docs/data_sources.md](docs/data_sources.md)。
