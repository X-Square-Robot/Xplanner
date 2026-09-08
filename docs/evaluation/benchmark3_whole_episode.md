# V5.3 Benchmark3 与完整 episode 推理

这组入口把 `luhao/planner` 最新快照中的 Benchmark3 进度评测和完整 episode 推理整理到 `x_planner` 包。训练数据、episode spec、checkpoint 和 checkpoint pin 都是外部输入，不随仓库分发。

## 入口

| 用途 | 入口 |
| --- | --- |
| 确定性选择 20 个 Benchmark3 Action episode | `scripts/evaluation/prepare_benchmark3_specs.py` |
| Benchmark3 task/Action progress 评测 | `x_planner.evaluation.benchmark3.progress_parity` |
| 单 episode 推理 | `scripts/evaluation/run_whole_episode.py` |
| 多 GPU 批量推理 | `scripts/evaluation/run_whole_episode_batch.py` |
| 结果和 Initial Plan 审计 | `scripts/evaluation/audit_whole_episode.py` |
| 视频渲染 | `x_planner.evaluation.whole_episode.video` |

运行环境需要安装项目使用的 torch/transformers、PyAV、Pillow。仓库已经固定了
`third_party/x2robot_dataset_v2` submodule，首次 checkout 后执行
`git submodule update --init --recursive` 即可；也可以通过 `XPLANNER_DATASET_REPO`
覆盖数据层路径。

## Benchmark3

`prepare_benchmark3_specs.py` 会生成固定的 episode bundle。默认路径可通过环境变量覆盖，也可以直接传入输出目录：

```bash
python scripts/evaluation/prepare_benchmark3_specs.py \
  --bundle /path/to/new_benchmark3_bundle
```

进度评测使用五种 context（无 memory、oracle memory、oracle memory 加 oracle plan、rolling memory、rolling memory 加 model plan），并同时比较 Action 中点和固定 stride 的 anchor schedule。非法输出按协议计入最大误差。

```bash
python -m x_planner.evaluation.benchmark3.progress_parity \
  --batch-spec /path/to/new_benchmark3_bundle/batch_spec.json \
  --checkpoint-pin /path/to/checkpoint_pin \
  --output-dir /path/to/progress_results \
  --devices cuda:0 \
  --dense-stride 10
```

## 完整 episode

单集推理会为每个 anchor 保持独立的模型预测 memory；设置 `--anchor-stride-frames` 可启用固定步长采样，省略时使用标签中点。

```bash
python scripts/evaluation/run_whole_episode.py \
  --episode-spec /path/to/episode_spec.json \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/episode_results \
  --device cuda:0 \
  --anchor-stride-frames 10 \
  --initial-max-new-tokens 4096 \
  --execution-max-new-tokens 512 \
  --render
```

批量 runner 默认使用 joint Action + Segment；Action-only 数据可以传入 `--profile action_only`：

```bash
python scripts/evaluation/run_whole_episode_batch.py \
  --batch-spec /path/to/batch_spec.json \
  --checkpoint /path/to/checkpoint \
  --output-dir /path/to/batch_results \
  --devices cuda:0,cuda:1 \
  --expected-episodes 20 \
  --anchor-stride-frames 10
```

输出目录包含运行 manifest、checkpoint pin、逐集预测、汇总指标以及可选的视频结果。评测代码使用 `x_planner.data.event_states` 中统一的 schema、prompt、memory 和 episode materializer，避免同一协议出现两套模块命名。
