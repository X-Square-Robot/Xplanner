# 长度均衡的序列打包（Length-Balanced Packing）

Qwen3.5-VL SFT 用序列打包（一行装多条 doc，块对角注意力）。要让打包在 DDP 下高效且不崩，需要：

1. 每步/每卡的 token 数尽量相等 —— 否则 all-reduce 被最慢的卡拖住，且长行更易 OOM；
2. 知道每条 doc 的 token 长度 —— 才能把它们均匀装进固定预算的行里。

三个组件配合：**长度预估器 → 长度感知采样器 → 打包**。

---

## 1. 长度预估器（`x_planner/data/length_estimator.py`）

对每条 doc（"frame"）估算它进模型后的 token 数，**只读元数据、不解码**：

- 文本：真实 tokenize。
- 图像：读图像头拿宽高 →`smart_resize`（与训练同一套 `min/max_pixels`）→ 网格 token 数。
- 视频：`av` 探针拿帧数/时长/fps → 若 JSONL 带闭区间帧号 `start_frame`/`end_frame` 则先取该窗口，再按采样规则算帧数 → 网格 token 数（不解码像素）。

相对真实 epilogue 的误差 ≤ 1 token。要点：

- **无法测量的帧记 `-1`（丢弃），绝不编造长度。** 编造的短长度会撑爆它所在的打包行，导致
  epilogue 右截断、连累同一行里其它 doc 的 label。损坏原因写进 `lengths.*.report.jsonl` 旁路
  文件；丢弃率超阈值会告警。
- 图像宽高**总是从真实文件读**（忽略 JSONL 里的 width/height）。
- I/O 读取带重试，避免 CPFS 抖动被误判为损坏。

### 缓存

每个 source 一个 `.npy`。缓存 key = 分辨率 / 视频参数 / 模型 / `max_seq_length` / 估算逻辑版本
的哈希 + source 签名（**不含 source 路径**），所以媒体不变时重指向 source 缓存仍命中。缓存目录 =
配置里的 `sampler.length_cache_dir`（默认 `./length_cache`，相对训练启动目录）。

### 离线预热（可选，加速启动）

```bash
python -m x_planner.tools.precompute_lengths \
    --config workspace/local_planner_sft.yml --cutoff 8192 --workers 40
```

不预热的话，首次训练启动时现算并落缓存。

---

## 2. 长度感知采样器（`x_planner/data/length_samplers.py`）

import 该模块即把 `knapsack_packed` 采样器注册进 dataset_v2 的 sampler 注册表（`from_config`
按 `sampler.type` 构建）。它子类化 `X2RobotSampler`，保留其 per-rank 分片、任务平衡、O(1) 续训，
并先 `_filter_unestimable` 丢掉 length<0 的 doc。

`knapsack_packed`：贪心装箱（next-fit-decreasing），把多条 doc 装进一行直到 token 预算 `cutoff`，
每行尽量填满 → 每步 token 数均匀。一个 bin = 一行 = 一步。

```yaml
sampler: { type: knapsack_packed, cutoff: 8192 }
```

启动脚本 `--per_device_train_batch_size` **必须 = 1**（bin 本身就是 batch）。

### DDP 正确性

bin 数补齐到 `world_size` 整数倍，各卡步数相等；否则某卡提前没数据 → 集合通信卡死。

---

## 3. 打包（dataset_v2 epilogue + 模型补丁）

采样器只决定"哪些 doc 进同一行"，真正拼行在 dataset_v2 的 epilogue（`packing: true`）：把这些 doc
折成一个 `bsz=1` 的行，每条 doc 的 `position_ids` 各自从 0 重排，拼接 `pixel_values` 等。

模型侧由 `x_planner/data/packing.py::apply_qwen3_5_packing_patch()` 打补丁，让全注意力层（显式
`cu_seq_lens`）和 GatedDeltaNet 线性注意力层都在 doc 边界重置，防止跨 doc 泄漏。详见 README §6。

---

## 4. Trainer 接线

`QwenVLTrainer.get_train_dataloader` 按 `sampler.yields_batches` 分派。`knapsack_packed` 的
`yields_batches=True`：每次迭代产出一个 bin → 一个打包行，DataLoader 用 `batch_sampler=`；此时
`--per_device_train_batch_size` 必须为 1（否则续训 `consumed` 计数会错位）。分派只看这个通用标志、
不针对具体类。

---

## 续训

采样器在 `state_dict` 里记 `lengths_config_hash`，恢复时哈希不一致会报错，防止用错缓存续训。
