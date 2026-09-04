# 视频 GOP-30 转码

## 为什么

训练用序列打包，某个 rank 若卡在 DataLoader 里解码一个又长又高清的视频，会拖垮整个 DDP
（NCCL 超时）。dataset_v2 的视频解码为绕开 mpeg4 关键帧 seek 的坑，默认走 forced-sequential
（整条流解码）—— 采样 32 帧却要解码几千帧，长视频就是灾难。

GOP-30 转码（每 30 帧一个关键帧）让关键帧 seek 变得便宜且有界。**两件事都要做**：

1. 把视频 source 转码成 GOP-30（本目录脚本 `x_planner/tools/gop30/`）；
2. 训练时设 `X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128`，让解码走关键帧 seek 而非整流顺序解码
   （默认训练脚本已设）。

非破坏性：原视频、原 source 目录都不动，GOP-30 副本 + 新 source 目录写到共享盘。

---

## 脚本（`x_planner/tools/gop30/`）

脚本顶部的常量（`NEW_BASE` / `SRC_ROOT` / `SOURCES` / `FFMPEG` 路径）按你的环境改。

### 1. 转码 `convert_gop30_batch.py`

```bash
# 单机
python x_planner/tools/gop30/convert_gop30_batch.py \
  --source-root /path/to/sources --output-root /path/to/gop30 \
  --strip-prefix /path/to/media --jobs 24 --gpus 0,1,2,3,4,5,6,7
# 多机：每个节点跑一个不相交切片，写同一共享目录，可断点续跑
python x_planner/tools/gop30/convert_gop30_batch.py \
  --source-root /path/to/sources --output-root /path/to/gop30 \
  --strip-prefix /path/to/media --shard i/N --jobs 24 --gpus 0,1,2,3,4,5,6,7
```

常用 flag：

| flag | 说明 |
|---|---|
| `--fps 4` | 低帧率重编码；时间戳保真（时长不变→采样时间戳不变），体积/解码 15–30× 更省 |
| `--downscale 640` | 长边封顶（训练本就降到很小分辨率） |
| `--jobs / --threads` | 并行 ffmpeg 进程数 / 每进程线程数 |
| `--preset / --crf` | 编码档位 |
| `--timeout` | 单视频秒数超时则跳过 |
| `--limit N` | 只转前 N 个（冒烟） |

可断点续跑（跳过已完成）。NVDEC 解码需 cuda 版 ffmpeg（脚本按绝对路径找，回退到 PATH）。

### 2. 重指向 source `repoint_jsonl_gop30.py`

```bash
python x_planner/tools/gop30/repoint_jsonl_gop30.py \
  --source-root /path/to/sources --output-root /path/to/gop30 \
  --strip-prefix /path/to/media
```

再把数据 YAML 里的视频 source `path` 指向新目录（脚本末尾会打印）。GOP-30 副本缺失的视频保留
原路径作安全回退（慢但不会崩）。

### 3. 训练时开启关键帧 seek

```bash
export X2ROBOT_AV_SEQUENTIAL_SPAN_MAX=128
```

不设的话 GOP-30 视频仍走顺序解码，没有提速。默认训练脚本已设。

---

## 效果

解码变成有界（~30 帧/目标，与视频总长无关）：中位 ~0.9s、p90 ~2s。之前会卡死（30 分钟超时）的
1920×1440 HEVC 降到 ~8.6s；加 `--downscale 640` 可再降到 ~1–2s。

保持原生 fps/分辨率（不加 `--fps/--downscale`）时帧数/时长不变，长度缓存仍然命中，无需重建。
