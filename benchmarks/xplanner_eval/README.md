# XPlanner Benchmark

The public benchmark is available at
[x-square-robot/xplanner-benchmark](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark).
The September 16, 2026 release contains **1,500 episodes** selected deterministically from 1,654
candidates, **3,490 MP4 videos**, and episode-level task, subtask, action, and scene metadata.

The corresponding inference model is
[X-Planner-9B-0916](https://huggingface.co/x-square-robot/X-Planner-9B-0916).

## Download and video preview

```bash
hf download x-square-robot/xplanner-benchmark --repo-type dataset \
  --local-dir data/xplanner-benchmark
```

For a reproducible download, add `--revision 0e75c5a91667f8587b9f527a685406832244b59a`,
the release revision with the verified media and typed video preview.

In [Dataset Preview](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark/viewer/default/test),
each row represents one episode. The `face_view`, `left_wrist_view`, `right_wrist_view`,
`global_view`, and `side_view` columns show playable videos when that camera is available.
Missing camera views are empty cells. Preview video references are pinned to the verified media
revision.

## Published layout

```text
xplanner-benchmark/
├── README.md
├── data/
│   └── episodes.parquet
├── metadata/
│   ├── manifest.jsonl
│   ├── episode_index.jsonl
│   ├── summary.json
│   ├── source_ledger.json
│   ├── source_datasets.tsv
│   └── release_checksums.json
└── media/
    └── <episode-id>/<view>.mp4
```

`metadata/manifest.jsonl` is the canonical episode metadata. Its `camera_videos` entries use
repository-relative paths and include media sizes and SHA-256 hashes. `data/episodes.parquet`
provides the typed video preview. The dataset card and source ledger describe the contributing
sources and their respective licensing terms.

This release supports offline video analysis and planning research. Episode-level metadata and
inferred action labels do not replace full temporal ground truth or a fixed scoring protocol.
The raw benchmark manifest is also not the prepared event snapshot required by
`scripts/inference/run_event_planner.py --snapshot`.

## Full-annotation export contract

The existing [`item.schema.json`](item.schema.json) and
`scripts/data/prepare_evaluation_release.py` define a **separate export contract** for data with
complete temporal annotations. That contract produces `manifest.json`, `items.jsonl`,
`checksums.sha256`, and `media/`; it does not describe the currently published Hugging Face layout.
The required full-annotation bundle has not been published as part of this release.

To audit a compatible source selection manifest before exporting that bundle:

```bash
python scripts/data/prepare_evaluation_release.py \
  --manifest /path/to/selection_manifest.jsonl \
  --audit-report /path/to/release_audit.json \
  --path-map /logical/video/root=/locally/mounted/video/root \
  --path-map /logical/label/root=/locally/mounted/label/root
```

The report must say `release_ready: true`, `episode_count: 1500`,
`annotation_available_count: 1500`, and `missing_video_count: 0` before materialization:

```bash
python scripts/data/prepare_evaluation_release.py \
  --manifest /path/to/selection_manifest.jsonl \
  --audit-report /path/to/release_audit.final.json \
  --release-output /path/to/xplanner-eval-with-annotations-v1 \
  --path-map /logical/video/root=/locally/mounted/video/root \
  --path-map /logical/label/root=/locally/mounted/label/root
```

Export aborts before copying if a declared video or annotation is missing. New exports must
record source attribution, licensing, and an immutable release revision independently of the
existing media release.
