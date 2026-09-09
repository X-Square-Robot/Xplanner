# X-Planner evaluation release

The 1,500-episode evaluation collection is selected deterministically from 1,654 candidates. The
private source folder contains the selection report, plots, task/label references, and paths to the
original camera videos. That folder is an input inventory, not yet a publishable artifact: its
manifests contain machine-specific paths and the media is not self-contained.

## Public artifact layout

```text
xplanner-eval-v1/
├── README.md
├── manifest.json
├── items.jsonl
├── checksums.sha256
└── media/
    └── <episode-id>/<view>.mp4
```

Every `items.jsonl` row follows [`item.schema.json`](item.schema.json). It contains an evaluation
instruction, temporal annotations, the original source UID, analysis attributes, and only
relative media paths. Every media file has a SHA-256 checksum. Training snapshots, robot action
arrays, private paths, credentials, and unrelated files from the source episode directories are not
part of this artifact.

## Audit and export

First produce a fail-closed audit without copying media:

```bash
  python scripts/data/prepare_evaluation_release.py \
  --manifest /path/to/selection_manifest.jsonl \
  --audit-report /path/to/release_audit.json \
  --path-map /logical/video/root=/locally/mounted/video/root \
  --path-map /logical/label/root=/locally/mounted/label/root
```

The report must say `release_ready: true`, `episode_count: 1500`,
`annotation_available_count: 1500`, and `missing_video_count: 0`. Only then materialize the immutable
release directory:

```bash
  python scripts/data/prepare_evaluation_release.py \
  --manifest /path/to/selection_manifest.jsonl \
  --audit-report /path/to/release_audit.final.json \
  --release-output /path/to/xplanner-eval-v1 \
  --path-map /logical/video/root=/locally/mounted/video/root \
  --path-map /logical/label/root=/locally/mounted/label/root
```

Export aborts before copying if any declared video or annotation is missing. The result still needs
a benchmark dataset card, one license/attribution entry per contributing source, privacy review,
and an immutable external release ID before it can be announced publicly.
