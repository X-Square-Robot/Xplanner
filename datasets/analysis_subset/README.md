# Coverage view of the evaluation collection

This artifact supports the report's data-coverage figures. It is a deterministic subset of 1,500
episodes selected from 1,654 candidates while preserving all 167 contributing dataset identifiers,
31 task classes, and observed semantic/action-label tails.

This directory is the lightweight anchor-frame coverage view. It is not the runnable evaluation
release; use [`../../benchmarks/xplanner_eval/`](../../benchmarks/xplanner_eval/) for the full media,
annotation, checksum, and release contract.

## Release layout

```text
xplanner-analysis-v1/
├── README.md
├── manifest.json
├── items.jsonl
├── checksums.sha256
└── media/
    └── <sample-id>/<view>.jpg
```

Each JSONL row follows `schema.json`. Paths are relative to the artifact root, and every image has a
SHA-256. The exporter copies only the exact anchor frame used for analysis; it never exposes source
mount paths.

```bash
python scripts/data/export_analysis_subset.py \
  --manifest /path/to/private_analysis_manifest.jsonl \
  --output /path/to/xplanner-analysis-v1 \
  --path-map /logical/source/root=/locally/mounted/root
```

The internal manifest currently references source videos. On the audited machine, only a subset of
those references resolves, so the public artifact must not be announced until all approved media is
materialized and the exporter completes without missing files.

## Publication gates

1. Fill one row per contributing source in `docs/data_sources.md`.
2. Remove or replace samples without redistribution permission.
3. Run privacy review on every image and annotation field.
4. Validate all rows against `schema.json` and verify `checksums.sha256`.
5. Publish the full artifact outside Git and pin its immutable release ID here.
