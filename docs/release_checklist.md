# Open-source release checklist

## Release boundary

The Git repository contains source code, schemas, configurations with portable placeholders, and
documentation. It must not contain model weights, full datasets, private paths, credentials,
generated smoke outputs, generated caches, or complete per-sample logs.
Deployment-specific DLC submission scripts remain in the private deployment repository; the public
tree keeps portable local/distributed training entry points only.

Publish the evaluation media and annotations as a separately versioned artifact. The event-grounded
training dataset is not part of the open-source release. Publish model weights in a model repository
with a model card. Pin the evaluation artifact and model release from the code repository by
immutable release ID and SHA-256.

## Required gates

1. Rotate every credential ever committed to the source history.
2. Rewrite the public branch so removed credentials are not reachable from its Git history.
3. Replace machine-specific paths and private service endpoints with configuration.
4. Select a code license and confirm ownership of all copied or derived source files.
5. Complete `docs/data_sources.md` for every redistributed dataset.
6. Export only approved analysis and evaluation media; verify relative paths and checksums.
7. Add the complete real-robot rubrics, inputs, trial records, and evaluator.
8. Add the Staircase Decoding implementation or narrow the public code-release claim to event mode.
9. Run unit tests, a fixture-sized end-to-end test, secret scanning, and a clean-environment install.

## Versioning

- Code: semantic versions such as `v0.1.0`.
- Dataset: immutable releases such as `xplanner-analysis-v1.0`.
- Manifest rows: stable IDs that never depend on local absolute paths.
- Every release: manifest checksum, file checksums, schema version, dataset card, and changelog.

## Current review state

- Done: responsibility-based package and file names; internal revision numbers removed from paths.
- Done: model weights, generated inventories, deployment launchers, and complete media excluded.
- Done: analysis-subset and real-robot benchmark concepts separated according to the report.
- Done: public examples use relative or `/path/to/...` values.
- Done: project page and paper PDF are included in the public source tree.
- Done: benchmark and model artifact names are fixed before external upload.
- Pending: publish an X-Planner-compatible public `x2robot_dataset_v2` backend or remove the
  event-state runtime's private processor dependency.
- Pending: credential rotation and removal of the previous remote branch/MR object that may retain
  the old commit.
- Pending: code license, source ownership review, and data redistribution approvals.
- Pending: complete evaluation media/annotations, real-robot benchmark artifacts, model weights, and
  Staircase Decoding code.
