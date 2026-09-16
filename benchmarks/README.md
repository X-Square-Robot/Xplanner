# Evaluation assets

This repository keeps evaluation protocols and small machine-readable records in Git. Public
model weights and benchmark videos are hosted on Hugging Face:

- **Model:** [x-square-robot/X-Planner-9B-0916](https://huggingface.co/x-square-robot/X-Planner-9B-0916)
  — the 9B-class BF16 inference release.
- **Dataset:** [x-square-robot/xplanner-benchmark](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)
  — 1,500 episodes, 3,490 MP4s, episode-level metadata, and playable Dataset Preview.

## Published offline benchmark

The benchmark supports video-conditioned planning research and offline analysis of task,
subtask, and action predictions. The `test` preview has one row per episode, with separate video
columns for each available camera. Full downloads include relative media paths and checksums.
See [`xplanner_eval/`](xplanner_eval/) for download commands and the published layout.

This media and metadata release does not yet provide the full temporal scoring annotations or a
fixed end-to-end scoring protocol. It is separate from the real-robot Task Progress experiment
below. The paper's historical results are not new measurements of `X-Planner-9B-0916`.

## Real-robot benchmark

[`real_robot/`](real_robot/) mirrors the evaluation section of the X-Planner report:

- **Reasoning Manipulation**: five tasks testing classification, ordering, matching, and
  instruction-conditioned selection.
- **Generalization**: four instructions executed in a shared cluttered scene.
- **Metric**: Task Progress, a task-specific score from 0 to 100 that credits partial completion.

The checked-in JSON contains the suite/task registry and the aggregate values reported in the
paper. It does not yet make the benchmark independently runnable: per-task scoring rubrics,
initial-state specifications, evaluation images/episodes, trial-level records, and the rollout
adapter still require release approval.

## Evaluation holdout

Training accepts a local, immutable evaluation-holdout manifest and its SHA-256. The holdout is a
leakage fence, not a dataset committed to this repository. Paths are normalized to logical episode
identities before comparison so alternate storage mounts do not bypass the check.

The 1,500-episode evaluation collection is published on
[Hugging Face](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark). Its current
layout and the separate full-annotation export contract are documented in
[`xplanner_eval/`](xplanner_eval/). The anchor-frame coverage view in
[`../datasets/analysis_subset/`](../datasets/analysis_subset/) is not a substitute for that full
artifact.
