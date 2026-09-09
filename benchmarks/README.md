# Evaluation assets

This repository keeps evaluation protocols and small machine-readable records in Git. Large media,
model rollouts, and private test labels belong in a separately versioned artifact.

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

The 1,500-episode evaluation collection has a separate path-portable release contract in
[`xplanner_eval/`](xplanner_eval/). The anchor-frame coverage view in
[`../datasets/analysis_subset/`](../datasets/analysis_subset/) is not a substitute for that full
artifact.
