# Release artifacts

The source repository and the large external artifacts have separate release boundaries. These
names are fixed so that links in the code, project page, and future model cards remain stable.

| Purpose | Platform | Canonical name | Status |
| --- | --- | --- | --- |
| Source code | GitHub | `X-Square-Robot/Xplanner` | This repository |
| Evaluation benchmark | Hugging Face dataset | `X-Square-Robot/XPlanner-OpenBenchmark` | Reserved |
| Evaluation benchmark archive | External release | `XPlanner-OpenBenchmark-v1.0` | Pending approval |
| Event planner checkpoint | Hugging Face model | `X-Square-Robot/XPlanner-V5.3-EventPlanner` | Reserved |
| Checkpoint revision | Model revision | `v5.3` (`checkpoint-80500`) | Pending upload |

The benchmark name describes the public evaluation collection and is independent of the private
training data. The model name identifies the event-planner checkpoint used by the V5.3 evaluation
snapshot; it does not imply that private training data is redistributed.

Until the Hugging Face repositories are created, do not replace these names with temporary paths or
job-specific checkpoint directories. When an upload is approved, add the immutable dataset/model
revision and SHA-256 values here and in the corresponding model or dataset card.

## Expected external layout

The benchmark repository should contain the versioned manifest, item records, checksums, media
metadata, and a dataset card. Large media files should use the storage mechanism provided by the
hosting platform rather than Git history.

The model repository should contain the model card, configuration/tokenizer files, checkpoint
weights, evaluation provenance, and a license statement. The source repository should reference a
specific immutable revision instead of only the mutable default branch.
