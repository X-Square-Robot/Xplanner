<div align="center">

<h1><strong>X-Planner</strong></h1>

<h3>Event-Structured Task Planning for Embodied Intelligence</h3>

</div>

<div id="top" align="center">

[![Homepage](https://img.shields.io/badge/Homepage-%F0%9F%8C%90-116466?style=flat)](https://x-square-robot.github.io/Xplanner/)
[![Code](https://img.shields.io/badge/Code-GitHub-181717?style=flat&logo=github)](https://github.com/X-Square-Robot/Xplanner)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b?style=flat&logo=adobeacrobatreader&logoColor=white)](docs/paper/X_Planner_Event_Structured_Task_Planning_for_Embodied_Intelligence.pdf)
[![Model](https://img.shields.io/badge/Model-X--Planner--9B--0916-ffd21e?style=flat&logo=huggingface)](https://huggingface.co/x-square-robot/X-Planner-9B-0916)
[![Benchmark](https://img.shields.io/badge/Benchmark-xplanner--benchmark-4c8bf5?style=flat&logo=huggingface)](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br>

<strong>English</strong> | <a href="README_zh.md">简体中文</a>

</div>

**X-Planner** is a task-planning front end for **long-horizon robot manipulation**. Given a
high-level instruction, synchronized multi-view observations, and optional execution history, it
represents the next behavior as an action-grounded event and passes that representation to a
downstream world-action model.

<div align="center">
  <img src="assets/X-Planner.jpg" alt="X-Planner overview" width="90%">
</div>

**Core ideas:**
- **Event-grounded data.** Demonstrations are synchronized and organized as a nested
  Task/Subtask/Action/Segment hierarchy.
- **Structured planning states.** Training examples materialize an initial plan, an ongoing event
  state, or an episode-end state as deterministic JSON.
- **Two planning interfaces.** Event mode exposes readable event states; unified mode uses compact
  latent planning states with Staircase Decoding.

## Updates

- 2026-09-16: released [X-Planner-9B-0916](https://huggingface.co/x-square-robot/X-Planner-9B-0916)
  (BF16) and the [XPlanner benchmark](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)
  with 1,500 episodes, 3,490 videos, and playable multi-view Dataset Preview.
- 2026-09: repository structure aligned with the X-Planner report and prepared for an initial
  open-source review.

## Repository layout

| Path | Contents |
| --- | --- |
| `x_planner/modeling/` | Qwen3.5-VL modeling extensions and vision-tower support |
| `x_planner/trainer/` | Distributed SFT launcher, model/data assembly, and trainer |
| `x_planner/data/pipeline/` | Episode normalization, hierarchy validation, and snapshot utilities |
| `x_planner/data/discovery/` | Scalable source discovery, validation, sampling, and immutable snapshots |
| `x_planner/data/context/` | Initial-plan and history-conditioned data construction |
| `x_planner/data/event_states/` | Current structured event-state materialization, holdout, training, and inference |
| `x_planner/evaluation/rollout/` | Offline prediction, rollout analysis, and review galleries |
| `scripts/` | User-facing training, inference, data-export, and evaluation entry points |
| `workspace/example/` | Portable example configuration; replace `/path/to/...` values locally |
| `benchmarks/xplanner_eval/` | Format and schema for the evaluation data |
| `benchmarks/real_robot/` | Reported suites, Task Progress protocol, and aggregate results |

The public names intentionally describe responsibilities rather than internal experiment revisions.
Saved files carry explicit schema versions so older snapshots can still be checked.

## Installation

Create the project environment from the checked-in configuration:

```bash
conda env create -f environment.yml
conda activate xplanner
python -m pip install --no-deps -e .
python -c 'import transformers, x2robot_dataset_v2; print("xplanner runtime ready")'
```

The environment file pins the CUDA-oriented torch/transformers stack and the runtime dependencies.
The data backend is installed separately so that the code repository does not bundle private
dataset sources. The public [`xDataset`](https://github.com/X-Square-Robot/xDataset) repository
provides the generic event-level video/action dataset API used by WALL-WM.

To install that public backend beside the checkout:

```bash
git clone https://github.com/X-Square-Robot/xDataset.git ../xDataset
python -m pip install --no-deps -e ../xDataset
```

For the exact CUDA-oriented environment used during development, see `environment.yml`. Install
FlashAttention separately when the target GPU supports it.

The training launcher accepts optional `XPLANNER_ENV_ROOT`, `XPLANNER_PYTHON`, and
`XPLANNER_DATASET_REPO` overrides. It checks an explicitly supplied backend path, then a sibling
`../x2robot_dataset_v2` checkout, and finally a sibling `../xDataset`.
Before starting training it verifies that the selected interpreter can import both
`transformers` and the X-Planner data-backend contract.

As of September 10, 2026, the public `xDataset/main` snapshot does not yet expose all of the
JSONL/Qwen3.5 processor modules used by the X-Planner event-state runtime. It can be installed for
the generic video/action path, but it is not yet a drop-in backend for X-Planner event-state
training and whole-episode inference. The launcher fails early with the missing module names
instead of reporting a misleading model or data error. A compatible public backend snapshot must be
published before those commands can be reproduced from a clean GitHub checkout.

## Data preparation

Copy the example configuration and point its single source at a local indexed multimodal JSONL
dataset:

```bash
cp workspace/example/data/planner_sft.yml workspace/local_planner_sft.yml
```

Each event-state record contains synchronized views, a task instruction, optional plan/history
context, and one compact JSON target. The current schema and rendering logic live in
`x_planner/data/event_states/schema.py` and `x_planner/data/event_states/prompt.py`.

The full materialization workflow enforces an evaluation holdout by stable episode identity and
manifest SHA-256:

```bash
cp .env.example .env
# Fill XPLANNER_MODEL_PATH, XPLANNER_EVALUATION_MANIFEST,
# XPLANNER_EVALUATION_SHA256, and XPLANNER_DATASET_REPO with an
# X-Planner-compatible x2robot_dataset_v2 checkout.

bash scripts/train/train_event_planner.sh prepare \
  /path/to/event_snapshot /path/to/prepared_data
```

## Training

Run Qwen3.5-VL supervised fine-tuning with a portable data config:

```bash
MODEL_PATH=/path/to/Qwen3.5-9B \
DATA_CONFIG=workspace/local_planner_sft.yml \
OUTPUT_DIR=work_dirs/x_planner_sft \
bash scripts/train/train_qwen35_sft.sh 1 8
```

The event-state launcher currently exposes bounded validation runs for the release candidate:

```bash
bash scripts/train/train_event_planner.sh unit
bash scripts/train/train_event_planner.sh smoke-single /path/to/event_snapshot
```

Its fail-closed checks verify the frozen snapshot digest, evaluation-holdout separation, loss-mask
contract, and resume metadata before optimization.

## Inference

Download the released [X-Planner-9B-0916](https://huggingface.co/x-square-robot/X-Planner-9B-0916)
inference checkpoint (9.41B stored parameters, BF16; approximately 18.82 GB of weights):

```bash
hf download x-square-robot/X-Planner-9B-0916 \
  --local-dir checkpoints/X-Planner-9B-0916
```

The model card includes a standalone Transformers loading example. For structured event-state
generation, install the compatible backend described above and prepare an event snapshot:

```bash
python scripts/inference/run_event_planner.py \
  --checkpoint checkpoints/X-Planner-9B-0916 \
  --snapshot /path/to/event_snapshot \
  --output-dir work_dirs/inference
```

Predictions are parsed and validated against the same compact JSON contract used for training.
The benchmark video manifest is not an event snapshot and cannot be passed directly to `--snapshot`.

## Evaluation

The release is organized into three clearly scoped parts:

1. [xplanner-benchmark on Hugging Face](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)
   provides the 1,500-episode collection: videos, episode-level metadata, checksums, and Dataset
   Preview. See [benchmarks/xplanner_eval/](benchmarks/xplanner_eval/) for its layout and the
   separate temporal-annotation export contract.
2. `benchmarks/real_robot/` records the Reasoning Manipulation and Generalization suites and their
   Task Progress protocol.
3. Training uses a separate local evaluation-holdout manifest; it is not training data and is not
   committed to this repository.

General multimodal evaluation wrappers are also provided:

```bash
CKPT=checkpoints/X-Planner-9B-0916 bash scripts/evaluation/run_lmms_eval.sh mmstar 0
CKPT=checkpoints/X-Planner-9B-0916 TASKS=erqa,vsibench \
  bash scripts/evaluation/run_embodied_benchmarks.sh
```

Download all benchmark assets with:

```bash
hf download x-square-robot/xplanner-benchmark --repo-type dataset \
  --local-dir data/xplanner-benchmark
```

Dataset Preview shows one row per episode and playable videos for each available camera view.
The published collection supports offline analysis and planning research; complete temporal
scoring annotations and the real-robot trial records remain separate releases. See
[benchmarks/README.md](benchmarks/README.md) for the evaluation scope. Reported paper results are
not new measurements of X-Planner-9B-0916.

## Citation

The accompanying report is available as
[`X_Planner_Event_Structured_Task_Planning_for_Embodied_Intelligence.pdf`](docs/paper/X_Planner_Event_Structured_Task_Planning_for_Embodied_Intelligence.pdf).

```bibtex
@article{xplanner2026event,
  title   = {X-Planner: Event-Structured Task Planning for Embodied Intelligence},
  author  = {{X Square Robot Team}},
  year    = {2026},
  note    = {Technical report}
}
```

## License

The source code is released under the [MIT License](LICENSE). The
[X-Planner-9B-0916 model weights](https://huggingface.co/x-square-robot/X-Planner-9B-0916)
are Apache-2.0-licensed. Benchmark media and annotations retain their upstream terms; consult the
[dataset card](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark) for provenance
and licensing details.
