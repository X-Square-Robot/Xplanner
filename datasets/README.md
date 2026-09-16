# Dataset artifacts

The Git repository contains schemas, examples, and export tools. Full datasets and model-generated
media should be published as immutable dataset releases with their own dataset cards, licenses,
manifests, and checksums.

- The published [xplanner-benchmark](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark)
  contains 1,500 episodes, 3,490 videos, episode-level metadata, and playable Dataset Preview.
  Download instructions and its current layout are in
  [`../benchmarks/xplanner_eval/`](../benchmarks/xplanner_eval/).
- [`analysis_subset/`](analysis_subset/) is the anchor-frame coverage view used by the report. The
  separate full-annotation export contract is in [`../benchmarks/xplanner_eval/`](../benchmarks/xplanner_eval/).
- Event-state training snapshots are generated locally by `x_planner.data.event_states` and must
  remain excluded from Git because they can contain private paths and annotations.

Published-source provenance and usage terms are described in the
[dataset card](https://huggingface.co/datasets/x-square-robot/xplanner-benchmark) and its source ledger.
For additional exports, see [`../docs/data_sources.md`](../docs/data_sources.md).
