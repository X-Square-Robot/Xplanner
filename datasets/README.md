# Dataset artifacts

The Git repository contains schemas, examples, and export tools. Full datasets and model-generated
media should be published as immutable dataset releases with their own dataset cards, licenses,
manifests, and checksums.

- [`analysis_subset/`](analysis_subset/) is the anchor-frame coverage view used by the report. The
  runnable evaluation release contract is in [`../benchmarks/xplanner_eval/`](../benchmarks/xplanner_eval/).
- Event-state training snapshots are generated locally by `x_planner.data.event_states` and must
  remain excluded from Git because they can contain private paths and annotations.

Source-by-source redistribution status is tracked in [`../docs/data_sources.md`](../docs/data_sources.md).
