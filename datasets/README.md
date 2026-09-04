# Dataset artifacts

The Git repository contains schemas, examples, and export tools. Full datasets and model-generated
media should be published as immutable dataset releases with their own dataset cards, licenses,
manifests, and checksums.

- [`analysis_subset/`](analysis_subset/) defines the 1,500-episode artifact used by the report's
  semantic and temporal coverage analysis.
- Event-state training snapshots are generated locally by `x_planner.data.event_states` and must
  remain excluded from Git because they can contain private paths and annotations.
