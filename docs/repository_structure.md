# Repository structure and naming

The public tree follows the same high-level pattern as WALL-WM: one brand package, a small set of
user-facing scripts, portable example configs, project assets, and explicit legal/release files.
Names come from the X-Planner report rather than experiment chronology.

| Report concept | Public code location |
| --- | --- |
| temporal synchronization and hierarchy | `x_planner/data/pipeline/` |
| deterministic source discovery and snapshots | `x_planner/data/discovery/` |
| initial plan and execution history | `x_planner/data/context/` |
| initial, ongoing, and episode-end states | `x_planner/data/event_states/` |
| event-mode generation | `x_planner/data/event_states/inference.py` |
| offline rollout evaluation | `x_planner/evaluation/rollout/` |
| 1,500-episode coverage analysis | `datasets/analysis_subset/` |
| Reasoning Manipulation and Generalization | `benchmarks/real_robot/` |

Internal labels such as `v10`, `v2`, `v3`, `v4`, `v5`, `v53`, dates, machine types, usernames, and
cluster names are not valid public filenames. Version suffixes remain only inside serialized schema
identifiers when required to distinguish artifact formats.

The current source snapshot implements the discrete event-state path. The report's unified latent
path and Staircase Decoding must be added before the repository can claim full method coverage.
