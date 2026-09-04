# Example workspace

Files in this directory are portable templates, not experiment outputs.

- `data/planner_sft.yml` is a minimal xDataset-backed multimodal SFT configuration.
- `training/deepspeed_zero1.json` is the default single-node DeepSpeed configuration.

Copy templates to an ignored local workspace before editing paths. Never commit generated data
lists, checkpoints, API keys, or absolute infrastructure paths.
