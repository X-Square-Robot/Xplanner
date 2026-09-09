# X-Planner Harbor adapter template

Copy `xplanner/template` into the Harbor adapter workspace and replace the
Dockerfile entrypoint with the deployment-specific rollout command. The
rollout must write `/logs/xplanner/result.json`; the verifier must independently
write `/logs/verifier/reward.json` and evidence.

Before running parallel jobs, validate one local Docker trial, then run a fixed
scene/seed matrix. Keep checkpoint, prompt version, scene config, simulator
image digest, and seed in the trial metadata so every distillation sample can
be replayed.
