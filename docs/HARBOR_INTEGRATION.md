# Harbor integration for X-Planner rollouts

Harbor is the harness/orchestrator around the existing X-Planner event-state
pipeline. In the upstream framework, the integration point is a Harbor task
(`task.toml`, `instruction.md`, environment, and verifier), plus a custom
agent adapter implementing `BaseAgent`; trial traces use ATIF. Harbor itself
already owns jobs, trials, traces, verifier outputs, and artifacts, so the
local recorder below is a compatibility/fallback layer for development.

## Trial contract

Each rollout writes one immutable directory:

```text
jobs/<job_id>/trials/<trial_id>/
  config.json result.json manifest.json
  planner/{decisions,plans,prompts}.jsonl
  executor/{requests,actions,latency}.jsonl
  environment/{states,events}.jsonl seed.json
  observations/ video/ verifier/{metrics.json,evidence.jsonl}
  failure/report.json
```

`config.json` must contain planner and executor checkpoint plus SHA-256,
repository commit, image digest, prompt version/hash, scene configuration and
seed. A trial is not uploadable unless these fields are present.

## Recorder API

The rollout runner should depend on a small recorder interface:

```python
trial = recorder.start(config)
trial.append("planner/decisions.jsonl", decision)
trial.append("executor/actions.jsonl", action)
trial.append("environment/states.jsonl", state)
trial.append("verifier/evidence.jsonl", evidence)
trial.finish(result)
trial.write_manifest()
```

Every record carries `step`, `timestamp`, and `trial_id`. Planner records also
carry `prompt_hash` and `plan_version`; executor records carry requested and
actual actions and latency. Observations are referenced by relative path.

## Failure report

Use a closed taxonomy: `planner_error`, `executor_failure`,
`verifier_failure`, `environment_error`, `simulator_crash`, `timeout`,
`invalid_prompt`, and `unknown`. Store the failed phase, subgoal, last
observation, whether replanning happened, and a replay command. This makes it
possible to distinguish planner mistakes from WA failures.

## Harbor adapter

Keep the local trial writer independent of Harbor. Add an adapter that uploads
the completed directory and registers metadata:

```text
upload_trial(trial_dir) -> harbor_trial_id
get_trial(harbor_trial_id)
download_artifact(harbor_trial_id, relative_path)
```

The adapter must be retryable and content-addressed using `manifest.json`; an
already uploaded file with the same checksum is skipped.

For a native integration, create `adapters/xplanner/` and a task template. The
task verifier writes `/logs/verifier/reward.json` (or `reward.txt`), while the
custom agent launches the X-Planner rollout and emits ATIF events. Use Harbor's
`harbor run --dataset xplanner@<version> --agent xplanner --n-concurrent N` to
execute trials; select `--env docker` first and switch to a cloud provider only
after the container is reproducible.

## Distillation export

Add an exporter that reads only completed Harbor trials, validates their
manifests, filters invalid/crashed episodes, and emits one JSONL sample per
planner decision. Each sample includes observation, history, current event,
teacher decision/plan, action, outcome, reward, failure category, and source
`trial_id`. Convert this JSONL into the existing event-state materialization
format, then run the repository validators before training.

Keep successful samples and labelled negative samples separately. Never train
on a failure without its failure category and correction/teacher label.

## Delivery order

1. Implement the recorder and replay command in the rollout runner.
2. Add manifest and failure validation; run a local multi-seed smoke test.
3. Implement the Harbor upload adapter and viewer metadata mapping.
4. Implement distillation export and connect it to `materialize.py` and the
   existing event-state training scripts.
5. Re-run a fixed scene/seed/checkpoint matrix and compare success, recovery,
   replanning rate, latency, and failure categories.

The acceptance condition is that every training sample links to one Harbor
trial and that the trial can be replayed from its recorded checkpoint, prompt,
scene configuration, and seed.
