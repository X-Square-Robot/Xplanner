# Native Harbor adapter plan

The upstream Harbor API models one execution as an **Agent** running one
**Task** inside an **Environment**, followed by a **Verifier**. Do not make the
X-Planner rollout runner upload files to Harbor directly. Make it a Harbor
agent and let Harbor own the job/trial lifecycle and ATIF trace.

## Repository layout

Create this in the Harbor checkout (or publish it as an adapter package):

```text
adapters/xplanner/
├── adapter.py
├── run_adapter.py
└── template/
    ├── task.toml
    ├── instruction.md
    ├── environment/Dockerfile
    └── tests/test_rollout.sh
```

`task.toml` declares timeout, resource requirements and metadata. The
environment image contains the simulator, X-Planner checkout and its Python
environment. `instruction.md` supplies the task/scene prompt. The verifier
must write `/logs/verifier/reward.json` and a machine-readable evidence file.

## Agent boundary

Implement Harbor's `BaseAgent` (normally `BaseInstalledAgent`) as
`xplanner`. Its `run()` method should:

1. Read the Harbor task instruction and scene variables.
2. Resolve the checkpoint and prompt version from environment variables.
3. Call the existing X-Planner rollout function.
4. Emit one ATIF event for each Planner decision and executed action.
5. Copy observations/video to the trial artifact directory.
6. Exit non-zero only for infrastructure errors; task failures go to the
   verifier so they remain analyzable trials.

Conceptual boundary:

```python
class XPlannerAgent(BaseAgent):
    async def run(self, instruction, environment, context):
        config = load_trial_config(instruction, context)
        result = await run_xplanner_rollout(config, trace=context.trajectory)
        await environment.write_file("/logs/xplanner/result.json", result)
```

The exact import paths should follow the Harbor version checked out by the
team; pin that version in the adapter package.

## Verifier contract

The verifier reads the rollout result and independent simulator state. It must
produce:

```json
{
  "reward": 1.0,
  "success": true,
  "failure_category": null,
  "evidence": [{"step": 12, "predicate": "drawer_open", "value": true}]
}
```

Use independent state for success/progress; never use the Planner's own
self-reported progress as the reward. Keep failure categories aligned with the
X-Planner schema: planner error, executor failure, verifier failure,
environment error, simulator crash, timeout, invalid prompt, and unknown.

## Commands

After installing the pinned Harbor release:

```bash
harbor adapters create xplanner       # or register the checked-in adapter
harbor run --dataset xplanner@v1 \
  --agent xplanner \
  --model local/<checkpoint> \
  --env docker \
  --n-concurrent 4
```

Start with Docker and one scene/seed. Increase concurrency only after the
image, simulator reset, checkpoint loading, and artifact collection are
reproducible. Cloud providers can then be selected with Harbor's `--env`
option.

## Distillation export

Read Harbor's ATIF traces plus verifier artifacts and emit the existing
X-Planner event-state JSONL. Join on `(trial_id, step)`:

```text
ATIF observation + Planner event
  + executed action
  + verifier outcome/evidence
  → xplanner_harbor_distill_v1
  → event_states materialize/validate
  → SFT or planner distillation
```

Keep successful and corrected negative samples in separate splits. Every row
must retain `source_trial`, checkpoint hash, prompt version, scene, and seed so
that a training example can be replayed through Harbor.
