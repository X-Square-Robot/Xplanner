# Real-robot evaluation

The X-Planner report evaluates the complete event-mode system on tabletop bimanual manipulation.
`tasks.json` records the two suites and nine task names; `results.json` records only the aggregate
Task Progress values that are currently available in the report source.

Task Progress is a dense score in `[0, 100]` defined by task-specific completion rubrics. It is more
informative than binary success for late failures, but it evaluates the full planner-conditioned
execution system and does not isolate planning from perception or control.

## Required for a runnable public release

- one versioned initial-state and language-instruction specification per task;
- the exact scoring rubric and evaluator for every task;
- approved observation images or episodes with stable IDs and relative paths;
- trial seeds, scene-randomization protocol, model/config identifiers, and trial-level results;
- a manifest and SHA-256 checksums for every released file.

Until those files are present, this directory is a paper-results record rather than a downloadable
benchmark implementation.
