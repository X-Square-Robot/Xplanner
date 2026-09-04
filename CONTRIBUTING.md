# Contributing

Thank you for improving X-Planner. Before opening a merge request:

1. keep generated datasets, checkpoints, credentials, and absolute infrastructure paths out of Git;
2. use report terminology and responsibility-based module names;
3. run `python -m compileall -q x_planner scripts`;
4. run `git diff --check` and review the staged diff for secrets and private paths.
