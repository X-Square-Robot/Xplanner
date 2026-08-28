#!/usr/bin/env bash
set -euo pipefail

USER_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao
REPO_ROOT=${USER_ROOT}/ACode/wall-vlm_B300
PYTHON_BIN=${USER_ROOT}/wall_wm_B300/bin/python
BASE_CATALOG=${USER_ROOT}/checkpoint/v10_continuous/merged/catalog_snapshots/merged-main005376-rev000245-robotwin000140-20260806T0145Z
ZHENGWEI_RUN=${USER_ROOT}/checkpoint/v10_continuous/zhengwei_accel_scans/scan-9bc52fe0f728
OPEN_ACTION_RUN=${USER_ROOT}/checkpoint/v10_continuous/open_action_delta_scans/scan-bfdc5446d968
ZHENGWEI_PID_FILE=${USER_ROOT}/checkpoint/v10_continuous/zhengwei_accel/scan.pid
OPEN_ACTION_PID_FILE=${USER_ROOT}/checkpoint/v10_continuous/open_action_delta/scan.pid
ZHENGWEI_LOG=${USER_ROOT}/checkpoint/v10_continuous/zhengwei_accel/scan.log
OPEN_ACTION_LOG=${USER_ROOT}/checkpoint/v10_continuous/open_action_delta/scan.log
OUTPUT_ROOT=${USER_ROOT}/checkpoint/v10_continuous/merged
CONTROL_ROOT=${USER_ROOT}/checkpoint/v10_continuous/all_sources_finalize

mkdir -p "$CONTROL_ROOT" "$OUTPUT_ROOT"

wait_for_scan() {
  local label=$1
  local pid_file=$2
  local log_file=$3
  local pid
  test -s "$pid_file"
  pid="$(<"$pid_file")"
  while kill -0 "$pid" 2>/dev/null; do
    echo "[finalize-all] waiting for ${label} scanner pid=${pid} utc=$(date -u +%FT%TZ)"
    sleep 30
  done
  if ! grep -q '"status": "complete"' "$log_file"; then
    echo "[finalize-all] ${label} scanner exited without completion marker" >&2
    tail -n 100 "$log_file" >&2 || true
    return 1
  fi
}

latest_catalog() {
  local run_root=$1
  local latest
  latest="$(find "$run_root/catalog_snapshots" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort -n | tail -n 1)"
  test -n "$latest"
  test -f "$run_root/catalog_snapshots/$latest/manifest.json"
  printf '%s\n' "$run_root/catalog_snapshots/$latest"
}

test -f "$BASE_CATALOG/manifest.json"
wait_for_scan zhengwei "$ZHENGWEI_PID_FILE" "$ZHENGWEI_LOG"
wait_for_scan open_action_delta "$OPEN_ACTION_PID_FILE" "$OPEN_ACTION_LOG"

ZHENGWEI_CATALOG="$(latest_catalog "$ZHENGWEI_RUN")"
OPEN_ACTION_CATALOG="$(latest_catalog "$OPEN_ACTION_RUN")"

read -r zhengwei_terminal open_action_terminal < <(
  "$PYTHON_BIN" - "$ZHENGWEI_CATALOG/manifest.json" "$OPEN_ACTION_CATALOG/manifest.json" <<'PY'
import json
import sys

values = []
for path in sys.argv[1:]:
    manifest = json.load(open(path, encoding="utf-8"))
    values.append(int(manifest["stats"]["terminal"]))
print(*values)
PY
)

stamp="$(date -u +%Y%m%dT%H%MZ)"
version="merged-allvalid-zhengwei$(printf '%06d' "$zhengwei_terminal")-open$(printf '%06d' "$open_action_terminal")-${stamp}"
result_tmp="$CONTROL_ROOT/.finalize_result.json.tmp"
result_final="$CONTROL_ROOT/finalize_result.json"

cd "$REPO_ROOT"
"$PYTHON_BIN" -m scripts.train.v10_continuous.merge_v10_catalogs \
  --catalog "$BASE_CATALOG" \
  --catalog "$ZHENGWEI_CATALOG" \
  --catalog "$OPEN_ACTION_CATALOG" \
  --output-run-root "$OUTPUT_ROOT" \
  --version "$version" \
  --complete \
  >"$result_tmp"
mv "$result_tmp" "$result_final"

snapshot="$OUTPUT_ROOT/training_snapshots/$version"
test -f "$snapshot/manifest.json"
pointer_tmp="$CONTROL_ROOT/.latest_complete_snapshot.txt.tmp"
printf '%s\n' "$snapshot" >"$pointer_tmp"
mv "$pointer_tmp" "$CONTROL_ROOT/latest_complete_snapshot.txt"
echo "[finalize-all] complete snapshot=$snapshot"
