#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/wall-vlm_B300
PYTHON_BIN=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall_wm_B300/bin/python
CONTROL_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/zhengwei_accel
OUTPUT_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/zhengwei_accel_scans
CONFIG_PATH="$REPO_ROOT/scripts/train/v10_continuous/configs/sources_zhengwei_accel.yml"
PID_FILE="$CONTROL_ROOT/scan.pid"
LOG_FILE="$CONTROL_ROOT/scan.log"
WORKERS="${V10_ZHENGWEI_WORKERS:-64}"

mkdir -p "$CONTROL_ROOT" "$OUTPUT_ROOT"

status() {
  if [[ -s "$PID_FILE" ]]; then
    local pid
    pid="$(<"$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      ps -p "$pid" -o pid=,etime=,%cpu=,%mem=,stat=,cmd=
      return 0
    fi
  fi
  echo "zhengwei accelerator is not running"
  if [[ -s "$LOG_FILE" ]]; then
    tail -n 20 "$LOG_FILE"
  fi
  return 1
}

start() {
  if status >/dev/null 2>&1; then
    status
    return 0
  fi
  cd "$REPO_ROOT"
  setsid nohup "$PYTHON_BIN" -m scripts.train.v10_continuous.scan_collection_accel \
    --config "$CONFIG_PATH" \
    --output-root "$OUTPUT_ROOT" \
    --workers "$WORKERS" \
    --episode-timeout 900 \
    --validation-ratio 0.10 \
    --publish-every 1000 \
    --publish-seconds 300 \
    --resume \
    >"$LOG_FILE" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" >"$PID_FILE"
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    tail -n 100 "$LOG_FILE"
    return 1
  fi
  status
}

case "${1:-status}" in
  start) start ;;
  status) status ;;
  *) echo "usage: $0 {start|status}" >&2; exit 2 ;;
esac
