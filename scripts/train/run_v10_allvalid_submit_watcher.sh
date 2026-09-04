#!/usr/bin/env bash
set -euo pipefail

USER_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao
SCRIPT=${USER_ROOT}/ACode/wall-vlm_B300/scripts/train/submit_v10_allvalid_when_ready.sh
CONTROL_ROOT=${USER_ROOT}/checkpoint/v10_continuous/all_sources_training_submit
PID_FILE=${CONTROL_ROOT}/watcher.pid
LOG_FILE=${CONTROL_ROOT}/watcher.log

mkdir -p "$CONTROL_ROOT"

status() {
  if [[ -s "$PID_FILE" ]]; then
    local pid
    pid="$(<"$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      ps -p "$pid" -o pid=,etime=,%cpu=,%mem=,stat=,cmd=
      return 0
    fi
  fi
  echo "all-valid submit watcher is not running"
  [[ -s "$LOG_FILE" ]] && tail -n 30 "$LOG_FILE"
  [[ -s "$CONTROL_ROOT/submit_receipt.json" ]]
}

start() {
  if status >/dev/null 2>&1; then
    status
    return 0
  fi
  setsid nohup bash "$SCRIPT" >"$LOG_FILE" 2>&1 < /dev/null &
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
