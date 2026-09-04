#!/usr/bin/env bash
set -euo pipefail

USER_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao
SCRIPT=${USER_ROOT}/ACode/wall-vlm_B300/scripts/train/finalize_v10_all_sources.sh
CONTROL_ROOT=${USER_ROOT}/checkpoint/v10_continuous/all_sources_finalize
PID_FILE=${CONTROL_ROOT}/finalize.pid
LOG_FILE=${CONTROL_ROOT}/finalize.log

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
  echo "all-sources finalizer is not running"
  if [[ -s "$LOG_FILE" ]]; then
    tail -n 30 "$LOG_FILE"
  fi
  return 1
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
