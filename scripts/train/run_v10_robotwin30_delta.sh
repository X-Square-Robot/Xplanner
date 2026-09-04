#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/wall-vlm_B300
PYTHON_BIN=${PYTHON_BIN:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall_wm_B300/bin/python}
CONFIG=${CONFIG:-${REPO_ROOT}/scripts/train/v10_continuous/configs/sources_robotwin30_arx_x5.yml}
OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/scans}
CONTROL_ROOT=${CONTROL_ROOT:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/checkpoint/v10_continuous/robotwin30_arx_x5_delta}
WORKERS=${WORKERS:-8}
PID_FILE=${CONTROL_ROOT}/scan.pid
LOG_FILE=${CONTROL_ROOT}/scan.log

status() {
    if [[ -s "${PID_FILE}" ]]; then
        local pid
        pid=$(<"${PID_FILE}")
        if kill -0 "${pid}" 2>/dev/null; then
            echo "running pid=${pid} log=${LOG_FILE}"
            return 0
        fi
    fi
    echo "not_running log=${LOG_FILE}"
    return 1
}

start() {
    mkdir -p "${CONTROL_ROOT}"
    if status >/dev/null 2>&1; then
        status
        return 0
    fi
    cd "${REPO_ROOT}"
    # A separate session keeps the delta scanner alive when the invoking shell
    # exits; nohup alone can still share the caller's process group.
    setsid nohup env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" \
        -m scripts.train.v10_continuous.scan_v10 \
        --config "${CONFIG}" \
        --output-root "${OUTPUT_ROOT}" \
        --workers "${WORKERS}" \
        --episode-timeout 900 \
        --validation-ratio 0.10 \
        --publish-every 25 \
        --publish-seconds 60 \
        --resume \
        >>"${LOG_FILE}" 2>&1 </dev/null &
    local pid=$!
    printf '%s\n' "${pid}" >"${PID_FILE}.tmp"
    mv "${PID_FILE}.tmp" "${PID_FILE}"
    echo "started pid=${pid} log=${LOG_FILE}"
}

case "${1:-start}" in
    start)
        start
        ;;
    status)
        status
        ;;
    *)
        echo "usage: $0 [start|status]" >&2
        exit 2
        ;;
esac
