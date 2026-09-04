#!/usr/bin/env bash
set -euo pipefail

WALL_REPO=${WALL_REPO:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/ACode/wall-vlm_B300}
PYTHON_BIN=${PYTHON_BIN:-/mnt/cpfs/zbl-cpfs-new/USERS/luhao/wall_wm_B300/bin/python}
command_name=${1:-help}
if [[ $# -gt 0 ]]; then
    shift
fi

cd "${WALL_REPO}"
export PYTHONPATH="${WALL_REPO}${PYTHONPATH:+:${PYTHONPATH}}"

case "${command_name}" in
    validate|scan|resume|status|merge|manifest|snapshot|all)
        exec "${PYTHON_BIN}" -m scripts.train.v10_continuous_v2.scan_v10_v2 "${command_name}" "$@"
        ;;
    scan-target)
        exec "${PYTHON_BIN}" -m scripts.train.v10_continuous_v2.scan_v10_v2 scan --discovery-mode fast "$@"
        ;;
    audit-data)
        exec "${PYTHON_BIN}" -m scripts.train.v10_continuous_v2.scan_v10_v2 scan --discovery-mode root "$@"
        ;;
    audit)
        exec "${PYTHON_BIN}" -m scripts.train.v10_continuous_v2.audit_samples_v2 "$@"
        ;;
    train-smoke)
        exec "${PYTHON_BIN}" -m scripts.train.v10_continuous_v2.train_smoke_v2 "$@"
        ;;
    help|-h|--help)
        exec "${PYTHON_BIN}" -m scripts.train.v10_continuous_v2.scan_v10_v2 --help
        ;;
    *)
        echo "Unknown command: ${command_name}" >&2
        echo "Usage: $0 {scan-target|audit-data|validate|scan|resume|status|merge|manifest|snapshot|all|audit|train-smoke} [options]" >&2
        exit 2
        ;;
esac
