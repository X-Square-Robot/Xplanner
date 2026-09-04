#!/usr/bin/env bash
set -Eeuo pipefail

# Run the supported embodied/spatial lmms-eval tasks.
# Judge-based tasks require an OpenAI-compatible endpoint.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TASKS=${1:-${TASKS:-unieqa,openeqa,erqa,vsibench}}
GPUS=${GPUS:-0}
LIMIT=${LIMIT:-}

if [[ ",${TASKS}," == *",unieqa,"* || ",${TASKS}," == *",openeqa,"* ]]; then
    if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_API_URL:-}" ]]; then
        echo "Judge-based tasks require OPENAI_API_KEY and OPENAI_API_URL." >&2
        exit 64
    fi
fi

export MODEL_VERSION=${MODEL_VERSION:-gpt-4o-mini}
exec bash "${SCRIPT_DIR}/run_lmms_eval.sh" "${TASKS}" "${GPUS}" "${LIMIT}"
