#!/usr/bin/env bash
set -euo pipefail

USER_ROOT=/mnt/cpfs/zbl-cpfs-new/USERS/luhao
PYTHON_BIN=${USER_ROOT}/wall_wm_B300/bin/python
AIHC_HELPER=${USER_ROOT}/.codex/skills/aihc-submit-jobs/scripts/aihc_job.py
AIHC_CONFIG=${USER_ROOT}/.aihc/config
SPEC=${USER_ROOT}/APlan/0805/aihc_v10_allvalid_full2epoch_8g_b6_auto_manifest.json
POINTER=${USER_ROOT}/checkpoint/v10_continuous/all_sources_finalize/latest_complete_snapshot.txt
CONTROL_ROOT=${USER_ROOT}/checkpoint/v10_continuous/all_sources_training_submit
RECEIPT=${CONTROL_ROOT}/submit_receipt.json

mkdir -p "$CONTROL_ROOT"
if [[ -s "$RECEIPT" ]]; then
  echo "[v10-allvalid-submit] already submitted; receipt=$RECEIPT"
  exit 0
fi

while [[ ! -s "$POINTER" ]]; do
  echo "[v10-allvalid-submit] waiting for validated all-source snapshot utc=$(date -u +%FT%TZ)"
  sleep 30
done

read -r snapshot <"$POINTER"
test -f "$snapshot/manifest.json"
test -f "$snapshot/train/data.jsonl"
test -f "$snapshot/train/data.index"

"$PYTHON_BIN" "$AIHC_HELPER" preflight \
  --spec "$SPEC" --config "$AIHC_CONFIG"

temporary=${CONTROL_ROOT}/.submit_receipt.json.tmp
"$PYTHON_BIN" "$AIHC_HELPER" submit \
  --spec "$SPEC" --config "$AIHC_CONFIG" --yes >"$temporary"
mv "$temporary" "$RECEIPT"
echo "[v10-allvalid-submit] submitted snapshot=$snapshot receipt=$RECEIPT"
