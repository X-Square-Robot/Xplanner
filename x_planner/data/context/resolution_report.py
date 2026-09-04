"""Join A/B/C GPU smoke evidence into the stratified resolution probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", required=True)
    args = parser.parse_args()
    result = json.loads(args.probe.read_text(encoding="utf-8"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report]
    by_mode = {str(report["resize_mode"]): report for report in reports}
    for mode, values in result["modes"].items():
        report = by_mode.get(mode)
        if report is None:
            continue
        values.update({
            "batch_6_peak_memory_bytes": (
                int(float(report["peak_gpu_memory_mib"]) * 1024 * 1024)
                if report.get("peak_gpu_memory_mib") is not None else None
            ),
            "batch_6_step_time_seconds": report.get("mean_step_time_seconds"),
            "gpu_status": "passed" if report.get("passed") else "failed",
            "gpu_smoke_report": report.get("log"),
            "oom_detected": bool(report.get("oom_detected")),
        })
    accepted = by_mode.get("B_auto_near_640")
    if accepted and accepted.get("passed") and not accepted.get("oom_detected"):
        result["selected_policy"] = "B_auto_near_640"
        result["acceptance"] = "passed: real single-GPU batch=6 training smoke completed without OOM"
    else:
        result["selected_policy"] = None
        result["acceptance"] = "failed: B batch=6 GPU smoke did not pass"
    result["schema_version"] = "memory_v3_resolution_probe_gpu_v1"
    write_json(str(args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
