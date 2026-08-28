from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ..gallery_v4 import build_gallery, classify_stage, discover_input_files
from ..prompt_v4 import sample_to_indexed_jsonl
from ..resource_monitor_v4 import (
    ConcurrencyPolicy,
    derive_sample,
    read_proc_snapshot,
    recommend_concurrency,
)


def _write_proc(root: Path, *, cpu: str, load1: float = 10.0, available_kib: int = 1_500_000_000) -> None:
    (root / "pressure").mkdir(parents=True, exist_ok=True)
    (root / "stat").write_text(f"cpu  {cpu}\n", encoding="utf-8")
    (root / "loadavg").write_text(f"{load1} 9.0 8.0 3/100 123\n", encoding="utf-8")
    (root / "meminfo").write_text(
        f"MemTotal: 2000000000 kB\nMemAvailable: {available_kib} kB\n"
        "MemFree: 1000000 kB\nCached: 2000000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n",
        encoding="utf-8",
    )
    pressure = "some avg10=0.10 avg60=0.20 avg300=0.30 total=100\n"
    for name in ("cpu", "io", "memory"):
        (root / "pressure" / name).write_text(pressure, encoding="utf-8")


def _row(index: int, stage: str) -> dict:
    terminal = stage == "terminal"
    task_type = "initial_plan" if stage == "initial" else "continuous"
    anchor = 0 if stage == "start" else 20 + index
    sample = {
        "schema_version": "memory_v4",
        "sample_key": f"memory-v4-{stage}-{index}",
        "sample_id": f"memory-v4-{stage}-{index}",
        "global_episode_key": f"collection:episode-{stage}-{index}",
        "episode_key": f"episode-{stage}-{index}",
        "task_type": task_type,
        "source_id": "collection",
        "split": "train",
        "profile": "L3L0",
        "task_instruction": "place the object",
        "unit_type": "segment",
        "anchor_frame": anchor,
        "anchor_grid_index": 0 if stage == "start" else index + 1,
        "is_terminal_window": terminal,
        "resize_policy_id": "test",
        "visible_long_memory_limit": 8,
        "long_memory": [],
        "short_memory": [],
        "images": [{
            "video": "/data/head.mp4", "frame": anchor,
            "relative_frame": 0, "view": "head",
        }],
    }
    if stage == "initial":
        sample["target"] = {
            "initial_plan": [{
                "index": 1,
                "l0": {"level": "L0", "source": "segment", "caption": "place the object"},
            }],
        }
    else:
        second_caption = "the task is complete" if terminal else "release the object"
        sample["target"] = {
            "task_progress_percent": 50,
            "predictions": [
                {
                    "index": 1,
                    "l0": {
                        "level": "L0", "source": "segment",
                        "caption": "place the object", "progress_percent": 50,
                    },
                },
                {
                    "index": 2,
                    "l0": {
                        "level": "L0", "source": "segment",
                        "caption": second_caption, "progress_percent": 0,
                    },
                },
            ],
        }
    return sample_to_indexed_jsonl(sample)


class ResourceMonitorV4Test(unittest.TestCase):
    def test_proc_delta_and_policy_recommendations(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_proc(root, cpu="100 0 100 700 0 0 0 0")
            previous = read_proc_snapshot(root)
            _write_proc(root, cpu="180 0 180 720 20 0 0 0")
            current = read_proc_snapshot(root)
            metrics = derive_sample(previous, current)
            self.assertAlmostEqual(metrics["cpu_busy_percent"], 80.0)
            self.assertAlmostEqual(metrics["iowait_percent"], 10.0)
            policy = ConcurrencyPolicy(reserve_logical_cpus=20)
            recommendation = recommend_concurrency(96, metrics, policy, logical_cpus=180)
            self.assertEqual(recommendation["action"], "decrease")
            self.assertEqual(recommendation["reason"], "iowait_soft_cap")

    def test_policy_honors_memory_and_cpu_reserves(self) -> None:
        metrics = {
            "cpu_busy_percent": 20.0,
            "iowait_percent": 0.0,
            "load": {"load1": 1.0},
            "memory": {"available_bytes": 100 * 1024**3},
        }
        recommendation = recommend_concurrency(
            150, metrics, ConcurrencyPolicy(), logical_cpus=180
        )
        self.assertEqual(recommendation["reason"], "memory_reserve_breached")
        self.assertLess(recommendation["recommended_concurrency"], 150)
        self.assertEqual(recommendation["effective_max_concurrency"], 160)


class GalleryV4Test(unittest.TestCase):
    def test_snapshot_gallery_is_exact_deterministic_and_skips_terminal_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            (snapshot / "_SUCCESS").parent.mkdir(parents=True)
            (snapshot / "_SUCCESS").write_text("{}\n", encoding="utf-8")
            rows = {stage: [_row(index, stage) for index in range(5)] for stage in ("initial", "start", "process", "terminal")}
            for task in ("continuous", "initial_plan", "terminal"):
                directory = snapshot / "datasets" / task / "train"
                directory.mkdir(parents=True)
                selected = rows["initial"] if task == "initial_plan" else (
                    rows["start"] + rows["process"] + rows["terminal"]
                )
                (directory / "data.jsonl").write_text(
                    "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in selected),
                    encoding="utf-8",
                )
            files = discover_input_files([snapshot])
            self.assertEqual(len(files), 2)
            output = root / "gallery"
            first = build_gallery(input_roots=[snapshot], output_root=output)
            keys_first = [item["sample_key"] for item in first["selected_samples"]]
            second = build_gallery(input_roots=[snapshot], output_root=output)
            self.assertEqual(
                keys_first,
                [item["sample_key"] for item in second["selected_samples"]],
            )
            self.assertEqual(first["rows_scanned"], 20)
            self.assertEqual(first["cells_observed"], 4)
            self.assertEqual(first["selections"], 12)
            for item in first["selected_samples"]:
                raw = (output / item["raw_file"]).read_text(encoding="utf-8")
                markdown = (output / item["markdown_file"]).read_text(encoding="utf-8")
                row = json.loads(raw)
                self.assertIn(row["text"][0]["text"], markdown)
                self.assertIn(row["text"][1]["text"], markdown)
                self.assertIn("/data/head.mp4", markdown)

    def test_incomplete_part_is_rejected_and_terminal_wins_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            part = root / "part-000"
            part.mkdir()
            (part / "rows.jsonl").write_text(json.dumps(_row(0, "start")) + "\n")
            with self.assertRaises(ValueError):
                discover_input_files([root])
        row = _row(0, "terminal")
        row["v4_sample"]["anchor_grid_index"] = 0
        self.assertEqual(classify_stage(row), "terminal")


if __name__ == "__main__":
    unittest.main()
