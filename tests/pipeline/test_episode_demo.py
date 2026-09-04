from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import av
from PIL import Image

from x_planner.data.pipeline.episode_demo import (
    EpisodeMemoryConfig,
    REQUIRED_CHECKPOINT_FILES,
    _frame_overlay_json,
    load_episode_bundle,
    render_episode_demo,
    run_episode_inference,
    select_latest_checkpoint,
)
from x_planner.data.pipeline.hierarchy import build_samples, canonicalize_episode
from x_planner.data.pipeline.models import TemporalUnit
from x_planner.data.pipeline.validation_runtime import OracleGenerator


def _write_video(path: Path, *, frames: int, color: tuple[int, int, int]) -> None:
    output = av.open(str(path), mode="w", format="mp4")
    stream = output.add_stream("libx264", rate=10)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    try:
        for index in range(frames):
            image = Image.new(
                "RGB",
                (32, 24),
                tuple(min(255, channel + index) for channel in color),
            )
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    finally:
        output.close()


def _episode_fixture(root: Path):
    video_paths = {}
    for view, color in (
        ("head", (200, 20, 20)),
        ("left_wrist", (20, 200, 20)),
        ("right_wrist", (20, 20, 200)),
    ):
        path = root / f"{view}.mp4"
        _write_video(path, frames=18, color=color)
        video_paths[view] = str(path)
    units = tuple(
        TemporalUnit(
            unit_id=f"L1-{index}",
            level="L1",
            caption=f"perform action number {index}",
            start_frame=index * 6,
            end_frame=(index + 1) * 6,
        )
        for index in range(3)
    )
    return canonicalize_episode(
        source="fixture",
        episode_key="fixture/demo/episode",
        episode_name="episode",
        split="validation",
        num_frames=18,
        task_caption="assemble the test object",
        raw_levels={"L1": units},
        videos=video_paths,
    )


def _catalog_fixture(root: Path):
    episode = _episode_fixture(root)
    samples = build_samples(episode)
    shard = root / "shard.json"
    shard.write_text(
        json.dumps({
            "episode": episode.to_dict(),
            "samples": [sample.to_dict() for sample in samples],
        }),
        encoding="utf-8",
    )
    catalog = root / "catalog"
    catalog.mkdir()
    (catalog / "accepted.jsonl").write_text(
        json.dumps({
            "episode_key": episode.episode_key,
            "status": "accepted",
            "shard_path": str(shard),
            "num_samples": len(samples),
        }) + "\n",
        encoding="utf-8",
    )
    return catalog, episode, samples


class EpisodeDemoTests(unittest.TestCase):
    def test_oracle_rollout_resume_and_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            catalog, episode, samples = _catalog_fixture(root)
            bundle = load_episode_bundle(catalog, episode.episode_key)
            output = root / "predictions.jsonl"
            memory_config = EpisodeMemoryConfig(
                done_threshold=75,
                stable_steps=2,
                similarity_threshold=0.85,
                short_memory_k=2,
                visible_long_memory_limit=3,
                vocabulary="closed",
            )
            partial = run_episode_inference(
                bundle=bundle,
                output_path=output,
                generator=OracleGenerator(),
                memory_config=memory_config,
                stop_after=4,
            )
            self.assertFalse(partial["completed"])
            self.assertTrue(output.with_suffix(".jsonl.partial").is_file())
            with self.assertRaises(ValueError):
                run_episode_inference(
                    bundle=bundle,
                    output_path=output,
                    generator=OracleGenerator(),
                    memory_config=EpisodeMemoryConfig(),
                )
            complete = run_episode_inference(
                bundle=bundle,
                output_path=output,
                generator=OracleGenerator(),
                memory_config=memory_config,
            )
            self.assertTrue(complete["completed"])
            self.assertEqual(complete["samples"], len(samples))
            self.assertEqual(complete["valid_json_rate"], 1.0)
            self.assertGreater(complete["memory_commits"], 0)
            self.assertGreater(complete["nonempty_memory_inputs"], 0)
            self.assertEqual(complete["memory_config"], memory_config.to_dict())
            before = output.read_bytes()
            repeated = run_episode_inference(
                bundle=bundle,
                output_path=output,
                generator=OracleGenerator(),
                memory_config=memory_config,
            )
            self.assertTrue(repeated["completed"])
            self.assertEqual(output.read_bytes(), before)

            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(rows[0]["input_long_memory"], [])
            self.assertEqual(rows[0]["memory_config"], memory_config.to_dict())
            self.assertTrue(all(max(row["image_frames"]) <= row["current_frame"] for row in rows))
            prediction_json, truth_json = _frame_overlay_json(
                episode, int(rows[0]["current_frame"]), rows[0]
            )
            self.assertEqual(json.loads(prediction_json), json.loads(rows[0]["assistant_json"]))
            self.assertIn("predictions", json.loads(truth_json))
            video = render_episode_demo(
                bundle=bundle,
                rows=rows,
                output_path=root / "demo.mp4",
            )
            self.assertEqual(video["frames"], episode.num_frames)
            self.assertEqual(video["fps"], 10.0)
            self.assertEqual(video["codec"], "h264")
            self.assertEqual(video["pixel_format"], "yuv420p")
            self.assertEqual(video["overlay"]["text1"], "full prediction JSON")
            self.assertEqual(
                video["overlay"]["text2"], "full per-frame ground-truth JSON"
            )

    def test_memory_config_validation_and_vocabulary(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = _episode_fixture(Path(temporary))
            default = EpisodeMemoryConfig()
            self.assertEqual(default.canonical_captions(episode), ())
            closed = EpisodeMemoryConfig(vocabulary="closed")
            self.assertEqual(
                closed.canonical_captions(episode),
                tuple(unit.caption for unit in episode.levels["L1"]),
            )
            self.assertEqual(closed.codec().short_memory_k, 1)
            self.assertEqual(closed.codec().visible_long_memory_limit, 8)
            for values in (
                {"done_threshold": 101},
                {"stable_steps": 1},
                {"similarity_threshold": -0.1},
                {"short_memory_k": 3},
                {"visible_long_memory_limit": 0},
                {"vocabulary": "unknown"},
            ):
                with self.assertRaises(ValueError):
                    EpisodeMemoryConfig(**values)

    def test_latest_complete_checkpoint_uses_completion_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            runs = Path(temporary)
            checkpoints = []
            for step, mtime in ((9000, 10_000_000_000), (1000, 20_000_000_000)):
                path = runs / "merged_fixture" / "train" / f"checkpoint-{step}"
                path.mkdir(parents=True)
                for name in REQUIRED_CHECKPOINT_FILES:
                    value = (
                        json.dumps({"global_step": step})
                        if name == "v10_checkpoint_meta.json"
                        else "fixture"
                    )
                    (path / name).write_text(value, encoding="utf-8")
                (path / "rng_state_0.pth").write_text("rng", encoding="utf-8")
                os.utime(path / "v10_checkpoint_meta.json", ns=(mtime, mtime))
                checkpoints.append(path)
            selected, report = select_latest_checkpoint(
                runs,
                min_step=0,
                minimum_large_file_bytes=0,
            )
            self.assertEqual(selected, checkpoints[1].resolve())
            self.assertEqual(report["selected"]["global_step"], 1000)


if __name__ == "__main__":
    unittest.main()
