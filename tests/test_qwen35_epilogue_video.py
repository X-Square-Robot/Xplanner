"""Functional tests for the Qwen3.5 epilogue video / mixed encode path.

Regression coverage for the 2026-07 fixes: video used-count (surplus decoded
videos must not attach), mixed image+video samples (images must survive the
video branch), and vision-safe truncation.  Loads the real Qwen3.5 processor
(tokenizer + image/video processors -- CPU-only, no model weights); skipped when
the model directory is absent.  Needs x2robot_dataset_v2 on PYTHONPATH.
"""

import os

import pytest
import torch
from PIL import Image

MODEL = "/mnt/data/x2robot_v2/Models/Qwen3.5-9B"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL), reason=f"Qwen3.5 processor not available at {MODEL}"
)

from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (  # noqa: E402
    IGNORE_INDEX,
    Qwen3_5MultimodalEpilogueProcessor,
)


@pytest.fixture(scope="module")
def epi():
    return Qwen3_5MultimodalEpilogueProcessor(
        params={"processor_path": MODEL, "max_seq_length": 8192}
    )


def _frames(n, size=64, color=(120, 30, 200)):
    return [Image.new("RGB", (size, size), color) for _ in range(n)]


def _img(size=64):
    return Image.new("RGB", (size, size), (10, 200, 60))


def _meta(n):
    return {"frames_indices": list(range(n)), "native_fps": 2.0, "duration": n / 2.0}


def _dialogue(user_text, answer="cars"):
    return [
        {"role": "user", "text": user_text},
        {"role": "assistant", "text": answer},
    ]


def _supervised_text(epi, ids, labels):
    mask = labels != IGNORE_INDEX
    return epi.tokenizer.decode(ids[mask])


class TestVideoEncode:
    def test_video_only_pads_match_grid(self, epi):
        ids, labels, pv, grid, pvv, vgrid = epi._encode_one(
            _dialogue("<video> what happens?"), [], [_frames(3)], [_meta(3)]
        )
        assert pv is None and grid is None
        assert pvv is not None and vgrid.shape[0] == 1
        t, h, w = (int(x) for x in vgrid[0].tolist())
        assert t == 2  # 3 frames pad up to 4 -> ceil(4/2): the processor's ceil
        n_pads = int((ids == epi.video_pad_id).sum())
        assert n_pads == t * h * w // (epi.video_merge_size**2)
        assert pvv.shape[0] == t * h * w
        assert "cars" in _supervised_text(epi, ids, labels)

    def test_surplus_videos_not_attached(self, epi):
        # 2 decoded videos, only 1 <video> tag -> exactly 1 video's features.
        ids, labels, pv, grid, pvv, vgrid = epi._encode_one(
            _dialogue("<video> describe"), [], [_frames(3), _frames(5)],
            [_meta(3), _meta(5)],
        )
        assert vgrid.shape[0] == 1
        n_pads = int((ids == epi.video_pad_id).sum())
        t, h, w = (int(x) for x in vgrid[0].tolist())
        assert n_pads == t * h * w // (epi.video_merge_size**2)
        assert pvv.shape[0] == t * h * w  # first video only

    def test_no_tag_attaches_nothing(self, epi):
        ids, labels, pv, grid, pvv, vgrid = epi._encode_one(
            _dialogue("no media in this prompt"), [], [_frames(3)], [_meta(3)]
        )
        assert pvv is None and vgrid is None
        assert int((ids == epi.video_pad_id).sum()) == 0

    def test_mixed_image_and_video_keeps_both(self, epi):
        ids, labels, pv, grid, pvv, vgrid = epi._encode_one(
            _dialogue("look <image> then <video> answer"),
            [_img()], [_frames(3)], [_meta(3)],
        )
        assert pv is not None and grid.shape[0] == 1
        assert pvv is not None and vgrid.shape[0] == 1
        n_img_pads = int((ids == epi.image_pad_id).sum())
        assert n_img_pads == int(grid[0].prod()) // 4  # Qwen merge_size 2
        n_vid_pads = int((ids == epi.video_pad_id).sum())
        t, h, w = (int(x) for x in vgrid[0].tolist())
        assert n_vid_pads == t * h * w // (epi.video_merge_size**2)
        assert "cars" in _supervised_text(epi, ids, labels)

    def test_safe_truncate_keeps_vision_end(self, epi):
        ids, labels, pv, grid, pvv, vgrid = epi._encode_one(
            _dialogue("<image> describe", answer="x " * 50), [_img(256)], [], []
        )
        old_max = epi.max_seq_length
        try:
            # Force the cut inside the vision span: floor at span end + its
            # <|vision_end|>, and (labels all -100) it warns instead of crashing.
            pad_pos = (ids == epi.image_pad_id).nonzero()
            epi.max_seq_length = int(pad_pos[0].item()) + 1
            t_ids, t_labels = epi._safe_truncate(ids, labels)
            assert int((t_ids == epi.image_pad_id).sum()) == int(
                (ids == epi.image_pad_id).sum()
            )
            assert int(t_ids[-1].item()) == epi.vision_end_id
            assert not bool((t_labels != IGNORE_INDEX).any())
        finally:
            epi.max_seq_length = old_max


class TestMixedActionVqaBatch:
    """Action co-train contract: robot rows never emit video keys, so a mixed
    batch collates them COMPACT (+ collator row mapping). The epilogue must
    scatter videos back to their true rows -- positional pairing would hand the
    VQA sample's video to the robot row."""

    @pytest.mark.skipif(not os.path.isdir(MODEL), reason="model absent")
    def test_compact_video_scatter(self, epi):
        import json as _json

        d_robot = [{"role": "user", "text": "robot sees <image>"},
                   {"role": "assistant", "text": "grip"}]
        d_video = [{"role": "user", "text": "<video> what?"},
                   {"role": "assistant", "text": "cars"}]
        meta = [{"frames_indices": [0, 1, 2], "native_fps": 2.0, "duration": 1.5}]
        batch = {
            "qwen_dialogues_json": [_json.dumps(d_robot), _json.dumps(d_video)],
            "image_observations": [[[_img()]], []],   # row0: one camera frame
            "video_observations": [[_frames(3)]],     # compact: only the VQA row
            "video_meta_json": [_json.dumps(meta)],
            "_video_observations_sample_indices": [1],
            "_video_meta_json_sample_indices": [1],
        }
        out = epi._process_batch_impl(batch)
        ids = out["input_ids"]
        assert ids.shape[0] == 2
        assert int((ids[0] == epi.video_pad_id).sum()) == 0      # robot row: no video
        assert int((ids[0] == epi.image_pad_id).sum()) > 0       # robot row: its image
        assert int((ids[1] == epi.image_pad_id).sum()) == 0
        n_vp = int((ids[1] == epi.video_pad_id).sum())
        assert n_vp == int(out["video_grid_thw"][0].prod()) // 4  # VQA row: its video

    @pytest.mark.skipif(not os.path.isdir(MODEL), reason="model absent")
    def test_missing_mapping_fails_loud(self, epi):
        import json as _json

        batch = {
            "qwen_dialogues_json": [_json.dumps([{"role": "user", "text": "a"}]),
                                    _json.dumps([{"role": "user", "text": "b"}])],
            "video_observations": [[_frames(2)]],  # 1 entry for a 2-row batch, no mapping
            "video_meta_json": [_json.dumps([{"frames_indices": [0, 1]}])],
        }
        with pytest.raises(ValueError, match="cannot align"):
            epi._process_batch_impl(batch)


class TestAllEmptyImageColumn:
    """A bin of rows that all carry image_observations with NO images (pure
    video / text-only) collates the column to [] with no compact mapping --
    that is the legitimate all-empty case, not a misalignment (regression:
    the strict alignment check raised on it)."""

    @pytest.mark.skipif(not os.path.isdir(MODEL), reason="model absent")
    def test_textonly_bin_passes(self, epi):
        import json as _json

        d = [{"role": "user", "text": "hi"}, {"role": "assistant", "text": "yo"}]
        batch = {
            "qwen_dialogues_json": [_json.dumps(d)] * 3,
            "image_observations": [],   # all-empty column collapsed by collate
            "video_observations": [],
            "video_meta_json": ["[]"] * 3,
        }
        out = epi._process_batch_impl(batch)
        assert out["input_ids"].shape[0] == 3
        assert "pixel_values" not in out and "pixel_values_videos" not in out
