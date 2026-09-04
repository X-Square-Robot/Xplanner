"""Per-frame video encoding for the pluggable (DINOv3-style) vision path.

Two layers, both CPU-only:

* tower: a fake 2D encoder validates the batched per-frame forward is
  byte-identical to encoding each frame separately (ordering/reshape is the
  real risk), for pure-video, pure-image and mixed grids;
* epilogue: the real Qwen3.5 tokenizer + a ``PluggableImageProcessor`` swapped
  in as ``hf_processor_instance`` validates the per-frame block builder
  (grid ``[t, gh, gw]``, one timestamp per frame, pad count == t*gh*gw).
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from PIL import Image

from x_planner.modeling.vision import PluggableImageProcessor, PluggableVisualTower

MODEL = "/data/Models/Qwen3.5-9B"


class _FakeEncoder(nn.Module):
    """DINOv3-shaped stand-in: patch-embed linear + CLS/register prefix."""

    def __init__(self, hidden=32, patch=16, regs=4):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden, patch_size=patch, num_register_tokens=regs
        )
        self.lin = nn.Linear(3 * patch * patch, hidden)

    def forward(self, pixel_values):
        b = pixel_values.shape[0]
        p = self.config.patch_size
        x = pixel_values.unfold(2, p, p).unfold(3, p, p)  # [B,3,gh,gw,p,p]
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(b, -1, 3 * p * p)
        feats = self.lin(x)
        prefix = feats.new_zeros(b, 1 + self.config.num_register_tokens, feats.shape[-1])
        return SimpleNamespace(last_hidden_state=torch.cat([prefix, feats], dim=1))


def _frames(n, size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [
        Image.fromarray(
            (torch.rand(size, size, 3, generator=g) * 255).byte().numpy()
        )
        for _ in range(n)
    ]


@pytest.fixture()
def tower():
    torch.manual_seed(0)
    return PluggableVisualTower(_FakeEncoder(), "fake", lm_hidden=48)


class TestTowerPerFrame:
    def test_video_grid_shapes(self, tower):
        ip = PluggableImageProcessor()
        enc = ip(images=_frames(3), return_tensors="pt")
        out = tower(enc["pixel_values"], torch.tensor([[3, 4, 4]]))
        assert out.pooler_output.shape == (3 * 16, 48)

    def test_batched_equals_per_frame(self, tower):
        ip = PluggableImageProcessor()
        enc = ip(images=_frames(3, seed=1), return_tensors="pt")
        pv = enc["pixel_values"]  # [48, 768], 16 rows per frame
        video_out = tower(pv, torch.tensor([[3, 4, 4]])).pooler_output
        per_frame = torch.cat(
            [
                tower(pv[f * 16 : (f + 1) * 16], torch.tensor([[1, 4, 4]])).pooler_output
                for f in range(3)
            ],
            dim=0,
        )
        assert torch.allclose(video_out, per_frame, atol=1e-6)

    def test_mixed_image_and_video_batch(self, tower):
        ip = PluggableImageProcessor()
        img = ip(images=_frames(1, seed=2), return_tensors="pt")
        vid = ip(images=_frames(2, seed=3), return_tensors="pt")
        pv = torch.cat([img["pixel_values"], vid["pixel_values"]], dim=0)
        grid = torch.tensor([[1, 4, 4], [2, 4, 4]])
        out = tower(pv, grid).pooler_output
        assert out.shape == (16 + 32, 48)
        # the image block must be unaffected by the trailing video block
        alone = tower(img["pixel_values"], torch.tensor([[1, 4, 4]])).pooler_output
        assert torch.allclose(out[:16], alone, atol=1e-6)


@pytest.fixture(scope="module")
def epi():
    from transformers import AutoProcessor

    from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (
        Qwen3_5MultimodalEpilogueProcessor,
    )

    proc = AutoProcessor.from_pretrained(MODEL)
    proc.image_processor = PluggableImageProcessor()
    return Qwen3_5MultimodalEpilogueProcessor(
        params={
            "hf_processor_instance": proc,
            "tokenizer_instance": proc.tokenizer,
            "max_seq_length": 8192,
        }
    )


@pytest.mark.skipif(
    not os.path.isdir(MODEL), reason=f"Qwen3.5 processor not available at {MODEL}"
)
class TestEpiloguePerFrame:

    def _dialogue(self, user_text, answer="ok"):
        return [
            {"role": "user", "text": user_text},
            {"role": "assistant", "text": answer},
        ]

    def test_per_frame_blocks_and_grid(self, epi):
        assert epi.per_patch_vision
        meta = {"frames_indices": [0, 1, 2], "native_fps": 2.0, "duration": 1.5}
        ids, labels, pv, grid, pvv, vgrid = epi._encode_one(
            self._dialogue("<video> what?"), [], [_frames(3)], [meta]
        )
        assert vgrid.tolist() == [[3, 4, 4]]  # t = FRAMES (no temporal pairing)
        assert pvv.shape == (3 * 16, 3 * 16 * 16)
        assert int((ids == epi.video_pad_id).sum()) == 3 * 16  # merge 1
        text = epi.tokenizer.decode(ids)
        for ts in ("<0.0 seconds>", "<0.5 seconds>", "<1.0 seconds>"):
            assert ts in text  # one timestamp per frame @ 2 fps

    def test_mixed_uses_per_patch_image_expansion(self, epi):
        meta = {"frames_indices": [0, 1], "native_fps": 2.0, "duration": 1.0}
        ids, labels, pv, grid, pvv, vgrid = epi._encode_one(
            self._dialogue("<image> and <video> ?"), [_frames(1)[0]], [_frames(2)], [meta]
        )
        assert grid.tolist() == [[1, 4, 4]]
        assert int((ids == epi.image_pad_id).sum()) == 16  # merge 1: gh*gw
        assert vgrid.tolist() == [[2, 4, 4]]
        assert int((ids == epi.video_pad_id).sum()) == 2 * 16
