"""Tests for the Qwen3.5 multimodal text-processor + epilogue.

Validates the data side of the Qwen3.5-VL SFT integration (no GPU / model
needed -- only the tokenizer + image processor):

- ``multimodal_jsonl_qwen3_5`` text processor + ``multimodal_qwen3_5`` epilogue
  are registered;
- grounding coords pass through unchanged ([0, 1000), the Qwen3-VL convention);
- the official chat template is used and ``<|image_pad|>`` expands to
  ``grid.prod() / merge**2`` tokens per image;
- labels supervise *every* assistant span (system/user/image/pad -> -100).

Skipped automatically when the local Qwen3.5 checkpoint is absent.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from x2robot_dataset_v2.processors.epilogue import EPILOGUE_REGISTRY
from x2robot_dataset_v2.processors.epilogue.base import EpilogueProcessor
from x2robot_dataset_v2.processors.text import TEXT_PROCESSOR_REGISTRY
from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue

QWEN35_PATH = os.environ.get("QWEN35_PATH", "/data/Models/Qwen3.5-9B")
_HAS_MODEL = os.path.isdir(QWEN35_PATH) and os.path.isfile(
    os.path.join(QWEN35_PATH, "config.json")
)
needs_model = pytest.mark.skipif(not _HAS_MODEL, reason="Qwen3.5 checkpoint absent")


def test_processors_registered():
    assert "multimodal_qwen3_5" in EPILOGUE_REGISTRY
    assert "multimodal_jsonl_qwen3_5" in TEXT_PROCESSOR_REGISTRY


def test_grounding_coords_passthrough():
    """[0,1000) coords must NOT be rescaled; <bbox> normalized to <box>."""
    dlg = [
        {"role": "user", "text": "<image>cup <bbox_question>"},
        {"role": "assistant", "text": "<bbox>[123, 456, 789, 900]</bbox>"},
    ]
    out = process_dialogue(dlg, seed=0, num_images=1)
    asst = out[-1]["text"]
    assert "123, 456, 789, 900" in asst
    assert "<box>" in asst and "<bbox>" not in asst


@pytest.fixture(scope="module")
def epilogue():
    return EpilogueProcessor.from_config({
        "type": "multimodal_qwen3_5",
        "params": {
            "processor_path": QWEN35_PATH,
            "max_seq_length": 4096,
            "padding_side": "right",
        },
    })


def _img(h, w):
    return Image.fromarray(np.uint8(np.random.rand(h, w, 3) * 255))


@needs_model
def test_epilogue_image_tokens_and_masking(epilogue):
    epi = epilogue
    tok = epi.tokenizer

    s0 = [{"role": "user", "text": "<image>What is the bounding box of the cup?"},
          {"role": "assistant", "text": "<box>[100, 200, 300, 400]</box>"}]
    s1 = [{"role": "user", "text": "<image>What is this?"},
          {"role": "assistant", "text": "A cat."},
          {"role": "user", "text": "What color is it?"},
          {"role": "assistant", "text": "Gray."}]
    s2 = [{"role": "user", "text": "Hello there, who are you?"},
          {"role": "assistant", "text": "I am an assistant."}]
    batch = {
        "qwen_dialogues_json": [json.dumps(s) for s in (s0, s1, s2)],
        "image_observations": [[[_img(224, 320)]], [[_img(256, 256)]], []],
    }
    out = epi.process_batch(batch)

    assert set(out.keys()) == {
        "input_ids", "attention_mask", "labels", "pixel_values", "image_grid_thw"
    }
    ii, lab, am, grid = (
        out["input_ids"], out["labels"], out["attention_mask"], out["image_grid_thw"]
    )
    assert ii.shape == lab.shape == am.shape

    # 2 images total; grids merge-aligned; image-pad count == sum(grid.prod/4).
    assert grid.shape[0] == 2
    for _, h, w in grid.tolist():
        assert h % 2 == 0 and w % 2 == 0
    n_pad = int((ii == epi.image_pad_id).sum())
    expect = int((grid[:, 0] * grid[:, 1] * grid[:, 2] // 4).sum())
    assert n_pad == expect

    def supervised(i):
        keep = lab[i] != -100
        return tok.decode(ii[i][keep])

    # S0: single-turn, coords intact in labels.
    assert "100, 200, 300, 400" in supervised(0)
    # S1: BOTH assistant turns supervised; user tokens not supervised.
    sup1 = supervised(1)
    assert "A cat." in sup1 and "Gray." in sup1
    assert "What color" not in sup1 and "What is this" not in sup1
    # image / pad tokens never supervised.
    assert int(((ii == epi.image_pad_id) & (lab != -100)).sum()) == 0
    assert int(((am == 0) & (lab != -100)).sum()) == 0
