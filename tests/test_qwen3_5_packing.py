"""Tests for Qwen3.5 epilogue sequence packing (neat-packing), data side.

Validates the packed-batch the epilogue emits (no GPU / model weights -- only the
tokenizer + image processor + a config-only ``get_rope_index``):

- packing folds the batch into one row (bsz == 1), ``attention_mask`` is None;
- per-document MRoPE ``position_ids`` restart at 0 and the explicit FA2
  ``cu_seq_lens_q`` match what the GDN patch derives from ``position_ids[0]``
  (so the linear-attention and full-attention layers agree on boundaries);
- labels supervise every assistant span across docs; image tokens never;
- ``<|image_pad|>`` count is preserved (== sum grid.prod/4);
- works together with the RVQ action epilogue (action + VQA packed in one row).

Skipped automatically when the Qwen3.5 checkpoint (and, for the action test, the
RVQ checkpoint) is absent.
"""

from __future__ import annotations

import functools
import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from qwenvl.data.packing import pack_sequences as _pack_sequences

from x2robot_dataset_v2.processors.epilogue.base import EpilogueProcessor

QWEN35_PATH = os.environ.get("QWEN35_PATH", "/mnt/data/x2robot_v2/Models/Qwen3.5-9B")
RVQ_CKPT = os.environ.get(
    "RVQ_CKPT",
    "/x2robot_v2/share/shiyanpei/x2robot_tokenizer/logs/x2robot_tokenizer_v3_2/"
    "v3_2_0312-1_vq_2_1024_26d_delta/checkpoints/latest.pth",
)
RVQ_CFG = os.environ.get(
    "RVQ_CFG",
    "/x2robot_v2/share/shiyanpei/x2robot_tokenizer/logs/x2robot_tokenizer_v3_2/"
    "v3_2_0312-1_vq_2_1024_26d_delta/configs",
)
_HAS_MODEL = os.path.isdir(QWEN35_PATH) and os.path.isfile(
    os.path.join(QWEN35_PATH, "config.json")
)
_HAS_RVQ = os.path.isfile(RVQ_CKPT)
needs_model = pytest.mark.skipif(not _HAS_MODEL, reason="Qwen3.5 checkpoint absent")
needs_all = pytest.mark.skipif(
    not (_HAS_MODEL and _HAS_RVQ), reason="Qwen3.5 or RVQ checkpoint absent"
)

HORIZON_AR = 32


def _img(h, w):
    return Image.fromarray(np.uint8(np.random.rand(h, w, 3) * 255))


def _config_only_rope_index():
    """``get_rope_index`` bound to the config only (no model weights)."""
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    cfg = AutoConfig.from_pretrained(QWEN35_PATH)

    class _Shim:
        pass

    shim = _Shim()
    shim.config = cfg
    return functools.partial(Qwen3_5Model.get_rope_index, shim)


def _assert_packed_consistency(out):
    """Shared checks: bsz==1, position resets, GDN cu == explicit FA2 cu."""
    from transformers.modeling_flash_attention_utils import (
        prepare_fa_kwargs_from_position_ids,
    )

    ii, pos, cu = out["input_ids"], out["position_ids"], out["cu_seq_lens_q"]
    assert out["attention_mask"] is None
    assert ii.shape[0] == 1
    total = ii.shape[1]
    assert pos.shape == (3, 1, total)
    assert cu[0].item() == 0 and cu[-1].item() == total
    assert torch.all(cu.diff() > 0)
    # The GDN patch derives cu_seqlens from position_ids[0]; it must equal the
    # explicit FA2 cu_seqlens we emit for the full-attention layers.
    cu_from_pos = prepare_fa_kwargs_from_position_ids(pos[0])[0][0]
    assert cu_from_pos.tolist() == cu.tolist()
    # every document restarts at temporal position 0
    starts = cu[:-1].tolist()
    assert all(int(pos[0, 0, s].item()) == 0 for s in starts)


@needs_model
def test_vqa_packing():
    epi = EpilogueProcessor.from_config({
        "type": "multimodal_qwen3_5",
        "params": {
            "processor_path": QWEN35_PATH,
            "max_seq_length": 8192,
            "padding_side": "right",
            "packing": True,
            "get_rope_index": _config_only_rope_index(),
            "pack_sequences_fn": _pack_sequences,
        },
    })
    tok = epi.tokenizer
    s0 = [{"role": "user", "text": "<image>What is the bounding box of the cup?"},
          {"role": "assistant", "text": "<box>[100, 200, 300, 400]</box>"}]
    s1 = [{"role": "user", "text": "Hello, who are you?"},
          {"role": "assistant", "text": "I am an assistant."}]
    s2 = [{"role": "user", "text": "<image>What is this?"},
          {"role": "assistant", "text": "A cat."}]
    batch = {
        "qwen_dialogues_json": [json.dumps(s) for s in (s0, s1, s2)],
        "image_observations": [[[_img(224, 320)]], [], [[_img(256, 256)]]],
    }
    out = epi.process_batch(batch)
    _assert_packed_consistency(out)

    ii, lab, grid = out["input_ids"], out["labels"], out["image_grid_thw"]
    # image pads preserved
    n_pad = int((ii == epi.image_pad_id).sum())
    assert n_pad == int((grid[:, 0] * grid[:, 1] * grid[:, 2] // 4).sum())
    # all 3 assistant spans supervised; image tokens never
    sup = tok.decode(ii[0][lab[0] != -100])
    assert "100, 200, 300, 400" in sup
    assert "I am an assistant." in sup and "A cat." in sup
    assert int(((ii == epi.image_pad_id) & (lab != -100)).sum()) == 0


@needs_all
def test_action_plus_packing():
    from qwenvl.data.rvq_tokenizer import RVQActionTokenizer

    rvq = RVQActionTokenizer(
        checkpoint_path=RVQ_CKPT, config_dir=RVQ_CFG, device="cpu", rvq_version="v3_2",
    )
    epi = EpilogueProcessor.from_config({
        "type": "multimodal_action_qwen3_5",
        "params": {
            "processor_path": QWEN35_PATH,
            "max_seq_length": 8192,
            "padding_side": "right",
            "action_tokenizer_instance": rvq,
            "action_horizon_ar": HORIZON_AR,
            "state_bins": 256,
            "packing": True,
            "get_rope_index": _config_only_rope_index(),
            "pack_sequences_fn": _pack_sequences,
        },
    })
    tok = epi.tokenizer
    tok.add_tokens(rvq.get_special_tokens())

    adlg = [{"role": "user", "text": "Observation: front view: <image>\n"
                                     "Instruction: pick up the cup\n"
                                     "Predict the next action. Proprioception: <|propri|>"},
            {"role": "assistant", "text": "<|action_ar|>"}]
    vdlg = [{"role": "user", "text": "<image>What is the bounding box of the cup?"},
            {"role": "assistant", "text": "<box>[100, 200, 300, 400]</box>"}]
    action_ar = torch.randn(1, HORIZON_AR, rvq.action_dim) * 0.05
    propri = torch.randn(1, 1, rvq.action_dim) * 0.05
    batch = {
        "qwen_dialogues_json": [json.dumps(adlg), json.dumps(vdlg)],
        "image_observations": [[[_img(224, 320)]], [[_img(256, 256)]]],
        "action_ar": action_ar,
        "proprioception": propri,
        "agent_pos_mask": torch.ones(1, 1, rvq.action_dim),
        "dataset_name": ["x2_normal", "x2_multimodal"],
        "_action_ar_sample_indices": [0],
    }
    dof = (~torch.isnan(action_ar)).float()
    expected = rvq.encode_to_tokens(action_ar.nan_to_num(0.0), dof_mask=dof, obs_state=propri)[0]
    eid = [tok.convert_tokens_to_ids(t) for t in expected]
    assert tok.unk_token_id not in eid

    out = epi.process_batch(batch)
    _assert_packed_consistency(out)

    ii, lab = out["input_ids"], out["labels"]
    sup_ids = ii[0][lab[0] != -100].tolist()
    # RVQ action tokens are supervised, in order, inside the packed row
    rvq_run = [t for t in sup_ids if t in set(eid)]
    assert rvq_run == eid
    # VQA answer in the same packed row is supervised too
    assert "100, 200, 300, 400" in tok.decode(ii[0][lab[0] != -100])
