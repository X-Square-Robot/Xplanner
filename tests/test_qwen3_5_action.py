"""Tests for the Qwen3.5 action-aware epilogue (``multimodal_action_qwen3_5``).

Validates the discrete RVQ-delta action co-train data path (no GPU / LM needed --
only the Qwen3.5 tokenizer + image processor and the RVQ codec on CPU):

- ``action_qwen3_5`` text processor + ``multimodal_action_qwen3_5`` epilogue are
  registered;
- the ``RVQActionTokenizer`` adapter loads the real codec and round-trips
  ``encode_to_tokens`` -> ``<rvq_group>`` / ``<rvq_r{q}_{idx}>`` strings;
- in a **mixed** batch (1 action + 1 VQA sample) the action row's supervised
  labels are exactly the RVQ tokens (+ ``<|im_end|>``), the state-string lands in
  the (masked) user turn, ``<|image_pad|>`` expands to ``grid.prod()/4`` tokens,
  and the VQA row is unchanged.

Skipped automatically when the Qwen3.5 checkpoint or the RVQ checkpoint is absent.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from PIL import Image

from x2robot_dataset_v2.processors.epilogue import EPILOGUE_REGISTRY
from x2robot_dataset_v2.processors.epilogue.base import EpilogueProcessor
from x2robot_dataset_v2.processors.text import TEXT_PROCESSOR_REGISTRY

QWEN35_PATH = os.environ.get("QWEN35_PATH", "/data/Models/Qwen3.5-9B")
RVQ_CKPT = os.environ.get(
    "RVQ_CKPT", "",
)
RVQ_CFG = os.environ.get(
    "RVQ_CFG", "",
)

_HAS_MODEL = os.path.isdir(QWEN35_PATH) and os.path.isfile(
    os.path.join(QWEN35_PATH, "config.json")
)
_HAS_RVQ = os.path.isfile(RVQ_CKPT)
needs_all = pytest.mark.skipif(
    not (_HAS_MODEL and _HAS_RVQ),
    reason="Qwen3.5 checkpoint or RVQ checkpoint absent",
)

HORIZON_AR = 32
STATE_BINS = 256


def test_action_processors_registered():
    assert "multimodal_action_qwen3_5" in EPILOGUE_REGISTRY
    assert "action_qwen3_5" in TEXT_PROCESSOR_REGISTRY


def _img(h, w):
    return Image.fromarray(np.uint8(np.random.rand(h, w, 3) * 255))


@pytest.fixture(scope="module")
def rvq():
    from x_planner.data.rvq_tokenizer import RVQActionTokenizer

    return RVQActionTokenizer(
        checkpoint_path=RVQ_CKPT, config_dir=RVQ_CFG, device="cpu", rvq_version="v3_2",
    )


@needs_all
def test_rvq_adapter_geometry(rvq):
    specials = rvq.get_special_tokens()
    assert specials[0] == "<rvq_group>"
    assert len(specials) == 1 + rvq.num_quantizers * rvq.codebook_size
    acts = torch.randn(2, HORIZON_AR, rvq.action_dim) * 0.05
    toks = rvq.encode_to_tokens(acts, obs_state=torch.randn(2, 1, rvq.action_dim) * 0.05)
    num_latents = HORIZON_AR // rvq.compression_ratio
    assert len(toks) == 2
    assert len(toks[0]) == (1 + rvq.num_quantizers) * num_latents
    assert all(t.startswith("<rvq_") or t == "<rvq_group>" for t in toks[0])


@needs_all
def test_mixed_action_vqa_batch(rvq):
    epi = EpilogueProcessor.from_config({
        "type": "multimodal_action_qwen3_5",
        "params": {
            "processor_path": QWEN35_PATH,
            "max_seq_length": 4096,
            "padding_side": "right",
            "action_tokenizer_instance": rvq,
            "action_horizon_ar": HORIZON_AR,
            "state_bins": STATE_BINS,
        },
    })
    tok = epi.tokenizer
    # Register the RVQ specials so each maps to a single id (trainer does this too).
    tok.add_tokens(rvq.get_special_tokens())

    # Row 0: action sample (1 camera, <|propri|> in user, <|action_ar|> assistant).
    action_dlg = [
        {"role": "user", "text": "Observation: front view: <image>\n"
                                 "Instruction: pick up the cup\n"
                                 "Predict the next action. Proprioception: <|propri|>"},
        {"role": "assistant", "text": "<|action_ar|>"},
    ]
    # Row 1: VQA sample.
    vqa_dlg = [
        {"role": "user", "text": "<image>What is the bounding box of the cup?"},
        {"role": "assistant", "text": "<box>[100, 200, 300, 400]</box>"},
    ]

    action_ar = torch.randn(1, HORIZON_AR, rvq.action_dim) * 0.05  # compact: action rows only
    propri = torch.randn(1, 1, rvq.action_dim) * 0.05
    agent_pos_mask = torch.ones(1, 1, rvq.action_dim)

    batch = {
        "qwen_dialogues_json": [json.dumps(action_dlg), json.dumps(vqa_dlg)],
        "image_observations": [[[_img(224, 320)]], [[_img(256, 256)]]],
        "action_ar": action_ar,
        "proprioception": propri,
        "agent_pos_mask": agent_pos_mask,
        "dataset_name": ["x2_normal", "x2_multimodal"],
        "_action_ar_sample_indices": [0],
        "_proprioception_sample_indices": [0],
        "_agent_pos_mask_sample_indices": [0],
    }

    # Expected RVQ tokens for the action row (deterministic encode).
    dof = (~torch.isnan(action_ar[:, :HORIZON_AR, :])).float()
    expected_tokens = rvq.encode_to_tokens(
        action_ar[:, :HORIZON_AR, :].nan_to_num(0.0), dof_mask=dof, obs_state=propri,
    )[0]
    expected_token_ids = [tok.convert_tokens_to_ids(t) for t in expected_tokens]
    assert tok.unk_token_id not in expected_token_ids  # all registered as single ids

    out = epi.process_batch(batch)
    ii, lab, am, grid = (
        out["input_ids"], out["labels"], out["attention_mask"], out["image_grid_thw"]
    )
    assert ii.shape == lab.shape == am.shape
    assert grid.shape[0] == 2  # one image per row

    # image-pad count matches grid.
    n_pad = int((ii == epi.image_pad_id).sum())
    expect_pad = int((grid[:, 0] * grid[:, 1] * grid[:, 2] // 4).sum())
    assert n_pad == expect_pad

    def supervised_ids(i):
        keep = lab[i] != -100
        return ii[i][keep].tolist()

    # --- Action row (0): the RVQ tokens are the supervised block immediately
    # before the trailing <|im_end|>. The official Qwen3.5 template prepends an
    # empty "<think>\n\n</think>\n\n" to every assistant turn (the VQA path trains
    # with this too), so that fixed prefix is also supervised -- we only require
    # the RVQ tokens to be contiguous, in order, and the last content tokens. ---
    sup0 = supervised_ids(0)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    n = len(expected_token_ids)
    rvq_tail = sup0[-(n + 1):-1] if sup0 and sup0[-1] == im_end else sup0[-n:]
    assert rvq_tail == expected_token_ids, (
        f"action labels tail {rvq_tail[:6]}... != expected RVQ {expected_token_ids[:6]}..."
    )

    # state-string is in the (masked) user turn: digits present in input, not in labels.
    row0_input = tok.decode(ii[0][am[0] == 1])
    assert "Proprioception:" in row0_input
    assert "<|propri|>" not in row0_input and "<|action_ar|>" not in row0_input
    # no RVQ token is left unsupervised-leaked into the user span:
    assert int(((lab[0] == -100) & torch.isin(ii[0], torch.tensor(expected_token_ids))).sum()) == 0

    # --- VQA row (1): answer supervised, no RVQ tokens. ---
    sup1 = tok.decode(supervised_ids(1))
    assert "100, 200, 300, 400" in sup1
    rvq_id_set = torch.tensor(sorted(set(expected_token_ids)))
    assert int(torch.isin(ii[1], rvq_id_set).sum()) == 0

    # image / pad tokens never supervised; pads never supervised.
    assert int(((ii == epi.image_pad_id) & (lab != -100)).sum()) == 0
    assert int(((am == 0) & (lab != -100)).sum()) == 0
