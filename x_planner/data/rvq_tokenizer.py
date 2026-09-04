"""Thin adapter around the external x2robot RVQ-delta action codec.

Wraps ``X2RobotTokenizer`` (v3.2) / ``X2RobotTokenizerV3_1`` (v3.1-delta) from the
``x2robot_tokenizer`` package and exposes the small contract the Qwen3.5 action
epilogue + the trainer's token-registration step need:

- :meth:`get_special_tokens` -> the ``<rvq_group>`` + ``<rvq_r{q}_{idx}>`` vocab
  to register on the LM tokenizer (``1 + num_quantizers * codebook_size`` tokens);
- :meth:`encode_to_tokens` -> ``List[List[str]]`` of those special tokens, one
  inner list per sample (``(1 + num_quantizers) * num_latents`` tokens each).

It deliberately does **not** import ``wall_x.model.tokenizer_mixin`` (which pulls
the v1 ``x2robot_dataset`` package, absent in this env). The ~30 lines of token
string formatting are replicated from ``X2RobotV31DeltaTokenizerMixin``
(``wall_x/wall_x/model/tokenizer_mixin.py``) so behaviour is identical.

The codec runs on CPU during training (``device="cpu"``) to avoid device
mismatches under distributed training; encoding is ``torch.no_grad``.
"""

from __future__ import annotations

import importlib
from typing import Any, List, Optional

import torch

# version -> (module path, class name) of the underlying codec API
_RVQ_API = {
    "v3_2": (
        "x2robot_tokenizer.x2robot_tokenizer_v3_2.api.tokenizer",
        "X2RobotTokenizer",
    ),
    "v3_1_delta": (
        "x2robot_tokenizer.x2robot_tokenizer_v3_1_delta.api.tokenizer",
        "X2RobotTokenizerV3_1",
    ),
}


class RVQActionTokenizer:
    """Adapter exposing ``encode_to_tokens`` / ``get_special_tokens``.

    Parameters
    ----------
    checkpoint_path :
        Path to the codec ``latest.pth`` checkpoint.
    config_dir :
        Optional config directory (``config.yaml`` / ``robot_types.yaml``).
        When ``None`` the codec reads config from the checkpoint.
    device :
        Torch device for the codec (``"cpu"`` for training).
    rvq_version :
        ``"v3_2"`` (default) or ``"v3_1_delta"`` -- selects which codec API to load.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_dir: Optional[str] = None,
        device: str = "cpu",
        rvq_version: str = "v3_2",
    ) -> None:
        if rvq_version not in _RVQ_API:
            raise ValueError(
                f"Unknown rvq_version {rvq_version!r}; expected one of "
                f"{sorted(_RVQ_API)}"
            )
        module_path, class_name = _RVQ_API[rvq_version]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                f"Could not import the RVQ codec ({module_path}.{class_name}). "
                "Install the x2robot_tokenizer package into this env."
            ) from exc
        codec_cls = getattr(module, class_name)
        self.codec = codec_cls(
            checkpoint_path=checkpoint_path,
            config_dir=config_dir,
            device=device,
        )
        self.device = device
        self.rvq_version = rvq_version

    # ------------------------------------------------------------------
    # Codec geometry (read straight off the loaded checkpoint)
    # ------------------------------------------------------------------

    @property
    def codebook_size(self) -> int:
        return int(self.codec.codebook_size)

    @property
    def num_quantizers(self) -> int:
        return int(self.codec.num_quantizers)

    @property
    def compression_ratio(self) -> int:
        return int(self.codec.compression_ratio)

    @property
    def action_dim(self) -> int:
        return int(self.codec.action_dim)

    # ------------------------------------------------------------------
    # Vocab + encoding
    # ------------------------------------------------------------------

    def get_special_tokens(self) -> List[str]:
        """Return ``<rvq_group>`` + every ``<rvq_r{q}_{idx}>`` token string.

        Length is ``1 + num_quantizers * codebook_size``. The order is fixed and
        deterministic, so registering these on the LM tokenizer is reproducible.
        """
        tokens = ["<rvq_group>"]
        for q in range(self.num_quantizers):
            for idx in range(self.codebook_size):
                tokens.append(f"<rvq_r{q}_{idx}>")
        return tokens

    @torch.no_grad()
    def encode_to_tokens(
        self,
        actions: Any,
        dof_mask: Optional[Any] = None,
        obs_state: Optional[Any] = None,
        robot_type_ids: Optional[Any] = None,
    ) -> List[List[str]]:
        """Encode normalized delta actions to RVQ special-token strings.

        Parameters
        ----------
        actions :
            ``(B, T, D)`` already-normalized delta actions (NaNs are zeroed).
        dof_mask :
            ``(B, T, D)`` validity mask (True/1 = valid). Optional.
        obs_state :
            ``(B, obs_horizon, D)`` normalized proprioception; the codec uses the
            last frame as the reference state. Optional.
        robot_type_ids :
            Optional ``(B,)`` robot-type ids. Left ``None`` mirrors the
            dataset backend's epilogue, which does not pass them.

        Returns
        -------
        ``List[List[str]]`` -- one inner list of ``(1 + num_quantizers) *
        num_latents`` token strings per sample.
        """
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        actions = actions.nan_to_num(nan=0.0).float()

        if obs_state is not None:
            if not isinstance(obs_state, torch.Tensor):
                obs_state = torch.as_tensor(obs_state)
            obs_state = obs_state.float()
            # codec expects [B, 1, D]; take the last observed frame.
            if obs_state.dim() == 3 and obs_state.shape[1] > 1:
                obs_state = obs_state[:, -1:, :]

        if dof_mask is not None and not isinstance(dof_mask, torch.Tensor):
            dof_mask = torch.as_tensor(dof_mask)

        bsz, horizon, _ = actions.shape
        padding_mask = torch.ones(bsz, horizon, dtype=torch.bool)

        robot_type_id_tensor = None
        if robot_type_ids is not None:
            robot_type_id_tensor = (
                robot_type_ids
                if isinstance(robot_type_ids, torch.Tensor)
                else torch.as_tensor(robot_type_ids, dtype=torch.long)
            )

        indices = self.codec.encode(
            actions,
            obs_state=obs_state,
            dof_mask=dof_mask.bool() if dof_mask is not None else None,
            padding_mask=padding_mask,
            robot_type_id=robot_type_id_tensor,
        )["indices"]  # [B, num_latents, num_quantizers]

        token_lists: List[List[str]] = []
        bsz, num_latents, num_quantizers = indices.shape
        for b in range(bsz):
            tokens: List[str] = []
            for t in range(num_latents):
                tokens.append("<rvq_group>")
                for q in range(num_quantizers):
                    tokens.append(f"<rvq_r{q}_{int(indices[b, t, q].item())}>")
            token_lists.append(tokens)
        return token_lists


__all__ = ["RVQActionTokenizer"]
