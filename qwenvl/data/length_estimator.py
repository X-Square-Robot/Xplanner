"""Length estimator -- metadata-only per-sample token-length estimates.

Shared component of the length-balanced packing plan
(``docs/qwenvl/length_balanced_packing.md``).  The ``knapsack_packed`` sampler needs,
for every frame in every source, an estimate of how many LM tokens that sample will
become *after* the Qwen3.5 epilogue tokenizes it.  The estimate must be:

* **metadata-only** -- no pixel/video decode (read image headers only;
  ``av.probe`` reads container metadata only), so a multi-hundred-k corpus scans
  in minutes, not hours;
* **training-config-aligned** -- it uses the *same* tokenizer, the *same*
  ``smart_resize`` image budget, and the *same* video sampling parameters as the
  running pipeline, so estimates track reality closely enough for load balancing;
* **cached & content-addressed** -- keyed by ``(source signature, config hash)``;
  change the resolution / video params / model and the hash changes, forcing a
  recompute, but an unchanged config loads instantly.

What is estimated (one integer per frame)::

    len = base_overhead                          (system prefix, measured once)
        + Σ_turn (turn_overhead + content_tokens)        (real tokenize, no decode)
        + Σ_turn (think_overhead if assistant)           (per-turn <think> block)
        + Σ_image (grid_h*grid_w // merge^2 + 2)          (path images: header dims)
        + Σ_video_frame (same as image)                   ({video,frame} refs: container H/W)
        + Σ_video (T * (seqlen + ts_overhead + 2))        (av.probe metadata only)

The image term is exact for the active VQA config (``smart_resize`` is idempotent
across the vision-processor and HF-processor resizes when both use the same
factor/min/max).  The text term carries a small, *constant* per-turn template
overhead that is measured against the real tokenizer at init, so the estimate is
within a handful of tokens of the true length -- accuracy that balancing does not
need to exceed (estimation drift only mildly degrades balance; see plan R1).

The lengths array returned for a list of sources is concatenated in source order
and is therefore aligned to the sampler's **global tagged-index** space
(``X2RobotSampler._offsets``): ``lengths[tagged_idx]`` is the estimate for the
frame the sampler/dataset resolves at ``tagged_idx``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import logging

import numpy as np

try:  # optional: progress bars for the offline precompute (no-op if absent)
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None

from qwenvl.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    EMPTY_THINK_BLOCK,
)

logger = logging.getLogger("qwenvl.data.length_estimator")

# Sentinel written when a frame's length CANNOT be measured truthfully: a media
# file could not be opened / read / parsed (after retries), a referenced media
# exposes no usable dimensions, or the dialogue could not be rendered.  We never
# fabricate a default length for these -- a wrong length silently corrupts
# length-balanced packing (an under-estimate overflows its bin and the epilogue
# right-truncates the packed row, dropping *other* samples).  Instead the sampler
# drops every frame whose length is this sentinel and the scan writes a report
# sidecar naming them (path + sample_idx + reason).
ESTIMATE_FAILED = -1

# NOTE: non-multimodal sources (action / subtask) also get ESTIMATE_FAILED for now
# (no decode-free length estimator yet) and are dropped -- we do NOT fabricate a
# constant placeholder length, which would corrupt packing.  Value data is
# source_type=multimodal and computes a real length, so it is unaffected.

# Transient networked-storage reads (image headers, video container open) are
# retried this many times before a frame is flagged ESTIMATE_FAILED, so a one-off
# CPFS/NFS hiccup does not permanently drop a good sample (the flag gets cached).
_IO_RETRIES = max(1, int(os.environ.get("X2ROBOT_LENGTH_IO_RETRIES", "3")))

# Loudly warn if more than this fraction of a source is flagged unestimable -- a
# high ratio usually means storage was unhealthy during the scan, not that the
# data is bad (delete the cache + report and rescan rather than train on drops).
_DROP_WARN_RATIO = float(os.environ.get("X2ROBOT_LENGTH_DROP_WARN_RATIO", "0.01"))


class _EstimateFailed(Exception):
    """Raised internally when a frame's length cannot be measured truthfully.

    Carries a short ``reason`` tag plus context (path / ref) for the scan report.
    The ONLY catch point is :func:`_estimate_episode`, which maps it to the
    ``ESTIMATE_FAILED`` sentinel and a report record -- so no default length is
    ever fabricated and no single bad frame crashes the scan.
    """

    def __init__(self, reason: str, **ctx: Any) -> None:
        super().__init__(reason)
        self.reason = reason
        self.ctx = ctx


@dataclass(frozen=True)
class LengthEstimatorConfig:
    """Picklable estimator configuration (also the cache-key material).

    Every field that changes the produced token length must live here so the
    ``config_hash`` invalidates stale caches.
    """

    processor_path: str
    image_factor: int = 32
    # Divisor from resized pixels -> LM vision tokens = patch_size * spatial_merge.
    # None -> use image_factor (Qwen: patch16*merge2 = 32 = the resize factor). A
    # pluggable per-patch backbone (DINOv3: patch16*merge1) sets this to 16 while the
    # resize factor stays the dataset vision processor's (32), so smart_resize matches
    # the actual image dims but each 16px patch is its own token.
    token_factor: Optional[int] = None
    image_min_pixels: int = 1024
    image_max_pixels: int = 589824
    # Multi-image samples split the per-sample area budget across images: each image
    # is capped to max(image_min_pixels, image_max_pixels // n_images), so N images
    # don't blow up the token count (mirrors the vision processor's same-named flag).
    max_pixels_split_by_images: bool = False
    # Value (V) anchor frames: hard long-side cap applied at train time by the
    # dataset_v2 ``value_video`` processor (``VALUE_IMAGE_LONG_SIDE``, a project
    # constant, not a yml param).  Mirrored here so the cache hash tracks it;
    # ``_estimate_value_item`` fails loud if the two ever drift.
    value_image_long_side: int = 448
    video_fps: float = 2.0
    video_maxlen: int = 128
    video_max_pixels: int = 256 * 256
    video_min_pixels: int = 16 * 16
    video_temporal_patch_size: int = 2
    max_seq_length: Optional[int] = None
    per_turn_think: bool = True
    image_tokens_mode: str = "header"  # "header" | "assume_max"
    # Bump when the estimation LOGIC changes (not just its params) so that
    # content-addressed caches produced by an older logic are invalidated.
    #   v2: no fabricated default lengths -- every unmeasurable frame is flagged
    #       ESTIMATE_FAILED (-1) and dropped; image dims always read from the file
    #       (JSONL width/height ignored); transient I/O retried.
    #   v3: token count uses token_factor (patch*merge) split from the resize factor,
    #       so pluggable per-patch backbones (DINOv3) estimate correctly.
    #   v4: video temporal grid = ceil(n_sampled / temporal_patch_size) -- the HF
    #       video processor pads the frame count UP by repeating the last frame,
    #       so the old floor under-estimated every odd-count video by one full
    #       temporal block.
    #   v5: per-frame video branch for per-patch backbones (token_factor set):
    #       T = n_sampled (no temporal patching), seqlen = (rh/tok)*(rw/tok) --
    #       matches the epilogue's _build_video_blocks_per_frame; replaces v4's
    #       flag-as-unestimable behavior for that combination.
    #   v6: value (V) items: anchor frames long-side-capped to
    #       value_image_long_side before the grid math (mirrors the hard cap in
    #       the dataset_v2 value_video processor), and text is joined the way
    #       the chat template assembles content chunks (strip around <image>
    #       tags) -- the old estimate over-counted +1 token per camera.
    #   v7: max_pixels_split_by_images -- a multi-image sample splits its area budget
    #       across images (each -> max(min_pixels, image_max_pixels // n_images)),
    #       mirroring the vision processor so multi-image samples don't blow up.
    #   v8: clip bounds are 0-based inclusive frame indices
    #       ``{"path","start_frame","end_frame"}``.
    #   v9: ``image`` entries of the form ``{"video","frame","view?"}``
    #       (dataset_v2 ``video_frame`` vision) count as images: probe container
    #       H/W and apply the image ``smart_resize`` budget (not video sampling).
    estimator_version: int = 9

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()


def read_image_size(path: str, peek_kb: int = 32) -> Optional[Tuple[int, int]]:
    """Return ``(h, w)`` reading only the file header -- fast on networked storage.

    ``PIL.Image.open(path).size`` issues *many* small seeking reads over the
    network to sniff the format and locate the size marker (measured ~3.8 ms/image
    on CPFS, vs ~0.1 ms for ``os.stat`` and ~0.3 ms for ``open()`` at concurrency).
    Instead we do **one** sequential read of the first ``peek_kb`` KB and parse the
    dimensions from memory -- ~1.7x faster and it scales far better under
    concurrency.  ~0.75% of images (e.g. JPEGs with a large leading EXIF) need the
    full-file fallback; dims are byte-identical to PIL either way.

    Transient reads are retried ``_IO_RETRIES`` times (a networked-storage hiccup
    should not permanently flag a good image).  Returns ``None`` only if the header
    still cannot be read after retries (missing / truncated / unreadable).
    """
    import io
    from PIL import Image

    def _once() -> Tuple[int, int]:
        with open(path, "rb") as f:
            head = f.read(max(1, peek_kb) * 1024)
        try:
            with Image.open(io.BytesIO(head)) as im:
                w, h = im.size
        except Exception:
            with Image.open(path) as im:  # fallback: full-file lazy open
                w, h = im.size
        return int(h), int(w)

    last: Optional[Exception] = None
    for attempt in range(_IO_RETRIES):
        try:
            return _once()
        except Exception as exc:  # retry any read/parse failure (transient or not)
            last = exc
            if attempt + 1 < _IO_RETRIES:
                time.sleep(0.05 * (attempt + 1))
    logger.debug("read_image_size failed for %s after %d attempts: %r", path, _IO_RETRIES, last)
    return None


# Success-only memo for _probe_video_hw: consecutive frames of one value episode
# share the same mp4s, so the offline scan probes each camera video roughly once.
_VIDEO_HW_CACHE: Dict[str, Tuple[int, int]] = {}
_VIDEO_HW_CACHE_MAX = 8192


def _probe_video_hw(path: str) -> Tuple[int, int]:
    """Return ``(h, w)`` from container metadata only (no decode). (0,0) on failure.

    Successful probes are memoized per path; failures are deliberately NOT
    cached -- an ``lru_cache`` here would let one transient storage blip on the
    first frame permanently flag every later frame sharing that mp4 for the
    whole process lifetime, defeating the retry policy.  ``(0, 0)`` means
    "could not read dims" and the caller flags the frame ESTIMATE_FAILED
    (never a default size).
    """
    hit = _VIDEO_HW_CACHE.get(path)
    if hit is not None:
        return hit
    import av

    last: Optional[Exception] = None
    for attempt in range(_IO_RETRIES):
        try:
            container = av.open(path, "r")
            try:
                stream = next(s for s in container.streams if s.type == "video")
                h = int(stream.codec_context.height or 0)
                w = int(stream.codec_context.width or 0)
            finally:
                container.close()
            if h > 0 and w > 0:
                if len(_VIDEO_HW_CACHE) >= _VIDEO_HW_CACHE_MAX:
                    _VIDEO_HW_CACHE.clear()
                _VIDEO_HW_CACHE[path] = (h, w)
            return (h, w)
        except Exception as exc:
            last = exc
            if attempt + 1 < _IO_RETRIES:
                time.sleep(0.05 * (attempt + 1))
    logger.debug("_probe_video_hw failed for %s after %d attempts: %r", path, _IO_RETRIES, last)
    return (0, 0)


class LengthEstimator:
    """Estimate the tokenized length of one multimodal JSONL sample (no decode).

    The constant template overheads (per-turn open/close, think block, timestamp,
    base/system prefix) are derived once from the tokenizer/processor.  Pass
    ``overheads`` (from :meth:`overheads`) to skip the one-time
    ``AutoProcessor`` probe -- the scan does this so pool workers load only the
    (lighter) tokenizer, not a full processor each.
    """

    def __init__(
        self, config: LengthEstimatorConfig, overheads: Optional[Dict[str, Any]] = None
    ) -> None:
        self.cfg = config
        from transformers import AutoTokenizer
        from x2robot_dataset_v2.processors.vision.base import smart_resize

        self._smart_resize = smart_resize
        self._tok = AutoTokenizer.from_pretrained(config.processor_path)

        # Measure structural template overheads against the *real* tokenizer so
        # the estimate self-calibrates to whatever processor is configured.
        def n(s: str) -> int:
            return len(self._tok.encode(s, add_special_tokens=False))

        self._turn_open = {
            role: n(f"<|im_start|>{role}\n") for role in ("user", "assistant", "system")
        }
        self._turn_close = n("<|im_end|>\n")
        self._think_overhead = n(EMPTY_THINK_BLOCK) if config.per_turn_think else 0
        self._ts_overhead = n("<999.5 seconds>")  # per video temporal-token timestamp
        # Base overhead = whatever the chat template adds beyond per-turn content
        # (e.g. a default system prefix).  Measured once via a probe render; reused
        # across workers via ``overheads`` to avoid 1 AutoProcessor load per worker.
        if overheads is not None and "base" in overheads:
            self._base_overhead = int(overheads["base"])
        else:
            self._base_overhead = self._measure_base_overhead()

    def overheads(self) -> Dict[str, Any]:
        """The constant overheads (picklable) to hand to pool workers."""
        return {"base": self._base_overhead}

    # ------------------------------------------------------------------
    def _measure_base_overhead(self) -> int:
        """Render a 2-turn probe and back out the constant template overhead."""
        try:
            from transformers import AutoProcessor

            proc = AutoProcessor.from_pretrained(self.cfg.processor_path)
            probe = [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
            text = proc.apply_chat_template(
                probe, tokenize=False, add_generation_prompt=False
            )
            if self.cfg.per_turn_think:
                from x2robot_dataset_v2.processors.epilogue.qwen3_5_epilogue import (
                    inject_per_turn_think,
                )

                text = inject_per_turn_think(text)
            total = len(self._tok.encode(text, add_special_tokens=False))
            expected = (
                self._turn_open["user"] + 1 + self._turn_close
                + self._turn_open["assistant"] + 1 + self._turn_close
                + self._think_overhead
            )
            return max(0, total - expected)
        except Exception as exc:  # pragma: no cover - never fail estimation on this
            logger.warning("LengthEstimator: base-overhead probe failed (%s); using 0", exc)
            return 0

    # ------------------------------------------------------------------
    def _image_tokens(self, h: int, w: int, max_pixels: Optional[int] = None) -> int:
        """LM tokens for one image: (rh/tok)*(rw/tok) merged tokens + 2.

        ``smart_resize`` uses the (dataset vision processor) ``image_factor`` so the
        resized dims match the real pipeline; the token divisor is ``token_factor``
        (== patch_size*spatial_merge; falls back to image_factor for the Qwen path).
        ``max_pixels`` overrides the area cap (used by the multi-image budget split).
        """
        factor = self.cfg.image_factor
        tok = self.cfg.token_factor or factor
        rh, rw = self._smart_resize(
            h, w,
            factor=factor,
            min_pixels=self.cfg.image_min_pixels,
            max_pixels=self.cfg.image_max_pixels if max_pixels is None else max_pixels,
        )
        return (rh // tok) * (rw // tok) + 2  # +vision_start/+vision_end

    def _per_image_max_pixels(self, n_images: int) -> int:
        """Area cap for one image in an n-image sample (mirrors vision processor)."""
        mp = self.cfg.image_max_pixels
        if self.cfg.max_pixels_split_by_images and n_images > 1:
            mp = max(self.cfg.image_min_pixels, self.cfg.image_max_pixels // n_images)
        return mp

    def _image_tokens_for_refs(
        self,
        image_refs: List[str],
        jsonl_path: str,
        image_dims: Optional[Dict[str, Tuple[int, int]]] = None,
        max_pixels: Optional[int] = None,
    ) -> int:
        if not image_refs:
            return 0
        mp = (
            self._per_image_max_pixels(len(image_refs))
            if max_pixels is None
            else max_pixels
        )
        if self.cfg.image_tokens_mode == "assume_max":
            per = self._image_tokens(4096, 4096, max_pixels=mp)  # clamped to the budget
            return per * len(image_refs)

        from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
            resolve_jsonl_image_path,
        )

        # Always read dims from the actual image FILE -- JSONL width/height
        # annotations are ignored (they can be stale/wrong, and the runtime loader
        # opens the file too).  The per-episode concurrent prefetch (image_dims)
        # only serves the read faster; a miss reads here (with retries).  If the
        # file cannot be read at all, flag the frame instead of guessing a size.
        total = 0
        for ref in image_refs:
            path = resolve_jsonl_image_path(ref, jsonl_path)
            hw = image_dims.get(path) if image_dims is not None else None
            if hw is None:
                hw = read_image_size(path)
            if hw is None:
                raise _EstimateFailed("image_unreadable", ref=ref, media=path)
            total += self._image_tokens(hw[0], hw[1], max_pixels=mp)
        return total

    def _image_tokens_for_video_frame_refs(
        self,
        video_frame_refs: List[Dict[str, Any]],
        jsonl_path: str,
        max_pixels: Optional[int] = None,
    ) -> int:
        """Token count for ``video_frame`` vision refs (decoded frames as images).

        Matches dataset_v2 ``VideoFrameVisionProcessor``: each ``{video, frame}``
        becomes one image, resized with the *image* ``min/max_pixels`` budget
        (not ``video_*`` sampling / temporal patches).
        """
        if not video_frame_refs:
            return 0
        mp = (
            self._per_image_max_pixels(len(video_frame_refs))
            if max_pixels is None
            else max_pixels
        )
        if self.cfg.image_tokens_mode == "assume_max":
            return self._image_tokens(4096, 4096, max_pixels=mp) * len(video_frame_refs)

        from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
            resolve_jsonl_image_path,
        )

        total = 0
        for ref in video_frame_refs:
            video = ref.get("video")
            if not isinstance(video, str) or not video.strip():
                raise _EstimateFailed("video_frame_bad_ref", ref=ref)
            path = resolve_jsonl_image_path(video, jsonl_path)
            h, w = _probe_video_hw(path)
            if h <= 0 or w <= 0:
                raise _EstimateFailed(
                    "video_frame_no_dims", ref=ref, media=path
                )
            total += self._image_tokens(h, w, max_pixels=mp)
        return total

    def _probe_video_meta(
        self, path: str, ref: str
    ) -> Tuple[int, float, float, int, int]:
        """(n_frames, duration_s, native_fps, height, width) from container metadata.

        Retries transient open failures.  Raises ``_EstimateFailed`` if the
        container cannot be opened after retries (``video_unreadable``) or opens
        but exposes no frame dimensions (``video_no_dims`` -- reported, never
        defaulted, because a wrong size would mis-estimate and overflow the row).
        No pixel decode.
        """
        import av

        last: Optional[Exception] = None
        for attempt in range(_IO_RETRIES):
            try:
                container = av.open(path, "r")
                try:
                    stream = next(s for s in container.streams if s.type == "video")
                    n_frames = int(stream.frames or 0)
                    avg_rate = float(stream.average_rate) if stream.average_rate else 0.0
                    if stream.duration is not None and stream.time_base is not None:
                        dur = float(stream.duration * stream.time_base)
                    elif avg_rate > 0 and n_frames > 0:
                        dur = n_frames / avg_rate
                    else:
                        dur = 0.0
                    if dur > 0 and n_frames > 0:
                        native_fps = n_frames / dur
                    elif avg_rate > 0:
                        native_fps = avg_rate
                    else:
                        native_fps = 2.0
                    fh = int(stream.codec_context.height or 0)
                    fw = int(stream.codec_context.width or 0)
                finally:
                    container.close()
                if fh <= 0 or fw <= 0:
                    # Opened fine but no usable h/w -> report, do not guess a size.
                    raise _EstimateFailed("video_no_dims", ref=ref, media=path)
                return n_frames, dur, float(native_fps), fh, fw
            except _EstimateFailed:
                raise  # no-dims is deterministic; retrying will not add dims
            except Exception as exc:  # retry transient container-open failures
                last = exc
                if attempt + 1 < _IO_RETRIES:
                    time.sleep(0.05 * (attempt + 1))
        raise _EstimateFailed("video_unreadable", ref=ref, media=path, error=repr(last))

    def _video_tokens(self, video_refs: List[Any], jsonl_path: str) -> int:
        if not video_refs:
            return 0
        import math

        from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
            resolve_jsonl_image_path,
        )
        from x2robot_dataset_v2.utils.multimodal_video import (
            parse_video_ref,
            resolve_clip_window,
        )

        factor = self.cfg.image_factor
        # Unified video math -- two independent divisors, no mode flag:
        #   spatial:  seqlen = (rh // vtok) * (rw // vtok)
        #             (Qwen: vtok = image_factor = patch16*merge2; per-patch
        #              backbones: vtok = token_factor = 16)
        #   temporal: T = ceil(n_sampled / video_temporal_patch_size)
        #             (Qwen frame pairs: 2; per-frame backbones: 1)
        # The two are set INDEPENDENTLY (a future video backbone may pair
        # frames while staying per-patch spatially), so require the injection
        # to have made them consistent instead of inferring one from the other.
        vtok = int(self.cfg.token_factor or factor)
        tps = int(self.cfg.video_temporal_patch_size)
        if vtok != factor and tps != 1:
            raise ValueError(
                "per-patch backbone (token_factor != image_factor) does no "
                "temporal pairing: set video_temporal_patch_size=1 alongside "
                "token_factor (inject_length_estimator_cfg does this "
                "automatically; a hand-built config must too)."
            )
        total = 0
        for raw in video_refs:
            parsed = parse_video_ref(raw)
            if parsed is None:
                continue
            ref_path, start_f, end_f = parsed
            path = resolve_jsonl_image_path(ref_path, jsonl_path)
            n_frames, dur, native_fps, fh, fw = self._probe_video_meta(path, ref_path)
            _s, _e, clip_n, clip_dur = resolve_clip_window(
                n_frames, dur, native_fps, start_f, end_f
            )

            if clip_n <= 0:
                # Container reported no frame count -> LF "infinite video": runtime
                # samples exactly video_maxlen frames, so this is the true sampling
                # rule, NOT a fabricated default.
                n_sampled = self.cfg.video_maxlen
            else:
                n_sampled = min(
                    clip_n,
                    self.cfg.video_maxlen,
                    max(1, math.floor(clip_dur * self.cfg.video_fps)),
                )
            # CEIL: the HF video processor pads the sampled frames UP to a
            # multiple of temporal_patch_size (repeating the last frame) before
            # building the grid -- a floor under-estimates every odd-count
            # video by one temporal block. tps=1 (per-frame) reduces to T=n.
            T = max(1, -(-n_sampled // tps))

            # Area-cap the frame to [video_min, video_max] then smart_resize.
            area = fw * fh
            if area > self.cfg.video_max_pixels:
                rf = math.sqrt(self.cfg.video_max_pixels / area)
                fw, fh = max(1, int(fw * rf)), max(1, int(fh * rf))
            rh, rw = self._smart_resize(
                fh, fw, factor=factor,
                min_pixels=self.cfg.video_min_pixels,
                max_pixels=self.cfg.video_max_pixels,
            )
            # smart_resize(factor) makes rh/rw multiples of `factor`, so for the
            # Qwen path (vtok == factor) this equals the old (rh//16)(rw//16)/merge²
            # exactly -- the qwen config hash and cached values are unaffected.
            seqlen = (rh // vtok) * (rw // vtok)
            # per temporal token: <ts> + vision_start + seqlen*video_pad + vision_end
            total += T * (seqlen + self._ts_overhead + 2)
        return total

    # ------------------------------------------------------------------
    def _estimate_value_item(self, item: Dict[str, Any], jsonl_path: str) -> int:
        """Length of a *value* (V) source sample -- mp4 anchor frames + V Q/A.

        Uses the SAME renderer the value processors use (``build_value_dialogue``
        in ``multimodal_utils``), so the estimate matches train-time token counts
        exactly except for tiny per-camera resolution differences.  Frame dims come
        from container metadata (no decode), cached per path.
        """
        from x2robot_dataset_v2.processors.vision.value_video_vision_processor import (
            VALUE_IMAGE_LONG_SIDE,
            cap_value_frame_hw,
        )
        from x2robot_dataset_v2.utils.multimodal_utils import (
            build_value_dialogue,
            value_ordered_cams,
        )

        # The train-time cap is a dataset_v2 constant (hard-coded in the
        # value_video processor); the config field only mirrors it into the
        # cache hash.  A drift between the two would silently mis-size every
        # value row, so it is a config error, not a per-sample failure.
        if int(self.cfg.value_image_long_side) != int(VALUE_IMAGE_LONG_SIDE):
            raise ValueError(
                f"length_estimator.value_image_long_side="
                f"{self.cfg.value_image_long_side} but dataset_v2 hard-codes "
                f"VALUE_IMAGE_LONG_SIDE={VALUE_IMAGE_LONG_SIDE}; the two must "
                "match (change both, then rebuild the length cache)."
            )

        try:
            turns, _n_img = build_value_dialogue(item, jsonl_path)
        except Exception as exc:
            raise _EstimateFailed("value_dialogue_error", error=repr(exc))
        n = self._base_overhead
        for turn in turns:
            role = turn.get("role", "user")
            raw = (turn.get("text", "") or "").replace(DEFAULT_VIDEO_TOKEN, "")
            # build_qwen_messages splits the turn on <image> and
            # apply_chat_template strips each text chunk when assembling the
            # content list.  Tokenize the stripped chunks SEPARATELY: in the
            # real stream a <|vision_start|> special token sits between them
            # (a hard tokenizer boundary), so joining chunks first would let
            # BPE merge across the seam and under-count.
            n += self._turn_open.get(role, self._turn_open["user"])
            n += self._turn_close
            n += sum(
                len(self._tok.encode(p.strip(), add_special_tokens=False))
                for p in raw.split(DEFAULT_IMAGE_TOKEN)
            )
            if role == "assistant" and self.cfg.per_turn_think and "<think>" not in raw:
                n += self._think_overhead

        vision = 0
        for _cam, path in value_ordered_cams(item):
            h, w = _probe_video_hw(path)
            if h <= 0 or w <= 0:
                # Report, do not guess a size -- a wrong frame size mis-estimates
                # the row and can overflow its pack bin.
                raise _EstimateFailed("value_video_no_dims", media=path)
            vision += self._image_tokens(
                *cap_value_frame_hw(h, w, self.cfg.value_image_long_side)
            )
        n += vision

        if self.cfg.max_seq_length:
            n = min(n, max(int(self.cfg.max_seq_length), vision))
        return int(n)

    def estimate_item(
        self,
        item: Dict[str, Any],
        jsonl_path: str,
        image_dims: Optional[Dict[str, Tuple[int, int]]] = None,
    ) -> int:
        """Estimate the tokenized length of one JSONL sample dict.

        ``image_dims`` is an optional ``{abs_path: (h, w)}`` cache (a per-episode
        concurrent prefetch) so image header reads don't serialize per sample.
        """
        # Value (V) source records carry mp4 paths + value_logits, no text/image.
        if "value_logits" in item:
            return self._estimate_value_item(item, jsonl_path)

        from x2robot_dataset_v2.utils.multimodal_schema import (
            iter_multimodal_image_refs,
            normalize_multimodal_dialogues,
        )
        from x2robot_dataset_v2.utils.multimodal_utils import process_dialogue

        # Path strings (multimodal_jsonl) + {video, frame} dicts (video_frame).
        # Must match dataset_v2 vision/text image counting.
        image_media = iter_multimodal_image_refs(item.get("image"))
        path_refs = [r for r in image_media if isinstance(r, str)]
        video_frame_refs = [r for r in image_media if isinstance(r, dict)]
        num_images = len(image_media)

        from x2robot_dataset_v2.utils.multimodal_video import (
            iter_raw_video_refs,
            parse_video_ref,
        )

        video_refs = [
            raw for raw in iter_raw_video_refs(item) if parse_video_ref(raw) is not None
        ]

        try:
            dialogues = normalize_multimodal_dialogues(item)
            # Deterministic seed so estimates are stable across runs (cache + resume).
            processed = process_dialogue(dialogues, seed=0, num_images=num_images)
        except Exception as exc:
            # A conversation the renderer rejects would also crash the runtime text
            # processor (same normalize/process_dialogue), so flag it, don't guess.
            raise _EstimateFailed("dialogue_error", error=repr(exc))

        n = self._base_overhead
        for turn in processed:
            role = turn.get("role", "user")
            text = turn.get("text", "") or ""
            text = text.replace(DEFAULT_IMAGE_TOKEN, "").replace(DEFAULT_VIDEO_TOKEN, "")
            n += self._turn_open.get(role, self._turn_open["user"])
            n += self._turn_close
            n += len(self._tok.encode(text, add_special_tokens=False))
            if role == "assistant" and self.cfg.per_turn_think:
                # The template/epilogue only injects an empty <think> when the turn
                # does not already carry a real one.
                if "<think>" not in text:
                    n += self._think_overhead

        # Shared multi-image area budget across path + video_frame images.
        mp = self._per_image_max_pixels(num_images) if num_images else None
        vision = self._image_tokens_for_refs(
            path_refs, jsonl_path, image_dims, max_pixels=mp
        )
        vision += self._image_tokens_for_video_frame_refs(
            video_frame_refs, jsonl_path, max_pixels=mp
        )
        vision += self._video_tokens(video_refs, jsonl_path)
        n += vision

        if self.cfg.max_seq_length:
            # Mirror _safe_truncate: the row is right-truncated to max_seq_length
            # but never inside a vision span, so the floor is the vision footprint.
            n = min(n, max(int(self.cfg.max_seq_length), vision))
        return int(n)


# ======================================================================
# Multiprocess scan + per-source caching
# ======================================================================

# Per-worker estimator singleton (built lazily in the pool worker).
_WORKER_EST: Optional[LengthEstimator] = None
_WORKER_CFG: Optional[LengthEstimatorConfig] = None
_WORKER_OVERHEADS: Optional[Dict[str, Any]] = None
# Image-header reads release the GIL, so each process worker prefetches its
# episode's image dims with a thread pool -- the open()+chunk-read is latency-bound
# and scales with concurrency, while the GIL-bound tokenize stays parallel across
# processes.  Tunable; affects speed only, not the cached result (so not in config).
_WORKER_DIM_THREADS = int(os.environ.get("X2ROBOT_LENGTH_DIM_THREADS", "16"))


def _worker_init(cfg_dict: Dict[str, Any], overheads: Optional[Dict[str, Any]] = None) -> None:
    global _WORKER_EST, _WORKER_CFG, _WORKER_OVERHEADS
    _WORKER_CFG = LengthEstimatorConfig(**cfg_dict)
    _WORKER_OVERHEADS = overheads
    _WORKER_EST = LengthEstimator(_WORKER_CFG, overheads=overheads)


def _prefetch_image_dims(items: Dict[int, Any], jsonl_path: str) -> Dict[str, Tuple[int, int]]:
    """Concurrently read (h, w) for all images referenced by an episode's items.

    Also warms the video-HW cache for ``video_frame`` ``{video, frame}`` refs so
    the per-sample probe is a hit (container open is the expensive part).
    """
    from x2robot_dataset_v2.readers.multimodal_jsonl_reader import resolve_jsonl_image_path
    from x2robot_dataset_v2.utils.multimodal_schema import iter_multimodal_image_refs

    paths = set()
    video_paths = set()
    for item in items.values():
        for r in iter_multimodal_image_refs(item.get("image")):
            if isinstance(r, str):
                paths.add(resolve_jsonl_image_path(r, jsonl_path))
            elif isinstance(r, dict):
                video = r.get("video")
                if isinstance(video, str) and video.strip():
                    video_paths.add(resolve_jsonl_image_path(video, jsonl_path))
    from concurrent.futures import ThreadPoolExecutor

    if video_paths:
        vpaths = list(video_paths)
        threads = max(1, min(_WORKER_DIM_THREADS, len(vpaths)))
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(_probe_video_hw, vpaths))

    if not paths:
        return {}

    paths = list(paths)
    threads = max(1, min(_WORKER_DIM_THREADS, len(paths)))
    with ThreadPoolExecutor(threads) as ex:
        sizes = list(ex.map(read_image_size, paths))
    return {p: s for p, s in zip(paths, sizes) if s is not None}


def _estimate_episode(
    ep_info: Tuple[str, int, int, Optional[List[int]], int, int]
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Estimate every frame of one episode chunk.

    Returns ``(int64[num_frames], reports)``.  A frame whose length cannot be
    measured truthfully is set to ``ESTIMATE_FAILED`` (-1) and a report record is
    appended -- no default length is ever fabricated, and no single bad frame
    aborts the scan (a stray exception is caught and flagged like any other).
    """
    global _WORKER_EST
    from x2robot_dataset_v2.readers.multimodal_jsonl_reader import (
        load_indexed_jsonl_items,
    )

    path, st_index, num_frames, no_static, iter_st, _ = ep_info
    if _WORKER_EST is None:  # single-process fallback
        _worker_init(asdict(_WORKER_CFG), _WORKER_OVERHEADS)  # type: ignore[arg-type]

    if no_static is not None:
        sample_indices = [st_index + no_static[iter_st + i] for i in range(num_frames)]
    else:
        sample_indices = [st_index + i for i in range(num_frames)]

    items, errors = load_indexed_jsonl_items(path, sample_indices)
    image_dims = _prefetch_image_dims(items, path)
    out = np.empty(num_frames, dtype=np.int64)
    reports: List[Dict[str, Any]] = []
    for i, sidx in enumerate(sample_indices):
        item = items.get(sidx)
        if item is None:
            out[i] = ESTIMATE_FAILED
            err = errors.get(sidx)
            reports.append({
                "path": path, "sample_idx": int(sidx),
                "reason": "item_load_error", "error": repr(err) if err else None,
            })
            continue
        try:
            out[i] = _WORKER_EST.estimate_item(item, path, image_dims=image_dims)
        except _EstimateFailed as ef:
            out[i] = ESTIMATE_FAILED
            reports.append({"path": path, "sample_idx": int(sidx), "reason": ef.reason, **ef.ctx})
        except Exception as exc:  # never let one frame crash the whole scan
            out[i] = ESTIMATE_FAILED
            reports.append({
                "path": path, "sample_idx": int(sidx),
                "reason": "unexpected", "error": repr(exc),
            })
    return out, reports


def _source_signature(source: Any) -> str:
    """Content-address a source by its episode layout (path + chunk ranges).

    ``no_static_frames`` content is hashed too (not just implied by the count):
    the frame->sample mapping runs through that list (see ``_estimate_episode``),
    so a regenerated list with the same length but different indices describes
    *different samples* and must not reuse the old cache.
    """
    h = hashlib.blake2b(digest_size=8)
    h.update(source.name.encode("utf-8"))
    for ep in source.frame_index.iter_episodes():
        h.update(
            f"{ep.path}|{ep.st_index}|{ep.num_frames}|{ep.iter_st}".encode("utf-8")
        )
        ns = getattr(ep, "no_static_frames", None)
        if ns is not None:
            h.update(np.asarray(ns, dtype=np.int64).tobytes())
    return h.hexdigest()


def _episode_infos(source: Any) -> List[Tuple[str, int, int, Optional[List[int]], int, int]]:
    infos = []
    for ep in source.frame_index.iter_episodes():
        no_static = list(ep.no_static_frames) if ep.no_static_frames is not None else None
        infos.append(
            (str(ep.path), int(ep.st_index or 0), int(ep.num_frames), no_static, int(ep.iter_st), 0)
        )
    return infos


def _mp_context():
    """Pick a multiprocessing start method.

    ``fork`` is much cheaper (no re-import, no per-worker tokenizer reload -- the
    same reason PenguinVL's offline tool is fast), but it is **unsafe once CUDA is
    initialized** in the parent (the in-training case, where the model is already
    on GPU).  So: ``fork`` when CUDA is not initialized (offline precompute /
    pre-model-load), ``spawn`` otherwise.
    """
    import multiprocessing as mp

    try:
        import torch

        if torch.cuda.is_initialized():
            return mp.get_context("spawn")
    except Exception:
        pass
    try:
        return mp.get_context("fork")
    except ValueError:  # platform without fork
        return mp.get_context("spawn")


def _atomic_save(path: str, arr: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    np.save(tmp, arr)
    # np.save appends .npy to a path without extension; normalize.
    tmp_npy = tmp if tmp.endswith(".npy") else tmp + ".npy"
    os.replace(tmp_npy, path)


def _wait_for_file(path: str, timeout_s: Optional[float] = None) -> None:
    if timeout_s is None:
        timeout_s = float(os.environ.get("X2ROBOT_LENGTH_CACHE_TIMEOUT_S", "3600"))
    t0 = time.monotonic()
    while not os.path.isfile(path):
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError(
                f"length cache {path} not produced within {timeout_s:.0f}s. Rank 0 "
                "is likely still scanning a large source (raise "
                "X2ROBOT_LENGTH_CACHE_TIMEOUT_S, or pre-build the cache with "
                "`python -m qwenvl.tools.precompute_lengths`), or the cache dir "
                "is not on a filesystem shared by all nodes."
            )
        time.sleep(2.0)


def _cache_path(source: Any, cfg: LengthEstimatorConfig, cache_dir: str) -> str:
    sig = _source_signature(source)
    fname = f"lengths.{source.name}.{cfg.config_hash()}.{sig}.npy"
    return os.path.join(os.path.expanduser(cache_dir), fname)


def _report_path(source: Any, cfg: LengthEstimatorConfig, cache_dir: str) -> str:
    """Sidecar path (next to the .npy cache) listing every flagged frame."""
    sig = _source_signature(source)
    fname = f"lengths.{source.name}.{cfg.config_hash()}.{sig}.report.jsonl"
    return os.path.join(os.path.expanduser(cache_dir), fname)


def _write_report(report_path: str, reports: List[Dict[str, Any]]) -> None:
    """Atomically write the per-frame drop report (one JSON object per line)."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    tmp = f"{report_path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        for r in reports:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, report_path)


def _scan_infos(
    infos: List[Tuple[str, int, int, Optional[List[int]], int, int]],
    *,
    name: str,
    total: int,
    cfg_dict: Dict[str, Any],
    overheads: Dict[str, Any],
    pool: Any = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Estimate every frame for a list of episode infos (via ``pool`` or inline).

    Returns ``(int64[total], reports)``.  Shows a per-source tqdm bar (frames
    scanned) when tqdm is installed; otherwise falls back to logging progress
    every 30s (pool path only), as before.
    """
    t0 = time.monotonic()
    n_eps = len(infos)
    parts: List[np.ndarray] = [None] * n_eps  # type: ignore[list-item]
    reports: List[Dict[str, Any]] = []
    if n_eps > 0:
        if pool is not None:
            results = pool.imap(_estimate_episode, infos, chunksize=4)
        else:
            _worker_init(cfg_dict, overheads)  # in-process: prime the globals once
            results = map(_estimate_episode, infos)
        bar = None
        if _tqdm is not None:
            bar = _tqdm(total=total, unit="fr", unit_scale=True, desc=f"len:{name}"[:32], leave=False)
        done = 0
        last_log = t0
        for i, (part_arr, part_reports) in enumerate(results):
            parts[i] = part_arr
            if part_reports:
                reports.extend(part_reports)
            done += int(part_arr.shape[0])
            if bar is not None:
                bar.update(int(part_arr.shape[0]))
            elif pool is not None and time.monotonic() - last_log >= 30:
                now = time.monotonic()
                rate = done / max(now - t0, 1e-9)
                eta = (total - done) / rate if rate > 0 else 0.0
                logger.info(
                    "LengthEstimator: %r %d/%d frames (%.0f%%) %.0f fr/s eta %.0fs",
                    name, done, total, 100.0 * done / max(total, 1), rate, eta,
                )
                last_log = now
        if bar is not None:
            bar.close()
    arr = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    if arr.shape[0] != total:
        raise RuntimeError(
            f"LengthEstimator: source {name!r} produced {arr.shape[0]} lengths "
            f"but total_frames={total}."
        )
    valid = arr[arr >= 0]
    logger.info(
        "LengthEstimator: source %r done in %.1fs (mean=%.0f max=%d flagged=%d).",
        name, time.monotonic() - t0, float(valid.mean()) if valid.size else 0.0,
        int(valid.max()) if valid.size else 0, int((arr < 0).sum()),
    )
    return arr, reports


def estimate_source_lengths(
    source: Any,
    cfg: LengthEstimatorConfig,
    cache_dir: str,
    *,
    num_workers: int = 8,
    rank: int = 0,
    world_size: int = 1,
    pool: Any = None,
    overheads: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Return int64[total_frames] for one source (cached, decode-free).

    ``pool`` / ``overheads`` let a caller reuse one worker pool across many
    sources (see :func:`estimate_lengths_for_sources`); when omitted a private
    pool is built for this source.
    """
    total = int(source.total_frames)
    path = _cache_path(source, cfg, cache_dir)

    if os.path.isfile(path):
        return np.load(path)

    # Non-multimodal sources (action / subtask / ...) have no decode-free length
    # estimator yet, so we do NOT fabricate a length for them -- every frame is
    # flagged ESTIMATE_FAILED and the sampler drops it.  (Value data is
    # source_type=multimodal and DOES compute a real length via _estimate_value_item,
    # so it is unaffected.)  When an action length branch lands, route it like
    # multimodal here instead.
    if getattr(source, "source_type", "action") != "multimodal":
        logger.warning(
            "LengthEstimator: source %r is %s (no decode-free length estimator yet); "
            "flagging ALL %d frames as unestimable -- they will be DROPPED from "
            "length-balanced training until a %s length branch lands.",
            source.name, getattr(source, "source_type", "?"), total,
            getattr(source, "source_type", "?"),
        )
        arr = np.full(total, ESTIMATE_FAILED, dtype=np.int64)
        if rank == 0:
            _atomic_save(path, arr)
        return arr

    # Distributed: only rank 0 computes; other ranks wait for the atomic file.
    if world_size > 1 and rank != 0:
        _wait_for_file(path)
        return np.load(path)

    cfg_dict = asdict(cfg)
    if overheads is None:
        overheads = LengthEstimator(cfg).overheads()
    infos = _episode_infos(source)

    if pool is not None:
        arr, reports = _scan_infos(infos, name=source.name, total=total,
                                   cfg_dict=cfg_dict, overheads=overheads, pool=pool)
    elif num_workers and num_workers > 1 and len(infos) > 1:
        ctx = _mp_context()
        logger.info(
            "LengthEstimator: scanning %r (%d frames, %d eps) with %d workers ...",
            source.name, total, source.num_episodes, num_workers,
        )
        with ctx.Pool(
            num_workers, initializer=_worker_init, initargs=(cfg_dict, overheads)
        ) as own_pool:
            arr, reports = _scan_infos(infos, name=source.name, total=total,
                                       cfg_dict=cfg_dict, overheads=overheads, pool=own_pool)
    else:
        arr, reports = _scan_infos(infos, name=source.name, total=total,
                                   cfg_dict=cfg_dict, overheads=overheads, pool=None)

    # Persist the per-frame drop report and warn loudly if a large fraction was
    # flagged (usually storage was unhealthy during the scan, not bad data).
    n_failed = int((arr < 0).sum())
    if n_failed:
        rpath = _report_path(source, cfg, cache_dir)
        try:
            _write_report(rpath, reports)
        except Exception as exc:  # a report-write failure must not lose the cache
            logger.warning("LengthEstimator: could not write report %s (%s)", rpath, exc)
        ratio = n_failed / max(total, 1)
        logger.log(
            logging.WARNING if ratio > _DROP_WARN_RATIO else logging.INFO,
            "LengthEstimator: source %r flagged %d/%d frames (%.2f%%) unestimable "
            "(will be DROPPED); details -> %s",
            source.name, n_failed, total, 100.0 * ratio, rpath,
        )
        if ratio > _DROP_WARN_RATIO:
            logger.warning(
                "LengthEstimator: source %r drop ratio %.2f%% exceeds %.2f%% -- this "
                "usually means storage was unhealthy during the scan, not bad data. "
                "Consider deleting the cache + report (%s) and rescanning.",
                source.name, 100.0 * ratio, 100.0 * _DROP_WARN_RATIO, rpath,
            )

    _atomic_save(path, arr)
    return arr


def estimate_lengths_for_sources(
    sources: List[Any],
    cfg: LengthEstimatorConfig,
    cache_dir: str,
    *,
    num_workers: int = 8,
    rank: int = 0,
    world_size: int = 1,
) -> np.ndarray:
    """Concatenate per-source length arrays in source order (global tagged-index).

    ``out[tagged_idx]`` is the estimated token length of the frame the sampler /
    dataset resolves at ``tagged_idx`` (same source ordering as
    ``X2RobotSampler._offsets``).

    One worker pool is shared across every uncached source (so the per-source
    pool-startup cost is paid once, not once per source), and only built at all
    if some source actually needs scanning on this rank.
    """
    needs_scan = [
        src for src in sources
        if not os.path.isfile(_cache_path(src, cfg, cache_dir))
        and getattr(src, "source_type", "action") == "multimodal"
        and not (world_size > 1 and rank != 0)
    ]

    pool = None
    overheads = None
    try:
        if needs_scan and num_workers and num_workers > 1:
            overheads = LengthEstimator(cfg).overheads()
            ctx = _mp_context()
            logger.info(
                "LengthEstimator: starting shared pool (%d workers) for %d "
                "uncached source(s).", num_workers, len(needs_scan),
            )
            pool = ctx.Pool(
                num_workers, initializer=_worker_init,
                initargs=(asdict(cfg), overheads),
            )
        arrays = [
            estimate_source_lengths(
                src, cfg, cache_dir,
                num_workers=num_workers, rank=rank, world_size=world_size,
                pool=pool, overheads=overheads,
            )
            for src in sources
        ]
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if not arrays:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(arrays)


__all__ = [
    "LengthEstimator",
    "LengthEstimatorConfig",
    "read_image_size",
    "estimate_source_lengths",
    "estimate_lengths_for_sources",
]
