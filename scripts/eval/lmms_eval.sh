#!/usr/bin/env bash
# Evaluate a trained Qwen3.5-VL SFT checkpoint with lmms-eval.
#
# Isolated env: this uses the cloned `lmms-eval` conda env (cloned from the
# training wx at /mnt/data/x2robot_v2/.../envs/wx), so the training env is
# never touched. The clone has transformers 5.2.0 / torch 2.6.0 / accelerate
# 1.14.0 + lmms_eval installed.
#
# IMPORTANT: use attn_implementation=sdpa, NOT flash_attention_2.
#   flash-attn 2.7.4 crashes on this arch (head_dim=256, hybrid linear/full
#   attention) with "CUDA error: an illegal instruction was encountered".
#
# Usage:
#   bash scripts/eval/lmms_eval.sh [TASKS] [GPUS] [LIMIT]
#     TASKS  comma-separated lmms-eval tasks   (default: mmstar)
#     GPUS   CUDA_VISIBLE_DEVICES              (default: 1)   e.g. "1" or "1,2,3"
#     LIMIT  cap samples per task for testing  (default: empty = full set)
#
# Examples:
#   bash scripts/eval/lmms_eval.sh mmstar 1 8            # quick smoke, 8 samples
#   bash scripts/eval/lmms_eval.sh mmstar,mmmu_val,ai2d 1   # full, single GPU
#   bash scripts/eval/lmms_eval.sh mmstar 1,2,3          # full, 3-GPU data-parallel
set -euo pipefail

PY=/x2robot_v2/cyril/miniforge3/envs/lmms-eval/bin/python
CKPT="${CKPT:-/mnt/data/x2robot_v2/cyril/Penguin-VL/work_dirs/qwen3_5_vl_sft/sft9b}"
TASKS="${1:-mmstar}"
GPUS="${2:-1}"
LIMIT="${3:-}"
OUT="${OUT:-/mnt/data/x2robot_v2/cyril/Penguin-VL/eval_results}"

# NOTE (pluggable vision, e.g. --vision_backbone dinov3): a swapped-encoder
# checkpoint has custom keys in config.json (vision_backbone/vision_encoder_config)
# and tower weights the stock qwen3_5 backend cannot rebuild. To eval it, the
# lmms_eval qwen3_5 backend must load via
#   from qwenvl.model.vision import load_pluggable_qwen35
#   self._model = load_pluggable_qwen35(pretrained, attn_implementation="sdpa")
# instead of AutoModelForImageTextToText.from_pretrained. (qwenvl must be importable
# in the lmms-eval env, or copy loader.py + qwenvl/model/vision into it.) Stock
# "qwen" checkpoints load unchanged.
#
# Match the checkpoint's processor_config.json so eval resolution == training.
# (the wrapper's own defaults would otherwise override the saved processor.)
MAX_PIXELS="${MAX_PIXELS:-589824}"   # 768*768, from sft9b/processor_config.json
MIN_PIXELS="${MIN_PIXELS:-1024}"
MODEL_ARGS="pretrained=${CKPT},attn_implementation=sdpa,max_pixels=${MAX_PIXELS},min_pixels=${MIN_PIXELS}"

EXTRA=()
[ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")

export HF_HUB_ENABLE_HF_TRANSFER=1
export CUDA_VISIBLE_DEVICES="$GPUS"

# CRITICAL on PAI-DLC: clear pre-set distributed env vars. lmms_eval reads
# world_size from $WORLD_SIZE (evaluator.py:864), so a plain single-process run
# otherwise thinks it's N-rank -> gather()=scalar -> `max(int)` crash; the vars
# also poison accelerate rendezvous -> multi-GPU hang.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
      GROUP_RANK ROLE_RANK ROLE_NAME NODE_RANK NPROC_PER_NODE GROUP_WORLD_SIZE \
      TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS 2>/dev/null || true

# count GPUs -> data-parallel via accelerate when >1
NGPU=$(awk -F, '{print NF}' <<< "$GPUS")

COMMON=(
  --model qwen3_5
  --model_args "$MODEL_ARGS"
  --tasks "$TASKS"
  --batch_size 1
  --log_samples
  --output_path "$OUT"
)

if [ "$NGPU" -gt 1 ]; then
  echo ">> data-parallel on $NGPU GPUs ($GPUS)"
  "$PY" -m accelerate.commands.launch --num_processes "$NGPU" \
    -m lmms_eval eval "${COMMON[@]}" "${EXTRA[@]}"
else
  echo ">> single GPU ($GPUS)"
  "$PY" -m lmms_eval eval "${COMMON[@]}" "${EXTRA[@]}"
fi
