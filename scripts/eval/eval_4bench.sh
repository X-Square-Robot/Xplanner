#!/usr/bin/env bash
# Evaluate sft9b on 4 embodied/spatial benchmarks: unieqa, openeqa, erqa, vsibench.
#
# Env: isolated clone `lmms-eval` (training wx untouched). MUST use sdpa (FA2 crashes).
# Data: redirected in the task yamls to /mnt/cpfs/zbl-cpfs-new/open_data/.../EVALSET/.
#
# Judge: unieqa + openeqa score with an LLM-as-judge -> you MUST export an
#        OpenAI-compatible endpoint below. erqa + vsibench are rule-based (no API).
#
# Usage:
#   # one GPU, all four (judge tasks need the API vars set):
#   bash scripts/eval/eval_4bench.sh                 # GPU 0, all 4
#   GPUS=1 bash scripts/eval/eval_4bench.sh          # pick GPU
#   TASKS="erqa,vsibench" bash scripts/eval/eval_4bench.sh   # subset (no API needed)
#   LIMIT=8 bash scripts/eval/eval_4bench.sh erqa    # quick smoke of one task
set -uo pipefail

PY=/x2robot_v2/cyril/miniforge3/envs/lmms-eval/bin/python
CKPT="${CKPT:-/mnt/data/x2robot_v2/cyril/Penguin-VL/work_dirs/qwen3_5_vl_sft/sft9b_new}"
OUT="${OUT:-/mnt/data/x2robot_v2/cyril/Penguin-VL/eval_results}"
GPUS="${GPUS:-0}"
LIMIT="${LIMIT:-}"
# per-GPU generation batch size. >1 is fine (wrapper left-pads + length-groups);
# safe & accuracy-neutral for image tasks. Keep small (1-2) for video tasks
# (vsibench/cosmos) — 32-frame clips blow up per-sample vision tokens -> OOM.
BS="${BS:-1}"
# tasks: arg1 overrides TASKS env overrides default-all
TASKS="${1:-${TASKS:-unieqa,openeqa,erqa,vsibench}}"

# ---- LLM judge endpoint (REQUIRED for unieqa & openeqa) -----------------------
# Fill these in (or export before calling). Ignored by erqa/vsibench.
export API_TYPE="${API_TYPE:-openai}"
export MODEL_VERSION="${MODEL_VERSION:-gpt-4o-mini}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-PUT-YOUR-KEY}"
export OPENAI_API_URL="${OPENAI_API_URL:-https://api.openai.com/v1/chat/completions}"
# ------------------------------------------------------------------------------

export HF_HUB_ENABLE_HF_TRANSFER=1
# The DLC eval node has NO HuggingFace network access. Force offline so any stray
# HF lookup fails fast instead of hanging on retries (and killing a DP rank ->
# cascade). All tasks here are local / load_from_disk (embspatial was localized).
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="$GPUS"

# CRITICAL on PAI-DLC: the job shell pre-sets WORLD_SIZE/RANK/MASTER_ADDR/... .
# lmms_eval reads world_size straight from $WORLD_SIZE (evaluator.py:864), so a
# plain single-process run thinks it's N-rank -> gather() returns a scalar ->
# `max(int)` TypeError; and these vars also poison accelerate's rendezvous ->
# multi-GPU NCCL hang. Clear them; the accelerate-launch path sets fresh ones.
unset WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT \
      GROUP_RANK ROLE_RANK ROLE_NAME NODE_RANK NPROC_PER_NODE GROUP_WORLD_SIZE \
      TORCHELASTIC_RUN_ID TORCHELASTIC_RESTART_COUNT TORCHELASTIC_MAX_RESTARTS 2>/dev/null || true

NGPU=$(awk -F, '{print NF}' <<< "$GPUS")

# image resolution matches sft9b/processor_config.json; max_num_frames caps video.
MODEL_ARGS="pretrained=${CKPT},attn_implementation=sdpa"
MODEL_ARGS="${MODEL_ARGS},max_pixels=589824,min_pixels=1024"
MODEL_ARGS="${MODEL_ARGS},max_num_frames=${MAX_NUM_FRAMES:-32}"
# Thinking OFF for eval. sft9b_new was trained with per-turn <think>, so the wrapper's
# default enable_thinking=True makes it emit a long reasoning chain on EVERY question
# (max_new_tokens tokens of <think>...</think>) -> ~15-20s/sample instead of <1s.
# Eval doesn't want reasoning, so force it off. Set THINK=True to re-enable if ever needed.
MODEL_ARGS="${MODEL_ARGS},enable_thinking=${THINK:-False}"

EXTRA=()
[ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")

run () {
  local tasks="$1"
  echo ">>> tasks=$tasks  gpus=$GPUS  ckpt=$CKPT"
  local base=(--model qwen3_5 --model_args "$MODEL_ARGS" --tasks "$tasks"
              --batch_size "$BS" --log_samples --output_path "$OUT" "${EXTRA[@]}")
  if [ "$NGPU" -gt 1 ]; then
    # Single-node DP eval via MANUAL env:// spawn -- deliberately NOT torchrun /
    # accelerate launch: their torch-elastic dynamic rendezvous times out on this
    # DLC container (RendezvousTimeoutError / "waiting for multi-machine"). Here
    # rank0 just starts a static TCPStore on 127.0.0.1:$port and the ranks join
    # via env:// -- no elastic agent, no hostname/NIC guessing. Each rank picks
    # cuda:LOCAL_RANK within CUDA_VISIBLE_DEVICES. NCCL only does a tiny end gather
    # so disabling IB/P2P (common hang fix) costs ~nothing.
    export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    # surface NCCL/CUDA failures with their real (synchronous) stack instead of a
    # secondary "failed to recv" on another rank:
    export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
    local port="${MASTER_PORT_OVERRIDE:-$((20000 + RANDOM % 20000))}"
    local logdir="$OUT/dp_logs"; mkdir -p "$logdir"
    echo ">>> per-rank logs -> $logdir/rank<N>.log   (live: tail -f $logdir/rank0.log)"
    local pids=() r fail=0
    for ((r=0; r<NGPU; r++)); do
      RANK=$r LOCAL_RANK=$r WORLD_SIZE=$NGPU LOCAL_WORLD_SIZE=$NGPU \
      MASTER_ADDR=127.0.0.1 MASTER_PORT=$port \
        "$PY" -m lmms_eval eval "${base[@]}" > "$logdir/rank${r}.log" 2>&1 &
      pids+=($!)
    done
    for ((r=0; r<NGPU; r++)); do
      if ! wait "${pids[$r]}"; then
        fail=1
        echo "!! rank $r FAILED -- primary error from $logdir/rank${r}.log:" >&2
        grep -aE "Error|error|OutOfMemory|out of memory|Traceback|Exception|CUDA" "$logdir/rank${r}.log" | grep -avE "recv, got 0 bytes|ncclUniqueId|broadcastUniqueNCCLID" | tail -8 >&2
      fi
    done
    return $fail
  else
    "$PY" -m lmms_eval eval "${base[@]}"
  fi
}

# Warn if judge tasks requested without a real key.
if [[ ",$TASKS," == *",unieqa,"* || ",$TASKS," == *",openeqa,"* ]] && [ "$OPENAI_API_KEY" = "PUT-YOUR-KEY" ]; then
  echo "!! unieqa/openeqa need a judge: export OPENAI_API_KEY / OPENAI_API_URL / MODEL_VERSION first." >&2
fi

run "$TASKS"
