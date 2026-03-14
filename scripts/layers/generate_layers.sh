#!/usr/bin/env bash
# Run generate_layers.py over the base+instruct models, on either GPU layout.
# Layers use the flat parallelaave/multivalue datasets (SAE vs each dialect), so
# there is one sweep, not phases: every model runs the same
# dataset x dialect cross product.
#
# Two scheduling modes, because the scarce resource differs by machine:
#   --mode multi   — 8x24GB L4 server (default). The large models do not fit on
#                    one card, so jobs are packed by GPU COUNT via
#                    helpers/gpu_pool.sh: 1-4B → 2-3 GPUs, 8B → 4, 12B → 8.
#   --mode single  — one big A100/B200 card. Every model fits, so jobs are packed
#                    by MEMORY via helpers/gpu_mem_pool.sh and the small models
#                    backfill behind the 12Bs. Peak is bf16 weights + the
#                    all-layer hidden states extract_hidden_states holds for the
#                    batch (which scales with n_layers x hidden_dim, not just
#                    params).
#
# BATCH_SIZE stays at the config's 8 on purpose. Measured on an A100 with
# Qwen3-1.7B over coqa_aave: 4 -> 59.9 tex/s (5.7 GiB), 8 -> 60.0 tex/s
# (8.4 GiB), 16 -> 57.9 tex/s (14.3 GiB), 32 -> 53.9 tex/s (25.1 GiB). These
# texts are short and the pass is bound by writing every layer's hidden state,
# not by compute, so throughput plateaus at 8 and larger batches only add
# padding waste and memory.
#
# Usage:
#   bash scripts/layers/generate_layers.sh [--mode multi|single] [--dry-run]
#
# Env: BATCH_SIZE (default 8; raising it measured slower, see above),
#      single mode also reads GPU_INDEX / GPU_MAX_CONCURRENT /
#      GPU_MEM_RESERVE_MIB (see helpers/gpu_mem_pool.sh).

set -uo pipefail
cleanup() { trap - INT TERM; kill 0 2>/dev/null; }
trap cleanup INT TERM

MODE=multi
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ "$MODE" == multi || "$MODE" == single ]] || { echo "unknown mode: $MODE (multi|single)" >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO/scripts/layers/generate_layers.py"

BATCH_SIZE="${BATCH_SIZE:-8}"
DATASETS="parallelaave,multivalue"
DIALECTS="aave,appalachian,chicano,indian,singapore"

if [[ "$MODE" == single ]]; then
    source "$REPO/scripts/helpers/gpu_mem_pool.sh"
else
    source "$REPO/scripts/helpers/gpu_pool.sh"
fi

########
# JOBS #
########

# "n_gpus|peak_mib|model". Each mode reads the resource field it schedules on;
# the order is descending in both, so the largest job claims capacity first
# either way. peak_mib is anchored on a measured 8.4 GiB for qwen_1.7b at
# BATCH_SIZE=8, with the activation share scaled by n_layers x hidden_dim.
MODELS=(
    "8|42000|gemma_12b_base"
    "8|42000|gemma_12b_instruct"
    "4|30000|qwen_8b_base"
    "4|30000|qwen_8b_instruct"
    "4|28500|llama_8b_base"
    "4|28500|llama_8b_instruct"
    "3|17000|gemma_4b_base"
    "3|17000|gemma_4b_instruct"
    "2|16400|qwen_4b_base"
    "2|16400|qwen_4b_instruct"
    "2|14300|llama_3b_base"
    "2|14300|llama_3b_instruct"
    "2|8600|qwen_1.7b_base"
    "2|8600|qwen_1.7b_instruct"
    "2|5500|llama_1b_base"
    "2|5500|llama_1b_instruct"
    "2|4700|gemma_1b_base"
    "2|4700|gemma_1b_instruct"
)

JOBS=()
for spec in "${MODELS[@]}"; do
    n_gpus="${spec%%|*}"; rest="${spec#*|}"
    peak_mib="${rest%%|*}"; model="${rest#*|}"
    if [[ "$MODE" == single ]]; then JOBS+=("$peak_mib|$model"); else JOBS+=("$n_gpus|$model"); fi
done

# $1 is whatever the mode's scheduler allocates: comma-joined GPUs (multi) or the
# job's peak MiB (single).
run_job() {  # $1=gpus|peak_mib  $2=model
    local res="$1" model="$2" device
    local -a envp=()
    if [[ "$MODE" == single ]]; then
        # gpu_mem_pool.sh already pinned CUDA_VISIBLE_DEVICES to GPU_INDEX.
        device=$(gpu_mem_device "$res")
        echo "[$(date '+%H:%M:%S')] $model device=$device bsz=$BATCH_SIZE"
    else
        device=auto
        envp=(env "CUDA_VISIBLE_DEVICES=$res")
        echo "[$(date '+%H:%M:%S')] $model on GPUs $res (device=$device) bsz=$BATCH_SIZE"
    fi
    $DRY_RUN && return 0
    "${envp[@]}" python "$SCRIPT" --multirun \
        model="$model" device="$device" batch_size="$BATCH_SIZE" \
        dataset="$DATASETS" dialect="$DIALECTS"
}

echo "=== generate_layers (mode=$MODE, ${#JOBS[@]} jobs, bsz=$BATCH_SIZE) ==="
if [[ "$MODE" == single ]]; then
    gpu_mem_pool_run run_job || exit 1
else
    gpu_pool_run run_job
fi
echo "[$(date '+%H:%M:%S')] === All layer computations complete (mode=$MODE) ==="
