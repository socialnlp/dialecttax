#!/usr/bin/env bash
# Run generate_gradients.py over the base language models, on either GPU layout.
# Both phases sweep the same 9 base models on redial; they differ only in what
# they hold fixed:
#   base           — dialect=sae,aave, the dialect comparison itself.
#   perturbations  — dialect=sae with surface-form perturbations (swap, drop,
#                    insert, capitalize, capitalize_alternating) as a control
#                    baseline, written under .../sae/perturbed/{name}.
#
# Two scheduling modes, because the scarce resource differs by machine:
#   --mode multi   — 8x24GB L4 server (default). No model fits on one card, so
#                    jobs are packed by GPU COUNT via helpers/gpu_pool.sh.
#                    Backprop needs more memory than a forward pass, so sizes
#                    get more GPUs than the logits lane: 1-4B → 2-3, 8B → 4,
#                    12B → 8.
#   --mode single  — one big A100/B200 card. Every base model fits, so jobs are
#                    packed by MEMORY via helpers/gpu_mem_pool.sh and several run
#                    concurrently. Peak is ~2.8x bf16 weights (params +
#                    gradients, same dtype, + backprop activations); gemma_12b is
#                    the only job needing most of an 80GB card.
#
# Usage:
#   bash scripts/gradients/generate_gradients.sh [--mode multi|single]
#       [--phase base,perturbations] [--dry-run]
#
# Env: PERTURBATIONS (the perturbation sweep),
#      single mode also reads GPU_INDEX / GPU_MAX_CONCURRENT /
#      GPU_MEM_RESERVE_MIB (see helpers/gpu_mem_pool.sh).

set -uo pipefail
cleanup() { trap - INT TERM; kill 0 2>/dev/null; }
trap cleanup INT TERM

ALL_PHASES="base perturbations"

MODE=multi
PHASE_ARG="$ALL_PHASES"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --phase) PHASE_ARG="${2//,/ }"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ "$MODE" == multi || "$MODE" == single ]] || { echo "unknown mode: $MODE (multi|single)" >&2; exit 1; }

PHASES=()
for phase in $PHASE_ARG; do
    [[ " $ALL_PHASES " == *" $phase "* ]] || { echo "unknown phase: $phase" >&2; exit 1; }
    PHASES+=("$phase")
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO/scripts/gradients/generate_gradients.py"
PERTURBATIONS="${PERTURBATIONS:-swap,drop,insert,capitalize,capitalize_alternating}"

if [[ "$MODE" == single ]]; then
    # Serialize by default. 60% of a gradient sample is CountSketch scatter_add,
    # whose atomics already saturate the card: measured on an A100, aggregate
    # sketch throughput FALLS from 9.32 to 8.33 Gelem/s going from 1 to 4
    # concurrent processes. Packing this lane costs ~11% and buys nothing.
    # Override to explore.
    GPU_MAX_CONCURRENT="${GPU_MAX_CONCURRENT:-1}"
    source "$REPO/scripts/helpers/gpu_mem_pool.sh"
else
    source "$REPO/scripts/helpers/gpu_pool.sh"
fi

########
# JOBS #
########

# "n_gpus|peak_mib|model" — gradients cover the base models only. Each mode reads
# the resource field it schedules on; the order is descending in both, so the
# largest job claims capacity first either way.
MODELS=(
    "8|68000|gemma_12b_base"
    "4|47000|qwen_8b_base"
    "4|46000|llama_8b_base"
    "3|24700|gemma_4b_base"
    "2|23000|qwen_4b_base"
    "2|18400|llama_3b_base"
    "2|9800|qwen_1.7b_base"
    "2|7200|llama_1b_base"
    "2|5800|gemma_1b_base"
)

JOBS=()
for spec in "${MODELS[@]}"; do
    n_gpus="${spec%%|*}"; rest="${spec#*|}"
    peak_mib="${rest%%|*}"; model="${rest#*|}"
    if [[ "$MODE" == single ]]; then JOBS+=("$peak_mib|$model"); else JOBS+=("$n_gpus|$model"); fi
done

# PHASE is read from the enclosing loop; each phase reuses the same JOBS array.
# $1 is whatever the mode's scheduler allocates: comma-joined GPUs (multi) or the
# job's peak MiB (single).
run_job() {  # $1=gpus|peak_mib  $2=model
    local res="$1" model="$2" device
    local -a envp=()
    if [[ "$MODE" == single ]]; then
        # gpu_mem_pool.sh already pinned CUDA_VISIBLE_DEVICES to GPU_INDEX.
        device=$(gpu_mem_device "$res")
        echo "[$(date '+%H:%M:%S')] $model ($PHASE) device=$device"
    else
        device=auto
        envp=(env "CUDA_VISIBLE_DEVICES=$res")
        echo "[$(date '+%H:%M:%S')] $model ($PHASE) on GPUs $res (device=$device)"
    fi
    $DRY_RUN && return 0

    if [[ "$PHASE" == base ]]; then
        "${envp[@]}" python "$SCRIPT" --multirun \
            model="$model" device="$device" \
            dataset=redial task=math,algorithm,logic,planning dialect=sae,aave
    else
        "${envp[@]}" python "$SCRIPT" --multirun \
            model="$model" device="$device" \
            dataset=redial task=math,algorithm,logic,planning dialect=sae \
            perturbation="$PERTURBATIONS" \
            'hydra.sweep.subdir=${model.name}/${dataset.name}/${task.name}/sae/perturbed/${perturbation.name}'
    fi
}

##########
# DRIVER #
##########

for phase in "${PHASES[@]}"; do
    PHASE="$phase"
    if [[ "$PHASE" == base ]]; then
        echo "=== gradients: base sweep (llama/gemma/qwen base, redial, mode=$MODE, ${#JOBS[@]} jobs) ==="
    else
        echo "=== gradients: perturbations (dialect=sae, $PERTURBATIONS, mode=$MODE, ${#JOBS[@]} jobs) ==="
    fi
    if [[ "$MODE" == single ]]; then
        gpu_mem_pool_run run_job || exit 1
    else
        gpu_pool_run run_job
    fi
done

echo "[$(date '+%H:%M:%S')] === All gradient computations complete (${PHASES[*]}) ==="
