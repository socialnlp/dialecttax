#!/usr/bin/env bash
# Reward models: extract hidden states, then benchmark. One entry point for every
# reward-model group and both GPU layouts.
#
# Phases (--phase, comma-separated, default "hidden,benchmark"):
#   hidden    — generate_rewards_hidden_states.py. ReDial only: the separability
#               analysis needs sample-level rows with task/dialect labels, which
#               only ReDial provides.
#   benchmark — benchmark_rewards.py over all three datasets and every dialect.
#               REQUIRES generate_words output first (run scripts/1_cpu_tokens.sh).
#
# Model groups (--group, default "all"):
#   small       — <=8B (skywork llama_3b/qwen_4b/llama_8b/qwen_8b, qrm_llama_8b,
#                 ai2_llama_8b, ai2_llama_8b_base)
#   gemma_qwen  — the Gemma-27B RMs (skywork_gemma_27b, qrm_gemma_27b). There is
#                 no Qwen-32B reward model, so this pair is the whole tier.
#   llama70b    — ai2_llama_70b
#   all         — every group above
#
# Two scheduling modes:
#   --mode multi   — 8x24GB L4 server (default), packed by GPU COUNT via
#                    helpers/gpu_pool.sh: <=8B → 1 GPU, 27B → 4, 70B → 8.
#   --mode single  — one big A100/B200 card, packed by MEMORY via
#                    helpers/gpu_mem_pool.sh. peak_mib is ~1.3x bf16 weights, this
#                    being a forward-only sequence classifier.
#
# ai2_llama_70b is ~131 GiB in bf16: it fits a 180GB B200 natively but not an
# 80GB A100. Unlike the character lanes there is no 256-step decode here -- reward
# scoring is a single forward pass -- so a CPU-offloaded run is slow but not
# absurd (weights stream over PCIe once per forward, not once per generated
# token). Single mode still refuses by default; set FORCE_OFFLOAD=1 to accept it.
#
# Reward scoring exposes no batch_size in either config, so there is no batch knob.
#
# Usage:
#   bash scripts/rewards/generate_rewards.sh [--mode multi|single]
#       [--group small|gemma_qwen|llama70b|all] [--phase hidden,benchmark] [--dry-run]
#
# Env: FORCE_OFFLOAD=1 (allow a CPU-offloaded 70B run in single mode),
#      GPU_INDEX / GPU_MAX_CONCURRENT / GPU_MEM_RESERVE_MIB (single mode).

set -uo pipefail
cleanup() { trap - INT TERM; kill 0 2>/dev/null; }
trap cleanup INT TERM

ALL_GROUPS="llama70b gemma_qwen small"
ALL_PHASES="hidden benchmark"

MODE=multi
GROUP=all
PHASE_ARG="$ALL_PHASES"
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --phase) PHASE_ARG="${2//,/ }"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done
[[ "$MODE" == multi || "$MODE" == single ]] || { echo "unknown mode: $MODE (multi|single)" >&2; exit 1; }
[[ "$GROUP" == all || " $ALL_GROUPS " == *" $GROUP "* ]] \
    || { echo "unknown group: $GROUP (small|gemma_qwen|llama70b|all)" >&2; exit 1; }

PHASES=()
for phase in $PHASE_ARG; do
    [[ " $ALL_PHASES " == *" $phase "* ]] || { echo "unknown phase: $phase" >&2; exit 1; }
    PHASES+=("$phase")
done

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HIDDEN_SCRIPT="$REPO/scripts/rewards/generate_rewards_hidden_states.py"
BENCH_SCRIPT="$REPO/scripts/rewards/benchmark_rewards.py"
TASKS="math,algorithm,logic,planning"

if [[ "$MODE" == single ]]; then
    source "$REPO/scripts/helpers/gpu_mem_pool.sh"
else
    source "$REPO/scripts/helpers/gpu_pool.sh"
fi

########
# JOBS #
########

# "n_gpus|peak_mib|reward_model|multi_device". Each mode reads the resource field
# it schedules on; ordered largest-first so freed capacity backfills with smaller
# jobs either way.
LLAMA70B_MODELS=(
    "8|138000|ai2_llama_70b|auto"
)
GEMMA_QWEN_MODELS=(
    "4|72000|skywork_gemma_27b|auto"
    "4|72000|qrm_gemma_27b|auto"
)
SMALL_MODELS=(
    "1|21000|skywork_llama_8b|cuda:0"
    "1|21000|skywork_qwen_8b|cuda:0"
    "1|21000|qrm_llama_8b|cuda:0"
    "1|21000|ai2_llama_8b|cuda:0"
    "1|21000|ai2_llama_8b_base|cuda:0"
    "1|11000|skywork_qwen_4b|cuda:0"
    "1| 9000|skywork_llama_3b|cuda:0"
)

SPECS=()
[[ "$GROUP" == all || "$GROUP" == llama70b ]] && SPECS+=("${LLAMA70B_MODELS[@]}")
[[ "$GROUP" == all || "$GROUP" == gemma_qwen ]] && SPECS+=("${GEMMA_QWEN_MODELS[@]}")
[[ "$GROUP" == all || "$GROUP" == small ]] && SPECS+=("${SMALL_MODELS[@]}")

JOBS=()
for spec in "${SPECS[@]}"; do
    n_gpus="${spec%%|*}"; rest="${spec#*|}"
    peak_mib="${rest%%|*}"; rest="${rest#*|}"
    model="${rest%%|*}"; multi_device="${rest#*|}"
    peak_mib="${peak_mib// /}"
    if [[ "$MODE" == single ]]; then JOBS+=("$peak_mib|$model|$multi_device"); else JOBS+=("$n_gpus|$model|$multi_device"); fi
done

# Echo the device this job should load onto, or return 1 to bail. In single mode a
# job larger than the budget would silently fall back to CPU offload; for the 70B
# that is a deliberate decision, not a default.
_device_for() {  # $1=res  $2=model  $3=peak_mib
    local device
    if [[ "$MODE" != single ]]; then echo "$1"; return 0; fi
    device=$(gpu_mem_device "$3")
    if [[ "$device" == auto && "${FORCE_OFFLOAD:-0}" != 1 ]]; then
        echo "ERROR: $2 needs ~${3} MiB but the budget is ${GPU_MEM_BUDGET_MIB} MiB." >&2
        echo "       Run on a B200, or set FORCE_OFFLOAD=1 to stream weights from host RAM." >&2
        return 1
    fi
    echo "$device"
}

# PHASE is read from the enclosing loop; both phases reuse the same JOBS array.
# $1 is whatever the mode's scheduler allocates: comma-joined GPUs (multi) or the
# job's peak MiB (single).
run_job() {  # $1=gpus|peak_mib  $2="model|multi_device"
    local res="$1" model dev_field device
    local -a envp=()
    IFS='|' read -r model dev_field <<< "$2"
    device=$(_device_for "$dev_field" "$model" "$res") || return 1
    if [[ "$MODE" == single ]]; then
        echo "[$(date '+%H:%M:%S')] $PHASE $model device=$device (~${res} MiB)"
    else
        envp=(env "CUDA_VISIBLE_DEVICES=$res")
        echo "[$(date '+%H:%M:%S')] $PHASE $model on GPUs $res (device=$device)"
    fi
    $DRY_RUN && return 0

    if [[ "$PHASE" == hidden ]]; then
        "${envp[@]}" python "$HIDDEN_SCRIPT" --multirun \
            reward_model="$model" device="$device" \
            dataset=redial task="$TASKS" dialect=sae,aave
    else
        "${envp[@]}" python "$BENCH_SCRIPT" --multirun \
            reward_model="$model" device="$device" \
            dataset=redial,parallelaave,multivalue task="$TASKS" \
            dialect=sae,aave,appalachian,chicano,indian,singapore
    fi
}

##########
# DRIVER #
##########

for phase in "${PHASES[@]}"; do
    PHASE="$phase"
    if [[ "$PHASE" == hidden ]]; then
        echo "=== rewards: hidden states (group=$GROUP, mode=$MODE, redial, ${#JOBS[@]} jobs) ==="
    else
        echo "=== rewards: benchmark (group=$GROUP, mode=$MODE, ${#JOBS[@]} jobs; REQUIRES 1_cpu_tokens.sh / generate_words done) ==="
    fi
    if [[ "$MODE" == single ]]; then
        gpu_mem_pool_run run_job || exit 1
    else
        gpu_pool_run run_job
    fi
done

echo "[$(date '+%H:%M:%S')] === Rewards lane complete ($GROUP: ${PHASES[*]}) ==="
