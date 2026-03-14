#!/usr/bin/env bash
# Character-level tokenization on ReDial: generate, then verify. One entry point
# for every model group and both GPU layouts.
#
# Phases (--phase, comma-separated, default "generate,verify"):
#   generate — generate_characters.py, forced character-level tokenization.
#   verify   — verify_characters.py, which RE-PROBES generate's saved entropy, so
#              it must run after generate and is only meaningful on a real (non
#              dry) run. Small group only: there is no verify table for the
#              27B/32B/70B models, so the big groups skip it.
#
# Model groups (--group, default "small"):
#   small       — instruct models <=12B. generate covers the 3 largest per family
#                 (naive only); verify covers all 9 and sweeps naive,cot.
#   llama70b    — Llama-70B instruct.
#   gemma_qwen  — Gemma-27B + Qwen-32B instruct. (There is no Qwen character
#                 model above 32B, so this is the pair.)
#   all         — every group above.
#
# The big groups run compute_hidden=false: that keeps input + answer entropy while
# skipping the per-step hidden-state materialization that dominates long char
# sequences.
#
# Two scheduling modes:
#   --mode multi   — 8x24GB L4 server (default). Char sequences are ~5x longer
#                    than canonical, so memory is tight and models are sharded by
#                    GPU COUNT via helpers/gpu_pool.sh.
#   --mode single  — one big A100/B200 card, packed by MEMORY via
#                    helpers/gpu_mem_pool.sh.
#
# !! The llama70b and gemma_27b jobs DELETE their existing
#    {experiments}/generate_characters/{model}/redial tree before running (they
#    pair a wipe with rerun=true to regenerate from scratch). qwen_32b and the
#    whole small group do not. The default group is small, so the default path
#    wipes nothing.
#
# Usage:
#   bash scripts/characters/generate_characters.sh [--mode multi|single]
#       [--group small|llama70b|gemma_qwen|all] [--phase generate,verify] [--dry-run]
#
# Env: BATCH_SIZE (generate; default 1 in multi = the config default, 4 in
#        single), RERUN (small generate, default false),
#      VERIFY_BATCH_SIZE / VERIFY_COMPUTE_BATCH_SIZE (override the verify batches),
#      FORCE_OFFLOAD=1 (let llama70b run CPU-offloaded in single mode),
#      GPU_INDEX / GPU_MAX_CONCURRENT / GPU_MEM_RESERVE_MIB (single mode).

set -uo pipefail
cleanup() { trap - INT TERM; kill 0 2>/dev/null; }
trap cleanup INT TERM

ALL_GROUPS="small llama70b gemma_qwen"
ALL_PHASES="generate verify"

MODE=multi
GROUP=small
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

SEL_GROUPS=()
if [[ "$GROUP" == all ]]; then
    SEL_GROUPS=($ALL_GROUPS)
else
    [[ " $ALL_GROUPS " == *" $GROUP "* ]] || { echo "unknown group: $GROUP (small|llama70b|gemma_qwen|all)" >&2; exit 1; }
    SEL_GROUPS=("$GROUP")
fi

PHASES=()
for phase in $PHASE_ARG; do
    [[ " $ALL_PHASES " == *" $phase "* ]] || { echo "unknown phase: $phase" >&2; exit 1; }
    PHASES+=("$phase")
done
has_phase() { [[ " ${PHASES[*]} " == *" $1 "* ]]; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GEN_SCRIPT="$REPO/scripts/characters/generate_characters.py"
VERIFY_SCRIPT="$REPO/scripts/characters/verify_characters.py"
TASKS="math,algorithm,logic,planning"

# Char sequences are ~5x longer than canonical, so the sharded L4 lane stays at
# the config's batch_size=1; one big card takes 4.
if [[ "$MODE" == single ]]; then
    BATCH_SIZE="${BATCH_SIZE:-4}"
    # Three at a time. One model leaves an 80GB card at ~64% utilisation because
    # decode is launch-latency bound (1024 sequential steps per batch); extra
    # processes overlap into that slack. WANTS MPS -- without it CUDA contexts
    # merely time-slice, so concurrency cannot beat 1/0.64 = 1.56x. Start it once
    # per boot: nvidia-cuda-mps-control -d
    GPU_MAX_CONCURRENT="${GPU_MAX_CONCURRENT:-3}"
    # 10 GiB, not 8: peak_mib is sized on the worst batch, but decode transients
    # and three CUDA contexts still need slack. The memory model has been wrong
    # twice (once by 2-3x on the table, once by 3.9x on prompt length).
    GPU_MEM_RESERVE_MIB="${GPU_MEM_RESERVE_MIB:-10240}"
    source "$REPO/scripts/helpers/gpu_mem_pool.sh"
else
    BATCH_SIZE="${BATCH_SIZE:-1}"
    source "$REPO/scripts/helpers/gpu_pool.sh"
fi

# generate is idempotent: rerun=false makes _is_config_done() skip any config whose
# metadata.jsonl is already complete. (cfg.rerun used to be read by nothing at all,
# so an earlier rerun=true here silently did nothing; generate_characters.py now
# honors it, which is why this defaults to false rather than true.)
RERUN="${RERUN:-false}"

EXPERIMENTS_DIR=$(python -c "import dialecttax.utils; print(dialecttax.utils.load_config('default')['directories']['experiments'])" | tail -n 1)

wipe_model() {  # $1=model
    local target="$EXPERIMENTS_DIR/generate_characters/$1/redial"
    if [[ -n "$EXPERIMENTS_DIR" && -d "$target" ]]; then
        echo "[$(date '+%H:%M:%S')] Removing existing results: $target"
        $DRY_RUN || rm -rf "$target"
    fi
}

############
# GENERATE #
############

# "n_gpus|peak_mib|model|reasoning|compute_hidden|rerun|wipe".
#
# small peak_mib scales with BATCH_SIZE (intercept + slope*bsz, MiB) so the packer
# stays honest when the knob is turned: step 3 of compute_all_char_generate
# re-applies lm_head across the WHOLE prompt, so peak carries a
# (bsz, prompt_len, vocab) fp32 logit tensor. Measured on an A100 over 8 ReDial
# logic/cot rows: gemma_12b bsz1 24824 | bsz2 26293 | bsz4 29244 MiB, fitting
# 23350 + 1474*bsz exactly; qwen_1.7b bsz4 5929 | bsz8 8541.
gen_jobs() {  # $1=group -> fills GEN_SPECS
    case "$1" in
        small) GEN_SPECS=(
            "4|$(( 23350 + 1550 * BATCH_SIZE ))|gemma_12b_instruct|naive|true|$RERUN|false"
            "2|$(( 16100 + 1140 * BATCH_SIZE ))|qwen_8b_instruct|naive|true|$RERUN|false"
            "2|$(( 15800 +  990 * BATCH_SIZE ))|llama_8b_instruct|naive|true|$RERUN|false"
        ) ;;
        llama70b) GEN_SPECS=(
            "8|140000|llama_70b_instruct|naive,cot|false|true|true"
        ) ;;
        # gemma first: multi ties on n_gpus and keeps array order, so this
        # preserves the old lane's order. single sorts by peak_mib regardless.
        gemma_qwen) GEN_SPECS=(
            "4|59000|gemma_27b_instruct|naive,cot|false|true|true"
            "4|70000|qwen_32b_instruct|naive,cot|false|false|false"
        ) ;;
    esac
}

# One --multirun per model, sweeping task x reasoning x dialect internally.
run_gen() {  # $1=gpus|peak_mib  $2="model|reasoning|compute_hidden|rerun"
    local res="$1" model reasoning compute_hidden rerun device
    local -a envp=()
    IFS='|' read -r model reasoning compute_hidden rerun <<< "$2"
    if [[ "$MODE" == single ]]; then
        device=$(gpu_mem_device "$res")
        # A CPU-offloaded 70B decode is ~38 days for this sweep; refuse by default.
        if [[ "$device" == auto && "$model" == llama_70b_instruct && "${FORCE_OFFLOAD:-0}" != 1 ]]; then
            echo "ERROR: $model needs ~${res} MiB but the budget is ${GPU_MEM_BUDGET_MIB} MiB." >&2
            echo "       A CPU-offloaded decode is ~38 days for this sweep. Run on a B200," >&2
            echo "       or set FORCE_OFFLOAD=1 to proceed anyway." >&2
            return 1
        fi
        echo "[$(date '+%H:%M:%S')] generate $model device=$device bsz=$BATCH_SIZE rerun=$rerun (~${res} MiB)"
    else
        device=auto
        envp=(env "CUDA_VISIBLE_DEVICES=$res")
        echo "[$(date '+%H:%M:%S')] generate $model on GPUs $res (device=$device) bsz=$BATCH_SIZE rerun=$rerun"
    fi
    $DRY_RUN && return 0
    "${envp[@]}" python "$GEN_SCRIPT" --multirun \
        model="$model" device="$device" batch_size="$BATCH_SIZE" dataset=redial \
        task="$TASKS" reasoning="$reasoning" dialect=sae,aave \
        compute_hidden="$compute_hidden" rerun="$rerun"
}

# One process per (task, reasoning, dialect). main() reloads the model per Hydra
# config anyway, so this costs no extra load time and keeps both 4-GPU slots of
# an 8-GPU box busy across the whole sweep, with finer restart granularity.
run_gen_config() {  # $1=gpus  $2="model|task|reasoning|dialect|compute_hidden|rerun"
    local gpus="$1" model task reasoning dialect compute_hidden rerun
    IFS='|' read -r model task reasoning dialect compute_hidden rerun <<< "$2"
    echo "[$(date '+%H:%M:%S')] generate $model/$task/$reasoning/$dialect on GPUs $gpus"
    $DRY_RUN && return 0
    env "CUDA_VISIBLE_DEVICES=$gpus" python "$GEN_SCRIPT" \
        model="$model" device=auto dataset=redial \
        task="$task" reasoning="$reasoning" dialect="$dialect" \
        compute_hidden="$compute_hidden" rerun="$rerun"
}

##########
# VERIFY #
##########

# "n_gpus|peak_mib|model|multi_bsz|multi_device|probe_bsz|compute_bsz".
#
# DO NOT change a verify batch so it differs from the batch generate used: the
# probe compares saved entropies against a fresh one with
# _entropy_close(rtol=1e-3), and a batch change alters padding and can trip a
# false MISMATCH -- which DELETES the config's outputs and recomputes it in full.
# The two single-mode batches are decoupled: probe_bsz feeds the 16-sample,
# 1-token probe and MUST match what generate wrote (4) for the three models with a
# saved naive/sae input_entropy.npz; compute_bsz feeds the full 1024-step decode,
# which nothing compares. Multi's batches (1/2/4 by size) mirror its own generate
# sizing -- on a mismatch verify re-runs the full config and needs that headroom.
#
# peak_mib is evaluated for a COT config (max_tokens_new=1024) at the worst
# observed prompt length (algorithm's 3426 char-level tokens), +10% margin.
# gemma_1b is NOT cheap: it shares the 12B's 262k vocab, so its scores and logit
# tensors dwarf its 1907 MiB of weights.
VERIFY_SPECS=(
    "2|54100|gemma_12b_instruct|1|auto|4|4"
    "2|32600|qwen_8b_instruct|1|auto|4|4"
    "2|30100|llama_8b_instruct|1|auto|4|4"
    "1|32000|gemma_4b_instruct|2|cuda:0|4|4"
    "1|23400|qwen_4b_instruct|2|cuda:0|4|4"
    "1|19300|llama_3b_instruct|2|cuda:0|4|4"
    "1|22500|gemma_1b_instruct|4|cuda:0|4|4"
    "1|17700|qwen_1.7b_instruct|4|cuda:0|4|4"
    "1|13100|llama_1b_instruct|4|cuda:0|4|4"
)

run_verify() {  # $1=gpus|peak_mib  $2="model|multi_bsz|multi_device|probe_bsz|compute_bsz"
    local res="$1" model multi_bsz multi_device probe_bsz compute_bsz device
    local -a envp=() extra=()
    IFS='|' read -r model multi_bsz multi_device probe_bsz compute_bsz <<< "$2"
    if [[ "$MODE" == single ]]; then
        probe_bsz="${VERIFY_BATCH_SIZE:-$probe_bsz}"
        compute_bsz="${VERIFY_COMPUTE_BATCH_SIZE:-$compute_bsz}"
        device=$(gpu_mem_device "$res")
        extra=("compute_batch_size=$compute_bsz")
        echo "[$(date '+%H:%M:%S')] verify $model device=$device probe_bsz=$probe_bsz compute_bsz=$compute_bsz (~${res} MiB)"
    else
        probe_bsz="${VERIFY_BATCH_SIZE:-$multi_bsz}"
        device="$multi_device"
        envp=(env "CUDA_VISIBLE_DEVICES=$res")
        echo "[$(date '+%H:%M:%S')] verify $model on GPUs $res (device=$device, bsz=$probe_bsz)"
    fi
    $DRY_RUN && return 0
    "${envp[@]}" python "$VERIFY_SCRIPT" --multirun \
        model="$model" device="$device" batch_size="$probe_bsz" "${extra[@]}" \
        dataset=redial task="$TASKS" reasoning=naive,cot dialect=sae
}

##########
# DRIVER #
##########

pool_run() {  # $1=launch_fn
    if [[ "$MODE" == single ]]; then gpu_mem_pool_run "$1" || exit 1; else gpu_pool_run "$1"; fi
}

for group in "${SEL_GROUPS[@]}"; do
    if has_phase generate; then
        gen_jobs "$group"
        per_config=false
        [[ "$MODE" == multi && "$group" == gemma_qwen ]] && per_config=true

        # Wipes happen up front, before anything in this group dispatches.
        for spec in "${GEN_SPECS[@]}"; do
            IFS='|' read -r _n _mib model _reasoning _ch _rerun wipe <<< "$spec"
            [[ "$wipe" == true ]] && wipe_model "$model"
        done

        JOBS=()
        for spec in "${GEN_SPECS[@]}"; do
            IFS='|' read -r n_gpus peak_mib model reasoning compute_hidden rerun _wipe <<< "$spec"
            res="$n_gpus"; [[ "$MODE" == single ]] && res="$peak_mib"
            if $per_config; then
                for task in ${TASKS//,/ }; do
                    for r in ${reasoning//,/ }; do
                        for dialect in sae aave; do
                            JOBS+=("$res|$model|$task|$r|$dialect|$compute_hidden|$rerun")
                        done
                    done
                done
            else
                JOBS+=("$res|$model|$reasoning|$compute_hidden|$rerun")
            fi
        done

        echo "=== characters: generate ($group, mode=$MODE, ${#JOBS[@]} jobs, bsz=$BATCH_SIZE) ==="
        if $per_config; then pool_run run_gen_config; else pool_run run_gen; fi
    fi

    if has_phase verify; then
        if [[ "$group" != small ]]; then
            echo "skipping verify for $group: verify_characters covers the <27B instruct models only" >&2
            continue
        fi
        JOBS=()
        for spec in "${VERIFY_SPECS[@]}"; do
            IFS='|' read -r n_gpus peak_mib rest <<< "$spec"
            res="$n_gpus"; [[ "$MODE" == single ]] && res="$peak_mib"
            JOBS+=("$res|$rest")
        done
        echo "=== characters: verify ($group, mode=$MODE, ${#JOBS[@]} jobs) ==="
        if [[ "$MODE" == single ]]; then
            if [ -S /tmp/nvidia-mps/control ]; then
                echo "[$(date '+%H:%M:%S')] MPS: on (clients co-execute)"
            else
                echo "[$(date '+%H:%M:%S')] MPS: OFF -- concurrency caps at ~1.56x. Start it: nvidia-cuda-mps-control -d" >&2
            fi
        fi
        pool_run run_verify
    fi
done

echo "[$(date '+%H:%M:%S')] === Characters lane complete (${SEL_GROUPS[*]}: ${PHASES[*]}) ==="
