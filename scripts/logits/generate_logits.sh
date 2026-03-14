#!/usr/bin/env bash
# Logits sweeps for the language models, scheduled onto whatever GPUs are free.
# Single entry point for all four lanes: base+instruct of a size share one job
# (Hydra --multirun) and each job packs onto the free-GPU pool instead of pinning
# fixed indices per family. Lanes run one after another, each with its own pass
# over the pool.
#
# Lanes (--lane, comma-separated, default "all" = every lane in the order below):
#   dialects            — generate_logits.py, full per-token logits/hidden across
#                         the dialect arms: redial (task x reasoning x sae,aave),
#                         parallelaave (sae,aave), multivalue (6 dialects).
#   mca                 — generate_logits_mca.py, forced-choice answer-probability
#                         over the same dialect arms. ReDial only (MCA has no
#                         other dataset).
#   transformations     — generate_logits.py on the full transform set: 6
#                         structural perturbations (swap/capitalize/
#                         capitalize_alternating/drop/drop_05/insert) + 6
#                         translations (french/chinese/hindi/polish/khmer/
#                         yoruba), on all three datasets (preprocessed text
#                         exists for each). Two sweeps back-to-back per job
#                         because the datasets differ in shape: redial nests by
#                         task x reasoning; multivalue/parallelaave are flat and
#                         forward-only. Attention: sdpa — no eager, since these
#                         Gemmas are Gemma 3, which has no attention-logit
#                         softcapping, so the Gemma-2 workaround doesn't apply.
#   mca_transformations — generate_logits_mca.py on the same transform set, so
#                         the outputs land in generate_logits_mca/{model}/redial/
#                         {task}/{reasoning}/sae/perturbed/{name}. ReDial only.
#
# Transformations are SAE-only: both python scripts skip any perturbation when
# dialect != "sae", so those two lanes run dialect=sae exclusively.
#
# Model selection — either a group or an explicit list:
#   --group small       — <=12B (llama 1b/3b/8b, gemma 1b/4b/12b, qwen 1.7b/4b/8b)
#   --group llama70b    — Llama-70B base+instruct (8 GPUs)
#   --group gemma_qwen  — Gemma-27B + Qwen-32B (4 GPUs each)
#   --group all         — every group above (default)
#   --models a,b,...    — size stems (gemma_4b -> base+instruct, one job) and/or
#                         full names (gemma_4b_instruct -> that model alone).
#                         Overrides --group; GPU sizing still comes from the
#                         size table below.
#
# GPU sizing: <=8B → 1 GPU; 12B → 2; 27B/32B → 4; 70B → 8.
#
# --dataset restricts the sweep (default all three). The MCA lanes are ReDial-only
# and are skipped when redial is filtered out.
#
# ReDial reasoning arms (--reasoning, default "naive,cot"). The arms differ
# hugely in cost: naive is one scored forward pass per sample; cot generates up
# to ~1k tokens per sample first. mca_transformations is the largest cross
# product (transforms x tasks x reasoning), so when --group / --reasoning are not
# given explicitly that lane defaults to small / naive instead.
#
# --batch-size overrides the config's batch_size for every job.
#
# Usage:
#   bash scripts/logits/generate_logits.sh [--lane dialects,mca,transformations,mca_transformations|all]
#       [--group small|llama70b|gemma_qwen|all] [--models stem|name,...]
#       [--dataset redial,parallelaave,multivalue] [--reasoning naive|cot|naive,cot]
#       [--batch-size N] [--dry-run]
#
#   # the unperturbed cot baseline gap-fill, for named models on ReDial
#   bash scripts/logits/generate_logits.sh --lane dialects,mca \
#       --models qwen_8b,qwen_4b --dataset redial --reasoning cot --batch-size 32

set -uo pipefail
cleanup() { trap - INT TERM; kill 0 2>/dev/null; }
trap cleanup INT TERM

# Fragmentation is the difference between fitting and an OOM at high occupancy.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

ALL_LANES="dialects mca transformations mca_transformations"
ALL_DATASETS="redial parallelaave multivalue"

LANE_ARG=all
GROUP=""
MODELS_ARG=""
DATASET_ARG=""
REASONING=""
BATCH_SIZE=""
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lane) LANE_ARG="$2"; shift 2 ;;
        --group) GROUP="$2"; shift 2 ;;
        --models) MODELS_ARG="$2"; shift 2 ;;
        --dataset) DATASET_ARG="$2"; shift 2 ;;
        --reasoning) REASONING="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

LANES=()
if [[ "$LANE_ARG" == all ]]; then
    LANES=($ALL_LANES)
else
    for lane in ${LANE_ARG//,/ }; do
        [[ " $ALL_LANES " == *" $lane "* ]] || { echo "unknown lane: $lane" >&2; exit 1; }
        LANES+=("$lane")
    done
fi

DATASETS=($ALL_DATASETS)
if [[ -n "$DATASET_ARG" ]]; then
    DATASETS=()
    for ds in ${DATASET_ARG//,/ }; do
        [[ " $ALL_DATASETS " == *" $ds "* ]] || { echo "unknown dataset: $ds" >&2; exit 1; }
        DATASETS+=("$ds")
    done
fi
has_dataset() { [[ " ${DATASETS[*]} " == *" $1 "* ]]; }

# Empty unless --batch-size is given, so the config default stands otherwise.
BATCH_ARG=()
[[ -n "$BATCH_SIZE" ]] && BATCH_ARG=("batch_size=$BATCH_SIZE")

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/helpers/gpu_pool.sh"
SCRIPT="$REPO/scripts/logits/generate_logits.py"
SCRIPT_MCA="$REPO/scripts/logits/generate_logits_mca.py"

TASKS="math,algorithm,logic,planning"
PERTURBATIONS="swap,capitalize,capitalize_alternating,drop,drop_05,insert,translate_french,translate_chinese,translate_hindi,translate_polish,translate_khmer,translate_yoruba"

#########
# JOBS  #
#########

# Size stem -> "n_gpus|device|models". Base+instruct of a size share one job;
# qwen_32b is instruct-only (there is no base checkpoint).
declare -A SIZE_JOB=(
    [llama_70b]="8|auto|llama_70b_base,llama_70b_instruct"
    [gemma_27b]="4|auto|gemma_27b_base,gemma_27b_instruct"
    [qwen_32b]="4|auto|qwen_32b_instruct"
    [gemma_12b]="2|auto|gemma_12b_base,gemma_12b_instruct"
    [llama_8b]="1|cuda:0|llama_8b_base,llama_8b_instruct"
    [qwen_8b]="1|cuda:0|qwen_8b_base,qwen_8b_instruct"
    [llama_3b]="1|cuda:0|llama_3b_base,llama_3b_instruct"
    [qwen_4b]="1|cuda:0|qwen_4b_base,qwen_4b_instruct"
    [gemma_4b]="1|cuda:0|gemma_4b_base,gemma_4b_instruct"
    [llama_1b]="1|cuda:0|llama_1b_base,llama_1b_instruct"
    [qwen_1.7b]="1|cuda:0|qwen_1.7b_base,qwen_1.7b_instruct"
    [gemma_1b]="1|cuda:0|gemma_1b_base,gemma_1b_instruct"
)
GROUP_LLAMA70B=(llama_70b)
GROUP_GEMMA_QWEN=(gemma_27b qwen_32b)
GROUP_SMALL=(gemma_12b llama_8b qwen_8b llama_3b qwen_4b gemma_4b llama_1b qwen_1.7b gemma_1b)

# Turn size stems / model names into JOBS entries ("n_gpus|models|device"). A bare
# stem takes the size's whole base+instruct pair; an explicit *_base / *_instruct
# is scheduled on its own but keeps that size's GPU count.
build_jobs() {  # $@=stems and/or model names
    JOBS=()
    local item stem spec n device models
    for item in "$@"; do
        case "$item" in
            *_base) stem="${item%_base}" ;;
            *_instruct) stem="${item%_instruct}" ;;
            *) stem="$item" ;;
        esac
        spec="${SIZE_JOB[$stem]:-}"
        [[ -n "$spec" ]] || { echo "unknown model or size stem: $item" >&2; exit 1; }
        IFS='|' read -r n device models <<< "$spec"
        [[ "$item" == "$stem" ]] || models="$item"
        JOBS+=("$n|$models|$device")
    done
}

# The MCA transform lane is the most expensive cross product; without an explicit
# flag it stays on the small models and the naive arm (the 18-model set the MCA
# dialect baselines exist for).
lane_group() {
    if [[ -n "$GROUP" ]]; then echo "$GROUP"
    elif [[ "$1" == mca_transformations ]]; then echo small
    else echo all; fi
}
lane_reasoning() {
    if [[ -n "$REASONING" ]]; then echo "$REASONING"
    elif [[ "$1" == mca_transformations ]]; then echo naive
    else echo naive,cot; fi
}

# Fill JOBS for a lane: an explicit --models list wins over the group tables.
build_lane_jobs() {  # $1=group
    if [[ -n "$MODELS_ARG" ]]; then
        build_jobs ${MODELS_ARG//,/ }
        return
    fi
    local stems=()
    [[ "$1" == all || "$1" == llama70b ]] && stems+=("${GROUP_LLAMA70B[@]}")
    [[ "$1" == all || "$1" == gemma_qwen ]] && stems+=("${GROUP_GEMMA_QWEN[@]}")
    [[ "$1" == all || "$1" == small ]] && stems+=("${GROUP_SMALL[@]}")
    if (( ${#stems[@]} == 0 )); then
        echo "unknown group: $1 (small|llama70b|gemma_qwen|all)" >&2; exit 1
    fi
    build_jobs "${stems[@]}"
}

#########
# LANES #
#########

run_dialects() {  # $1=gpus  $2="models|device|dataset"
    local gpus="$1" models device dataset
    IFS='|' read -r models device dataset <<< "$2"
    echo "[$(date '+%H:%M:%S')] $models on GPUs $gpus (device=$device) [$dataset]"
    $DRY_RUN && return 0
    case "$dataset" in
        redial)
            CUDA_VISIBLE_DEVICES="$gpus" python "$SCRIPT" --multirun \
                model="$models" device="$device" "${BATCH_ARG[@]}" \
                dataset=redial task="$TASKS" reasoning="$LANE_REASONING" dialect=sae,aave ;;
        parallelaave)
            CUDA_VISIBLE_DEVICES="$gpus" python "$SCRIPT" --multirun \
                model="$models" device="$device" "${BATCH_ARG[@]}" \
                dataset=parallelaave dialect=sae,aave ;;
        multivalue)
            CUDA_VISIBLE_DEVICES="$gpus" python "$SCRIPT" --multirun \
                model="$models" device="$device" "${BATCH_ARG[@]}" \
                dataset=multivalue dialect=sae,aave,appalachian,chicano,indian,singapore ;;
    esac
}

run_mca() {  # $1=gpus  $2="models|device"
    local gpus="$1" models device
    IFS='|' read -r models device <<< "$2"
    echo "[$(date '+%H:%M:%S')] MCA $models on GPUs $gpus (device=$device) [redial]"
    $DRY_RUN && return 0
    CUDA_VISIBLE_DEVICES="$gpus" python "$SCRIPT_MCA" --multirun \
        model="$models" device="$device" "${BATCH_ARG[@]}" \
        dataset=redial task="$TASKS" reasoning="$LANE_REASONING" dialect=sae,aave
}

run_transformations() {  # $1=gpus  $2="models|device"
    local gpus="$1" models device
    IFS='|' read -r models device <<< "$2"
    echo "[$(date '+%H:%M:%S')] $models on GPUs $gpus (device=$device) [${DATASETS[*]}, perturbations + translations]"
    # ReDial: full set across task × reasoning.
    if has_dataset redial; then
        if $DRY_RUN; then
            echo "  [dry-run] redial:              model=$models attn=sdpa dataset=redial task=$TASKS reasoning=$LANE_REASONING dialect=sae perturbation=$PERTURBATIONS"
        else
            CUDA_VISIBLE_DEVICES="$gpus" python "$SCRIPT" --multirun \
                model="$models" device="$device" attn_implementation=sdpa "${BATCH_ARG[@]}" \
                dataset=redial task="$TASKS" reasoning="$LANE_REASONING" \
                dialect=sae perturbation="$PERTURBATIONS"
        fi
    fi
    # MultiVALUE + ParallelAAVE: full set, forward-only; flat (task/reasoning pinned).
    local flat=()
    has_dataset multivalue && flat+=(multivalue)
    has_dataset parallelaave && flat+=(parallelaave)
    if (( ${#flat[@]} )); then
        local flat_csv; flat_csv="$(IFS=,; echo "${flat[*]}")"
        if $DRY_RUN; then
            echo "  [dry-run] multivalue/parallel: model=$models attn=sdpa dataset=$flat_csv dialect=sae perturbation=$PERTURBATIONS"
        else
            CUDA_VISIBLE_DEVICES="$gpus" python "$SCRIPT" --multirun \
                model="$models" device="$device" attn_implementation=sdpa "${BATCH_ARG[@]}" \
                dataset="$flat_csv" task=math reasoning=naive \
                dialect=sae perturbation="$PERTURBATIONS"
        fi
    fi
}

run_mca_transformations() {  # $1=gpus  $2="models|device"
    local gpus="$1" models device
    IFS='|' read -r models device <<< "$2"
    echo "[$(date '+%H:%M:%S')] MCA $models on GPUs $gpus (device=$device) [redial, perturbations + translations, reasoning=$LANE_REASONING]"
    if $DRY_RUN; then
        echo "  [dry-run] model=$models dataset=redial task=$TASKS reasoning=$LANE_REASONING dialect=sae perturbation=$PERTURBATIONS"
        return 0
    fi
    CUDA_VISIBLE_DEVICES="$gpus" python "$SCRIPT_MCA" --multirun \
        model="$models" device="$device" "${BATCH_ARG[@]}" \
        dataset=redial task="$TASKS" reasoning="$LANE_REASONING" \
        dialect=sae perturbation="$PERTURBATIONS"
}

##########
# DRIVER #
##########

for lane in "${LANES[@]}"; do
    group="$(lane_group "$lane")"
    LANE_REASONING="$(lane_reasoning "$lane")"

    # The MCA lanes have no dataset but ReDial.
    if [[ "$lane" == mca || "$lane" == mca_transformations ]] && ! has_dataset redial; then
        echo "skipping $lane: ReDial-only, and redial is not in --dataset (${DATASETS[*]})" >&2
        continue
    fi

    build_lane_jobs "$group"

    # The dialect lane is the only one whose datasets are separate jobs; the
    # others sweep their datasets inside a single job.
    if [[ "$lane" == dialects ]]; then
        expanded=()
        for job in "${JOBS[@]}"; do
            for dataset in "${DATASETS[@]}"; do expanded+=("$job|$dataset"); done
        done
        JOBS=("${expanded[@]}")
    fi

    models_desc="${MODELS_ARG:-group=$group}"
    echo "=== $lane ($models_desc, reasoning=$LANE_REASONING, free-GPU pool, ${#JOBS[@]} jobs) ==="
    gpu_pool_run "run_$lane"
    echo "[$(date '+%H:%M:%S')] === $lane complete ($models_desc) ==="
done

echo "[$(date '+%H:%M:%S')] === All logits lanes complete (${LANES[*]}) ==="
