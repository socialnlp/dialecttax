#!/usr/bin/env bash
# GPU lane 3: logits for one model group, on whatever GPUs are free.
#
# Delegates the sweep to scripts/logits/generate_logits.sh, which runs three
# lanes in order:
#   1. dialects        — full per-token logits/hidden, all 3 datasets.
#   2. mca             — forced-choice answer-prob (redial).
#   3. transformations — perturbations + translations on all 3 datasets
#                        (redial, multivalue, parallelaave), SAE.
#
# This script's own job is the model cache: prewarm sequentially, then run the
# parallel jobs offline so the many from_pretrained() calls don't storm the HF
# Hub API (429). Cached models verify instantly; uncached ones download once
# here. The prewarm is scoped to the group, so a lane downloads only what it
# runs.
#
# Model groups (--group, default "small"):
#   small       — <=12B (llama 1b/3b/8b, gemma 1b/4b/12b, qwen 1.7b/4b/8b)
#   llama70b    — Llama-70B base+instruct
#   gemma_qwen  — Gemma-27B + Qwen-32B
#   all         — every group above, in one run
#
# The big models are a separate group (rather than a separate script) so they can
# be run later on their own schedule without re-touching the small sweep.
#
# --models a,b,... names size stems / models directly instead of a group, and
# scopes the prewarm to them. Every other flag is passed through to
# generate_logits.sh, so --lane, --dataset, --reasoning and --batch-size work
# here too.
#
# Usage:
#   conda activate dialecttax
#   bash scripts/3_gpu_logits.sh                      # small models
#   bash scripts/3_gpu_logits.sh --group llama70b
#   bash scripts/3_gpu_logits.sh --group gemma_qwen --dry-run
#
#   # unperturbed cot baseline gap-fill (was gpu_2_logits_cot_baseline.sh).
#   # ReDial pinned output_subdir to naive until 2026-07-11, so every
#   # reasoning=cot config resolved onto its naive twin's directory and was
#   # skipped by the rerun=false gate; this refills exactly that hole. Strictly
#   # additive -- the gates skip anything already written.
#   bash scripts/3_gpu_logits.sh --lane dialects,mca --models qwen_8b,qwen_4b \
#       --dataset redial --reasoning cot --batch-size 32

set -uo pipefail
cleanup() { trap - INT TERM; kill 0 2>/dev/null; }
trap cleanup INT TERM

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GROUP=small
MODELS=""
DRY_RUN=false
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; shift 2 ;;
        --models) MODELS="$2"; PASS+=("$1" "$2"); shift 2 ;;
        --dry-run) DRY_RUN=true; PASS+=("$1"); shift ;;
        *) PASS+=("$1"); shift ;;
    esac
done

# Prewarm filters: an explicit --models list wins, matched as a substring glob so
# a bare size stem pulls that size's base+instruct. Otherwise scope by group
# ("all" passes no filter, i.e. every model config).
if [[ -n "$MODELS" ]]; then
    PREWARM=()
    for m in ${MODELS//,/ }; do PREWARM+=(--include-name "*$m*"); done
else
    case "$GROUP" in
        small)      PREWARM=(--exclude-name '*_70b*' --exclude-name '*_27b*' --exclude-name '*_32b*') ;;
        llama70b)   PREWARM=(--include-name '*_70b*') ;;
        gemma_qwen) PREWARM=(--include-name '*_27b*' --include-name '*_32b*') ;;
        all)        PREWARM=() ;;
        *) echo "unknown group: $GROUP (small|llama70b|gemma_qwen|all)" >&2; exit 1 ;;
    esac
fi

if ! $DRY_RUN; then
    python "$REPO/scripts/helpers/prewarm_hf.py" --model-config-dir "$REPO/configs/generate_logits/model" \
        "${PREWARM[@]}" \
        || { echo "ERROR: model prewarm failed; aborting before offline run." >&2; exit 1; }
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --group is only meaningful without --models; generate_logits.sh ignores it when
# an explicit list is given, and PASS may carry its own --lane.
echo "=== logits lanes (${MODELS:-group=$GROUP}, 8 GPUs) ==="
bash "$REPO/scripts/logits/generate_logits.sh" \
    --lane dialects,mca,transformations --group "$GROUP" "${PASS[@]}"

echo "[$(date '+%H:%M:%S')] === Logits lane complete (${MODELS:-$GROUP}) ==="
