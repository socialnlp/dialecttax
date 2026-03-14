#!/usr/bin/env bash
# GPU lane 6: character-level tokenization (generate + verify).
#
# This script's own job is the model cache: prewarm sequentially, then run the
# parallel jobs offline so the many from_pretrained() calls don't storm the HF
# Hub API (429). Cached models verify instantly; uncached ones download once
# here. The prewarm is scoped to the group. The sweep itself is
# scripts/characters/generate_characters.sh, which every flag below is passed
# through to:
#
#   --group small|llama70b|gemma_qwen|all   (default small)
#   --mode multi|single                     (default multi)
#   --phase generate,verify
#
# !! --group llama70b and --group gemma_qwen DELETE the existing results tree for
#    llama_70b_instruct / gemma_27b_instruct before running. The default group is
#    small, which wipes nothing.
#
# Usage:
#   conda activate dialecttax
#   bash scripts/6_gpu_characters.sh                        # small, 8 GPUs
#   bash scripts/6_gpu_characters.sh --group llama70b
#   bash scripts/6_gpu_characters.sh --mode single --group gemma_qwen
#   bash scripts/6_gpu_characters.sh --dry-run

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GROUP=small
DRY_RUN=false
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; PASS+=("$1" "$2"); shift 2 ;;
        --dry-run) DRY_RUN=true; PASS+=("$1"); shift ;;
        *) PASS+=("$1"); shift ;;
    esac
done

case "$GROUP" in
    small)      PREWARM=(--exclude-name '*_70b*' --exclude-name '*_27b*' --exclude-name '*_32b*') ;;
    llama70b)   PREWARM=(--include-name '*_70b_instruct*') ;;
    gemma_qwen) PREWARM=(--include-name '*_27b_instruct*' --include-name '*_32b_instruct*') ;;
    all)        PREWARM=() ;;
    *) echo "unknown group: $GROUP (small|llama70b|gemma_qwen|all)" >&2; exit 1 ;;
esac

if ! $DRY_RUN; then
    python "$REPO/scripts/helpers/prewarm_hf.py" --model-config-dir "$REPO/configs/generate_characters/model" \
        "${PREWARM[@]}" \
        || { echo "ERROR: model prewarm failed; aborting before offline run." >&2; exit 1; }
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash "$REPO/scripts/characters/generate_characters.sh" "${PASS[@]}"

echo "[$(date '+%H:%M:%S')] === Characters lane complete ($GROUP) ==="
