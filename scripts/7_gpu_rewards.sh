#!/usr/bin/env bash
# GPU lane 7: reward models — hidden states + benchmark.
#
# This script's own job is the model cache: prewarm sequentially, then run the
# parallel jobs offline so the many from_pretrained() calls don't storm the HF
# Hub API (429). Cached models verify instantly; uncached ones download once
# here. The prewarm is scoped to the group. The sweep itself is
# scripts/rewards/generate_rewards.sh, which every flag below is passed through to:
#
#   --group small|gemma_qwen|llama70b|all   (default small)
#   --mode multi|single                     (default multi)
#   --phase hidden,benchmark
#
# NOTE: the benchmark phase requires generate_words output first
#       (run scripts/1_cpu_tokens.sh before it).
#
# Usage:
#   conda activate dialecttax
#   bash scripts/7_gpu_rewards.sh                          # small RMs, 8 GPUs
#   bash scripts/7_gpu_rewards.sh --group llama70b
#   bash scripts/7_gpu_rewards.sh --mode single --group gemma_qwen
#   bash scripts/7_gpu_rewards.sh --phase hidden --dry-run

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GROUP=small
DRY_RUN=false
PASS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --group) GROUP="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; PASS+=("$1"); shift ;;
        *) PASS+=("$1"); shift ;;
    esac
done

case "$GROUP" in
    small)      PREWARM=(--exclude-name '*_70b*' --exclude-name '*_27b*' --exclude-name '*_32b*') ;;
    gemma_qwen) PREWARM=(--include-name '*_27b*') ;;
    llama70b)   PREWARM=(--include-name '*_70b*') ;;
    all)        PREWARM=() ;;
    *) echo "unknown group: $GROUP (small|gemma_qwen|llama70b|all)" >&2; exit 1 ;;
esac

if ! $DRY_RUN; then
    python "$REPO/scripts/helpers/prewarm_hf.py" --reward-config-dir "$REPO/configs/benchmark_rewards/reward_model" \
        "${PREWARM[@]}" \
        || { echo "ERROR: model prewarm failed; aborting before offline run." >&2; exit 1; }
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# The lane defaults to the small RMs; the sweep script itself defaults to all.
bash "$REPO/scripts/rewards/generate_rewards.sh" --group "$GROUP" "${PASS[@]}"

echo "[$(date '+%H:%M:%S')] === Rewards lane complete ($GROUP) ==="
