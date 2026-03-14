#!/usr/bin/env bash
# GPU lane 4: gradients (base sweep + perturbations).
#
# This script's own job is the model cache: prewarm sequentially, then run the
# parallel jobs offline so the many from_pretrained() calls don't storm the HF
# Hub API (429). Cached models verify instantly; uncached ones download once
# here. The sweep itself is scripts/gradients/generate_gradients.sh, which every
# flag below is passed through to:
#
#   --mode multi   — 8x24GB L4 server (default), packed by GPU count.
#   --mode single  — one big A100/B200 card, packed by memory.
#   --phase base,perturbations
#
# Usage:
#   conda activate dialecttax
#   bash scripts/4_gpu_gradients.sh
#   bash scripts/4_gpu_gradients.sh --mode single
#   bash scripts/4_gpu_gradients.sh --phase perturbations --dry-run

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=false
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=true; done

if ! $DRY_RUN; then
    python "$REPO/scripts/helpers/prewarm_hf.py" --model-config-dir "$REPO/configs/generate_gradients/model" \
        --exclude-name '*_70b*' --exclude-name '*_27b*' --exclude-name '*_32b*' \
        || { echo "ERROR: model prewarm failed; aborting before offline run." >&2; exit 1; }
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash "$REPO/scripts/gradients/generate_gradients.sh" "$@"

echo "[$(date '+%H:%M:%S')] === Gradients lane complete ==="
