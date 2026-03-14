#!/usr/bin/env bash
# GPU lane 5: layers (all 18 base+instruct models).
#
# This script's own job is the model cache: prewarm sequentially, then run the
# parallel jobs offline so the many from_pretrained() calls don't storm the HF
# Hub API (429). Cached models verify instantly; uncached ones download once
# here. The sweep itself is scripts/layers/generate_layers.sh, which every flag
# below is passed through to:
#
#   --mode multi   — 8x24GB L4 server (default), packed by GPU count.
#   --mode single  — one big A100/B200 card, packed by memory.
#
# Usage:
#   conda activate dialecttax
#   bash scripts/5_gpu_layers.sh
#   bash scripts/5_gpu_layers.sh --mode single
#   bash scripts/5_gpu_layers.sh --dry-run

set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DRY_RUN=false
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=true; done

if ! $DRY_RUN; then
    python "$REPO/scripts/helpers/prewarm_hf.py" --model-config-dir "$REPO/configs/generate_layers/model" \
        --exclude-name '*_70b*' --exclude-name '*_27b*' --exclude-name '*_32b*' \
        || { echo "ERROR: model prewarm failed; aborting before offline run." >&2; exit 1; }
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

bash "$REPO/scripts/layers/generate_layers.sh" "$@"

echo "[$(date '+%H:%M:%S')] === Layers lane complete ==="
