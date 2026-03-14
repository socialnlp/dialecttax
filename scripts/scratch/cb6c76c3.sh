#!/usr/bin/env bash
# Hold all 8 GPUs with light Qwen3-8B forward passes (one model per GPU).
#
# Usage:
#   bash scripts/scratch/idle_gpus_6_7.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python "$SCRIPT_DIR/52d73e1e.py" \
    --gpus 0 1 2 3 4 5 6 7 \
    --model_ids Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-8B Qwen/Qwen3-8B
