#!/usr/bin/env bash
# CPU lane 1: generate_tokens + generate_words across all datasets, parallelized
# across cores. These are tokenizer-only (no model, no GPU), so they run on the
# 192-core box concurrently with the GPU lanes.
#
# Parallelism: each (script, dataset[, tokenizer]) is one Hydra --multirun job
# (which sweeps task x dialect internally). Jobs are dispatched up to MAX_JOBS at
# a time. Invalid dataset/dialect/task combos and existing outputs are skipped by
# the scripts themselves.
#
# Usage:
#   conda activate dialecttax
#   bash scripts/1_cpu_tokens.sh
#   MAX_JOBS=64 bash scripts/1_cpu_tokens.sh
#   bash scripts/1_cpu_tokens.sh --dry-run

set -uo pipefail
cleanup() { trap - INT TERM; kill 0 2>/dev/null; }
trap cleanup INT TERM

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENS="$REPO_ROOT/scripts/tokens/generate_tokens.py"
WORDS="$REPO_ROOT/scripts/tokens/generate_words.py"

DATASETS=(redial parallelaave multivalue)
TOKENIZERS=(bpe gpt2 gemma llama qwen unigram wordpiece)
MAX_JOBS="${MAX_JOBS:-32}"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

run_job() {
    echo "[$(date '+%H:%M:%S')] $*"
    $DRY_RUN && return 0
    "$@"
}

throttle() {  # block until fewer than MAX_JOBS background jobs are running
    while (( $(jobs -rp | wc -l) >= MAX_JOBS )); do wait -n || true; done
}

# AutoTokenizer.from_pretrained() hits the HF Hub API on every call (even for
# already-cached models), and words/tokens jobs reload tokenizers per
# task x dialect. Dozens of concurrent jobs blow past HF's 1000-req / 5-min
# rate limit (429). Fix: warm the cache once (sequentially, a few API calls),
# then run the parallel jobs fully offline so they read the local cache and
# make zero HF requests.
if ! $DRY_RUN; then
    echo "=== Prewarming tokenizer cache (sequential, one-time) ==="
    if ! python - <<'PY'
import tiktoken
from dialecttax.tokenizers.tokenization import (
    TOKENIZER_NAME_MAP, TOKENIZER_NAME_TO_TYPE, get_tokenizer,
)
for name, target in TOKENIZER_NAME_MAP.items():
    print(f"  caching {name} ({target})", flush=True)
    if TOKENIZER_NAME_TO_TYPE[name] == "gpt":
        tiktoken.encoding_for_model(target)  # tiktoken cache, not HF
    else:
        get_tokenizer(name)
PY
    then
        echo "ERROR: failed to prewarm tokenizer cache; aborting before offline run." >&2
        exit 1
    fi
fi

# Force the parallel jobs to read tokenizers from the local cache only.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "=== generate_tokens (dataset x tokenizer) ==="
for dataset in "${DATASETS[@]}"; do
    for tok in "${TOKENIZERS[@]}"; do
        throttle
        run_job python "$TOKENS" --multirun \
            dataset="$dataset" tokenizer="$tok" task="glob(*)" dialect="glob(*)" &
    done
done

echo "=== generate_words (dataset; iterates tokenizers internally) ==="
for dataset in "${DATASETS[@]}"; do
    throttle
    run_job python "$WORDS" --multirun \
        dataset="$dataset" task="glob(*)" dialect="glob(*)" &
done

wait
echo "[$(date '+%H:%M:%S')] === CPU lane (tokens + words) complete ==="
