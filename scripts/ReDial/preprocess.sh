#!/usr/bin/env bash
#
# Stage 1 - Preprocess ReDial into the `preprocessed` data directory.
#
# Runs, in order:
#   1) preprocess_redial.py     -> preprocessed/ReDial/{task}_{sae,aave}.jsonl
#   2) preprocess_redial_qa.py  -> preprocessed/ReDial/{task}_{sae,aave}_qa.jsonl
#
# ReDial and most source datasets are pulled live from HuggingFace, but two raw
# inputs have no downloader and must already be present under {datasets}/:
#   - mbpp/sanitized-mbpp.json
#   - logicbench/           (LogicBench(Aug)/ and LogicBench(Eval)/{BQA,MCQA}/)
# Both preprocessing steps also call OpenRouter, so the API key file must exist.
#
# Usage:
#   scripts/ReDial/preprocess.sh              # config=default
#   scripts/ReDial/preprocess.sh --rewrite    # overwrite existing outputs
#   scripts/ReDial/preprocess.sh --config external
#
set -euo pipefail

#########
# SETUP #
#########

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CONFIG="default"
REWRITE=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    --rewrite) REWRITE=(--rewrite); shift ;;
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

###########
# PREFLIGHT #
###########

# Confirm the package is importable (conda env active + `pip install -e .` done).
if ! python -c "import dialecttax" 2>/dev/null; then
  echo "ERROR: cannot import dialecttax. Activate the env first: conda activate dialecttax" >&2
  exit 1
fi

# Resolve config-driven paths (load_config's own print is routed to stderr).
eval "$(python - "$CONFIG" <<'PY'
import contextlib, shlex, sys
import dialecttax.utils as u
with contextlib.redirect_stdout(sys.stderr):
    cfg = u.load_config(sys.argv[1])
d, k = cfg["directories"], cfg["keys"]
print(f'DATASETS={shlex.quote(d["datasets"])}')
print(f'PREPROCESSED={shlex.quote(d["preprocessed"])}')
print(f'OPENROUTER_KEY={shlex.quote(k["openrouter"])}')
PY
)"

missing=0
check() {  # check <path> <human description>
  if [[ ! -e "$1" ]]; then
    echo "MISSING: $2" >&2
    echo "         expected at: $1" >&2
    missing=1
  fi
}
check_either() {  # check_either <human description> <path> [<path>...]
  local desc="$1"; shift
  for p in "$@"; do [[ -e "$p" ]] && return 0; done
  echo "MISSING: $desc" >&2
  printf '         looked in: %s\n' "$@" >&2
  missing=1
}

echo "Config:        $CONFIG"
echo "Datasets dir:  $DATASETS"
echo "Output dir:    $PREPROCESSED/ReDial"
echo

# Non-regenerable prerequisites (no script produces these).
check "$OPENROUTER_KEY"                            "OpenRouter API key (needed by both steps)"
check "$DATASETS/mbpp/sanitized-mbpp.json"         "mbpp/sanitized-mbpp.json (algorithm task)"
# logicbench: accept either the top-level layout or the upstream data/-nested one;
# preprocess_redial.py normalizes the nesting (via symlinks) at runtime.
check_either "logicbench/LogicBench(Aug) (logic task)" \
  "$DATASETS/logicbench/LogicBench(Aug)" "$DATASETS/logicbench/data/LogicBench(Aug)"
check_either "logicbench/LogicBench(Eval) (logic task)" \
  "$DATASETS/logicbench/LogicBench(Eval)" "$DATASETS/logicbench/data/LogicBench(Eval)"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN is not set; gated HuggingFace downloads may fail." >&2
fi

if [[ "$missing" -ne 0 ]]; then
  echo >&2
  echo "Aborting: place the missing prerequisites above, then re-run." >&2
  exit 1
fi

#######
# RUN #
#######

echo ">>> [1/2] preprocess_redial.py"
python scripts/ReDial/preprocess_redial.py --config "$CONFIG" "${REWRITE[@]}"

echo ">>> [2/2] preprocess_redial_qa.py"
python scripts/ReDial/preprocess_redial_qa.py --config "$CONFIG" "${REWRITE[@]}"

echo
echo "Done. ReDial JSONL written to: $PREPROCESSED/ReDial"
