#!/usr/bin/env bash
# CPU lane 0: put the raw and preprocessed datasets in place. Everything
# downstream (1_cpu_tokens.sh and the GPU lanes) reads what this stage produces.
#
# Phases (--phase, comma-separated, default "install,redial"):
#   install    — copy the raw inputs the repo vendors under data/datasets/ into
#                {datasets}/: parallelaave, mbpp, logicbench, multivalue. These
#                have no downloader anywhere -- they ship with the repo, which is
#                exactly the gap ReDial/preprocess.sh's header calls out for
#                mbpp/sanitized-mbpp.json and logicbench/. Existing files are left
#                alone unless --rewrite.
#   parallelaave — re-derive {datasets}/parallelaave/ from its original source
#                rather than from the vendored copy: the optional supplementary
#                material of Groenwold et al., "Investigating African-American
#                Vernacular English in Transformer-Based Text Generation"
#                (EMNLP 2020), https://aclanthology.org/2020.emnlp-main.473/
#                The archive holds EMNLP-AAVE-files/{sae,aave}_samples.txt (plus
#                __MACOSX/.DS_Store cruft this phase drops); those two files are
#                what data/datasets/parallelaave/ vendors, byte for byte. Not in
#                the default set, since `install` already puts them in place
#                offline -- use this to refresh from upstream.
#   redial     — scripts/ReDial/preprocess.sh: builds
#                {preprocessed}/ReDial/{task}_{dialect}[_qa].jsonl. Pulls
#                fangrulin/redial, fangrulin/asynchow, yale-nlp/FOLIO and gsm8k
#                live from HuggingFace, reads mbpp/ and logicbench/ locally, and
#                calls OpenRouter -- so the API key file must exist.
#   multivalue — regenerate {datasets}/multivalue/coqa_{dialect}.txt from CoQA.
#                NOT in the default set, and skipped when the corpus is already
#                present unless --rewrite, because Multi-VALUE's dialect rules
#                fire stochastically: a regeneration is internally consistent but
#                NOT byte-identical to the vendored corpus every published
#                multivalue result was computed on.
#
#                It also cannot run in the dialecttax env (Python 3.14): Multi-VALUE
#                needs spaCy on Python <=3.12 with torch<2.6 and
#                transformers<4.50. This phase therefore builds a THROWAWAY venv
#                at $MV_VENV with those exact pins and runs the generator there.
#                Do NOT install value-nlp into the dialecttax conda env.
#
# Usage:
#   conda activate dialecttax
#   bash scripts/0_cpu_datasets.sh                              # install + redial
#   bash scripts/0_cpu_datasets.sh --config server
#   bash scripts/0_cpu_datasets.sh --phase install
#   bash scripts/0_cpu_datasets.sh --phase parallelaave --rewrite   # refresh from upstream
#   bash scripts/0_cpu_datasets.sh --phase multivalue --rewrite  # regenerate the corpus
#   bash scripts/0_cpu_datasets.sh --dry-run
#
# Env: MV_VENV (default /tmp/mv-venv), MV_WORKERS (default 40), MV_PYTHON
#      (default python3.12)

set -uo pipefail

ALL_PHASES="install parallelaave redial multivalue"
DEFAULT_PHASES="install redial"

CONFIG=default
PHASE_ARG="$DEFAULT_PHASES"
REWRITE=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --phase) PHASE_ARG="${2//,/ }"; shift 2 ;;
        --rewrite) REWRITE=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

PHASES=()
for phase in $PHASE_ARG; do
    [[ " $ALL_PHASES " == *" $phase "* ]] || { echo "unknown phase: $phase ($ALL_PHASES)" >&2; exit 1; }
    PHASES+=("$phase")
done
has_phase() { [[ " ${PHASES[*]} " == *" $1 "* ]]; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDORED="$REPO/data/datasets"

DATASETS_DIR=$(python -c "import dialecttax.utils; print(dialecttax.utils.load_config('$CONFIG')['directories']['datasets'])" | tail -n 1)
[[ -n "$DATASETS_DIR" ]] || { echo "ERROR: could not resolve the datasets directory from configs/$CONFIG.yaml" >&2; exit 1; }
echo "[lane] config=$CONFIG  datasets=$DATASETS_DIR  phases=${PHASES[*]}"

###########
# INSTALL #
###########

if has_phase install; then
    echo "=== datasets: install vendored raw inputs -> $DATASETS_DIR ==="
    if [[ ! -d "$(dirname "$DATASETS_DIR")" ]]; then
        echo "ERROR: $(dirname "$DATASETS_DIR") does not exist. Wrong --config for this box?" >&2
        echo "       configs/$CONFIG.yaml points {datasets} at $DATASETS_DIR" >&2
        exit 1
    fi
    $DRY_RUN || mkdir -p "$DATASETS_DIR"
    for src in "$VENDORED"/*/; do
        name="$(basename "$src")"
        dest="$DATASETS_DIR/$name"
        if [[ -e "$dest" ]] && ! $REWRITE; then
            echo "  skip    $name (already present; --rewrite to overwrite)"
            continue
        fi
        echo "  install $name ($(du -sh "$src" | cut -f1))"
        $DRY_RUN && continue
        rm -rf "$dest"
        cp -r "$src" "$dest" || { echo "ERROR: failed to install $name" >&2; exit 1; }
    done
fi

################
# PARALLELAAVE #
################

if has_phase parallelaave; then
    PA_URL="https://aclanthology.org/attachments/2020.emnlp-main.473.OptionalSupplementaryMaterial.zip"
    PA_DIR="$DATASETS_DIR/parallelaave"
    echo "=== datasets: parallelaave from the EMNLP 2020 supplementary ==="
    if [[ -e "$PA_DIR/sae_samples.txt" ]] && ! $REWRITE; then
        echo "  skip: $PA_DIR already populated (--rewrite to refetch)"
    elif $DRY_RUN; then
        echo "  [dry-run] fetch $PA_URL -> $PA_DIR/{sae,aave}_samples.txt"
    else
        # unzip is not installed on every box; zipfile also lets us take exactly
        # the two members we want and skip the __MACOSX/.DS_Store entries.
        python - "$PA_URL" "$PA_DIR" <<'PY' || { echo "ERROR: parallelaave fetch failed." >&2; exit 1; }
import io
import os
import sys
import urllib.request
import zipfile

url, dest = sys.argv[1], sys.argv[2]
os.makedirs(dest, exist_ok=True)
print(f"  fetching {url}")
with urllib.request.urlopen(url, timeout=120) as r:
    blob = r.read()
z = zipfile.ZipFile(io.BytesIO(blob))
for name in ("sae_samples.txt", "aave_samples.txt"):
    member = f"EMNLP-AAVE-files/{name}"
    if member not in z.namelist():
        raise SystemExit(f"ERROR: {member} not in the archive; upstream layout changed.")
    data = z.read(member)
    out = os.path.join(dest, name)
    with open(out, "wb") as f:
        f.write(data)
    print(f"  wrote {out} ({len(data)} bytes)")
PY
    fi
fi

##########
# REDIAL #
##########

if has_phase redial; then
    echo "=== datasets: ReDial preprocessing (HuggingFace + OpenRouter) ==="
    REDIAL_ARGS=(--config "$CONFIG")
    $REWRITE && REDIAL_ARGS+=(--rewrite)
    if $DRY_RUN; then
        echo "  [dry-run] bash $REPO/scripts/ReDial/preprocess.sh ${REDIAL_ARGS[*]}"
    else
        bash "$REPO/scripts/ReDial/preprocess.sh" "${REDIAL_ARGS[@]}" \
            || { echo "ERROR: ReDial preprocessing failed." >&2; exit 1; }
    fi
fi

##############
# MULTIVALUE #
##############

if has_phase multivalue; then
    MV_VENV="${MV_VENV:-/tmp/mv-venv}"
    MV_WORKERS="${MV_WORKERS:-40}"
    MV_PYTHON="${MV_PYTHON:-python3.12}"
    MV_CORPUS="$DATASETS_DIR/multivalue/coqa_sae.txt"

    echo "=== datasets: regenerate the MultiVALUE corpus ==="
    if [[ -e "$MV_CORPUS" ]] && ! $REWRITE; then
        echo "  skip: $MV_CORPUS exists. Regeneration is stochastic and would replace"
        echo "        the vendored corpus the published results used. --rewrite to force."
    elif $DRY_RUN; then
        echo "  [dry-run] build $MV_VENV ($MV_PYTHON) then run generate_multivalue.py --workers $MV_WORKERS"
    else
        # Throwaway venv with the exact pins Multi-VALUE needs: torch<2.6 (>=2.6
        # flips torch.load weights_only=True and breaks stanza models),
        # transformers<4.50 (>=4.50 refuses torch.load unless torch>=2.6), and the
        # spaCy model from its release wheel (`spacy download` builds a bad URL).
        if [[ ! -x "$MV_VENV/bin/python" ]]; then
            command -v "$MV_PYTHON" >/dev/null 2>&1 \
                || { echo "ERROR: $MV_PYTHON not found; Multi-VALUE needs Python <=3.12." >&2; exit 1; }
            echo "  building $MV_VENV with $MV_PYTHON"
            "$MV_PYTHON" -m venv "$MV_VENV" || { echo "ERROR: venv creation failed." >&2; exit 1; }
            "$MV_VENV/bin/pip" install -q value-nlp datasets pyyaml \
                && "$MV_VENV/bin/pip" install -q "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu \
                && "$MV_VENV/bin/pip" install -q "transformers==4.46.3" \
                && "$MV_VENV/bin/pip" install -q "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl" \
                || { echo "ERROR: failed to build $MV_VENV." >&2; exit 1; }
        fi
        "$MV_VENV/bin/python" "$REPO/scripts/multivalue/generate_multivalue.py" \
            --config "$CONFIG" --workers "$MV_WORKERS" \
            || { echo "ERROR: MultiVALUE generation failed." >&2; exit 1; }
    fi
fi

echo "[$(date '+%H:%M:%S')] === Datasets lane complete (${PHASES[*]}) ==="
