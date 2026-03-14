#!/usr/bin/env bash
#
# One-time environment setup - create the config-declared data directories and
# fetch the raw datasets that have no live (HuggingFace) downloader.
#
# For every configs/{config}.yaml `directories:` entry, create the folder if it
# is missing (datasets, experiments, preprocessed, ...). Then populate
# {datasets}/ with the datasets that have no live (HuggingFace) downloader:
#   - mbpp/sanitized-mbpp.json              google-research/mbpp
#   - parallelaave/{sae,aave}_samples.txt   EMNLP 2020 paper 473 supplement
#   - logicbench/                           git clone of Mihir3009/LogicBench
#   - multivalue/coqa_{dialect}.txt         generated via Multi-VALUE (needs python3.12)
#
# The first three are plain downloads; multivalue is *generated* in a throwaway
# python3.12 venv (its pinned deps conflict with the dialecttax env) and takes
# ~15 min - skip it with --no-multivalue. See scripts/multivalue/generate_multivalue.py.
#
# Idempotent: anything already present is left untouched unless --rewrite is set.
#
# Usage:
#   scripts/helpers/setup.sh
#   scripts/helpers/setup.sh --config external
#   scripts/helpers/setup.sh --rewrite
#   scripts/helpers/setup.sh --no-multivalue   # skip the slow Multi-VALUE generation
set -euo pipefail

#########
# SETUP #
#########

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MBPP_URL="https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json"
PARALLELAAVE_URL="https://aclanthology.org/attachments/2020.emnlp-main.473.OptionalSupplementaryMaterial.zip"
LOGICBENCH_URL="https://github.com/Mihir3009/LogicBench.git"

CONFIG="default"
REWRITE=0
NO_MULTIVALUE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --config=*) CONFIG="${1#*=}"; shift ;;
    --rewrite) REWRITE=1; shift ;;
    --no-multivalue) NO_MULTIVALUE=1; shift ;;
    -h|--help) awk 'NR==1{next} /^#/{sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

#############
# PREFLIGHT #
#############

for cmd in python3 curl git; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "ERROR: '$cmd' is required but not on PATH." >&2; exit 1; }
done
python3 -c "import yaml" 2>/dev/null || { echo "ERROR: PyYAML is required (pip install pyyaml)." >&2; exit 1; }

# Resolve every directory from configs/{CONFIG}.yaml, with {hostname} substituted.
eval "$(python3 - "$REPO_ROOT" "$CONFIG" <<'PY'
import shlex, socket, sys, yaml
repo_root, name = sys.argv[1], sys.argv[2]
with open(f"{repo_root}/configs/{name}.yaml") as f:
    raw = f.read().replace("{hostname}", socket.gethostname().split(".")[0])
dirs = yaml.safe_load(raw)["directories"]
print("DATASETS=" + shlex.quote(dirs["datasets"]))
print("ALLDIRS=(" + " ".join(shlex.quote(v) for v in dirs.values()) + ")")
PY
)"

echo "Config:       $CONFIG"
echo "Datasets dir: $DATASETS"
echo

###############
# DIRECTORIES #
###############

for d in "${ALLDIRS[@]}"; do
  if [[ -d "$d" ]]; then
    echo "exists:  $d"
  else
    mkdir -p "$d"
    echo "created: $d"
  fi
done
echo

############
# DATASETS #
############

# MBPP: a single JSON file, straight download.
mbpp_json="$DATASETS/mbpp/sanitized-mbpp.json"
if [[ "$REWRITE" -eq 0 && -f "$mbpp_json" ]]; then
  echo "mbpp:         present, skipping ($mbpp_json)"
else
  echo "mbpp:         downloading sanitized-mbpp.json"
  mkdir -p "$DATASETS/mbpp"
  curl -fL --retry 3 -o "$mbpp_json" "$MBPP_URL"
fi

# ParallelAAVE: the EMNLP-2020 supplement zip holds EMNLP-AAVE-files/{sae,aave}_samples.txt.
pa_dir="$DATASETS/parallelaave"
if [[ "$REWRITE" -eq 0 && -f "$pa_dir/sae_samples.txt" && -f "$pa_dir/aave_samples.txt" ]]; then
  echo "parallelaave: present, skipping ($pa_dir)"
else
  echo "parallelaave: downloading + extracting samples"
  mkdir -p "$pa_dir"
  pa_zip="$(mktemp)"
  trap 'rm -f "$pa_zip"' EXIT
  curl -fL --retry 3 -o "$pa_zip" "$PARALLELAAVE_URL"
  # No `unzip` dependency: pull the two samples files out with stdlib zipfile.
  python3 - "$pa_zip" "$pa_dir" <<'PY'
import os, shutil, sys, zipfile
zip_path, out_dir = sys.argv[1], sys.argv[2]
wanted = {"sae_samples.txt", "aave_samples.txt"}
found = set()
with zipfile.ZipFile(zip_path) as z:
    for info in z.infolist():
        base = os.path.basename(info.filename)
        if base in wanted and "__MACOSX" not in info.filename:
            with z.open(info) as src, open(os.path.join(out_dir, base), "wb") as dst:
                shutil.copyfileobj(src, dst)
            found.add(base)
missing = wanted - found
if missing:
    raise SystemExit(f"ParallelAAVE zip missing expected files: {sorted(missing)}")
PY
  rm -f "$pa_zip"; trap - EXIT
fi

# LogicBench: cloned straight into logicbench/ (i.e. renamed from LogicBench).
lb_dir="$DATASETS/logicbench"
if [[ "$REWRITE" -eq 0 && -d "$lb_dir" ]]; then
  echo "logicbench:   present, skipping ($lb_dir)"
else
  echo "logicbench:   cloning $LOGICBENCH_URL"
  [[ "$REWRITE" -eq 1 ]] && rm -rf "$lb_dir"
  git clone --depth 1 "$LOGICBENCH_URL" "$lb_dir"
fi

# The loader (dialecttax.data.redial.load_logicbench) expects LogicBench(Aug)/(Eval)
# at the top level; upstream nests them under data/. Link them if only nested.
for name in "LogicBench(Aug)" "LogicBench(Eval)"; do
  if [[ -d "$lb_dir/data/$name" && ! -e "$lb_dir/$name" && ! -L "$lb_dir/$name" ]]; then
    ln -s "data/$name" "$lb_dir/$name"
    echo "logicbench:   linked $name -> data/$name"
  fi
done

##############
# MULTIVALUE #
##############

# MultiVALUE: the CoQA dialect corpus is generated (not downloaded) by Multi-VALUE,
# whose heavy, version-pinned deps conflict with the dialecttax env, so it runs in a
# throwaway python3.12 venv. See scripts/multivalue/generate_multivalue.py for the
# reasoning behind each pin.
mv_dir="$DATASETS/multivalue"
mv_have=1
for dialect in sae aave appalachian chicano indian singapore; do
  [[ -f "$mv_dir/coqa_$dialect.txt" ]] || mv_have=0
done
if [[ "$NO_MULTIVALUE" -eq 1 ]]; then
  echo "multivalue:   skipped (--no-multivalue)"
elif [[ "$REWRITE" -eq 0 && "$mv_have" -eq 1 ]]; then
  echo "multivalue:   present, skipping ($mv_dir)"
elif ! command -v python3.12 >/dev/null 2>&1; then
  echo "multivalue:   SKIPPED - python3.12 not found (needed for the isolated Multi-VALUE venv)." >&2
  echo "              Install python3.12 and re-run, or generate it manually per" >&2
  echo "              scripts/multivalue/generate_multivalue.py, then re-run to finish." >&2
else
  echo "multivalue:   building isolated python3.12 venv + generating corpus (~15 min)"
  mv_venv="$(mktemp -d)"
  trap 'rm -rf "$mv_venv"' EXIT
  python3.12 -m venv "$mv_venv"
  # venv activate/deactivate reference unset vars; relax `set -u` around them.
  set +u; source "$mv_venv/bin/activate"; set -u
  pip install --upgrade pip
  pip install value-nlp datasets pyyaml
  pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
  pip install "transformers==4.46.3"
  pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
  python "$REPO_ROOT/scripts/multivalue/generate_multivalue.py" --config "$CONFIG"
  set +u; deactivate; set -u
  rm -rf "$mv_venv"; trap - EXIT
fi

echo
echo "Done. Datasets ready under: $DATASETS"
