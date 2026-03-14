#!/usr/bin/env bash
# GPU lane 2 (8x24GB L4): generate_embeddings.py across whatever GPUs are free.
#
# Paths come from configs/generate_embeddings/config.yaml's `project_config`,
# which this script does NOT override: `server` resolves
# /data/{hostname}/ellang/dialecttax/... on this box (gemini/lynx), `default` the
# layout in configs/default.yaml. Retarget the lane by editing that one key.
#
# EmbeddingGemma-300m weights are ~1.2 GB, so N_GPUS=1 per job here and the
# count-based pool in helpers/gpu_pool.sh keeps all 8 L4s busy. That 1 is a
# property of THIS model, not of the lane: the pool's "<n_gpus>|<payload>" field
# exists because a model too large for one 24 GB card must be given several (a
# 12B needs all 8). Raise N_GPUS if this lane ever points at a bigger encoder.
#
# The unperturbed baseline sweeps all dialects (for the cross-dialect analysis);
# perturbed variants are SAE-only, matching what the analysis consumes
# (sim(SAE_original, SAE_perturbed)). Each job sweeps task x dim internally.
#
# WHY BSZ=32, not the config's 256. Peak attention memory scales with the
# LONGEST sequence in a batch, and the heavily-fragmented variants (translate to
# khmer/yoruba, capitalize-random/alternating) tokenize the same passages into
# far more tokens -- the tax this project measures. At 256 those four OOM on a
# 24 GB L4 mid-encode; the allocator then holds the memory, so every later config
# in the same --multirun reports the misleading "No GPU with >= 2 GB free". The
# datasets are ~400 samples, so a smaller batch costs seconds and nothing else.
#
# RESUMABLE. generate_embeddings.py skips a config whose embeddings-{dim}.npy
# already exists (rerun=false), so re-running this script backfills only what is
# missing. That matters: on 2026-07-06 four multivalue jobs (khmer, yoruba, both
# capitalizations) died with "No GPU with >= 2 GB free" while redial and
# parallelaave saturated the pool, and the analysis silently plotted the gap.
# The VERIFY pass below now reports any variant that failed to produce output.
#
# Usage:
#   conda activate dialecttax
#   bash scripts/2_gpu_embeddings.sh [--dry-run] [dataset...]
#
#   bash scripts/2_gpu_embeddings.sh              # all three datasets
#   bash scripts/2_gpu_embeddings.sh multivalue   # backfill one dataset

set -uo pipefail
set -f  # keep Hydra 'glob(*)' literal when job specs word-split in the dispatcher
cleanup() { trap - INT TERM; kill 0 2>/dev/null; }
trap cleanup INT TERM

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO_ROOT/scripts/helpers/gpu_pool.sh"
SCRIPT="$REPO_ROOT/scripts/embeddings/generate_embeddings.py"
DIMS="768,512,256,128"
DATASETS=(redial parallelaave multivalue)
BSZ="${BSZ:-32}"
N_GPUS="${N_GPUS:-1}"   # per job; EmbeddingGemma-300m fits on a single L4

# Fragmentation is the difference between fitting and an OOM at high occupancy.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Remaining positional args restrict the sweep to those datasets.
DRY_RUN=false
ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--dry-run" ]] && { DRY_RUN=true; continue; }
    ARGS+=("$arg")
done
(( ${#ARGS[@]} )) && DATASETS=("${ARGS[@]}")

# Prewarm the model cache sequentially, then run the parallel jobs offline so the
# many from_pretrained() calls don't storm the HF Hub API (429). Cached models are
# verified instantly; uncached ones download once here.
if ! $DRY_RUN; then
    python "$REPO_ROOT/scripts/helpers/prewarm_hf.py" google/embeddinggemma-300m \
        || { echo "ERROR: model prewarm failed; aborting before offline run." >&2; exit 1; }
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# One job per (dataset, perturbation variant). Baseline sweeps all dialects; the
# perturbed variants are SAE-only (dialect=sae). Each job is 1 GPU and sweeps
# task x dim internally. Payload = the per-job Hydra args.
JOBS=()
for dataset in "${DATASETS[@]}"; do
    JOBS+=("$N_GPUS|dataset=$dataset dialect=glob(*)")
    for p in swap drop insert; do JOBS+=("$N_GPUS|dataset=$dataset perturbation=$p dialect=sae"); done
    for lang in chinese french hindi khmer polish yoruba; do
        JOBS+=("$N_GPUS|dataset=$dataset perturbation=translate language=$lang dialect=sae")
    done
    for c in random alternating; do
        JOBS+=("$N_GPUS|dataset=$dataset perturbation=capitalize capitalization_type=$c dialect=sae")
    done
done

run_job() {  # $1=comma-joined gpus  $2=hydra-arg-string
    local gpu="$1" args="$2"
    echo "[$(date '+%H:%M:%S')] GPU $gpu: $args"
    $DRY_RUN && return 0
    CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" --multirun \
        task="glob(*)" dim="$DIMS" device=cuda:0 batch_size="$BSZ" $args
}

echo "=== generate_embeddings (free-GPU pool, ${#JOBS[@]} jobs) ==="
gpu_pool_run run_job
echo "[$(date '+%H:%M:%S')] === All embeddings dispatched ==="

##########
# VERIFY #
##########

# Report SAE perturbation variants that produced no embeddings, so a partially
# failed sweep can't masquerade as a complete one in the downstream analysis.
$DRY_RUN && exit 0
python - "$REPO_ROOT" "${DATASETS[@]}" <<'PY'
import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(sys.argv[1], "src"))
import dialecttax.utils

repo_root, datasets = sys.argv[1], sys.argv[2:]
cfg_path = os.path.join(repo_root, "configs/generate_embeddings/config.yaml")
experiments = dialecttax.utils.load_config(OmegaConf.load(cfg_path).project_config)["directories"]["experiments"]
emb_dir = os.path.join(experiments, "generate_embeddings")

variants = ["swap-0.05", "drop-0.15", "insert-0.05", "capitalize-random", "capitalize-alternating"]
variants += [f"translate-{lang}" for lang in ["chinese", "french", "hindi", "khmer", "polish", "yoruba"]]
dims = [768, 512, 256, 128]


def sae_subdirs(dataset: str) -> list[str]:
    """Expand a dataset's output_subdir into its concrete SAE directories.

    multivalue/parallelaave use '${dialect.name}' (-> 'sae'); redial uses
    '${task.name}/${dialect.name}' (-> 'math/sae', ...), so the layout has to be
    read per dataset rather than assumed.

    Args:
        dataset: Dataset name, matching configs/generate_embeddings/dataset/{name}.yaml.

    Returns:
        Output subdirectories relative to {emb_dir}/{dataset}.
    """
    # resolve=False: output_subdir interpolates ${task.name}/${dialect.name}, which
    # only exist once Hydra composes the full config. We want the raw template.
    cfg = OmegaConf.to_container(
        OmegaConf.load(os.path.join(repo_root, f"configs/generate_embeddings/dataset/{dataset}.yaml")),
        resolve=False,
    )
    subdir = cfg["output_subdir"].replace("${dialect.name}", "sae")
    if "${task.name}" not in subdir:
        return [subdir]
    return [subdir.replace("${task.name}", task) for task in cfg.get("tasks", [])]


expected, missing = 0, []
for dataset in datasets:
    for subdir in sae_subdirs(dataset):
        for variant in variants:
            for dim in dims:
                rel = os.path.join(dataset, subdir, variant, f"embeddings-{dim}.npy")
                expected += 1
                if not os.path.exists(os.path.join(emb_dir, rel)):
                    missing.append(rel)
if missing:
    print(f"\nWARNING: {len(missing)}/{expected} missing SAE perturbation embeddings under {emb_dir}:")
    for path in missing:
        print(f"  {path}")
    sys.exit(1)
print(f"\nVerified: all {expected} SAE perturbation embeddings present.")
PY
