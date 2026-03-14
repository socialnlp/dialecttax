"""Pre-download HuggingFace model snapshots so parallel GPU jobs can run offline.

Each model is fetched sequentially, so the many ``from_pretrained()`` calls the
GPU lanes make never storm the HF Hub API concurrently and trip the 429 rate
limit. After prewarming, the caller sets ``HF_HUB_OFFLINE=1`` and every job reads
purely from the local cache (zero Hub requests). Already-cached models are
verified instantly; uncached ones download once here.

Usage:
    python scripts/prewarm_hf.py google/embeddinggemma-300m
    python scripts/prewarm_hf.py --model-config-dir configs/generate_logits/model
    python scripts/prewarm_hf.py --reward-config-dir configs/benchmark_rewards/reward_model
"""

import argparse
import concurrent.futures
import fnmatch
import glob
import logging
import os

import yaml
from huggingface_hub import snapshot_download

log = logging.getLogger(__name__)

# Skip weight duplicates transformers never loads: Llama's original/*.pth
# consolidated checkpoints plus GGUF/ONNX/Flax/TF variants.
IGNORE_PATTERNS = ["original/*", "*.pth", "*.gguf", "*.onnx", "*.msgpack", "*.h5"]


############
# COLLECT  #
############

def _keep(name: str, include: list[str], exclude: list[str]) -> bool:
    """Whether a config `name` passes the include/exclude glob filters.

    Args:
        name: The config's `name` field.
        include: If non-empty, `name` must match one of these globs.
        exclude: `name` must not match any of these globs.

    Returns:
        True if the model should be prewarmed.
    """
    if include and not any(fnmatch.fnmatch(name, p) for p in include):
        return False
    if exclude and any(fnmatch.fnmatch(name, p) for p in exclude):
        return False
    return True


def _model_ids_from_configs(config_dirs: list[str], include: list[str], exclude: list[str]) -> list[str]:
    """Read `model_id` from every YAML whose `name` passes the filters.

    Args:
        config_dirs: Directories of Hydra model configs (each has a `model_id`).
        include: Name globs to keep (empty = keep all).
        exclude: Name globs to drop.

    Returns:
        List of HF repo ids, in sorted-per-dir order.
    """
    ids = []
    for d in config_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            if cfg and "model_id" in cfg and _keep(cfg.get("name", ""), include, exclude):
                ids.append(cfg["model_id"])
    return ids


def _model_ids_from_reward_configs(config_dirs: list[str], include: list[str], exclude: list[str]) -> list[str]:
    """Resolve each reward config's `name` (passing the filters) via REWARD_MODELS.

    Args:
        config_dirs: Directories of reward-model configs (each has a `name`).
        include: Name globs to keep (empty = keep all).
        exclude: Name globs to drop.

    Returns:
        List of HF repo ids (falls back to the raw name if unmapped).
    """
    if not config_dirs:
        return []
    from dialecttax.rewards import REWARD_MODELS  # lazy: avoids importing torch for LM lanes

    ids = []
    for d in config_dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.yaml"))):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            name = cfg.get("name") if cfg else None
            if name is not None and _keep(name, include, exclude):
                ids.append(REWARD_MODELS.get(name, name))
    return ids


########
# MAIN #
########

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model_ids", nargs="*", help="Explicit HF repo ids to prewarm.")
    parser.add_argument(
        "--model-config-dir", action="append", default=[],
        help="Dir of *.yaml configs each with a 'model_id' field (repeatable).",
    )
    parser.add_argument(
        "--reward-config-dir", action="append", default=[],
        help="Dir of reward *.yaml configs whose 'name' resolves via REWARD_MODELS (repeatable).",
    )
    parser.add_argument(
        "--include-name", action="append", default=[], metavar="GLOB",
        help="Only prewarm configs whose 'name' matches one of these globs (repeatable).",
    )
    parser.add_argument(
        "--exclude-name", action="append", default=[], metavar="GLOB",
        help="Skip configs whose 'name' matches one of these globs (repeatable).",
    )
    parser.add_argument(
        "--jobs", "-j", type=int, default=8,
        help="Models to download concurrently (default 8). Downloads are network-bound, not "
             "GPU-bound; each model also fetches its own files in parallel.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-file HTTP lines

    ids = list(args.model_ids)
    ids += _model_ids_from_configs(args.model_config_dir, args.include_name, args.exclude_name)
    ids += _model_ids_from_reward_configs(args.reward_config_dir, args.include_name, args.exclude_name)

    # Dedupe, preserving first-seen order.
    seen = set()
    ids = [x for x in ids if not (x in seen or seen.add(x))]
    if not ids:
        parser.error("no model ids to prewarm (pass ids or --model-config-dir/--reward-config-dir)")

    jobs = max(1, min(args.jobs, len(ids)))
    log.info(f"Prewarming {len(ids)} model(s), {jobs} at a time, into the HF cache:")
    if jobs > 1:
        # Per-file progress bars from concurrent downloads interleave unreadably.
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()

    failed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(snapshot_download, mid, ignore_patterns=IGNORE_PATTERNS): mid for mid in ids}
        for n, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            mid = futures[fut]
            try:
                fut.result()
                log.info(f"  [{n}/{len(ids)}] ok: {mid}")
            except Exception as e:
                log.error(f"  [{n}/{len(ids)}] FAILED: {mid}: {e}")
                failed.append(mid)

    if failed:
        raise SystemExit(f"prewarm failed for {len(failed)} model(s): {failed}")
    log.info("Prewarm complete.")


if __name__ == "__main__":
    main()
