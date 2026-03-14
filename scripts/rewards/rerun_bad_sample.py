"""One-shot: re-extract a single sample whose hidden vector came back NaN."""

import gc
import json
import logging
import os
import sys

import numpy as np
import torch

import dialecttax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_rewards import load_reward_model, _build_redial_prompt  # noqa: E402
from generate_rewards_hidden_states import extract_score_hidden, _patch_gemma_attention  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


###########
# CONFIG  #
###########

RM_NAME = "ai2_llama_70b"
DATASET = "redial"
TASK = "algorithm"
DIALECT = "aave"
TARGET_UID = "redial-algorithm_vanilla_aave-11"

EXPERIMENT_DIR = "/data/gemini/ellang/dialecttax/experiments/generate_rewards_hidden_states"
NPZ_PATH = os.path.join(EXPERIMENT_DIR, RM_NAME, DATASET, TASK, "naive", DIALECT, "hidden.npz")
META_PATH = os.path.join(EXPERIMENT_DIR, RM_NAME, DATASET, TASK, "naive", DIALECT, "metadata.jsonl")


########
# MAIN #
########

def main():
    project_config = dialecttax.utils.load_config()
    mod = dialecttax.data.DATASET_MODULES[DATASET]
    dir_root = project_config["directories"]["preprocessed"]
    path_file = os.path.join(mod.DIRECTORY_NAME, mod.FILE_NAME_FORMAT.format(task=TASK, dialect=DIALECT))
    ds = mod.load_dataset(dir_root, path_file)

    idx = next(i for i, s in enumerate(ds) if s["unique_id"] == TARGET_UID)
    sample = ds[idx]
    log.info(f"Target index {idx}, uid {sample['unique_id']}")

    prompt = _build_redial_prompt(ds, idx, TASK, "naive", DIALECT)
    response = str(sample["answer"])
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    log.info(f"Loading {RM_NAME} ...")
    rm = load_reward_model(RM_NAME, device="auto")
    rm = _patch_gemma_attention(rm)

    log.info("Extracting hidden state ...")
    vec = extract_score_hidden(rm, conversation)
    n_nan = int(np.isnan(vec).sum())
    n_inf = int(np.isinf(vec).sum())
    log.info(f"vec shape={vec.shape} dtype={vec.dtype} NaN={n_nan} Inf={n_inf} "
             f"min={float(np.nanmin(vec)):.4g} max={float(np.nanmax(vec)):.4g}")

    if n_nan or n_inf:
        log.error("Re-run produced non-finite values; not patching npz.")
        return

    log.info(f"Loading existing npz: {NPZ_PATH}")
    existing = dict(np.load(NPZ_PATH))
    if TARGET_UID not in existing:
        log.error(f"{TARGET_UID} not present in existing archive; aborting.")
        return
    existing[TARGET_UID] = vec.astype(np.float32)
    np.savez_compressed(NPZ_PATH, **existing)
    log.info(f"Patched {TARGET_UID} into {NPZ_PATH}")

    # Metadata likely already has the entry; touch it only if missing.
    seen_uids = set()
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            for line in f:
                seen_uids.add(json.loads(line)["unique_id"])
    if TARGET_UID not in seen_uids:
        with open(META_PATH, "a") as f:
            f.write(json.dumps({"unique_id": TARGET_UID, "hidden_dim": int(vec.shape[0])}) + "\n")
        log.info(f"Appended metadata entry for {TARGET_UID}")

    del rm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
