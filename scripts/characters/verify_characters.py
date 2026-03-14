"""Verify saved input_entropy.npz against a fresh probe and re-run on mismatch.

For each (model x task x reasoning x dialect) config matched by Hydra:
  1. If model is >=27B, skip — verification only targets the small/medium
     models that the verify_characters_*.sh launchers cover.
  2. If saved input_entropy.npz is missing, run the full config inline (same
     code path as generate_characters.main).
  3. Otherwise probe the first PROBE_N samples through the current code path
     and compare to the saved arrays via np.allclose. Only entropy is
     compared: input_log_probs and input_entropy are derived from the same
     prompt_logits tensor, so an entropy match implies a logprob match up to
     floating-point noise.
  4. On match: log VERIFIED and return.
  5. On mismatch: delete stale outputs and re-run inline (same process, same
     loaded model). The probe already used the current code path, so the
     loaded model is the right one to redo the full config with — no need to
     spawn a subprocess (which would fight the parent for GPU memory and OOM
     on sharded 8B/12B configs).

Usage:
    python scripts/characters/verify_characters.py \\
        model=qwen_8b_instruct task=math reasoning=naive dialect=sae dataset=redial
    python scripts/characters/verify_characters.py --multirun \\
        model=qwen_8b_instruct task=math,algorithm,logic,planning \\
        reasoning=naive,cot dialect=sae dataset=redial
"""

import logging
import os
import re
import sys

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_characters import (
    _build_samples,
    _compute_and_save,
    _load_model,
    _project_config,
    compute_all_char_generate,
)

import dialecttax
import dialecttax.data

log = logging.getLogger(__name__)


###########
# CONSTS  #
###########

PROBE_N = 16
PROBE_MAX_TOKENS = 1
RTOL = 1e-3
ATOL = 1e-4
MAX_MODEL_SIZE_B = 27.0
VERIFIED_MARKER = ".verified"
STALE_FILES = ("input_log_probs.npz", "input_entropy.npz", "hidden.npz", "metadata.jsonl")


###########
# COMPARE #
###########

def _model_size_b(model_name: str) -> float | None:
    """Parse the parameter count (in B) from a model name like 'gemma_12b_instruct'."""
    m = re.search(r"_(\d+(?:\.\d+)?)b_", model_name)
    return float(m.group(1)) if m else None


def _load_saved_entropy(npz_path: str, n: int) -> "list[np.ndarray]":
    """Load first n positional arrays (arr_0..arr_{n-1}) from input_entropy.npz."""
    with np.load(npz_path) as data:
        return [data[f"arr_{i}"] for i in range(min(n, len(data.files)))]


def _entropy_close(saved: "list[np.ndarray]", computed: "list[np.ndarray]") -> bool:
    if len(saved) != len(computed):
        return False
    for s, c in zip(saved, computed):
        if s.shape != c.shape or not np.allclose(s, c, rtol=RTOL, atol=ATOL):
            return False
    return True


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_characters", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    task = cfg.task.name
    dialect = cfg.dialect.name
    reasoning = cfg.reasoning.name

    size_b = _model_size_b(cfg.model.name)
    if size_b is not None and size_b >= MAX_MODEL_SIZE_B:
        log.info(f"Skipping (model '{cfg.model.name}' is >={MAX_MODEL_SIZE_B:.0f}B)")
        return

    valid_dialects = list(cfg.dataset.dialects)
    if dialect not in valid_dialects:
        log.info(f"Skipping (dialect '{dialect}' not in {valid_dialects})")
        return
    if "tasks" in cfg.dataset and task not in list(cfg.dataset.tasks):
        log.info(f"Skipping (task '{task}' not valid for dataset '{dataset_name}')")
        return

    out_dir = HydraConfig.get().runtime.output_dir
    if os.path.exists(os.path.join(out_dir, VERIFIED_MARKER)):
        log.info(f"Skipping (already verified): {out_dir}")
        return

    saved_path = os.path.join(out_dir, "input_entropy.npz")
    saved_exists = os.path.exists(saved_path)

    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    dir_root = _project_config["directories"][cfg.dataset.dir_key]
    fmt = getattr(mod, "FILE_NAME_QA_FORMAT", mod.FILE_NAME_FORMAT)
    path_file = fmt.format(task=task, dialect=dialect)
    path_file = os.path.join(mod.DIRECTORY_NAME, path_file)
    path = os.path.join(dir_root, path_file)
    if not os.path.exists(path):
        log.error(f"Dataset not found: {path}")
        return
    ds = mod.load_dataset(dir_root, path_file)
    family = cfg.model.name.split("_")[0]
    samples = _build_samples(ds, task, dialect, reasoning=reasoning, family=family)

    try:
        model, tokenizer = _load_model(cfg.model.name, cfg.model.model_id, device=cfg.device)
    except OSError as e:
        log.warning(f"Skipping model '{cfg.model.name}' ({cfg.model.model_id}): {e}")
        return

    if not saved_exists:
        log.info(f"No saved input_entropy.npz at {out_dir}; computing fresh")
        _compute_and_save(cfg, model, tokenizer, samples, out_dir)
        return

    instruct = cfg.model.name.endswith("_instruct")
    batch_size = int(cfg.get("batch_size", 1))

    probe = samples[:PROBE_N]
    label = f"{dataset_name}/{task}/{reasoning}/{dialect}/probe"
    log.info(f"Probing first {len(probe)} samples for {out_dir}")
    probe_results = compute_all_char_generate(
        model, tokenizer, probe,
        instruct=instruct, max_tokens_new=PROBE_MAX_TOKENS,
        batch_size=batch_size, label=label, answer_only=False,
    )

    saved = _load_saved_entropy(saved_path, len(probe))
    if _entropy_close(saved, probe_results["input_entropy"]):
        log.info(f"VERIFIED ({len(probe)}/{len(probe)} probe entropies match): {out_dir}")
        with open(os.path.join(out_dir, VERIFIED_MARKER), "w") as f:
            f.write("")
        return

    log.info(f"MISMATCH at {out_dir}; deleting stale outputs and re-running inline")
    for fname in STALE_FILES:
        path_stale = os.path.join(out_dir, fname)
        if os.path.exists(path_stale):
            os.remove(path_stale)
    del probe_results
    _compute_and_save(cfg, model, tokenizer, samples, out_dir)
    with open(os.path.join(out_dir, VERIFIED_MARKER), "w") as f:
        f.write("")


if __name__ == "__main__":
    main()
