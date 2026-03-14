"""
Generate perturbed versions of dataset texts using surface-form perturbations.

Uses Hydra for configuration. Loads a dataset, applies a perturbation function,
and saves the perturbed texts as a JSON file.

Usage:
    python scripts/perturbations/generate_perturbations.py
    python scripts/perturbations/generate_perturbations.py perturbation=drop
    python scripts/perturbations/generate_perturbations.py --multirun perturbation=swap,drop,insert
"""

import argparse
import json
import logging
import os
import random

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import dialecttax


#########
# SETUP #
#########

def setup_hydra():
    """Register OmegaConf resolver and apply Hydra compatibility patches.

    Returns:
        Project config dict from dialecttax.utils.load_config().
    """
    project_config = dialecttax.utils.load_config()
    OmegaConf.register_new_resolver("project", lambda key: project_config["directories"][key], replace=True)

    # Hydra 1.3.2 + Python 3.14 compatibility patch
    if hasattr(argparse.ArgumentParser, "_check_help"):
        _orig_check_help = argparse.ArgumentParser._check_help

        def _patched_check_help(self, action):
            if action.help is not None and not isinstance(action.help, str):
                action.help = repr(action.help)
            _orig_check_help(self, action)

        argparse.ArgumentParser._check_help = _patched_check_help

    return project_config


_project_config = setup_hydra()

log = logging.getLogger(__name__)


########
# DATA #
########

def load_dataset(dataset_name, dir_root, task, dialect, **kwargs):
    """Load a dataset split by dispatching to the appropriate data module.

    Args:
        dataset_name: Key into DATASET_MODULES (e.g. "redial", "parallelaave").
        dir_root: Root directory for this dataset's files.
        task: Task name (used only for datasets with TASKS attribute).
        dialect: Dialect name.
        **kwargs: Forwarded to the underlying mod.load_dataset().

    Returns:
        List of sample dicts or strings, or None if the file is missing.
    """
    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    fmt_kwargs = {"dialect": dialect}
    if hasattr(mod, "TASKS"):
        fmt_kwargs["task"] = task
    path_file = mod.FILE_NAME_FORMAT.format(**fmt_kwargs)
    path_file = os.path.join(mod.DIRECTORY_NAME, path_file)
    path = os.path.join(dir_root, path_file)
    if not os.path.exists(path):
        log.error(f"Dataset not found: {path}")
        return None
    ds = mod.load_dataset(dir_root, path_file, **kwargs)
    log.info(f"Loaded {len(ds)} samples from {os.path.abspath(path)}")
    return ds


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_perturbations", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    dir_root = _project_config["directories"][cfg.dataset.dir_key]
    task = cfg.task.name
    dialect = cfg.dialect.name
    row_text = cfg.task.row_text if "row_text" in cfg.task else cfg.dataset.row_text
    perturbation_name = cfg.perturbation.name
    perturbation_fn = cfg.perturbation.fn
    perturbation_kwargs = OmegaConf.to_container(cfg.perturbation.kwargs, resolve=True)
    seed = cfg.seed

    # Skip invalid dialect/dataset or task/dataset combos
    valid_dialects = list(cfg.dataset.dialects)
    if dialect not in valid_dialects:
        log.info(f"Skipping (dialect '{dialect}' not in {valid_dialects} for dataset '{dataset_name}')")
        return
    if "tasks" in cfg.dataset and task not in list(cfg.dataset.tasks):
        log.info(f"Skipping (task '{task}' not in {list(cfg.dataset.tasks)} for dataset '{dataset_name}')")
        return

    # Check for existing results (also skips redundant runs for taskless datasets)
    preprocessed_dir = _project_config["directories"]["preprocessed"]
    output_subdir = cfg.dataset.output_subdir
    out_path = os.path.join(preprocessed_dir, "perturbations", perturbation_name, dataset_name, f"{output_subdir}.jsonl")
    if not cfg.rerun and os.path.exists(out_path):
        log.info(f"Skipping (results exist: {out_path})")
        return

    log.info(
        f"\n{'=' * 60}\n"
        f"  Dataset:       {dataset_name}\n"
        f"  Task:          {task}\n"
        f"  Dialect:       {dialect}\n"
        f"  Row text:      {row_text}\n"
        f"  Perturbation:  {perturbation_name}\n"
        f"  Kwargs:        {perturbation_kwargs}\n"
        f"  Seed:          {seed}\n"
        f"{'=' * 60}"
    )

    # Load dataset
    ds = load_dataset(dataset_name, dir_root, task, dialect, return_id=False)
    if ds is None:
        return

    # Extract texts
    if isinstance(ds[0], str):
        texts = ds
    else:
        texts = [row[row_text] for row in ds]
    log.info(f"Perturbing {len(texts)} texts with '{perturbation_name}'")

    # Seed and perturb
    random.seed(seed)
    np.random.seed(seed)
    fn = getattr(dialecttax.perturbations, perturbation_fn)
    if perturbation_fn == "translate":
        api_key = dialecttax.utils.get_api_key(_project_config["keys"]["gcloud"])
        perturbation_kwargs["api_key"] = api_key
        log.info(f"Translating to: {perturbation_kwargs['target_language']}")
    perturbed = fn(texts, **perturbation_kwargs)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for text in perturbed:
            f.write(json.dumps(text, ensure_ascii=False) + "\n")

    log.info(f"Saved {len(perturbed)} perturbed texts: {out_path}")


if __name__ == "__main__":
    main()
