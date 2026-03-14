"""
Generate per-sample embeddings for datasets using EmbeddingGemma.

Uses Hydra for configuration. Loads a dataset, encodes each sample's text
field, and saves embeddings as a raw .npy array.

Usage:
    python scripts/embeddings/generate_embeddings.py
    python scripts/embeddings/generate_embeddings.py dataset=parallelaave dialect=aave
    python scripts/embeddings/generate_embeddings.py --multirun dataset=parallelaave dialect=sae,aave
"""

import argparse
import json
import logging
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
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
    # The `project:` resolver must be registered before Hydra composes the config
    # (hydra.run.dir needs it), so project_config is read straight from the YAML
    # rather than from cfg.
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs/generate_embeddings/config.yaml")
    project_config = dialecttax.utils.load_config(OmegaConf.load(config_path).project_config)
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

def load_dataset(dataset_name, dir_root, task, dialect, perturbation_name=None, **kwargs):
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

    if perturbation_name is None:
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
    else:
        if hasattr(mod, "TASKS"):
            dataset_name = os.path.join(dataset_name, task)
        path = os.path.join(
            dir_root,
            dialecttax.perturbations.DIRECTORY_NAME,
            perturbation_name,
            dataset_name,
            f"{dialect}.jsonl"
        )
        if not os.path.exists(path):
            # translate perturbations exist for SAE only; any other missing file is a real gap.
            if perturbation_name.startswith("translate") and dialect != "sae":
                log.info(f"Skipping (translate is SAE-only; no '{dialect}' variant): {path}")
            else:
                log.error(f"Dataset not found: {path}")
            return None
        with open(path, "r") as f:
            ds = [json.loads(line) for line in f]
        log.info(f"Loaded {len(ds)} samples from {os.path.abspath(path)}")
    return ds

########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_embeddings", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    dir_root = _project_config["directories"][cfg.dataset.dir_key]
    task = cfg.task.name
    dialect = cfg.dialect.name
    row_text = cfg.task.row_text if "row_text" in cfg.task else cfg.dataset.row_text
    dim = cfg.dim
    batch_size = cfg.batch_size
    perturbation_name = cfg.perturbation.name if "perturbation" in cfg else None

    # Pick GPU with most free memory (need >=2 GB)
    MIN_FREE_BYTES = 2 * 1024**3
    n_gpus = torch.cuda.device_count()
    if n_gpus > 0:
        free = [torch.cuda.mem_get_info(i)[0] for i in range(n_gpus)]
        best = free.index(max(free))
        if free[best] >= MIN_FREE_BYTES:
            device = f"cuda:{best}"
        else:
            log.error(f"No GPU with >= 2 GB free (best: {free[best] / 1024**3:.1f} GB on cuda:{best})")
            return
    else:
        device = "cpu"

    # Skip invalid dialect/dataset or task/dataset combos
    valid_dialects = list(cfg.dataset.dialects)
    if dialect not in valid_dialects:
        log.info(f"Skipping (dialect '{dialect}' not in {valid_dialects} for dataset '{dataset_name}')")
        return
    if "tasks" in cfg.dataset and task not in list(cfg.dataset.tasks):
        log.info(f"Skipping (task '{task}' not in {list(cfg.dataset.tasks)} for dataset '{dataset_name}')")
        return

    log.info(
        f"\n{'=' * 60}\n"
        f"  Dataset:    {dataset_name}\n"
        f"  Task:       {task}\n"
        f"  Dialect:    {dialect}\n"
        f"  Row text:   {row_text}\n"
        f"  Perturb:    {perturbation_name}\n"
        f"  Dim:        {dim}\n"
        f"  Batch size: {batch_size}\n"
        f"  Device:     {device}\n"
        f"{'=' * 60}"
    )

    # Check for existing results
    out_dir = HydraConfig.get().runtime.output_dir
    if perturbation_name is not None:
        out_path = os.path.join(out_dir, perturbation_name, f"embeddings-{dim}.npy")
    else:
        out_path = os.path.join(out_dir, f"embeddings-{dim}.npy")
    if not cfg.rerun and os.path.exists(out_path):
        log.info("Skipping (results exist)")
        return

    # Load dataset
    if perturbation_name is not None:
        dir_preprocessed = _project_config["directories"]["preprocessed"]
        ds = load_dataset(dataset_name, dir_preprocessed, task, dialect, perturbation_name=perturbation_name, return_id=False)
    else:
        ds = load_dataset(dataset_name, dir_root, task, dialect, return_id=False)
    if ds is None:
        return

    # Extract texts
    if isinstance(ds[0], str):
        texts = ds
    else:
        texts = [row[row_text] for row in ds]
    log.info(f"Encoding {len(texts)} texts")

    # Encode
    model = dialecttax.embeddings.load_embedding_gemma(device=device)
    embeddings = dialecttax.embeddings.encode(model, texts, dim=dim, batch_size=batch_size)

    # Save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, embeddings)

    log.info(f"Saved embeddings {embeddings.shape} to: {out_path}")


if __name__ == "__main__":
    main()
