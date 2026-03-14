"""
Generate per-sample token information for datasets across tokenizers.

Uses Hydra for configuration. Loads a dataset, tokenizes each sample's text
fields, and saves token data as JSONL.

Usage:
    python scripts/tokens/generate_tokens.py
    python scripts/tokens/generate_tokens.py dataset=parallelaave tokenizer=unigram dialect=aave
    python scripts/tokens/generate_tokens.py --multirun dataset=multivalue tokenizer=gpt2,unigram dialect=sae,aave
"""

import argparse
import json
import logging
import math
import os

import hydra
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
    project_config = dialecttax.utils.load_config()
    OmegaConf.register_new_resolver(
        "project", lambda key: project_config["directories"][key],
        replace=True,
    )

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

def load_dataset(dataset_name, dir_root, task, dialect):
    """Load a dataset split by dispatching to the appropriate data module.

    Args:
        dataset_name: Key into DATASET_MODULES (e.g. "redial", "parallelaave").
        dir_root: Root directory for this dataset's files.
        task: Task name (used only for datasets with TASKS attribute).
        dialect: Dialect name.

    Returns:
        List of sample dicts, or None if the file is missing.
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
    ds = mod.load_dataset(dir_root, path_file)
    log.info(f"Loaded {len(ds)} samples from {os.path.abspath(path)}")
    return ds


##########
# SAVING #
##########

def save_results(results, path):
    """Save tokenization results as JSONL.

    Args:
        results: List of result dicts.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in results:
            # Convert NaN to null for valid JSON
            cleaned = {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in row.items()
            }
            f.write(json.dumps(cleaned) + "\n")


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_tokens", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    dir_root = _project_config["directories"][cfg.dataset.dir_key]
    tokenizer_name = cfg.tokenizer.name
    task = cfg.task.name
    dialect = cfg.dialect.name
    row_text = cfg.task.row_text if ("tasks" in cfg.dataset and "row_text" in cfg.task) else cfg.dataset.row_text
    row_names = list(cfg.dataset.row_names)

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
        f"  Tokenizer:  {tokenizer_name}\n"
        f"  Task:       {task}\n"
        f"  Dialect:    {dialect}\n"
        f"  Row text:   {row_text}\n"
        f"{'=' * 60}"
    )

    # Check for existing results
    out_dir = HydraConfig.get().runtime.output_dir
    out_path = os.path.join(out_dir, "tokens.jsonl")
    if not cfg.rerun and os.path.exists(out_path):
        log.info("Skipping (results exist)")
        return

    # Load dataset
    ds = load_dataset(dataset_name, dir_root, task, dialect)
    if ds is None:
        return

    # Compute results
    tokenizer_type = dialecttax.tokenizers.tokenization.TOKENIZER_NAME_TO_TYPE[tokenizer_name]
    results_fn = dialecttax.tokenizers.get_tokenizer_results(tokenizer_type)
    log.info(f"Computing results with: {results_fn.__name__} (tokenizer='{tokenizer_name}')")
    results = results_fn(ds, row_names, tokenizer_name=tokenizer_name, row_text=row_text)

    # Save
    save_results(results, out_path)

    n_tokens = sum(r["n_tokens"] for r in results if isinstance(r.get("n_tokens"), (int, float)) and not (isinstance(r["n_tokens"], float) and math.isnan(r["n_tokens"])))
    log.info(f"Tokenized {len(results)} samples ({n_tokens:.0f} total tokens)")
    print(f"Saved {len(results)} samples to: {out_path}")


if __name__ == "__main__":
    main()
