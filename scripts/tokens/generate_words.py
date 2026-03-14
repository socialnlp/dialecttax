"""
Generate unique words for datasets by splitting text on whitespace and delimiters.

Uses Hydra for configuration. Loads a dataset, splits each sample's text
field into words (on whitespace, "/", etc.), and saves unique words as a text file.

Usage:
    python scripts/tokens/generate_words.py
    python scripts/tokens/generate_words.py dataset=parallelaave dialect=aave
    python scripts/tokens/generate_words.py --multirun dataset=multivalue dialect=sae,aave
"""

import argparse
import json
import logging
import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import dialecttax
from dialecttax.tokenizers.tokenization import extract_tokens, TOKENIZER_NAME_MAP
from dialecttax.tokenizers.words import extract_words


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


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_words", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    dir_root = _project_config["directories"][cfg.dataset.dir_key]
    task = cfg.task.name
    dialect = cfg.dialect.name
    row_text = cfg.task.row_text if ("tasks" in cfg.dataset and "row_text" in cfg.task) else cfg.dataset.row_text

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
        f"{'=' * 60}"
    )

    # Check which outputs still need generating
    out_dir = HydraConfig.get().runtime.output_dir
    words_path = os.path.join(out_dir, "words.json")
    skip_words = not cfg.rerun and os.path.exists(words_path)
    tokens_todo = [
        name for name in TOKENIZER_NAME_MAP
        if cfg.rerun or not os.path.exists(os.path.join(out_dir, f"tokens_{name}.json"))
    ]

    if skip_words and not tokens_todo:
        log.info("Skipping (all results exist)")
        return

    # Load dataset
    ds = load_dataset(dataset_name, dir_root, task, dialect)
    if ds is None:
        return

    # Extract and save words
    if skip_words:
        log.info("Skipping words (results exist)")
    else:
        word_counts = extract_words(ds, row_text)
        os.makedirs(out_dir, exist_ok=True)
        with open(words_path, "w") as f:
            json.dump(word_counts, f)
        log.info(f"Saved {len(word_counts)} unique words to: {words_path}")

    # Extract and save tokens for each tokenizer
    for tokenizer_name in tokens_todo:
        tokens_path = os.path.join(out_dir, f"tokens_{tokenizer_name}.json")
        log.info(f"Tokenizing with {tokenizer_name} ({TOKENIZER_NAME_MAP[tokenizer_name]})")
        token_counts = extract_tokens(ds, row_text, tokenizer_name)
        with open(tokens_path, "w") as f:
            json.dump(token_counts, f)
        log.info(f"Saved {len(token_counts)} unique tokens to: {tokens_path}")


if __name__ == "__main__":
    main()
