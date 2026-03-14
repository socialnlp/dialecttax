"""
Compute per-layer cosine similarity between dialect and SAE hidden states.

For each parallel pair (dialect, SAE), extracts mean-pooled hidden states at
every transformer layer, then computes per-pair cosine similarity to measure
how representations converge across layers.

Usage:
    python scripts/layers/generate_layers.py
    python scripts/layers/generate_layers.py model=llama_8b_base dataset=parallelaave dialect=aave
    python scripts/layers/generate_layers.py --multirun model=llama_8b_base,llama_3b_base dialect=sae,aave
"""

import argparse
import gc
import json
import logging
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import dialecttax
import dialecttax.layers


#########
# SETUP #
#########

def setup_hydra():
    """Register OmegaConf resolver and apply Hydra compatibility patches.

    Returns:
        Project config dict from dialecttax.utils.load_config().
    """
    project_config = dialecttax.utils.load_config(os.environ.get("DIALECTTAX_CONFIG", "default"))
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


#########
# MODEL #
#########

_loaded_model = None
_loaded_model_name = None


def _load_model(name, model_id, device="auto"):
    """Load a language model, reusing across multirun iterations.

    Args:
        name: Short model name (e.g. "llama_8b").
        model_id: HuggingFace model ID.
        device: Device string (default "auto" for multi-GPU via accelerate).

    Returns:
        Tuple of (model, tokenizer).
    """
    global _loaded_model, _loaded_model_name
    if _loaded_model is not None and _loaded_model_name == name:
        log.info(f"Reusing loaded model: {name}")
        return _loaded_model

    if _loaded_model is not None:
        log.info(f"Unloading previous model: {_loaded_model_name}")
        del _loaded_model
        _loaded_model = None
        _loaded_model_name = None
        torch.cuda.empty_cache()
        gc.collect()

    log.info(f"Loading model: {name} ({model_id}, device={device})")
    model, tokenizer = dialecttax.layers.load_model(model_id, device=device)
    _loaded_model = (model, tokenizer)
    _loaded_model_name = name
    return model, tokenizer


########
# DATA #
########

def _load_parallel_texts(cfg):
    """Load dialect and SAE texts aligned by index.

    Args:
        cfg: Hydra DictConfig.

    Returns:
        Tuple of (dialect_texts, sae_texts, unique_ids).
    """
    dataset_name = cfg.dataset.name
    dialect = cfg.dialect.name
    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    dir_root = _project_config["directories"][cfg.dataset.dir_key]

    # Load dialect samples
    dialect_file = mod.FILE_NAME_FORMAT.format(task=cfg.task.name, dialect=dialect)
    dialect_path = os.path.join(mod.DIRECTORY_NAME, dialect_file)
    dialect_ds = mod.load_dataset(dir_root, dialect_path)

    # Load SAE samples
    sae_file = mod.FILE_NAME_FORMAT.format(task=cfg.task.name, dialect="sae")
    sae_path = os.path.join(mod.DIRECTORY_NAME, sae_file)
    sae_ds = mod.load_dataset(dir_root, sae_path)

    n = min(len(dialect_ds), len(sae_ds))
    dialect_texts = [dialect_ds[i]["text"] for i in range(n)]
    sae_texts = [sae_ds[i]["text"] for i in range(n)]
    unique_ids = [dialect_ds[i]["unique_id"] for i in range(n)]

    return dialect_texts, sae_texts, unique_ids


def _load_perturbation_texts(preprocessed_dir, perturbation_name, dataset_name):
    """Load pre-generated perturbed SAE texts for a flat dataset.

    Files are JSONL with one JSON-encoded string per line, line-parallel to the
    SAE dataset (same format as generate_gradients).

    Args:
        preprocessed_dir: Root preprocessed directory.
        perturbation_name: Perturbation name (e.g. "swap-0.05", "translate-french").
        dataset_name: Dataset name ("parallelaave" or "multivalue").

    Returns:
        List of perturbed text strings, or None if the file is missing.
    """
    path = os.path.join(preprocessed_dir, "perturbations", perturbation_name, dataset_name, "sae.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f]


def _load_perturbed_pair(cfg, preprocessed_dir):
    """Load perturbed-SAE and SAE texts aligned by index.

    The "dialect" arm becomes SAE text under a perturbation; SAE stays the
    unperturbed control, so the per-layer similarity measures what a
    dialect-agnostic transformation does to the representation.

    Args:
        cfg: Hydra DictConfig.
        preprocessed_dir: Root preprocessed directory.

    Returns:
        Tuple of (perturbed_texts, sae_texts, unique_ids), or (None, None, None)
        if the perturbation file is missing.
    """
    dataset_name = cfg.dataset.name
    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    dir_root = _project_config["directories"][cfg.dataset.dir_key]

    sae_file = mod.FILE_NAME_FORMAT.format(task=cfg.task.name, dialect="sae")
    sae_path = os.path.join(mod.DIRECTORY_NAME, sae_file)
    sae_ds = mod.load_dataset(dir_root, sae_path)

    perturbed = _load_perturbation_texts(preprocessed_dir, cfg.perturbation, dataset_name)
    if perturbed is None:
        return None, None, None

    n = min(len(sae_ds), len(perturbed))
    perturbed_texts = perturbed[:n]
    sae_texts = [sae_ds[i]["text"] for i in range(n)]
    unique_ids = [sae_ds[i]["unique_id"] for i in range(n)]

    return perturbed_texts, sae_texts, unique_ids


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_layers", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    dialect = cfg.dialect.name
    perturbation = cfg.get("perturbation", "none")

    if perturbation != "none":
        # Control arm: perturbed-SAE vs SAE, the noise baseline for §5.3.
        out_dir = os.path.join(
            _project_config["directories"]["experiments"], cfg.experiment,
            cfg.model.name, dataset_name, "perturbed", perturbation,
        )
        label = f"{perturbation} vs SAE"
    else:
        # Skip SAE-vs-SAE (no comparison needed)
        if dialect == "sae":
            log.info("Skipping (dialect=sae, no comparison needed)")
            return
        # Skip invalid dialect/dataset combos
        valid_dialects = list(cfg.dataset.dialects)
        if dialect not in valid_dialects:
            log.info(f"Skipping (dialect '{dialect}' not in {valid_dialects} for dataset '{dataset_name}')")
            return
        out_dir = HydraConfig.get().runtime.output_dir
        label = f"{dialect} vs SAE"

    # Skip if already computed
    sim_exists = os.path.exists(os.path.join(out_dir, "layer_similarity.npy"))
    meta_exists = os.path.exists(os.path.join(out_dir, "metadata.jsonl"))
    if not cfg.rerun and sim_exists and meta_exists:
        log.info(f"Skipping (outputs exist, rerun=false): {out_dir}")
        return

    log.info(
        f"\n{'=' * 60}\n"
        f"  Model:       {cfg.model.name} ({cfg.model.model_id})\n"
        f"  Dataset:     {dataset_name}\n"
        f"  Condition:   {label}\n"
        f"  Batch size:  {cfg.batch_size}\n"
        f"{'=' * 60}"
    )

    # Load texts: dialect arm, or perturbed-SAE arm
    if perturbation != "none":
        dialect_texts, sae_texts, unique_ids = _load_perturbed_pair(
            cfg, _project_config["directories"]["preprocessed"],
        )
        if dialect_texts is None:
            log.error(f"Perturbation file not found: {perturbation}/{dataset_name}")
            return
    else:
        dialect_texts, sae_texts, unique_ids = _load_parallel_texts(cfg)
    n = len(dialect_texts)
    log.info(f"Loaded {n} parallel pairs ({dataset_name}, {label})")

    # Load model
    model, tokenizer = _load_model(cfg.model.name, cfg.model.model_id, device=cfg.device)

    # Extract hidden states
    log.info(f"Extracting dialect hidden states ({n} samples)...")
    hidden_dialect = dialecttax.layers.extract_hidden_states(model, tokenizer, dialect_texts, batch_size=cfg.batch_size)

    log.info(f"Extracting SAE hidden states ({n} samples)...")
    hidden_sae = dialecttax.layers.extract_hidden_states(model, tokenizer, sae_texts, batch_size=cfg.batch_size)

    # Compute per-layer cosine similarity: (n_samples, n_layers)
    similarities = dialecttax.layers.compute_pairwise_cosine(hidden_dialect, hidden_sae)
    n_layers = similarities.shape[1]

    # Summary stats
    mean_sim = similarities.mean(axis=0)  # (n_layers,)
    log.info(f"Layer similarities (mean across {n} pairs):")
    for layer_idx in range(n_layers):
        log.info(f"  Layer {layer_idx:3d}: {mean_sim[layer_idx]:.4f}")

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "layer_similarity.npy"), similarities)

    metadata = []
    for i in range(n):
        metadata.append({
            "unique_id": unique_ids[i],
            "mean_similarity": float(similarities[i].mean()),
            "n_layers": n_layers,
        })
    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    log.info(f"Saved layer_similarity.npy ({n} x {n_layers}) to: {out_dir}")
    log.info(f"Saved {n} metadata entries to: {out_dir}/metadata.jsonl")


if __name__ == "__main__":
    main()
