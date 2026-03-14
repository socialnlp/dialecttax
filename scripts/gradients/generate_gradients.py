"""
Compute per-document projected gradients and pairwise cosine similarities.

For each document in the dataset, computes the full-parameter gradient of the
causal LM cross-entropy loss, projects it to a low-dimensional vector via
CountSketch, and saves projections + cosine similarity matrix.

Supports an optional perturbation applied to SAE text before gradient
computation, providing a surface-form control baseline (e.g. swap, drop, insert)
against which dialect-induced gradient divergence can be compared.

Usage:
    python scripts/gradients/generate_gradients.py
    python scripts/gradients/generate_gradients.py model=llama_8b_base
    python scripts/gradients/generate_gradients.py perturbation=swap dialect=sae
    python scripts/gradients/generate_gradients.py --multirun model=llama_8b_base,llama_3b_base dataset=redial dialect=sae,aave
    python scripts/gradients/generate_gradients.py --multirun perturbation=swap,drop,insert dialect=sae
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
import dialecttax.gradients


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
    model, tokenizer = dialecttax.gradients.load_model(model_id, device=device)
    _loaded_model = (model, tokenizer)
    _loaded_model_name = name
    return model, tokenizer


########
# DATA #
########

def _build_redial_prompt(ds, i, task, dialect):
    """Build a ReDial prompt for a single sample (same as benchmark_rewards).

    Args:
        ds: Dataset list.
        i: Sample index.
        task: Task name.
        dialect: Dialect name.

    Returns:
        Formatted prompt string.
    """
    reasoning = "naive"
    instructions = dialecttax.prompts.INSTS[task][reasoning][dialect]
    if "{choices}" in instructions:
        choices = ds[i]["choices"]
        choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
        instructions = instructions.format(choices=choices_str)

    formatter = dialecttax.prompts.FORMAT_PROMPTS_REGISTRY[task]
    template = dialecttax.prompts.PROMPTS[task][reasoning][dialect]
    body = formatter(template)(ds, i)
    return dialecttax.prompts.get_prompt(body, instructions=instructions)


def _build_samples(ds, task, dialect, dataset_name):
    """Build (unique_id, text) pairs for each sample in the dataset.

    For ReDial: concatenates prompt + answer as plain text.
    For text-only datasets (parallelaave, multivalue): uses raw text.

    Args:
        ds: Dataset list of dicts.
        task: Task name.
        dialect: Dialect name.
        dataset_name: Name of the dataset.

    Returns:
        List of dicts with keys: unique_id, text.
    """
    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    samples = []

    if hasattr(mod, "TASKS"):
        # ReDial-style: concatenate prompt + answer
        for i, sample in enumerate(ds):
            prompt = _build_redial_prompt(ds, i, task, dialect)
            answer = str(sample["answer"])
            samples.append({
                "unique_id": sample["unique_id"],
                "text": prompt + answer,
            })
    else:
        # Text-only datasets
        for sample in ds:
            samples.append({
                "unique_id": sample["unique_id"],
                "text": sample["text"],
            })

    return samples


##################
# PERTURBATIONS  #
##################

def _load_perturbation_texts(preprocessed_dir, perturbation_name, dataset_name, task, dialect):
    """Load pre-generated perturbed texts from disk.

    Perturbation files are JSONL with one JSON-encoded string per line,
    line-parallel to the original dataset.

    Args:
        preprocessed_dir: Root preprocessed directory.
        perturbation_name: Name of perturbation (e.g. "swap-0.05").
        dataset_name: Name of dataset (e.g. "redial").
        task: Task name (e.g. "math").
        dialect: Dialect name (e.g. "sae").

    Returns:
        List of perturbed text strings, or None if file not found.
    """
    path = os.path.join(
        preprocessed_dir, "perturbations", perturbation_name, dataset_name, task, f"{dialect}.jsonl"
    )
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f]


#############
# GRADIENTS #
#############

def compute_all_gradients(model, tokenizer, samples, projection_dim, seed, label=""):
    """Compute projected gradients for all samples.

    Args:
        model: CausalLM with gradient checkpointing.
        tokenizer: Corresponding tokenizer.
        samples: List of dicts with unique_id and text.
        projection_dim: CountSketch projection dimensionality.
        seed: Base seed for CountSketch.
        label: Description string for progress display.

    Returns:
        Tuple of (projections, metadata) where projections is ndarray of shape
        (n_samples, projection_dim) and metadata is list of dicts.
    """
    input_device = next(model.parameters()).device
    projections = np.zeros((len(samples), projection_dim), dtype=np.float32)
    metadata = []

    for i, sample in enumerate(samples):
        inputs = tokenizer(sample["text"], return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)
        projected, loss, grad_norm = dialecttax.gradients.compute_projected_gradient(
            model, input_ids, projection_dim, seed,
        )
        projections[i] = projected.numpy()
        metadata.append({
            "unique_id": sample["unique_id"],
            "loss": loss,
            "grad_norm": grad_norm,
            "n_tokens": input_ids.shape[1],
        })
        print(f"\r  Computing sample ({label}) {i + 1}/{len(samples)}", end="", flush=True)
    print()

    return projections, metadata


def compute_cosine_similarity(projections):
    """Compute pairwise cosine similarity matrix.

    Args:
        projections: ndarray of shape (n_samples, projection_dim).

    Returns:
        Cosine similarity matrix of shape (n_samples, n_samples).
    """
    norms = np.linalg.norm(projections, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = projections / norms
    return normalized @ normalized.T


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_gradients", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    task = cfg.task.name
    dialect = cfg.dialect.name
    perturbation_fn = cfg.perturbation.fn
    perturbation_name = cfg.perturbation.name

    # Skip invalid dialect/dataset or task/dataset combos
    valid_dialects = list(cfg.dataset.dialects)
    if dialect not in valid_dialects:
        log.info(f"Skipping (dialect '{dialect}' not in {valid_dialects} for dataset '{dataset_name}')")
        return
    if "tasks" in cfg.dataset and task not in list(cfg.dataset.tasks):
        log.info(f"Skipping (task '{task}' not in {list(cfg.dataset.tasks)} for dataset '{dataset_name}')")
        return

    # Perturbations only apply to SAE text (surface-form control baseline)
    if perturbation_fn is not None and dialect != "sae":
        log.info(f"Skipping (perturbation '{perturbation_name}' only applies to dialect 'sae', got '{dialect}')")
        return

    if perturbation_fn is not None:
        experiments_dir = _project_config["directories"]["experiments"]
        out_dir = os.path.join(
            experiments_dir, "generate_gradients", cfg.model.name, dataset_name, task, "sae", "perturbed", perturbation_name,
        )
    else:
        out_dir = HydraConfig.get().runtime.output_dir

    # Skip if already computed
    dim = cfg.projection_dim
    proj_exists = os.path.exists(os.path.join(out_dir, f"projections-{dim}.npy"))
    meta_exists = os.path.exists(os.path.join(out_dir, "metadata.jsonl"))
    if not cfg.rerun and proj_exists and meta_exists:
        log.info(f"Skipping (outputs exist, rerun=false): {out_dir}")
        return

    log.info(
        f"\n{'=' * 60}\n"
        f"  Model:          {cfg.model.name} ({cfg.model.model_id})\n"
        f"  Dataset:        {dataset_name}\n"
        f"  Task:           {task}\n"
        f"  Dialect:        {dialect}\n"
        f"  Perturbation:   {perturbation_name}\n"
        f"  Projection dim: {cfg.projection_dim}\n"
        f"  Seed:           {cfg.seed}\n"
        f"{'=' * 60}"
    )

    # Load dataset
    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    dir_root = _project_config["directories"][cfg.dataset.dir_key]
    path_file = mod.FILE_NAME_FORMAT.format(task=task, dialect=dialect)
    path_file = os.path.join(mod.DIRECTORY_NAME, path_file)
    path = os.path.join(dir_root, path_file)
    if not os.path.exists(path):
        log.error(f"Dataset not found: {path}")
        return
    ds = mod.load_dataset(dir_root, path_file)
    log.info(f"Loaded {len(ds)} samples from {os.path.abspath(path)}")

    # Load and apply perturbation if specified (substitutes the text field in-place)
    if perturbation_fn is not None:
        preprocessed_dir = _project_config["directories"]["preprocessed"]
        perturbed_texts = _load_perturbation_texts(
            preprocessed_dir, perturbation_name, dataset_name, task, dialect,
        )
        if perturbed_texts is None:
            # translate perturbations exist for SAE only; any other missing file is a real gap.
            if perturbation_name.startswith("translate") and dialect != "sae":
                log.info(f"Skipping (translate is SAE-only; no '{dialect}' variant): {perturbation_name}/{dataset_name}/{task}/{dialect}")
            else:
                log.error(f"Perturbation file not found for {perturbation_name}/{dataset_name}/{task}/{dialect}")
            return
        row_text = cfg.task.row_text
        for i, text in enumerate(perturbed_texts):
            ds[i][row_text] = text
        log.info(f"Loaded {len(perturbed_texts)} perturbed texts ({perturbation_name})")

    # Build samples
    samples = _build_samples(ds, task, dialect, dataset_name)
    log.info(f"Built {len(samples)} samples for gradient computation")

    # Load model
    model, tokenizer = _load_model(cfg.model.name, cfg.model.model_id, device=cfg.device)

    # Compute projected gradients
    label = f"{dataset_name}/{task}/{dialect}"
    if perturbation_fn is not None:
        label += f"/{perturbation_name}"
    projections, metadata = compute_all_gradients(
        model, tokenizer, samples, cfg.projection_dim, cfg.seed, label=label,
    )

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"projections-{dim}.npy"), projections)
    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    log.info(f"Saved {len(samples)} projections ({dim}-d) to: {out_dir}/projections-{dim}.npy")
    log.info(f"Saved {len(samples)} metadata entries to: {out_dir}/metadata.jsonl")
    log.info(
        f"Done: dataset={dataset_name}, task={task}, dialect={dialect}, "
        f"perturbation={perturbation_name}, n_samples={len(samples)}"
    )


if __name__ == "__main__":
    main()
