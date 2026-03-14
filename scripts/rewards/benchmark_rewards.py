"""
Benchmark reward models on word-level, token-level, and sample-level scoring.

Uses Hydra for configuration. Loads unique words/tokens from generate_words outputs,
scores each with the reward model, and caches results globally per reward model.

Usage:
    python scripts/rewards/benchmark_rewards.py
    python scripts/rewards/benchmark_rewards.py reward_model=qrm_llama_8b
    python scripts/rewards/benchmark_rewards.py --multirun reward_model=skywork_llama_8b,qrm_llama_8b dataset=redial,parallelaave
"""

import argparse
import glob
import json
import logging
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


################
# REWARD MODEL #
################

PROVIDER_CLASSES = {
    "skywork": dialecttax.rewards.SkyworkRewardModel,
    "qrm": dialecttax.rewards.QRMRewardModel,
    "ai2": dialecttax.rewards.Ai2RewardModel,
}


_loaded_rm = None
_loaded_rm_name = None


def load_reward_model(name, device="auto"):
    """Instantiate and load a reward model, reusing across multirun iterations.

    Control visible GPUs via CUDA_VISIBLE_DEVICES before launching the process.

    Args:
        name: Reward model name (e.g. "skywork_llama_8b").
        device: Device string (default "auto" for multi-GPU via accelerate).

    Returns:
        Loaded RewardModel instance.
    """
    global _loaded_rm, _loaded_rm_name
    if _loaded_rm is not None and _loaded_rm_name == name:
        log.info(f"Reusing loaded reward model: {name}")
        return _loaded_rm

    # Free previous model before loading a new one
    if _loaded_rm is not None:
        log.info(f"Unloading previous reward model: {_loaded_rm_name}")
        del _loaded_rm
        _loaded_rm = None
        _loaded_rm_name = None
        import torch
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    provider = name.split("_")[0]
    cls = PROVIDER_CLASSES[provider]
    rm = cls(name, device=device)
    log.info(f"Loading reward model: {name} (provider={provider}, device={device})")
    rm.load()
    _loaded_rm = rm
    _loaded_rm_name = name
    return rm


#########
# CACHE #
#########

def _load_cache(path):
    """Load a JSON cache file, returning empty dict if missing."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache, path):
    """Save a dict as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f)


###########
# SCORING #
###########

def score_items(rm, prompt, items, cache, label="item"):
    """Score items not already in cache, update cache in place.

    Args:
        rm: Loaded RewardModel instance.
        prompt: The user prompt text.
        items: List of item strings to score.
        cache: Dict of already-scored items {item: score}.
        label: Label for logging ("word" or "token").

    Returns:
        Number of newly scored items.
    """
    uncached = [item for item in items if item not in cache]
    if not uncached:
        log.info(f"  All {len(items)} {label}s already cached")
        return 0

    log.info(f"  Scoring {len(uncached)} new {label}s ({len(items) - len(uncached)} cached)")
    for item in uncached:
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": item},
        ]
        cache[item] = rm.score(conversation)
    return len(uncached)


def score_samples(rm, ds, task, dialect):
    """Score prompt-completion samples for a ReDial dataset.

    Args:
        rm: Loaded RewardModel instance.
        ds: List of sample dicts.
        task: Task name (math, algorithm, logic, planning).
        dialect: Dialect name (sae, aave).

    Returns:
        List of result dicts with unique_id, prompt, response, score.
    """
    reasoning = "naive"
    results = []
    log.info(f"  Scoring {len(ds)} samples")
    for i, sample in enumerate(ds):
        prompt = _build_redial_prompt(ds, i, task, reasoning, dialect)
        response = str(sample["answer"])
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        score = rm.score(conversation)
        results.append({
            "unique_id": sample["unique_id"],
            "prompt": prompt,
            "response": response,
            "score": score,
        })
    return results


def _build_redial_prompt(ds, i, task, reasoning, dialect):
    """Build a ReDial prompt for a single sample.

    Args:
        ds: Dataset list.
        i: Sample index.
        task: Task name.
        reasoning: Reasoning strategy (always "naive").
        dialect: Dialect name.

    Returns:
        Formatted prompt string.
    """
    instructions = dialecttax.prompts.INSTS[task][reasoning][dialect]
    if "{choices}" in instructions:
        choices = ds[i]["choices"]
        choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
        instructions = instructions.format(choices=choices_str)

    formatter = dialecttax.prompts.FORMAT_PROMPTS_REGISTRY[task]
    template = dialecttax.prompts.PROMPTS[task][reasoning][dialect]
    body = formatter(template)(ds, i)

    return dialecttax.prompts.get_prompt(body, instructions=instructions)


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/benchmark_rewards", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    task = cfg.task.name
    dialect = cfg.dialect.name
    output_subdir = cfg.dataset.output_subdir

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
        f"  Reward Model: {cfg.reward_model.name}\n"
        f"  Dataset:      {dataset_name}\n"
        f"  Task:         {task}\n"
        f"  Dialect:      {dialect}\n"
        f"{'=' * 60}"
    )

    # Directories
    experiments_dir = _project_config["directories"]["experiments"]
    gw_dir = os.path.join(experiments_dir, "generate_words", dataset_name, output_subdir)
    out_dir = HydraConfig.get().runtime.output_dir
    rm_root = os.path.join(experiments_dir, "benchmark_rewards", cfg.reward_model.name)

    # Check generate_words outputs exist
    words_path = os.path.join(gw_dir, "words.json")
    if not os.path.exists(words_path):
        log.error(f"words.json not found: {words_path}")
        return

    # Determine which outputs still need to be computed
    skip = not cfg.rerun
    need_words = not skip or not os.path.exists(os.path.join(out_dir, "words.json"))
    token_files = sorted(glob.glob(os.path.join(gw_dir, "tokens_*.json")))
    need_tokens = [f for f in token_files if not skip or not os.path.exists(os.path.join(out_dir, os.path.basename(f)))]
    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    need_samples = hasattr(mod, "TASKS") and (not skip or not os.path.exists(os.path.join(out_dir, "samples.jsonl")))

    if not need_words and not need_tokens and not need_samples:
        log.info(f"Skipping (all outputs exist, rerun=false): {out_dir}")
        return

    # Load reward model (control GPUs via CUDA_VISIBLE_DEVICES)
    rm = load_reward_model(cfg.reward_model.name, device=cfg.device)
    os.makedirs(out_dir, exist_ok=True)

    #########
    # WORDS #
    #########

    if need_words:
        log.info("Scoring words...")
        word_counts = json.load(open(words_path))
        words = list(word_counts.keys())

        word_cache_path = os.path.join(rm_root, "words.json")
        word_cache = _load_cache(word_cache_path)
        n_new = score_items(rm, dialecttax.prompts.PROMPT_REWARD_WORDS, words, word_cache, label="word")
        if n_new > 0:
            _save_cache(word_cache, word_cache_path)

        local_word_scores = {w: word_cache[w] for w in words}
        with open(os.path.join(out_dir, "words.json"), "w") as f:
            json.dump(local_word_scores, f)
        log.info(f"Saved {len(local_word_scores)} word scores to: {out_dir}/words.json")
    else:
        log.info("Skipping words (already exists, rerun=false)")

    ##########
    # TOKENS #
    ##########

    if need_tokens:
        token_cache_path = os.path.join(rm_root, "tokens.json")
        token_cache = _load_cache(token_cache_path)
        for token_file in need_tokens:
            basename = os.path.basename(token_file)
            tokenizer_name = basename.replace("tokens_", "").replace(".json", "")
            log.info(f"Scoring tokens ({tokenizer_name})...")

            token_counts = json.load(open(token_file))
            tokens = list(token_counts.keys())

            n_new = score_items(rm, dialecttax.prompts.PROMPT_REWARD_TOKENS, tokens, token_cache, label="token")
            if n_new > 0:
                _save_cache(token_cache, token_cache_path)

            local_token_scores = {t: token_cache[t] for t in tokens}
            with open(os.path.join(out_dir, basename), "w") as f:
                json.dump(local_token_scores, f)
            log.info(f"Saved {len(local_token_scores)} token scores to: {out_dir}/{basename}")
    else:
        log.info("Skipping tokens (already exist, rerun=false)")

    ###########
    # SAMPLES #
    ###########

    if need_samples:
        log.info("Scoring samples...")
        dir_root = _project_config["directories"][cfg.dataset.dir_key]
        path_file = mod.FILE_NAME_FORMAT.format(task=task, dialect=dialect)
        path_file = os.path.join(mod.DIRECTORY_NAME, path_file)
        path = os.path.join(dir_root, path_file)
        if not os.path.exists(path):
            log.error(f"Dataset not found: {path}")
            return
        ds = mod.load_dataset(dir_root, path_file)
        log.info(f"Loaded {len(ds)} samples from {os.path.abspath(path)}")

        sample_results = score_samples(rm, ds, task, dialect)
        samples_path = os.path.join(out_dir, "samples.jsonl")
        with open(samples_path, "w") as f:
            for r in sample_results:
                f.write(json.dumps(r) + "\n")
        log.info(f"Saved {len(sample_results)} sample scores to: {samples_path}")
    elif hasattr(mod, "TASKS"):
        log.info("Skipping samples (already exist, rerun=false)")
    else:
        log.info(f"Skipping sample scoring (no TASKS for dataset '{dataset_name}')")

    log.info("Done.")


if __name__ == "__main__":
    main()
