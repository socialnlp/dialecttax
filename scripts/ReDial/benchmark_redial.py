"""
Benchmark ReDial across tasks, dialects, models, and reasoning strategies via OpenRouter.

Uses Hydra for configuration. Results saved as JSONL per (model, reasoning, task, dialect).

Usage:
    python scripts/ReDial/benchmark_redial.py
    python scripts/ReDial/benchmark_redial.py model=gemma_instruct reasoning=cot task=logic dialect=aave
    python scripts/ReDial/benchmark_redial.py --multirun model=llama_instruct,gemma_instruct reasoning=naive,cot task=math,logic,planning dialect=sae,aave
"""

import argparse
import json
import logging
import os
import sys

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


##########
# CONFIG #
##########

def resolve_generation_config(cfg: DictConfig):
    """Extract generation parameters from a Hydra config.

    Resolves max_tokens_new, max_tokens_reasoning, and reasoning_effort
    with the precedence chain: reasoning defaults < task overrides < model
    overrides.

    Args:
        cfg: Hydra DictConfig with model, task, reasoning sub-configs.

    Returns:
        Dict with keys: model_name, family, reasoning, task, dialect,
        max_tokens_new, max_tokens_reasoning, reasoning_effort,
        temperature, max_workers, print_tag.
    """
    model_info = dialecttax.endpoints.MODELS[cfg.model.name]
    model_name = model_info["id_openrouter"]
    family = model_info["family"]
    reasoning = cfg.reasoning.name
    task = cfg.task.name
    dialect = cfg.dialect.name

    # Resolve max tokens: reasoning defaults < task overrides < model overrides
    max_tokens_new = int(cfg.reasoning.max_tokens_new)
    max_tokens_reasoning = cfg.reasoning.max_tokens_reasoning
    if "max_tokens_new" in cfg.task:
        max_tokens_new = int(cfg.task.max_tokens_new[reasoning])
    if "max_tokens_reasoning" in cfg.task:
        max_tokens_reasoning = cfg.task.max_tokens_reasoning[reasoning]
    if "max_tokens_new" in cfg.model:
        if isinstance(cfg.model.max_tokens_new, int):
            max_tokens_new = int(cfg.model.max_tokens_new)
        elif reasoning in cfg.model.max_tokens_new:
            max_tokens_new = int(cfg.model.max_tokens_new[reasoning])
    if "max_tokens_reasoning" in cfg.model:
        if isinstance(cfg.model.max_tokens_reasoning, int) or cfg.model.max_tokens_reasoning is None:
            max_tokens_reasoning = cfg.model.max_tokens_reasoning
        elif reasoning in cfg.model.max_tokens_reasoning:
            max_tokens_reasoning = cfg.model.max_tokens_reasoning[reasoning]
    max_tokens_reasoning = int(max_tokens_reasoning) if max_tokens_reasoning is not None else None
    reasoning_effort = None
    if "reasoning_effort" in cfg.model and reasoning in cfg.model.reasoning_effort:
        reasoning_effort = str(cfg.model.reasoning_effort[reasoning])
    temperature = float(cfg.temperature)
    max_workers = int(cfg.max_workers)
    print_tag = f"[{cfg.model.name}/{task}/{dialect}/{reasoning}]"

    return {
        "model_name": model_name,
        "family": family,
        "reasoning": reasoning,
        "task": task,
        "dialect": dialect,
        "max_tokens_new": max_tokens_new,
        "max_tokens_reasoning": max_tokens_reasoning,
        "reasoning_effort": reasoning_effort,
        "temperature": temperature,
        "max_workers": max_workers,
        "print_tag": print_tag,
    }


###########
# PROMPTS #
###########

def get_prompt(ds, i, task, reasoning, dialect):
    """Build a single prompt for a dataset sample."""
    instructions = dialecttax.prompts.INSTS[task][reasoning][dialect]
    if "{choices}" in instructions:
        choices = ds[i]["choices"]
        choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
        instructions = instructions.format(choices=choices_str)

    formatter = dialecttax.prompts.FORMAT_PROMPTS_REGISTRY[task]
    template = dialecttax.prompts.PROMPTS[task][reasoning][dialect]
    body = formatter(template)(ds, i)

    return dialecttax.prompts.get_prompt(body, instructions=instructions)


def build_messages(ds, task, reasoning, dialect, family, indices=None):
    """Build list of messages for dataset samples.

    Args:
        ds: Dataset (list of dicts).
        task: Task name.
        reasoning: Reasoning strategy name.
        dialect: Dialect name.
        family: Model family name.
        indices: Sample indices to build for. None for all.

    Returns:
        List of message dicts.
    """
    if indices is None:
        indices = range(len(ds))
    messages = []
    for i in indices:
        prompt = get_prompt(ds, i, task, reasoning, dialect)
        system = dialecttax.prompts.get_system_prompt(dialect, reasoning, family=family)
        messages.append(dialecttax.endpoints.get_message(prompt, system))
    return messages


########
# DATA #
########

def load_redial_dataset(dir_preprocessed, task, dialect, print_tag):
    """Load a ReDial dataset split.

    Args:
        dir_preprocessed: Root directory for preprocessed data.
        task: Task name.
        dialect: Dialect name.
        print_tag: Logging prefix.

    Returns:
        List of sample dicts, or None if the file is missing.
    """
    path_file = dialecttax.data.redial.FILE_NAME_FORMAT.format(task=task, dialect=dialect)
    path_file = os.path.join(dialecttax.data.redial.DIRECTORY_NAME, path_file)
    path = os.path.join(dir_preprocessed, path_file)
    if not os.path.exists(path):
        log.error(f"{print_tag} {path} not found")
        return None
    ds = dialecttax.data.redial.load_dataset(dir_preprocessed, path_file)
    log.info(f"{print_tag} Loaded {len(ds)} samples from: {os.path.abspath(path)}")
    return ds


###########
# GRADING #
###########

def grade_task(task, completions, ds):
    """Grade completions for a given task."""
    grader = dialecttax.data.graders.GRADERS[task]
    if task == "algorithm":
        gold = [(row["test_imports"], row["tests"]) for row in ds]
    else:
        gold = [str(row["answer"]) for row in ds]
    return grader.grade_completions(completions, gold)


##########
# SAVING #
##########

def save_results(results, path):
    """Save results as JSONL."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/benchmark_redial", config_name="config")
def main(cfg: DictConfig):
    api_key = dialecttax.utils.get_api_key(_project_config["keys"]["openrouter"])
    dir_preprocessed = _project_config["directories"]["preprocessed"]

    rc = resolve_generation_config(cfg)
    model_name = rc["model_name"]
    family = rc["family"]
    reasoning = rc["reasoning"]
    task = rc["task"]
    dialect = rc["dialect"]
    max_tokens_new = rc["max_tokens_new"]
    max_tokens_reasoning = rc["max_tokens_reasoning"]
    reasoning_effort = rc["reasoning_effort"]
    temperature = rc["temperature"]
    max_workers = rc["max_workers"]
    print_tag = rc["print_tag"]

    log.info(
        f"\n{'=' * 60}\n"
        f"  Model:            {cfg.model.name} ({model_name})\n"
        f"  Task:             {task}\n"
        f"  Dialect:          {dialect}\n"
        f"  Reasoning:        {reasoning}\n"
        f"  Tokens:           max_tokens_new={max_tokens_new}\n"
        f"                    max_tokens_reasoning={max_tokens_reasoning}\n"
        f"                    reasoning_effort={reasoning_effort}\n"
        f"  Hyperparameters:  temperature={temperature}\n"
        f"{'=' * 60}"
    )

    # Capture stderr; only write .err file if non-empty
    hydra_cfg = HydraConfig.get()
    out_dir = hydra_cfg.runtime.output_dir
    log_path = os.path.join(out_dir, hydra_cfg.job.name + ".log")
    err_path = log_path.replace(".log", ".err")
    os.makedirs(out_dir, exist_ok=True)
    _orig_stderr = sys.stderr
    sys.stderr = open(err_path, "w")
    log.info(f"Logging run to: {out_dir}")

    # Check for existing results
    out_path = os.path.join(out_dir, "results.jsonl")
    if not cfg.rerun and os.path.exists(out_path):
        log.info(f"{print_tag} Skipping (results exist)")
        return

    # Load data
    ds = load_redial_dataset(dir_preprocessed, task, dialect, print_tag)
    if ds is None:
        return
    if cfg.run_test:
        ds = ds[:5]

    # Build and send
    messages = build_messages(ds, task, reasoning, dialect, family)
    resp_path = os.path.join(out_dir, "responses.jsonl")
    log.info(f"Saving responses to: {resp_path}")
    responses = dialecttax.endpoints.generate(
        messages,
        api_key,
        model=model_name,
        max_tokens_new=max_tokens_new,
        max_tokens_reasoning=max_tokens_reasoning,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_workers=max_workers,
        path_save=resp_path,
    )
    n_errors = sum(1 for r in responses if "error" in r)
    log.info(f"PROCESSED {len(responses)} OF {len(responses)} [ERRORS: {n_errors}]")

    # Grade
    completions = dialecttax.endpoints.get_completions(responses)
    results = grade_task(task, completions, ds)

    # Save results
    save_results(results, out_path)

    # Summary
    n_correct = sum(1 for r in results if r.get("correct"))
    acc = n_correct / len(results) * 100 if results else 0
    log.info(f"{print_tag} Accuracy: {n_correct}/{len(results)} ({acc:.0f}%)")
    log.info(f"Saved results to: {out_path}")

    # Remove .err file if empty
    sys.stderr.close()
    sys.stderr = _orig_stderr
    if os.path.getsize(err_path) == 0:
        os.remove(err_path)


if __name__ == "__main__":
    main()
