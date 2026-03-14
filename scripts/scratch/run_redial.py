"""
Benchmark preprocessed ReDial datasets on small (<4B) base models.

Evaluates greedy and temperature-sampled generation on the preprocessed
ReDial splits (math_sae, math_aave, etc.) stored at
config["directories"]["preprocessed"]/ReDial/.

Models:
    Llama-3.2-1B, Llama-3.2-3B
    Gemma-3-1B, Gemma-3-4B
    Qwen3-0.6B, Qwen3-4B

Usage:
    # Single run (defaults: llama_base + sae + naive)
    python scripts/ReDial/benchmark_redial.py

    # Override one dimension
    python scripts/ReDial/benchmark_redial.py model=gemma_base dialect=aave

    # Override generation param
    python scripts/ReDial/benchmark_redial.py max_new_tokens=256 max_samples=10

    # Sweep all combinations
    python scripts/ReDial/benchmark_redial.py -m \
      model=llama_base,llama_instruct dialect=sae,aave reasoning=naive,cot
"""

import gc
import json
import logging
import os
import time

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

def _resolve_project_dir(key: str) -> str:
    return dialecttax.utils.load_config("default")["directories"][key]

OmegaConf.register_new_resolver("project", _resolve_project_dir, use_cache=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

import dialecttax

log = logging.getLogger(__name__)

# Hydra 1.3.2 + Python 3.14 compatibility patch
# Python 3.14 argparse._check_help requires help to be a real string.
import argparse
if hasattr(argparse.ArgumentParser, "_check_help"):
    _orig_check_help = argparse.ArgumentParser._check_help
    def _patched_check_help(self, action):
        if action.help is not None and not isinstance(action.help, str):
            action.help = repr(action.help)
        _orig_check_help(self, action)
    argparse.ArgumentParser._check_help = _patched_check_help


#########
# MODEL #
#########

def load_model_and_tokenizer(model_id):
    """Load model and tokenizer with automatic device placement."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="bfloat16", device_map="auto")
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return tokenizer, model


##########
# PROMPT #
##########

def build_prompts(
    ds: list[dict],
    *,
    reasoning: str,
    dialect: str,
    model_type: str,
    n_few_shot: int,
    tokenizer,
    model_family: str,
    rng=None,
    demo_indices: list[int] | None = None,
) -> tuple[list[str], list[int]]:
    """Build prompt strings for every sample in a dataset.

    Samples demo_indices once (shared across all model types and dialects).
    Demo samples are excluded from ds (in-place) for both base and instruct
    models to ensure a fair comparison.

    For base models, constructs few-shot demos from the demo samples.
    For instruct models, wraps prompts with the tokenizer's chat template.

    Returns (prompts, demo_indices).
    """
    # Sample demo indices once, reuse across dialects and model types
    if demo_indices is None:
        demo_indices = dialecttax.prompts.get_demo_indices(len(ds), n_few_shot, rng=rng)

    # Build demos for base models (before popping)
    if model_type == "base":
        format_demos = dialecttax.prompts.format_prompts_math(dialecttax.prompts.MATH_DEMO[reasoning][dialect])
        demos = dialecttax.prompts.get_demos(ds, format_demos, demo_indices)
    else:
        demos = None

    # Exclude demo samples from ds (pop in reverse order to preserve indices)
    for i in sorted(demo_indices, reverse=True):
        ds.pop(i)

    instructions = dialecttax.prompts.MATH_INST[reasoning][dialect]
    format_prompt = dialecttax.prompts.format_prompts_math(dialecttax.prompts.MATH_PROMPT[reasoning][dialect])
    prompts = [
        dialecttax.prompts.get_prompt(format_prompt(ds, i), demos=demos, instructions=instructions)
        for i in range(len(ds))
    ]

    if model_type == "instruct":
        system_prompt = dialecttax.prompts.get_system_prompt(dialect, reasoning=reasoning, family=model_family)
        enable_thinking = bool(reasoning == "cot")
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
            for prompt in prompts
        ]

    return prompts, demo_indices


##############
# GENERATION #
##############

def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    temperature: float = 1.0,
    do_sample: bool = False,
) -> dict:
    """Generate completions for a list of prompts in batches.

    Left-pads inputs so that the actual tokens are right-aligned,
    which is required for correct causal-LM generation with padding.

    Returns dict with keys:
        completions: list[str] — decoded completion texts.
        input_metrics: list[dict] — per-sample input token metrics.
        generation_metrics: list[dict] — per-sample generation token metrics.
    """
    tokenizer.padding_side = "left"
    all_completions = []
    all_input_metrics = []
    all_generation_metrics = []

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for batch_idx, start in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start:(start + batch_size)]
        log.info("Batch %d/%d (%d samples)", batch_idx + 1, n_batches, len(batch_prompts))
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        # Input metrics (forward pass on prompt)
        batch_input_metrics = dialecttax.logits.compute_input_metrics(
            model, inputs.input_ids, inputs.attention_mask,
        )
        all_input_metrics.extend(batch_input_metrics)

        # Generation with scores
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode completions
        completions = tokenizer.batch_decode(
            outputs.sequences[:, input_len:], skip_special_tokens=True,
        )
        all_completions.extend(completions)

        # Generation metrics from scores
        batch_gen_metrics = dialecttax.logits.compute_generation_metrics(
            outputs.scores,
            outputs.sequences,
            prompt_len=input_len,
            eos_token_id=tokenizer.eos_token_id,
        )
        all_generation_metrics.extend(batch_gen_metrics)

        # Free batch memory
        del inputs, outputs
        torch.cuda.empty_cache()

    results = {
        "completions": all_completions,
        "input_metrics": all_input_metrics,
        "generation_metrics": all_generation_metrics,
    }
    return results


##################
# CLI
##################

@hydra.main(version_base=None, config_path="../../configs/benchmark_redial", config_name="config")
def main(cfg: DictConfig) -> None:
    # Clean up memory from previous Hydra sweep job
    gc.collect()
    torch.cuda.empty_cache()

    # Skip if all output files already exist (unless rerun=true)
    output_dir = HydraConfig.get().runtime.output_dir
    expected_files = [f"results_{d.name}.jsonl" for d in cfg.dialects]
    if not cfg.rerun and all(
        os.path.exists(os.path.join(output_dir, f)) for f in expected_files
    ):
        log.info("Skipping %s/%s — outputs already exist in %s",
                 cfg.model.name, cfg.reasoning.name, output_dir)
        return

    # Set NumPy seed to ensure the same demos for few-shot prompting are discarded
    rng = np.random.default_rng(cfg.seed)

    # Load configs
    project_config = dialecttax.utils.load_config(cfg.project_config)
    reasoning = cfg.reasoning.name

    # Load model
    log.info("Loading model: %s (%s)", cfg.model.name, cfg.model.hf_id)
    tokenizer, model = load_model_and_tokenizer(cfg.model.hf_id)

    # Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    demo_indices = None
    for dialect_cfg in cfg.dialects:
        dialect = dialect_cfg.name
        log.info("Processing dialect: %s", dialect)
        ds = dialecttax.data.redial.load_dataset(project_config["directories"]["preprocessed"], dialect_cfg.path_file)
        log.info("Loaded %d samples", len(ds))

        # Prompts (build_prompts pops demo samples from ds in-place)
        prompts, demo_indices = build_prompts(
            ds,
            reasoning=reasoning,
            dialect=dialect,
            model_type=cfg.model.type,
            n_few_shot=cfg.n_few_shot,
            tokenizer=tokenizer,
            model_family=cfg.model.family,
            rng=rng,
            demo_indices=demo_indices,
        )
        log.info("Evaluating %d samples (%d excluded as demos)", len(ds), len(demo_indices))

        # Generate + collect token-level metrics
        gen_output = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=cfg.reasoning.max_new_tokens,
            batch_size=cfg.batch_size,
            temperature=cfg.temperature,
        )
        completions = gen_output["completions"]

        # Grade
        labels = [sample["answer"] for sample in ds]
        grading_results = dialecttax.data.graders.math.grade_completions(completions, labels)

        # Per-sample metrics
        sample_metrics = [
            dialecttax.logits.compute_sample_metrics(
                grading_results[i],
                gen_output["input_metrics"][i],
                gen_output["generation_metrics"][i],
                completions[i],
            )
            for i in range(len(completions))
        ]

        # Run-level metrics
        run_metrics = dialecttax.logits.compute_run_metrics(sample_metrics)
        run_metrics["model"] = cfg.model.name
        run_metrics["dialect"] = dialect
        run_metrics["reasoning"] = reasoning

        # Save per-sample results as JSONL
        sample_path = os.path.join(output_dir, f"results_{dialect}.jsonl")
        with open(sample_path, "w") as f:
            for s in sample_metrics:
                f.write(json.dumps(s, default=str) + "\n")

        # Save run-level summary
        summary_path = os.path.join(output_dir, f"summary_{dialect}.json")
        with open(summary_path, "w") as f:
            json.dump(run_metrics, f, indent=2, default=str)

        # Log results
        log.info("[%s] %s / %s", dialect, cfg.model.name, reasoning)
        log.info("  accuracy:           %.3f", run_metrics["accuracy"])
        log.info("  acceptance_rate:    %.3f", run_metrics["acceptance_rate"])
        log.info("  mean_resp_len:      %.1f", run_metrics["mean_response_length"])
        log.info("  mean_gen_log_prob:  %.3f", run_metrics["mean_gen_log_prob"])
        log.info("  mean_gen_entropy:   %.3f", run_metrics["mean_gen_entropy"])
        log.info("  mean_inp_log_prob:  %.3f", run_metrics["mean_input_log_prob"])
        log.info("  mean_inp_entropy:   %.3f", run_metrics["mean_input_entropy"])
        log.info("  Saved: %s, %s", sample_path, summary_path)

        # Free dialect iteration memory
        del ds, prompts, gen_output, completions, labels, grading_results, sample_metrics
        torch.cuda.empty_cache()

    # Free model memory before Hydra launches next job
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
