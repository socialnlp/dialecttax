"""
Compute per-document token-level logits, log-probs, and entropy.

For ReDial MQA datasets: generates a predicted answer, grades it, and
extracts input/generation metrics from a single generate() call.
For text-only datasets: runs a forward pass and saves per-token metrics.

Usage:
    python scripts/logits/generate_logits.py
    python scripts/logits/generate_logits.py model=llama_8b_base
    python scripts/logits/generate_logits.py --multirun model=llama_8b_base,llama_3b_base dataset=redial dialect=sae,aave
"""

import argparse
import gc
import json
import re
import logging
import os
import warnings

warnings.filterwarnings("ignore", message="resource_tracker:.*leaked semaphore")

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import dialecttax
import dialecttax.data.graders.mqa as mqa_grader
import dialecttax.logits


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


def _load_model(name, model_id, device="auto", attn_implementation=None):
    """Load a language model, reusing across multirun iterations.

    Args:
        name: Short model name (e.g. "llama_8b_base").
        model_id: HuggingFace model ID.
        device: Device string (default "auto" for multi-GPU via accelerate).
        attn_implementation: Optional override (e.g. "eager"); HF default if None.

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

    log.info(f"Loading model: {name} ({model_id}, device={device}, attn={attn_implementation})")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"dtype": torch.bfloat16, "device_map": device}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _loaded_model = (model, tokenizer)
    _loaded_model_name = name
    return model, tokenizer


########
# DATA #
########

def _build_redial_prompt(ds, i, task, dialect, reasoning="naive"):
    """Build system and user prompt components for a ReDial MQA sample.

    Args:
        ds: Dataset list.
        i: Sample index.
        task: Task name.
        dialect: Dialect name.
        reasoning: Reasoning strategy ("naive" or "cot").

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    instructions = dialecttax.prompts.INSTS_MQA[task][reasoning][dialect]

    # Add choices
    choices = ds[i]["choices"]
    if task == "algorithm":
        choices_str = "\n\n".join(f"{k}.\n```\n{v}\n```" for k, v in choices.items())
    else:
        choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
    instructions = instructions.format(choices=choices_str)

    system = dialecttax.prompts.get_system_prompt(dialect)
    formatter = dialecttax.prompts.FORMAT_PROMPTS_REGISTRY[task]
    template = dialecttax.prompts.PROMPTS[task][reasoning][dialect]
    body = formatter(template)(ds, i)
    user_prompt = dialecttax.prompts.get_prompt(body, instructions=instructions)
    return system, user_prompt


def _build_samples(ds, task, dialect, dataset_name, reasoning="naive"):
    """Build sample dicts for each entry in the dataset.

    For ReDial MQA: returns system prompt, user prompt, and gold answer.
    For text-only datasets (parallelaave, multivalue): returns raw text.

    Args:
        ds: Dataset list of dicts.
        task: Task name.
        dialect: Dialect name.
        dataset_name: Name of the dataset.
        reasoning: Reasoning strategy ("naive" or "cot").

    Returns:
        List of dicts. ReDial samples have keys: unique_id, system, prompt, answer.
        Text-only samples have keys: unique_id, text.
    """
    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    samples = []

    if hasattr(mod, "TASKS"):
        # ReDial MQA: separate system, prompt, and answer for generation
        for i, sample in enumerate(ds):
            system, prompt = _build_redial_prompt(ds, i, task, dialect, reasoning=reasoning)
            samples.append({
                "unique_id": sample["unique_id"],
                "system": system,
                "prompt": prompt,
                "answer": str(sample["answer"]),
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

    Args:
        preprocessed_dir: Root preprocessed directory.
        perturbation_name: Name of perturbation (e.g. "swap-0.05").
        dataset_name: Name of dataset (e.g. "redial").
        task: Task name (e.g. "math").
        dialect: Dialect name (e.g. "sae").

    Returns:
        List of perturbed text strings, or None if file not found.
    """
    # ReDial nests perturbed text by task; flat datasets (MultiVALUE /
    # ParallelAAVE) do not, so try the task-qualified path then fall back.
    candidates = [
        os.path.join(preprocessed_dir, "perturbations", perturbation_name, dataset_name, task, f"{dialect}.jsonl"),
        os.path.join(preprocessed_dir, "perturbations", perturbation_name, dataset_name, f"{dialect}.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return [json.loads(line) for line in f]
    return None


##########
# LOGITS #
##########


def _find_answer_step(predicted, gen_ids, tokenizer):
    """Find the generation step whose token completes '#### <answer>'.

    Args:
        predicted: The extracted answer string (e.g. "B"), or None.
        gen_ids: List of generated token IDs.
        tokenizer: Tokenizer with a decode method.

    Returns:
        The step index (int) of the token that completes the answer, or None.
    """
    if predicted is None:
        return None
    running = ""
    for k in range(len(gen_ids)):
        running += tokenizer.decode([gen_ids[k]])
        if re.search(r"####\s*" + re.escape(predicted), running):
            return k
    return None


def _answer_entropy_at_step(entropies, answer_step):
    """Return the scalar entropy at the extracted answer step, if available."""
    if answer_step is None or answer_step < 0 or answer_step >= len(entropies):
        return None
    value = entropies[answer_step]
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _metadata_has_field(path, field):
    """Return True when every non-empty JSONL row contains ``field``."""
    if not os.path.exists(path):
        return False

    found_row = False
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                found_row = True
                if field not in json.loads(line):
                    return False
    except (OSError, json.JSONDecodeError):
        return False
    return found_row


def compute_all_generate(model, tokenizer, samples, instruct=False, max_tokens_new=256, answer_only=False, batch_size=8, label=""):
    """Generate answers and compute metrics with batched KV-cached generation.

    Uses a single batched forward pass on prompts to compute input logits and
    initialize the KV cache, then generates autoregressively reusing the cache.
    Eliminates the redundant second forward pass of the original implementation.

    Args:
        model: CausalLM in eval mode.
        tokenizer: Corresponding tokenizer.
        samples: List of dicts with unique_id, system, prompt, and answer.
        instruct: If True, format with chat template; otherwise plain text.
        max_tokens_new: Maximum tokens to generate per sample.
        answer_only: If True, only retain answer-step entropy (faster for cot).
            If False, also retain full generation log-probs and entropy (naive).
        batch_size: Number of samples to process simultaneously.
        label: Description string for progress display.

    Returns:
        Dict with keys: input_log_probs, input_entropy, hidden, metadata.
        If answer_only=False, also includes gen_log_probs and gen_entropy.
    """
    input_device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id

    # Tokenize all samples up front
    texts = []
    for sample in samples:
        if instruct:
            messages = dialecttax.models.get_message(sample["prompt"].strip(), system=sample["system"].strip())
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=answer_only)
        else:
            text = f"{sample['system']}\n\n{sample['prompt']}\n"
        texts.append(text)

    all_input_lp = []
    all_input_ent = []
    all_gen_lp = []
    all_gen_ent = []
    all_hidden = []
    metadata = []
    n_correct = 0

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    for batch_start in range(0, len(samples), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_texts = texts[batch_start:batch_end]
        batch_samples = samples[batch_start:batch_end]
        B = len(batch_texts)

        # Tokenize batch with left-padding
        batch_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True)
        input_ids = batch_inputs["input_ids"].to(input_device)
        attention_mask = batch_inputs["attention_mask"].to(input_device)
        seq_len = input_ids.shape[1]
        prompt_lens = attention_mask.sum(dim=1).tolist()
        pad_lens = [seq_len - pl for pl in prompt_lens]

        # Pre-allocate attention mask for full generation length
        full_attn = torch.ones(B, seq_len + max_tokens_new, device=input_device, dtype=attention_mask.dtype)
        full_attn[:, :seq_len] = attention_mask

        #####################
        # PROMPT FORWARD    #
        #####################

        with torch.no_grad():
            prompt_out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
            )

        # Extract per-sample input logits (unpadded)
        for b in range(B):
            padl = pad_lens[b]
            sample_logits = prompt_out.logits[b, padl:seq_len - 1, :].cpu()
            shifted_ids = input_ids[b, padl + 1:seq_len].cpu()
            inp_lp = dialecttax.logits.compute_log_probs(sample_logits, shifted_ids)
            inp_ent = dialecttax.logits.compute_entropy(sample_logits)
            all_input_lp.append(inp_lp.float().numpy())
            all_input_ent.append(inp_ent.float().numpy())

        # First generated token (produced by logits at last prompt position)
        first_logits = prompt_out.logits[:, -1, :]  # (B, V)
        first_hidden = prompt_out.hidden_states[-1][:, -1, :].cpu().float()  # (B, D)
        kv_cache = prompt_out.past_key_values
        del prompt_out

        first_ids = first_logits.argmax(dim=-1)  # (B,)

        # Per-sample generation storage
        gen_ids_batch = [[] for _ in range(B)]
        # step_logits[b][k] = logits that predicted gen_ids[k] (naive only)
        step_logits_batch = [[] for _ in range(B)] if not answer_only else None
        # step_hidden[b][k] = hidden state at position that predicted gen_ids[k]
        step_hidden_batch = [[] for _ in range(B)]
        # step_ent[b][k] = entropy at position that predicted gen_ids[k] (cot only)
        step_ent_batch = [[] for _ in range(B)] if answer_only else None
        finished = [False] * B

        # Store data for first generated token
        for b in range(B):
            tid = first_ids[b].item()
            if eos_id is not None and tid == eos_id:
                finished[b] = True
                continue
            gen_ids_batch[b].append(tid)
            step_hidden_batch[b].append(first_hidden[b].numpy())
            if not answer_only:
                step_logits_batch[b].append(first_logits[b].cpu())
            if answer_only:
                ent_val = dialecttax.logits.compute_entropy(first_logits[b:b + 1].cpu())[0].item()
                step_ent_batch[b].append(ent_val)

        del first_logits, first_hidden

        ##############
        # GENERATION #
        ##############

        cur_ids = first_ids.unsqueeze(1)  # (B, 1)
        gen_step = 1

        for step in range(1, max_tokens_new):
            if all(finished):
                break

            with torch.no_grad():
                step_out = model(
                    input_ids=cur_ids,
                    past_key_values=kv_cache,
                    attention_mask=full_attn[:, :seq_len + gen_step],
                    output_hidden_states=True,
                    use_cache=True,
                )

            kv_cache = step_out.past_key_values
            cur_logits = step_out.logits[:, -1, :]  # (B, V)
            cur_hidden = step_out.hidden_states[-1][:, -1, :].cpu().float()  # (B, D)
            next_ids = cur_logits.argmax(dim=-1)  # (B,)
            del step_out

            for b in range(B):
                if finished[b]:
                    continue
                tid = next_ids[b].item()
                if eos_id is not None and tid == eos_id:
                    finished[b] = True
                    continue
                gen_ids_batch[b].append(tid)
                step_hidden_batch[b].append(cur_hidden[b].numpy())
                if not answer_only:
                    step_logits_batch[b].append(cur_logits[b].cpu())
                if answer_only:
                    ent_val = dialecttax.logits.compute_entropy(cur_logits[b:b + 1].cpu())[0].item()
                    step_ent_batch[b].append(ent_val)

            cur_ids = next_ids.unsqueeze(1)
            gen_step += 1

        del kv_cache
        torch.cuda.empty_cache()

        ################
        # POST-PROCESS #
        ################

        for b in range(B):
            sample = batch_samples[b]
            gen_ids = gen_ids_batch[b]
            n_gen = len(gen_ids)
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Grade
            predicted = mqa_grader.extract_answer(gen_text)
            gold = sample["answer"]
            correct = mqa_grader.grade(predicted, gold)
            if correct:
                n_correct += 1

            # Find answer step (token whose generation completes "#### <answer>")
            answer_step = _find_answer_step(predicted, gen_ids, tokenizer)

            # Hidden state at answer position
            emb = None
            if answer_step is not None and answer_step < len(step_hidden_batch[b]):
                emb = step_hidden_batch[b][answer_step]
            all_hidden.append(emb)

            # Metadata
            idx = batch_start + b
            meta = {
                "unique_id": sample["unique_id"],
                "n_prompt_tokens": prompt_lens[b],
                "n_gen_tokens": n_gen,
                "gold_answer": gold,
                "predicted_answer": predicted,
                "correct": correct,
                "completion": gen_text,
                "answer_step": answer_step,
                "input_mean_log_prob": float(all_input_lp[idx].mean()) if len(all_input_lp[idx]) > 0 else 0.0,
                "input_mean_entropy": float(all_input_ent[idx].mean()) if len(all_input_ent[idx]) > 0 else 0.0,
            }

            if answer_only:
                answer_ent = _answer_entropy_at_step(step_ent_batch[b], answer_step)
            else:
                if n_gen > 0 and step_logits_batch[b]:
                    stacked = torch.stack(step_logits_batch[b][:n_gen], dim=0)
                    ids_t = torch.tensor(gen_ids[:n_gen])
                    gen_lp = dialecttax.logits.compute_log_probs(stacked, ids_t)
                    gen_ent = dialecttax.logits.compute_entropy(stacked)
                else:
                    gen_lp = torch.tensor([])
                    gen_ent = torch.tensor([])
                all_gen_lp.append(gen_lp.float().numpy())
                all_gen_ent.append(gen_ent.float().numpy())
                meta["gen_mean_log_prob"] = float(gen_lp.mean()) if len(gen_lp) > 0 else 0.0
                meta["gen_mean_entropy"] = float(gen_ent.mean()) if len(gen_ent) > 0 else 0.0
                answer_ent = _answer_entropy_at_step(gen_ent, answer_step)

            meta["answer_entropy"] = answer_ent

            metadata.append(meta)
            acc = n_correct / (idx + 1) * 100
            print(f"\r  ({label}) {idx + 1}/{len(samples)} acc={acc:.1f}% pred={predicted} gold={gold}", end="", flush=True)

    print()
    tokenizer.padding_side = orig_padding_side

    result = {
        "input_log_probs": all_input_lp,
        "input_entropy": all_input_ent,
        "hidden": all_hidden,
        "metadata": metadata,
    }
    if not answer_only:
        result["gen_log_probs"] = all_gen_lp
        result["gen_entropy"] = all_gen_ent
    return result


def compute_all_logits(model, tokenizer, samples, label=""):
    """Compute per-token log-probs and entropy for all samples.

    Used for text-only datasets (no generation/grading).

    Args:
        model: CausalLM in eval mode.
        tokenizer: Corresponding tokenizer.
        samples: List of dicts with unique_id and text.
        label: Description string for progress display.

    Returns:
        Dict with keys: log_probs, entropy, metadata.
    """
    input_device = next(model.parameters()).device
    all_log_probs = []
    all_entropy = []
    metadata = []

    for i, sample in enumerate(samples):
        inputs = tokenizer(sample["text"], return_tensors="pt")
        input_ids = inputs["input_ids"].to(input_device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids)

        logits = outputs.logits[0]  # (seq_len, vocab)

        # Shift: logits at position t predict token at position t+1
        shifted_logits = logits[:-1]  # (seq_len-1, vocab)
        shifted_ids = input_ids[0, 1:]  # (seq_len-1,)

        lp = dialecttax.logits.compute_log_probs(shifted_logits, shifted_ids)
        ent = dialecttax.logits.compute_entropy(shifted_logits)

        all_log_probs.append(lp.cpu().float().numpy())
        all_entropy.append(ent.cpu().float().numpy())
        metadata.append({
            "unique_id": sample["unique_id"],
            "n_tokens": input_ids.shape[1],
            "mean_log_prob": lp.mean().item(),
            "mean_entropy": ent.mean().item(),
        })
        print(f"\r  ({label}) {i + 1}/{len(samples)}", end="", flush=True)
    print()

    return {
        "log_probs": all_log_probs,
        "entropy": all_entropy,
        "metadata": metadata,
    }


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_logits", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    task = cfg.task.name
    dialect = cfg.dialect.name
    reasoning = cfg.reasoning.name
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

    # Perturbations only apply to SAE text
    if perturbation_fn is not None and dialect != "sae":
        log.info(f"Skipping (perturbation '{perturbation_name}' only applies to dialect 'sae', got '{dialect}')")
        return

    if perturbation_fn is not None:
        experiments_dir = _project_config["directories"]["experiments"]
        if "tasks" in cfg.dataset:
            # ReDial: nest under task/reasoning, mirroring the dialect outputs.
            out_dir = os.path.join(
                experiments_dir, "generate_logits", cfg.model.name, dataset_name, task, reasoning, "sae", "perturbed", perturbation_name,
            )
        else:
            # Flat datasets (MultiVALUE / ParallelAAVE): no task/reasoning nesting.
            out_dir = os.path.join(
                experiments_dir, "generate_logits", cfg.model.name, dataset_name, "sae", "perturbed", perturbation_name,
            )
    else:
        out_dir = HydraConfig.get().runtime.output_dir

    # Skip if already computed — check all expected outputs for this mode.
    forward_files = ["input_log_probs.npz", "input_entropy.npz", "metadata.jsonl"]
    generate_files = [
        "input_log_probs.npz", "input_entropy.npz",
        "gen_log_probs.npz", "gen_entropy.npz",
        "hidden.npz", "metadata.jsonl",
    ]
    generate_answer_only_files = [
        "input_log_probs.npz", "input_entropy.npz",
        "hidden.npz", "metadata.jsonl",
    ]
    metadata_path = os.path.join(out_dir, "metadata.jsonl")
    is_generate_dataset = "tasks" in cfg.dataset
    if is_generate_dataset:
        expected_files = generate_answer_only_files if reasoning == "cot" else generate_files
        outputs_complete = (
            all(os.path.exists(os.path.join(out_dir, f)) for f in expected_files)
            and _metadata_has_field(metadata_path, "answer_entropy")
        )
    else:
        outputs_complete = all(
            os.path.exists(os.path.join(out_dir, f)) for f in forward_files
        )
    if not cfg.rerun and outputs_complete:
        log.info(f"Skipping (outputs exist, rerun=false): {out_dir}")
        return

    log.info(
        f"\n{'=' * 60}\n"
        f"  Model:          {cfg.model.name} ({cfg.model.model_id})\n"
        f"  Dataset:        {dataset_name}\n"
        f"  Task:           {task}\n"
        f"  Reasoning:      {reasoning}\n"
        f"  Dialect:        {dialect}\n"
        f"  Perturbation:   {perturbation_name}\n"
        f"{'=' * 60}"
    )

    # Load dataset
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
        # Flat datasets store text under the dataset's row_text ("text"); ReDial
        # reads the task-specific field, so fall back to cfg.task.row_text.
        row_text = cfg.dataset.get("row_text") or cfg.task.row_text
        for i, text in enumerate(perturbed_texts):
            ds[i][row_text] = text
        log.info(f"Loaded {len(perturbed_texts)} perturbed texts ({perturbation_name})")

    # Build samples
    samples = _build_samples(ds, task, dialect, dataset_name, reasoning=reasoning)
    log.info(f"Built {len(samples)} samples for logits computation")

    # Load model
    try:
        model, tokenizer = _load_model(
            cfg.model.name, cfg.model.model_id, device=cfg.device,
            attn_implementation=cfg.get("attn_implementation"),
        )
    except OSError as e:
        log.warning(f"Skipping model '{cfg.model.name}' ({cfg.model.model_id}): {e}")
        return

    # Compute logits
    label = f"{dataset_name}/{task}/{reasoning}/{dialect}"
    if perturbation_fn is not None:
        label += f"/{perturbation_name}"
    generate_mode = "prompt" in samples[0]

    if generate_mode:
        instruct = cfg.model.name.endswith("_instruct")
        # Reasoning default, then model-level override (e.g. Qwen cot → 2048)
        variant = "instruct" if instruct else "base"
        max_tokens_new = int(cfg.reasoning.max_tokens_new[variant])
        if "max_tokens_new" in cfg.model and reasoning in cfg.model.max_tokens_new:
            max_tokens_new = int(cfg.model.max_tokens_new[reasoning])
        answer_only = (reasoning == "cot")
        results = compute_all_generate(model, tokenizer, samples, instruct=instruct, max_tokens_new=max_tokens_new, answer_only=answer_only, batch_size=int(cfg.batch_size), label=label)
    else:
        results = compute_all_logits(model, tokenizer, samples, label=label)

    # Save outputs
    os.makedirs(out_dir, exist_ok=True)

    if generate_mode:
        np.savez(os.path.join(out_dir, "input_log_probs.npz"), *results["input_log_probs"])
        np.savez(os.path.join(out_dir, "input_entropy.npz"), *results["input_entropy"])
        if "gen_log_probs" in results:
            np.savez(os.path.join(out_dir, "gen_log_probs.npz"), *results["gen_log_probs"])
            np.savez(os.path.join(out_dir, "gen_entropy.npz"), *results["gen_entropy"])
        # Save hidden as .npz since some entries may be None (no valid answer)
        valid_hidden = {str(i): h for i, h in enumerate(results["hidden"]) if h is not None}
        np.savez(os.path.join(out_dir, "hidden.npz"), **valid_hidden)
        log.info(f"Saved {len(valid_hidden)}/{len(samples)} hidden vectors to: {out_dir}/hidden.npz")
    else:
        np.savez(os.path.join(out_dir, "input_log_probs.npz"), *results["log_probs"])
        np.savez(os.path.join(out_dir, "input_entropy.npz"), *results["entropy"])

    metadata = results["metadata"]
    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    log.info(f"Saved {len(samples)} metadata entries to: {out_dir}/metadata.jsonl")
    if generate_mode:
        n_correct = sum(1 for m in metadata if m["correct"])
        log.info(f"Accuracy: {n_correct}/{len(metadata)} ({n_correct / len(metadata) * 100:.1f}%)")
    log.info(f"Done: dataset={dataset_name}, task={task}, dialect={dialect}, n_samples={len(samples)}")


if __name__ == "__main__":
    main()
