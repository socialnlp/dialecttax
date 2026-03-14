#!/usr/bin/env python3
"""MCMC Power Sampling for LLMs on the ReDial dialect fairness dataset.

Implements the power sampling method from:
  "Reasoning with Sampling: Your Base Model is Smarter Than You Think"
  (arXiv:2510.14901, Karan & Du 2025)

The method samples from p(x)^alpha using Metropolis-Hastings MCMC, where
proposals resample from a random position onward.  This "sharpens" the
model distribution to concentrate on higher-likelihood sequences.

Reference implementation (toy):
  https://github.com/aakaran/reasoning-with-sampling/blob/main/toy_composition.py

Dataset: https://huggingface.co/datasets/fangrulin/redial

Base (pretrained) models are used, not instruction-tuned variants, since
the paper's key claim is that base LMs already encode reasoning ability and
MCMC power sampling can surface it without any fine-tuning.

Usage:
    # MCMC power sampling (default)
    python scripts/mcmc.py --model llama-3.2-3b --max-samples 20

    # Greedy baseline for comparison
    python scripts/mcmc.py --model llama-3.2-3b --method greedy --max-samples 20

    # Tune MCMC hyperparameters
    python scripts/mcmc.py --model qwen2.5-7b --alpha 4.0 --n-mcmc 20

    # Specific splits
    python scripts/mcmc.py --splits math_vanilla_original math_vanilla_aave
"""

import argparse
import json
import math
import random
import re
import socket
import time
import os

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "fangrulin/redial"

DEFAULT_SPLITS = [
    "math_vanilla_original",
    "math_vanilla_aave",
    "comprehensive_vanilla_original",
    "comprehensive_vanilla_aave",
    "logic_vanilla_original",
    "logic_vanilla_aave",
    "algorithm_vanilla_original",
    "algorithm_vanilla_aave",
]

MODELS = {
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
    "qwen2.5-3b": "Qwen/Qwen2.5-3B",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B",
    "gemma-2-2b": "google/gemma-2-2b",
}

N_FEW_SHOT = 5


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_reasoning_dataset(splits: list[str]) -> dict[str, list[dict]]:
    """Load the reasoning-formatted ReDial dataset from local JSON files.

    Reads from {datasets_dir}/ReDial/{split}.json where datasets_dir is
    resolved from configs/default.yaml.
    """
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    ds_dir = os.path.join(
        cfg["directories"]["datasets"].format(hostname=socket.gethostname()),
        "ReDial",
    )

    dataset = {}
    for split in splits:
        path = os.path.join(ds_dir, f"{split}.json")
        if not os.path.exists(path):
            print(f"  Warning: {path} not found, skipping")
            continue
        with open(path) as f:
            dataset[split] = json.load(f)
    return dataset


# ---------------------------------------------------------------------------
# Answer extraction (shared logic with redial.py)
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Extract numerical answer from model output.

    Tries patterns in order of specificity:
      1. <answer>...</answer> tags (format requested by ReDial prompts)
      2. #### <number>
      3. "the answer is <number>"
      4. Last number in the response (fallback)
    """
    # Pattern 1: <answer> tags (what ReDial questions ask for)
    match = re.search(r"<answer>\s*([+-]?\d+(?:[.,]\d+)?)\s*</answer>", text)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 2: #### marker
    match = re.search(r"####\s*([+-]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Pattern 3: "the answer is ..."
    match = re.search(
        r"(?:the answer is|answer:)\s*[:\s]*([+-]?\d+(?:[.,]\d+)?)",
        text,
        re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "")

    # Pattern 4: last number in text
    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
    return numbers[-1] if numbers else None


def normalize_answer(answer: str) -> str:
    """Normalize answer string for comparison."""
    answer = answer.strip().replace(",", "")
    try:
        val = float(answer)
        return str(int(val)) if val == int(val) else str(val)
    except ValueError:
        return answer


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_model(
    model_id: str, dtype: torch.dtype = torch.bfloat16
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer with automatic device placement."""
    print(f"  Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def build_prompt_ids(
    question: str, demos: list[dict], tokenizer, device: torch.device
) -> torch.Tensor:
    """Build a 5-shot prompt for a base model.

    Each demo is formatted as: question text + " <answer>GOLD</answer>\\n\\n"
    The final question is appended without an answer.
    """
    parts = []
    for demo in demos:
        parts.append(f"{demo['question']} <answer>{demo['answer']}</answer>\n\n")
    parts.append(question)
    prompt = "".join(parts)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


# ---------------------------------------------------------------------------
# MCMC Power Sampling – core
# ---------------------------------------------------------------------------

def _get_stop_ids(tokenizer) -> list[int]:
    """Return token ids for the ``</answer`` prefix.

    We match on ``</answer`` (without the trailing ``>``) because BPE
    may merge ``>`` with whatever character follows (e.g. ``>\\n`` becomes
    a single token), making an exact ``</answer>`` match unreliable.
    The two-token prefix is stable across contexts.
    """
    return tokenizer.encode("</answer", add_special_tokens=False)


def _ends_with_stop(gen_tokens: list[int], stop_ids: list[int]) -> bool:
    """Check whether *gen_tokens* ends with the *stop_ids* sequence."""
    n = len(stop_ids)
    return len(gen_tokens) >= n and gen_tokens[-n:] == stop_ids


@torch.no_grad()
def _naive_temp(
    model,
    context: list[int],
    prompt_kv,
    prompt_logits: torch.Tensor,
    prompt_len: int,
    *,
    temp: float,
    seq_len: int,
    eos_token_id: int,
    stop_ids: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], list[float], list[float]]:
    """Generate tokens to reach total *seq_len*, matching the reference's naive_temp.

    Samples at temperature *temp* (where alpha = 1/temp).  Returns the full
    token sequence (context + new) and per-new-token log probs:
      - log_probs_norm:   log q(token)  (proposal, temperature-scaled)
      - log_probs_unnorm: (1/temp) * log p(token)  (target, pre-scaled by alpha)
    """
    n_new = seq_len - len(context)
    if n_new <= 0:
        return context, [], []

    # Lightweight wrapper — shares tensor data with prompt_kv but torch.cat
    # in update() creates new tensors, so the original is never mutated.
    kv = _wrap_kv_cache(prompt_kv)

    # Build KV for context by extending prompt KV with generated tokens
    gen_offset = len(context) - prompt_len  # how many tokens past prompt
    if gen_offset > 0:
        extra = torch.tensor(
            [context[prompt_len:]], device=device, dtype=dtype,
        )
        with torch.no_grad():
            ext_out = model(extra, past_key_values=kv, use_cache=True)
        past_kv = ext_out.past_key_values
        logits = ext_out.logits[:, -1, :]
    else:
        past_kv = kv
        logits = prompt_logits

    gen_tokens: list[int] = []
    log_probs_norm: list[float] = []    # log q(token)
    log_probs_unnorm: list[float] = []  # (1/temp) * log p(token)

    for _ in range(n_new):
        log_p = torch.log_softmax(logits, dim=-1)

        if temp != 1.0:
            log_q = torch.log_softmax(logits / temp, dim=-1)
        else:
            log_q = log_p

        probs = log_q.exp().clamp(min=1e-10)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        token = torch.multinomial(probs[0], 1)

        tid = token.item()
        log_probs_unnorm.append((1.0 / temp) * log_p[0, tid].item())
        log_probs_norm.append(log_q[0, tid].item())
        gen_tokens.append(tid)

        # Check EOS — reference lets model.generate handle this, so the
        # proposal can be shorter than seq_len (matching reference behavior)
        if tid == eos_token_id or _ends_with_stop(gen_tokens, stop_ids):
            break

        with torch.no_grad():
            outputs = model(
                token.unsqueeze(0), past_key_values=past_kv, use_cache=True,
            )
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    full_seq = context + gen_tokens
    return full_seq, log_probs_norm, log_probs_unnorm


@torch.no_grad()
def mcmc_power_sample(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    alpha: float = 2.0,
    n_mcmc: int = 10,
    max_new_tokens: int = 128,
    block_num: int = 16,
    proposal_temp: float | None = None,
    prompt_kv_cache: tuple | None = None,
    prompt_logits_cache: torch.Tensor | None = None,
) -> tuple[str, float]:
    """Sample from p(x)^alpha via block-based Metropolis-Hastings MCMC.

    Follows the reference implementation (aakaran/reasoning-with-sampling):
      1. Generation is split into *block_num* blocks of *jump_size* tokens.
      2. For each block:
         a. Generate *jump_size* new tokens at temperature tau = 1/alpha.
         b. Run *n_mcmc* MH refinement steps on the *entire* sequence so far.
            Each step picks a random position, regenerates from there to the
            current end, and accepts/rejects with the standard MH ratio.
      3. After each block, check for EOS and stop early if found.

    This interleaving lets refined early tokens influence later generation.

    If *prompt_kv_cache* and *prompt_logits_cache* are provided, they are
    reused instead of recomputing the prompt forward pass.

    Returns (decoded_text, acceptance_rate).
    """
    if proposal_temp is None:
        proposal_temp = 1.0 / alpha

    prompt_len = prompt_ids.shape[1]
    device = prompt_ids.device
    dtype = prompt_ids.dtype
    eos_id = tokenizer.eos_token_id
    stop_ids = _get_stop_ids(tokenizer)

    # Adjust block_num so jump_size is integral
    if max_new_tokens < block_num:
        block_num = max(1, max_new_tokens)
    jump_size = max_new_tokens // block_num

    # -- prompt KV cache (reuse if provided) ---------------------------------
    if prompt_kv_cache is not None and prompt_logits_cache is not None:
        prompt_kv = prompt_kv_cache
        prompt_logits = prompt_logits_cache
    else:
        with torch.no_grad():
            prompt_out = model(prompt_ids, use_cache=True)
        prompt_kv = prompt_out.past_key_values
        prompt_logits = prompt_out.logits[:, -1, :]

    # gen = full token sequence (prompt + generated)
    gen = prompt_ids[0].tolist()
    log_probs_norm: list[float] = []    # log q(token) per generated token
    log_probs_unnorm: list[float] = []  # (1/temp) * log p(token)

    n_attempts = 0
    n_accepted = 0

    for _block in range(block_num):
        # -- Generate jump_size new tokens -----------------------------------
        target_len = len(gen) + jump_size
        gen, lp_norm, lp_unnorm = _naive_temp(
            model, gen, prompt_kv, prompt_logits, prompt_len,
            temp=proposal_temp,
            seq_len=target_len,
            eos_token_id=eos_id,
            stop_ids=stop_ids,
            device=device,
            dtype=dtype,
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)

        # -- MCMC refinement on entire sequence so far -----------------------
        c = prompt_len  # context length (prompt)
        for _step in range(n_mcmc):
            n_attempts += 1
            t = len(gen)
            idx = random.randint(c, t - 1)

            # Propose: regenerate from position idx to current total length
            prop, lp_prop_norm, lp_prop_unnorm = _naive_temp(
                model, gen[:idx], prompt_kv, prompt_logits, prompt_len,
                temp=proposal_temp,
                seq_len=t,
                eos_token_id=eos_id,
                stop_ids=stop_ids,
                device=device,
                dtype=dtype,
            )
            s = len(prop)

            # Current log probs for the suffix being replaced
            # idx-c maps generated-token index, s-c is proposal gen length
            lp_cur_norm = log_probs_norm[idx - c : s - c]
            lp_cur_unnorm = log_probs_unnorm[idx - c : s - c]

            # MH ratio:
            #   log r = log pi(x') + log q(x|x') - log pi(x) - log q(x'|x)
            # where log pi = (1/temp) * log p = alpha * log p (pre-scaled)
            log_r = (
                sum(lp_prop_unnorm)
                + sum(lp_cur_norm)
                - sum(lp_cur_unnorm)
                - sum(lp_prop_norm)
            )

            if log_r >= 0 or np.random.rand() < np.exp(log_r):
                n_accepted += 1
                gen = prop
                log_probs_norm[idx - c :] = lp_prop_norm
                log_probs_unnorm[idx - c :] = lp_prop_unnorm

        # -- Check for EOS after this block ----------------------------------
        if eos_id in gen[prompt_len:]:
            eos_idx = gen.index(eos_id, prompt_len)
            gen = gen[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1 - prompt_len]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1 - prompt_len]
            break

    accept_rate = n_accepted / n_attempts if n_attempts > 0 else 0.0
    full_ids = torch.tensor([gen], device=device, dtype=dtype)
    return _decode_generated(full_ids, prompt_len, eos_id, stop_ids, tokenizer), accept_rate


def _decode_generated(
    ids: torch.Tensor,
    prompt_len: int,
    eos_id: int,
    stop_ids: list[int],
    tokenizer,
) -> str:
    """Decode only the generated portion, truncating at EOS or </answer>."""
    new_toks = ids[0, prompt_len:]

    # Truncate at first EOS
    eos_pos = (new_toks == eos_id).nonzero(as_tuple=True)[0]
    if len(eos_pos) > 0:
        new_toks = new_toks[: eos_pos[0]]

    # Truncate after first </answer (keep it for answer extraction;
    # include one extra token for the closing ">")
    tok_list = new_toks.tolist()
    n = len(stop_ids)
    for i in range(n, len(tok_list) + 1):
        if tok_list[i - n : i] == stop_ids:
            # keep up to one token past the match (the ">" / ">\n")
            end = min(i + 1, len(tok_list))
            new_toks = new_toks[:end]
            break

    return tokenizer.decode(new_toks, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def _clone_kv_cache(kv_cache):
    """Clone a DynamicCache by copying its tensors."""
    from transformers.cache_utils import DynamicCache
    clone = DynamicCache()
    for layer in kv_cache.layers:
        clone.update(layer.keys.clone(), layer.values.clone(), len(clone.layers))
    return clone


def _wrap_kv_cache(kv_cache):
    """Create a lightweight DynamicCache wrapper sharing the same tensor data.

    When the model extends the wrapper (via update → torch.cat), it creates
    new tensors, leaving the original cache unchanged.  This is O(n_layers)
    reference assignments with zero tensor copying — much cheaper than clone.
    """
    from transformers.cache_utils import DynamicCache
    wrapper = DynamicCache()
    for layer in kv_cache.layers:
        wrapper.update(layer.keys, layer.values, len(wrapper.layers))
    return wrapper


# ---------------------------------------------------------------------------
# Greedy baseline (with entropy collection)
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_generate(
    model,
    prompt_kv,
    first_logits: torch.Tensor,
    *,
    max_new_tokens: int = 128,
    eos_token_id: int,
    stop_ids: list[int],
) -> tuple[list[int], list[float]]:
    """Greedy decoding with per-position entropy collection.

    Returns (generated_token_ids, entropy_profile).
    """
    gen_tokens: list[int] = []
    entropies: list[float] = []
    logits = first_logits.clone()
    past_kv = prompt_kv

    for _ in range(max_new_tokens):
        logits_f32 = logits.float()
        log_p = torch.log_softmax(logits_f32, dim=-1)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=-1).item()
        entropies.append(entropy)

        token_id = logits.argmax(dim=-1).item()
        gen_tokens.append(token_id)

        if token_id == eos_token_id or _ends_with_stop(gen_tokens, stop_ids):
            break

        token_tensor = torch.tensor([[token_id]], device=first_logits.device)
        outputs = model(token_tensor, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    return gen_tokens, entropies


# ---------------------------------------------------------------------------
# Batched temperature sampling (for pass@k)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_batch_samples(
    model,
    prompt_kv,
    first_logits: torch.Tensor,
    *,
    n_samples: int,
    temperature: float = 0.7,
    max_new_tokens: int = 128,
    eos_token_id: int,
    stop_ids: list[int],
) -> list[list[int]]:
    """Generate n_samples completions in parallel from a shared prompt KV cache."""
    device = first_logits.device
    n_stop = len(stop_ids)

    batch_kv = _clone_kv_cache(prompt_kv)
    batch_kv.batch_repeat_interleave(n_samples)
    logits = first_logits.expand(n_samples, -1)

    generated = torch.zeros(n_samples, max_new_tokens, dtype=torch.long, device=device)
    lengths = torch.full((n_samples,), max_new_tokens, dtype=torch.long, device=device)
    done = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        probs = torch.softmax(logits / temperature, dim=-1)
        probs = probs.clamp(min=1e-10)
        tokens = torch.multinomial(probs, 1)
        generated[:, t] = tokens.squeeze(1)

        for i in range(n_samples):
            if done[i]:
                continue
            tid = tokens[i].item()
            if tid == eos_token_id:
                lengths[i] = t + 1
                done[i] = True
            elif t + 1 >= n_stop:
                seq = generated[i, t + 1 - n_stop : t + 1].tolist()
                if seq == stop_ids:
                    lengths[i] = t + 1
                    done[i] = True

        if done.all():
            break

        if t < max_new_tokens - 1:
            outputs = model(tokens, past_key_values=batch_kv, use_cache=True)
            batch_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    return [generated[i, : lengths[i]].tolist() for i in range(n_samples)]


# ---------------------------------------------------------------------------
# pass@k estimator
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k)."""
    if n < k:
        return float(c > 0)
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    log_ratio = 0.0
    for i in range(k):
        if n - c - i <= 0:
            return 1.0
        log_ratio += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_ratio)


PASS_AT_K_VALUES = [1, 5, 10, 25, 50]


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_split(
    model,
    tokenizer,
    model_name: str,
    split_name: str,
    dataset,
    *,
    method: str,
    alpha: float,
    n_mcmc: int,
    block_num: int = 16,
    proposal_temp: float | None = None,
    n_samples: int = 50,
    n_mcmc_samples: int = 8,
    temperature: float = 0.7,
    max_samples: int | None = None,
    max_new_tokens: int = 128,
) -> dict:
    """Evaluate one dataset split and return accuracy + per-example details."""
    data = dataset[split_name]
    stop_ids = _get_stop_ids(tokenizer)
    eos_id = tokenizer.eos_token_id

    # First N_FEW_SHOT examples are reserved as demos, evaluate on the rest.
    demos = [data[i] for i in range(N_FEW_SHOT)]
    eval_start = N_FEW_SHOT
    if max_samples:
        eval_end = min(eval_start + max_samples, len(data))
    else:
        eval_end = len(data)
    eval_indices = list(range(eval_start, eval_end))

    correct = 0
    total = len(eval_indices)
    results: list[dict] = []
    accept_rates: list[float] = []
    all_entropies: list[list[float]] = []
    device = next(model.parameters()).device

    for qi, idx in enumerate(eval_indices):
        example = data[idx]
        question = example["question"]
        gold = normalize_answer(example["answer"])
        prompt_ids = build_prompt_ids(question, demos, tokenizer, device)

        # Compute prompt KV cache once, reuse for greedy + batch samples
        prompt_out = model(prompt_ids, use_cache=True)
        prompt_kv = prompt_out.past_key_values
        first_logits = prompt_out.logits[:, -1, :]

        # 1. Greedy pass (entropy collection)
        greedy_tokens, entropy_profile = greedy_generate(
            model,
            _clone_kv_cache(prompt_kv),
            first_logits,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            stop_ids=stop_ids,
        )
        greedy_text = _decode_generated(
            torch.cat([prompt_ids, torch.tensor([greedy_tokens], device=device, dtype=prompt_ids.dtype)], dim=1),
            prompt_ids.shape[1], eos_id, stop_ids, tokenizer,
        )
        if entropy_profile:
            all_entropies.append(entropy_profile)

        # 2. Batched temperature samples (for pass@k)
        sample_token_lists = generate_batch_samples(
            model,
            prompt_kv,
            first_logits,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            stop_ids=stop_ids,
        )
        n_correct_samples = 0
        for sample_tokens in sample_token_lists:
            sample_ids = torch.tensor([sample_tokens], device=device, dtype=prompt_ids.dtype)
            full_ids = torch.cat([prompt_ids, sample_ids], dim=1)
            sample_text = _decode_generated(full_ids, prompt_ids.shape[1], eos_id, stop_ids, tokenizer)
            sample_pred = extract_answer(sample_text)
            sample_norm = normalize_answer(sample_pred) if sample_pred else None
            if sample_norm == gold:
                n_correct_samples += 1

        pk = {}
        for k in PASS_AT_K_VALUES:
            if k <= n_samples:
                pk[k] = pass_at_k(n_samples, n_correct_samples, k)

        # 3. MCMC (if selected) — reuses pre-computed prompt KV cache
        mcmc_pk = {}
        n_correct_mcmc = 0
        if method == "mcmc":
            mcmc_responses = []
            mcmc_ars = []
            for _ in range(n_mcmc_samples):
                resp, ar = mcmc_power_sample(
                    model,
                    tokenizer,
                    prompt_ids,
                    alpha=alpha,
                    n_mcmc=n_mcmc,
                    max_new_tokens=max_new_tokens,
                    block_num=block_num,
                    proposal_temp=proposal_temp,
                    prompt_kv_cache=prompt_kv,
                    prompt_logits_cache=first_logits,
                )
                mcmc_responses.append(resp)
                mcmc_ars.append(ar)

            accept_rates.append(float(np.mean(mcmc_ars)))

            # Use first chain as the primary answer
            response = mcmc_responses[0]

            # Count correct across all MCMC chains for pass@k
            for resp in mcmc_responses:
                pred = extract_answer(resp)
                pred_norm = normalize_answer(pred) if pred else None
                if pred_norm == gold:
                    n_correct_mcmc += 1

            for k in PASS_AT_K_VALUES:
                if k <= n_mcmc_samples:
                    mcmc_pk[k] = pass_at_k(n_mcmc_samples, n_correct_mcmc, k)
        else:
            response = greedy_text

        predicted = extract_answer(response)
        predicted_norm = normalize_answer(predicted) if predicted else None
        is_correct = predicted_norm == gold
        if is_correct:
            correct += 1

        results.append(
            {
                "index": idx,
                "question": (
                    question[:120] + "..." if len(question) > 120 else question
                ),
                "gold": gold,
                "predicted": predicted_norm,
                "correct": is_correct,
                "raw_response": response[:300],
                "mean_entropy": float(np.mean(entropy_profile)) if entropy_profile else None,
                "n_correct_samples": n_correct_samples,
                "n_samples": n_samples,
                "pass_at_k": pk,
                "n_correct_mcmc": n_correct_mcmc,
                "n_mcmc_samples": n_mcmc_samples,
                "mcmc_pass_at_k": mcmc_pk,
            }
        )

        acc_s = f"{correct / (qi + 1):.1%}"
        ar_s = (
            f"  accept={np.mean(accept_rates):.2f}" if accept_rates else ""
        )
        print(f"    [{split_name}] {qi + 1}/{total} — acc: {acc_s}{ar_s}")

    accuracy = correct / total if total > 0 else 0.0

    # Average pass@k across questions
    avg_pass_at_k = {}
    for k in PASS_AT_K_VALUES:
        if k <= n_samples:
            vals = [q["pass_at_k"].get(k, 0.0) for q in results]
            avg_pass_at_k[k] = float(np.mean(vals))

    # Entropy profile (per-position average)
    max_entropy_len = max((len(e) for e in all_entropies), default=0)
    entropy_profile_avg = []
    if max_entropy_len > 0:
        for pos in range(max_entropy_len):
            vals = [e[pos] for e in all_entropies if pos < len(e)]
            entropy_profile_avg.append(float(np.mean(vals)))

    mean_entropy = float(np.mean([
        r["mean_entropy"] for r in results if r["mean_entropy"] is not None
    ])) if results else 0.0

    # Average MCMC pass@k across questions
    avg_mcmc_pass_at_k = {}
    if method == "mcmc":
        for k in PASS_AT_K_VALUES:
            if k <= n_mcmc_samples:
                vals = [q["mcmc_pass_at_k"].get(k, 0.0) for q in results]
                avg_mcmc_pass_at_k[k] = float(np.mean(vals))

    out: dict = {
        "split": split_name,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "mean_entropy": mean_entropy,
        "entropy_profile": entropy_profile_avg,
        "pass_at_k": avg_pass_at_k,
        "mcmc_pass_at_k": avg_mcmc_pass_at_k,
        "results": results,
    }
    if accept_rates:
        out["mean_accept_rate"] = float(np.mean(accept_rates))
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print formatted summary tables to stdout."""
    print(f"\n{'=' * 70}")
    print("PER-SPLIT ACCURACY")
    print(f"{'=' * 70}")
    pivot = df.pivot_table(
        index="split", columns="model", values="accuracy", aggfunc="first"
    )
    print(pivot.to_string(float_format="{:.1%}".format))

    # Dialect gap
    print(f"\n{'=' * 70}")
    print("DIALECT GAP  (Original - AAVE)")
    print(f"{'=' * 70}")
    for name in df["model"].unique():
        mdf = df[df["model"] == name]
        orig = mdf[mdf["split"].str.endswith("_original")]
        aave = mdf[mdf["split"].str.endswith("_aave")]
        if orig.empty or aave.empty:
            continue
        o_acc = orig["correct"].sum() / orig["total"].sum()
        a_acc = aave["correct"].sum() / aave["total"].sum()
        gap = o_acc - a_acc
        print(
            f"  {name:<20}: Original={o_acc:.1%}  "
            f"AAVE={a_acc:.1%}  Gap={gap:+.1%}"
        )

    # pass@k (temperature sampling)
    pk_cols = [c for c in df.columns if c.startswith("pass@")]
    if pk_cols:
        print(f"\n{'=' * 70}")
        print("pass@k (temperature sampling)")
        print(f"{'=' * 70}")
        pk_df = df[["model", "split"] + pk_cols].set_index(["model", "split"])
        print(pk_df.to_string(float_format="{:.3f}".format))

    # MCMC pass@k
    mcmc_pk_cols = [c for c in df.columns if c.startswith("mcmc_pass@")]
    if mcmc_pk_cols:
        print(f"\n{'=' * 70}")
        print("pass@k (MCMC chains)")
        print(f"{'=' * 70}")
        mcmc_pk_df = df[["model", "split"] + mcmc_pk_cols].set_index(["model", "split"])
        print(mcmc_pk_df.to_string(float_format="{:.3f}".format))

    # Mean entropy
    if "mean_entropy" in df.columns:
        print(f"\n{'=' * 70}")
        print("MEAN ENTROPY (greedy path)")
        print(f"{'=' * 70}")
        for name in df["model"].unique():
            mdf = df[df["model"] == name]
            me = mdf["mean_entropy"].mean()
            print(f"  {name:<20}: {me:.3f}")

    # MCMC accept rate
    if "accept_rate" in df.columns:
        ar = df.dropna(subset=["accept_rate"])
        if not ar.empty:
            print(f"\n{'=' * 70}")
            print("MCMC ACCEPT RATE")
            print(f"{'=' * 70}")
            for name in ar["model"].unique():
                mdf = ar[ar["model"] == name]
                print(f"  {name:<20}: {mdf['accept_rate'].mean():.2f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MCMC power sampling for LLMs on the ReDial dataset."
    )
    parser.add_argument(
        "--model",
        default="llama-3.2-3b",
        choices=list(MODELS.keys()),
        help="Model to evaluate (default: llama-3.2-3b).",
    )
    parser.add_argument(
        "--method",
        default="mcmc",
        choices=["greedy", "mcmc"],
        help="Generation method (default: mcmc).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Dataset splits to evaluate (default: 8 vanilla splits).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap samples per split (useful for quick testing).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Temperature samples per question for pass@k (default: 50).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for pass@k (default: 0.7).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate per question (default: 128).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mcmc",
        help="Directory for result files (default: results/mcmc).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model weight dtype (default: bf16).",
    )
    # MCMC hyperparameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Sharpening exponent for p(x)^alpha (default: 2.0).",
    )
    parser.add_argument(
        "--n-mcmc",
        type=int,
        default=10,
        help="Number of MCMC refinement steps per block (default: 10).",
    )
    parser.add_argument(
        "--block-num",
        type=int,
        default=16,
        help="Number of generation blocks (default: 16).",
    )
    parser.add_argument(
        "--n-mcmc-samples",
        type=int,
        default=8,
        help="Number of independent MCMC chains per question for pass@k (default: 8).",
    )
    parser.add_argument(
        "--proposal-temp",
        type=float,
        default=None,
        help="Proposal sampling temperature tau (default: 1/alpha).",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Use reasoning-formatted dataset (stripped of instructional preamble).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # ------------------------------------------------------------------
    if args.reasoning:
        print("Loading ReDial reasoning dataset (local) ...")
        dataset = load_reasoning_dataset(args.splits)
    else:
        print("Loading ReDial dataset ...")
        dataset = load_dataset(DATASET_ID)

    model_name = args.model
    model_id = MODELS[model_name]

    pt = args.proposal_temp if args.proposal_temp else 1.0 / args.alpha
    print(f"\n{'=' * 60}")
    print(f"Model:  {model_name}  ({model_id})")
    print(f"Method: {args.method}")
    if args.method == "mcmc":
        print(f"Alpha:  {args.alpha}  |  n_mcmc: {args.n_mcmc}  |  blocks: {args.block_num}  |  tau: {pt:.3f}")
    print(f"{'=' * 60}")

    model, tokenizer = load_model(model_id, dtype=dtype)

    summary_rows: list[dict] = []
    all_results: dict = {}

    for split_name in args.splits:
        if split_name not in dataset:
            print(f"  Skipping unknown split: {split_name}")
            continue

        n = len(dataset[split_name])
        print(f"\n  Evaluating {split_name} ({n} samples) ...")
        t0 = time.time()

        result = evaluate_split(
            model,
            tokenizer,
            model_name,
            split_name,
            dataset,
            method=args.method,
            alpha=args.alpha,
            n_mcmc=args.n_mcmc,
            block_num=args.block_num,
            proposal_temp=args.proposal_temp,
            n_samples=args.n_samples,
            n_mcmc_samples=args.n_mcmc_samples,
            temperature=args.temperature,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0

        all_results[split_name] = result
        row: dict = {
            "model": model_name,
            "method": args.method,
            "split": split_name,
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "mean_entropy": result["mean_entropy"],
            "time_s": round(elapsed, 1),
        }
        for k, v in result["pass_at_k"].items():
            row[f"pass@{k}"] = v
        for k, v in result["mcmc_pass_at_k"].items():
            row[f"mcmc_pass@{k}"] = v
        if "mean_accept_rate" in result:
            row["accept_rate"] = result["mean_accept_rate"]
        if args.method == "mcmc":
            row["alpha"] = args.alpha
            row["n_mcmc"] = args.n_mcmc
        summary_rows.append(row)

        ar_str = ""
        if "mean_accept_rate" in result:
            ar_str = f"  accept={result['mean_accept_rate']:.2f}"
        print(
            f"  => {split_name}: {result['accuracy']:.1%} "
            f"({result['correct']}/{result['total']}) "
            f"in {elapsed:.1f}s{ar_str}"
        )

    # ------------------------------------------------------------------
    # Persist results
    tag = f"{model_name}_{args.method}"
    if args.method == "mcmc":
        tag += f"_a{args.alpha}_m{args.n_mcmc}"

    results_path = os.path.join(output_dir, f"{tag}_detailed.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results -> {results_path}")

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, f"{tag}_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary -> {csv_path}")

    # Entropy profiles
    entropy_profiles = {
        split_name: all_results[split_name]["entropy_profile"]
        for split_name in all_results
    }
    entropy_path = os.path.join(output_dir, f"{tag}_entropy_profiles.json")
    with open(entropy_path, "w") as f:
        json.dump(entropy_profiles, f, indent=2)
    print(f"Entropy profiles -> {entropy_path}")

    # ------------------------------------------------------------------
    print_summary(df)


if __name__ == "__main__":
    main()
