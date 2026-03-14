#!/usr/bin/env python3
"""Entropy-reweighted sampling for LLMs on the ReDial dialect fairness dataset.

Generates K candidate responses per question, computes per-token entropy
during generation, then selects the candidate with probability proportional
to exp(-beta * H(x)).  Optionally uses a hybrid score that also includes
the sequence log-likelihood: w(x) ∝ exp(alpha * log π(x) - beta * H(x)).

Based on src/dialecttax/sampling.py, adapted for ReDial evaluation with
5-shot prompting, answer extraction, and pass@k metrics.

Usage:
    # Default: Llama 3B base, all 8 splits
    python scripts/entropy_sampling.py

    # Quick test
    python scripts/entropy_sampling.py --max-samples 2 --K 4

    # Tune entropy parameters
    python scripts/entropy_sampling.py --beta 2.0 --entropy-window last_fraction --window-fraction 0.3
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
import torch.nn.functional as F
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

PASS_AT_K_VALUES = [1, 5, 10, 25, 50]


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
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """Extract numerical answer from model output."""
    match = re.search(r"<answer>\s*([+-]?\d+(?:[.,]\d+)?)\s*</answer>", text)
    if match:
        return match.group(1).replace(",", "")

    match = re.search(r"####\s*([+-]?\d+(?:[.,]\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    match = re.search(
        r"(?:the answer is|answer:)\s*[:\s]*([+-]?\d+(?:[.,]\d+)?)",
        text, re.IGNORECASE,
    )
    if match:
        return match.group(1).replace(",", "")

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


def _get_stop_ids(tokenizer) -> list[int]:
    """Return token ids for the ``</answer`` prefix."""
    return tokenizer.encode("</answer", add_special_tokens=False)


def _ends_with_stop(gen_tokens: list[int], stop_ids: list[int]) -> bool:
    """Check whether *gen_tokens* ends with the *stop_ids* sequence."""
    n = len(stop_ids)
    return len(gen_tokens) >= n and gen_tokens[-n:] == stop_ids


def _clone_kv_cache(kv_cache):
    """Clone a DynamicCache by copying its tensors."""
    from transformers.cache_utils import DynamicCache
    clone = DynamicCache()
    for layer in kv_cache.layers:
        clone.update(layer.keys.clone(), layer.values.clone(), len(clone.layers))
    return clone


def build_prompt_ids(
    question: str, demos: list[dict], tokenizer, device: torch.device
) -> torch.Tensor:
    """Build a 5-shot prompt for a base model."""
    parts = []
    for demo in demos:
        parts.append(f"{demo['question']} <answer>{demo['answer']}</answer>\n\n")
    parts.append(question)
    prompt = "".join(parts)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def _decode_generated(gen_tokens: list[int], eos_id: int, stop_ids: list[int], tokenizer) -> str:
    """Decode generated tokens, truncating at EOS or </answer>."""
    try:
        eos_pos = gen_tokens.index(eos_id)
        gen_tokens = gen_tokens[:eos_pos]
    except ValueError:
        pass

    n = len(stop_ids)
    for i in range(n, len(gen_tokens) + 1):
        if gen_tokens[i - n : i] == stop_ids:
            end = min(i + 1, len(gen_tokens))
            gen_tokens = gen_tokens[:end]
            break

    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Entropy-reweighted generation (adapted from src/dialecttax/sampling.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_k_with_entropy(
    model,
    prompt_kv,
    first_logits: torch.Tensor,
    *,
    K: int,
    temperature: float = 0.8,
    max_new_tokens: int = 128,
    eos_token_id: int,
    stop_ids: list[int],
    entropy_window: str = "all",
    window_fraction: float = 0.5,
    window_n: int = 64,
) -> list[dict]:
    """Generate K candidate completions, collecting entropy and log-prob.

    Each candidate is generated sequentially from the shared prompt KV cache.
    Per-token entropy and log-probability are collected during generation.

    Returns a list of K dicts, each with:
        tokens, text (None — decoded later), entropies, log_probs,
        windowed_entropy, avg_log_likelihood
    """
    candidates = []

    for _ in range(K):
        gen_tokens: list[int] = []
        token_entropies: list[float] = []
        token_log_probs: list[float] = []
        logits = first_logits.clone()
        past_kv = _clone_kv_cache(prompt_kv)

        for _ in range(max_new_tokens):
            # Entropy from untempered logits (float32 for stability)
            logits_f32 = logits.float()
            log_p = torch.log_softmax(logits_f32, dim=-1)
            p = log_p.exp()
            entropy = -(p * log_p).sum(dim=-1).item()
            token_entropies.append(entropy)

            # Sample with temperature
            scaled_logits = logits / temperature
            probs = torch.softmax(scaled_logits, dim=-1)
            probs = probs.clamp(min=1e-10)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            token_id = torch.multinomial(probs[0], 1).item()

            # Log-prob of chosen token under untempered distribution
            token_log_probs.append(log_p[0, token_id].item())

            gen_tokens.append(token_id)

            if token_id == eos_token_id or _ends_with_stop(gen_tokens, stop_ids):
                break

            token_tensor = torch.tensor([[token_id]], device=logits.device)
            outputs = model(token_tensor, past_key_values=past_kv, use_cache=True)
            past_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        # Compute windowed entropy
        ent = torch.tensor(token_entropies) if token_entropies else torch.tensor([0.0])
        if entropy_window == "last_fraction" and len(token_entropies) > 0:
            n_keep = max(1, int(len(token_entropies) * window_fraction))
            ent = ent[-n_keep:]
        elif entropy_window == "last_n" and len(token_entropies) > 0:
            n_keep = min(window_n, len(token_entropies))
            ent = ent[-n_keep:]

        windowed_entropy = ent.mean().item()
        avg_ll = sum(token_log_probs) / len(token_log_probs) if token_log_probs else 0.0

        candidates.append({
            "tokens": gen_tokens,
            "entropies": token_entropies,
            "log_probs": token_log_probs,
            "windowed_entropy": windowed_entropy,
            "avg_log_likelihood": avg_ll,
        })

    return candidates


def select_candidate(
    candidates: list[dict],
    beta: float = 1.0,
    alpha: float | None = None,
) -> tuple[int, list[float]]:
    """Select a candidate using entropy-reweighted importance sampling.

    Pure entropy:  w(x) ∝ exp(-beta * H(x))
    Hybrid:        w(x) ∝ exp(alpha * log π(x) - beta * H(x))

    Returns (selected_index, weights).
    """
    entropies = torch.tensor([c["windowed_entropy"] for c in candidates])
    log_weights = -beta * entropies

    if alpha is not None:
        ll = torch.tensor([c["avg_log_likelihood"] for c in candidates])
        log_weights = log_weights + alpha * ll

    weights = F.softmax(log_weights, dim=0)
    selected_idx = torch.multinomial(weights, num_samples=1).item()

    return selected_idx, weights.tolist()


# ---------------------------------------------------------------------------
# Greedy generation (for entropy profile + baseline accuracy)
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
    """Greedy decode with per-position entropy collection."""
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
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_split(
    model,
    tokenizer,
    model_name: str,
    split_name: str,
    dataset,
    *,
    K: int = 8,
    beta: float = 1.0,
    alpha: float | None = None,
    temperature: float = 0.7,
    entropy_window: str = "all",
    window_fraction: float = 0.5,
    window_n: int = 64,
    n_samples: int = 50,
    max_samples: int | None = None,
    max_new_tokens: int = 128,
) -> dict:
    """Evaluate entropy-reweighted sampling on one dataset split.

    For each question:
      1. Compute prompt KV cache once
      2. Greedy pass (entropy profile + baseline accuracy)
      3. Generate K candidates with entropy tracking, select best
      4. Batched temperature samples (for pass@k)
    """
    data = dataset[split_name]
    stop_ids = _get_stop_ids(tokenizer)
    eos_id = tokenizer.eos_token_id

    demos = [data[i] for i in range(N_FEW_SHOT)]
    eval_start = N_FEW_SHOT
    if max_samples:
        eval_end = min(eval_start + max_samples, len(data))
    else:
        eval_end = len(data)
    eval_indices = list(range(eval_start, eval_end))

    correct_greedy = 0
    correct_entropy = 0
    total = len(eval_indices)
    results: list[dict] = []
    all_entropies: list[list[float]] = []
    device = next(model.parameters()).device

    for qi, idx in enumerate(eval_indices):
        example = data[idx]
        question = example["question"]
        gold = normalize_answer(example["answer"])
        prompt_ids = build_prompt_ids(question, demos, tokenizer, device)

        # Compute prompt KV cache once
        prompt_out = model(prompt_ids, use_cache=True)
        prompt_kv = prompt_out.past_key_values
        first_logits = prompt_out.logits[:, -1, :]

        # 1. Greedy pass (entropy + baseline)
        greedy_tokens, entropy_profile = greedy_generate(
            model,
            _clone_kv_cache(prompt_kv),
            first_logits,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            stop_ids=stop_ids,
        )
        greedy_text = _decode_generated(greedy_tokens, eos_id, stop_ids, tokenizer)
        greedy_pred = extract_answer(greedy_text)
        greedy_norm = normalize_answer(greedy_pred) if greedy_pred else None
        greedy_correct = greedy_norm == gold
        if greedy_correct:
            correct_greedy += 1
        if entropy_profile:
            all_entropies.append(entropy_profile)

        # 2. Entropy-reweighted selection from K candidates
        candidates = generate_k_with_entropy(
            model,
            prompt_kv,
            first_logits,
            K=K,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            stop_ids=stop_ids,
            entropy_window=entropy_window,
            window_fraction=window_fraction,
            window_n=window_n,
        )
        selected_idx, weights = select_candidate(candidates, beta=beta, alpha=alpha)
        selected = candidates[selected_idx]
        selected_text = _decode_generated(selected["tokens"], eos_id, stop_ids, tokenizer)
        selected_pred = extract_answer(selected_text)
        selected_norm = normalize_answer(selected_pred) if selected_pred else None
        entropy_correct = selected_norm == gold
        if entropy_correct:
            correct_entropy += 1

        # Count how many of the K candidates are correct (for pass@k over K)
        n_correct_K = 0
        for cand in candidates:
            cand_text = _decode_generated(cand["tokens"], eos_id, stop_ids, tokenizer)
            cand_pred = extract_answer(cand_text)
            cand_norm = normalize_answer(cand_pred) if cand_pred else None
            if cand_norm == gold:
                n_correct_K += 1

        # pass@k from K entropy candidates
        pk_entropy = {}
        for k in PASS_AT_K_VALUES:
            if k <= K:
                pk_entropy[k] = pass_at_k(K, n_correct_K, k)

        # 3. Batched temperature samples (for pass@k)
        # Re-encode prompt since candidates consumed the KV cache
        prompt_out2 = model(prompt_ids, use_cache=True)
        prompt_kv2 = prompt_out2.past_key_values
        first_logits2 = prompt_out2.logits[:, -1, :]

        sample_token_lists = generate_batch_samples(
            model,
            prompt_kv2,
            first_logits2,
            n_samples=n_samples,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            stop_ids=stop_ids,
        )
        n_correct_samples = 0
        for sample_tokens in sample_token_lists:
            sample_text = _decode_generated(sample_tokens, eos_id, stop_ids, tokenizer)
            sample_pred = extract_answer(sample_text)
            sample_norm = normalize_answer(sample_pred) if sample_pred else None
            if sample_norm == gold:
                n_correct_samples += 1

        # pass@k from temperature samples
        pk_samples = {}
        for k in PASS_AT_K_VALUES:
            if k <= n_samples:
                pk_samples[k] = pass_at_k(n_samples, n_correct_samples, k)

        results.append({
            "index": idx,
            "question": question[:120] + "..." if len(question) > 120 else question,
            "gold": gold,
            "greedy_answer": greedy_norm,
            "greedy_correct": greedy_correct,
            "greedy_mean_entropy": float(np.mean(entropy_profile)) if entropy_profile else None,
            "entropy_answer": selected_norm,
            "entropy_correct": entropy_correct,
            "entropy_mean_entropy": selected["windowed_entropy"],
            "selected_idx": selected_idx,
            "selected_log_likelihood": selected["avg_log_likelihood"],
            "weights": weights,
            "candidate_entropies": [c["windowed_entropy"] for c in candidates],
            "n_correct_K": n_correct_K,
            "n_correct_samples": n_correct_samples,
            "n_samples": n_samples,
            "pass_at_k_samples": pk_samples,
            "pass_at_k_entropy": pk_entropy,
        })

        if (qi + 1) % 10 == 0 or (qi + 1) == total:
            g_acc = correct_greedy / (qi + 1)
            e_acc = correct_entropy / (qi + 1)
            print(
                f"    [{split_name}] {qi + 1}/{total} — "
                f"greedy: {g_acc:.1%}  entropy: {e_acc:.1%}"
            )

    accuracy_greedy = correct_greedy / total if total > 0 else 0.0
    accuracy_entropy = correct_entropy / total if total > 0 else 0.0

    # Average pass@k from temperature samples (greedy method's diversity measure)
    avg_pk_samples = {}
    for k in PASS_AT_K_VALUES:
        if k <= n_samples:
            vals = [q["pass_at_k_samples"].get(k, 0.0) for q in results]
            avg_pk_samples[k] = float(np.mean(vals))

    # Average pass@k from K entropy candidates
    avg_pk_entropy = {}
    for k in PASS_AT_K_VALUES:
        if k <= K:
            vals = [q["pass_at_k_entropy"].get(k, 0.0) for q in results]
            avg_pk_entropy[k] = float(np.mean(vals))

    # Greedy entropy profile (per-position average)
    max_entropy_len = max((len(e) for e in all_entropies), default=0)
    entropy_profile_avg = []
    if max_entropy_len > 0:
        for pos in range(max_entropy_len):
            vals = [e[pos] for e in all_entropies if pos < len(e)]
            entropy_profile_avg.append(float(np.mean(vals)))

    greedy_mean_entropy = float(np.mean([
        r["greedy_mean_entropy"] for r in results if r["greedy_mean_entropy"] is not None
    ])) if results else 0.0

    entropy_mean_entropy = float(np.mean([
        r["entropy_mean_entropy"] for r in results
    ])) if results else 0.0

    # Best-of-K oracle
    oracle_correct = sum(1 for r in results if r["n_correct_K"] > 0)
    oracle_acc = oracle_correct / total if total > 0 else 0.0

    return {
        "split": split_name,
        "accuracy_greedy": accuracy_greedy,
        "accuracy_entropy": accuracy_entropy,
        "accuracy_oracle_K": oracle_acc,
        "correct_greedy": correct_greedy,
        "correct_entropy": correct_entropy,
        "total": total,
        "greedy_mean_entropy": greedy_mean_entropy,
        "entropy_mean_entropy": entropy_mean_entropy,
        "entropy_profile": entropy_profile_avg,
        "pass_at_k_samples": avg_pk_samples,
        "pass_at_k_entropy": avg_pk_entropy,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print formatted summary tables to stdout."""
    # Accuracy pivot: method × split
    print(f"\n{'=' * 70}")
    print("PER-SPLIT ACCURACY")
    print(f"{'=' * 70}")
    pivot = df.pivot_table(
        index="split", columns="method", values="accuracy", aggfunc="first"
    )
    print(pivot.to_string(float_format="{:.1%}".format))

    # Dialect gap per method
    print(f"\n{'=' * 70}")
    print("DIALECT GAP  (Original - AAVE)")
    print(f"{'=' * 70}")
    for method in df["method"].unique():
        mdf = df[df["method"] == method]
        orig = mdf[mdf["split"].str.endswith("_original")]
        aave = mdf[mdf["split"].str.endswith("_aave")]
        if orig.empty or aave.empty:
            continue
        o_acc = orig["accuracy"].mean()
        a_acc = aave["accuracy"].mean()
        gap = o_acc - a_acc
        print(f"  {method:<20}: Original={o_acc:.1%}  AAVE={a_acc:.1%}  Gap={gap:+.1%}")

    # pass@k
    pk_cols = [c for c in df.columns if c.startswith("pass@")]
    if pk_cols:
        print(f"\n{'=' * 70}")
        print("pass@k")
        print(f"{'=' * 70}")
        pk_df = df[["method", "split"] + pk_cols].set_index(["method", "split"])
        print(pk_df.to_string(float_format="{:.3f}".format))

    # Mean entropy per method
    print(f"\n{'=' * 70}")
    print("MEAN ENTROPY")
    print(f"{'=' * 70}")
    for method in df["method"].unique():
        mdf = df[df["method"] == method]
        me = mdf["mean_entropy"].mean()
        print(f"  {method:<20}: {me:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Entropy-reweighted sampling on the ReDial dataset."
    )
    parser.add_argument(
        "--model",
        default="llama-3.2-3b",
        choices=list(MODELS.keys()),
        help="Model to evaluate (default: llama-3.2-3b).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=DEFAULT_SPLITS,
        help="Dataset splits to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap examples per split (for testing).",
    )
    # Entropy sampling parameters
    parser.add_argument(
        "--K", type=int, default=8,
        help="Number of candidate sequences (default: 8).",
    )
    parser.add_argument(
        "--beta", type=float, default=1.0,
        help="Entropy sharpening strength (default: 1.0).",
    )
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Log-likelihood weight for hybrid scoring (default: None = pure entropy).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature for candidate generation (default: 0.7).",
    )
    parser.add_argument(
        "--entropy-window",
        choices=["all", "last_fraction", "last_n"],
        default="all",
        help="Entropy windowing strategy (default: all).",
    )
    parser.add_argument(
        "--window-fraction", type=float, default=0.5,
        help="Fraction for last_fraction window (default: 0.5).",
    )
    parser.add_argument(
        "--window-n", type=int, default=64,
        help="Number of tokens for last_n window (default: 64).",
    )
    # pass@k parameters
    parser.add_argument(
        "--n-samples", type=int, default=50,
        help="Temperature samples per question for pass@k (default: 50).",
    )
    # General
    parser.add_argument(
        "--max-new-tokens", type=int, default=128,
        help="Maximum tokens to generate per question (default: 128).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/entropy_sampling",
        help="Directory for result files.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model weight dtype (default: bf16).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42).",
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

    if args.reasoning:
        print("Loading ReDial reasoning dataset (local) ...")
        dataset = load_reasoning_dataset(args.splits)
    else:
        print("Loading ReDial dataset ...")
        dataset = load_dataset(DATASET_ID)

    model_name = args.model
    model_id = MODELS[model_name]

    print(f"\n{'=' * 60}")
    print(f"Model:  {model_name}  ({model_id})")
    print(f"K={args.K}  beta={args.beta}  alpha={args.alpha}  temp={args.temperature}")
    print(f"Entropy window: {args.entropy_window}", end="")
    if args.entropy_window == "last_fraction":
        print(f" (fraction={args.window_fraction})")
    elif args.entropy_window == "last_n":
        print(f" (n={args.window_n})")
    else:
        print()
    print(f"{'=' * 60}")

    model, tokenizer = load_model(model_id, dtype=dtype)

    summary_rows: list[dict] = []
    all_results: dict = {}

    for split_name in args.splits:
        if split_name not in dataset:
            print(f"  Skipping unknown split: {split_name}")
            continue

        n = len(dataset[split_name])
        print(f"\n  Evaluating {split_name} ({n} examples) ...")
        t0 = time.time()

        result = evaluate_split(
            model,
            tokenizer,
            model_name,
            split_name,
            dataset,
            K=args.K,
            beta=args.beta,
            alpha=args.alpha,
            temperature=args.temperature,
            entropy_window=args.entropy_window,
            window_fraction=args.window_fraction,
            window_n=args.window_n,
            n_samples=args.n_samples,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
        )
        elapsed = time.time() - t0

        all_results[split_name] = result

        # Emit two rows per split: one for greedy, one for entropy
        greedy_row: dict = {
            "model": model_name,
            "method": "greedy",
            "split": split_name,
            "accuracy": result["accuracy_greedy"],
            "correct": result["correct_greedy"],
            "total": result["total"],
            "mean_entropy": result["greedy_mean_entropy"],
            "K": args.K,
            "beta": args.beta,
            "time_s": round(elapsed, 1),
        }
        for k, v in result["pass_at_k_samples"].items():
            greedy_row[f"pass@{k}"] = v
        summary_rows.append(greedy_row)

        entropy_row: dict = {
            "model": model_name,
            "method": f"entropy_K{args.K}",
            "split": split_name,
            "accuracy": result["accuracy_entropy"],
            "correct": result["correct_entropy"],
            "total": result["total"],
            "mean_entropy": result["entropy_mean_entropy"],
            "K": args.K,
            "beta": args.beta,
            "time_s": round(elapsed, 1),
        }
        for k, v in result["pass_at_k_entropy"].items():
            entropy_row[f"pass@{k}"] = v
        summary_rows.append(entropy_row)

        print(
            f"  => {split_name}: greedy={result['accuracy_greedy']:.1%}  "
            f"entropy={result['accuracy_entropy']:.1%}  "
            f"oracle@{args.K}={result['accuracy_oracle_K']:.1%}  "
            f"in {elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Save results
    tag = f"{model_name}_K{args.K}_b{args.beta}"
    if args.alpha is not None:
        tag += f"_a{args.alpha}"

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

    # Console summary
    if not df.empty:
        print_summary(df)


if __name__ == "__main__":
    main()
