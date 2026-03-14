#!/usr/bin/env python3
"""Mode collapse analysis: base vs instruct models on the ReDial dataset.

Measures how instruction tuning affects output diversity (mode collapse) and
dialect fairness by comparing base and instruct variants on the ReDial dialect
fairness benchmark.

Metrics:
  - Accuracy (greedy)
  - Dialect gap (Original - AAVE accuracy)
  - pass@k curves (unbiased estimator from temperature samples)
  - Per-token entropy profiles (from greedy-path logits)

Usage:
    # Full run (all 4 models, all 4 splits)
    python scripts/mode_collapse.py

    # Quick smoke test
    python scripts/mode_collapse.py --models llama-3.1-8b-base --max-samples 2 --n-samples 5

    # Specific models and splits
    python scripts/mode_collapse.py --models llama-3.1-8b-base llama-3.1-8b-instruct \
        --splits math_vanilla_original math_vanilla_aave
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
    "logic_vanilla_original",
    "logic_vanilla_aave",
]

MODELS = {
    "llama-3.2-3b-base": ("meta-llama/Llama-3.2-3B", "base"),
    "llama-3.2-3b-instruct": ("meta-llama/Llama-3.2-3B-Instruct", "instruct"),
    "llama-3.1-8b-base": ("meta-llama/Llama-3.1-8B", "base"),
    "llama-3.1-8b-instruct": ("meta-llama/Llama-3.1-8B-Instruct", "instruct"),
    "qwen2.5-7b-base": ("Qwen/Qwen2.5-7B", "base"),
    "qwen2.5-7b-instruct": ("Qwen/Qwen2.5-7B-Instruct", "instruct"),
}

N_FEW_SHOT = 5

INSTRUCT_SYSTEM_PROMPT = "Provide your final answer inside `<answer></answer>` tags."

# Known categorical answers for logic splits
LOGIC_ANSWERS = [
    "necessarily true", "necessarily false",
    "yes", "no",
    "a", "b", "c", "d",
]

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
# Model helpers (reused from mcmc.py)
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
    """Clone a DynamicCache by copying its tensors (avoids deepcopy issues)."""
    from transformers.cache_utils import DynamicCache
    clone = DynamicCache()
    for layer in kv_cache.layers:
        clone.update(layer.keys.clone(), layer.values.clone(), len(clone.layers))
    return clone


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_few_shot_prompt(
    question: str,
    demos: list[dict],
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Build a 5-shot prompt for base models.

    Each demo is formatted as: question text + " <answer>GOLD</answer>\n\n"
    The final question is appended without an answer.
    """
    parts = []
    for demo in demos:
        parts.append(f"{demo['question']} <answer>{demo['answer']}</answer>\n\n")
    parts.append(question)
    prompt = "".join(parts)
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


def build_instruct_prompt(
    question: str,
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """Build a chat-template prompt for instruct models."""
    messages = [
        {"role": "system", "content": INSTRUCT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer(prompt, return_tensors="pt").input_ids.to(device)


# ---------------------------------------------------------------------------
# Answer extraction and normalization
# ---------------------------------------------------------------------------

def extract_answer(text: str, split_name: str) -> tuple[str | None, str]:
    """Extract answer from model output.

    For math splits: numeric answers.
    For logic splits: categorical answers.

    Returns (answer, extraction_method) where extraction_method is one of:
        "tag", "last_number", "logic_keyword", "none".
    """
    is_math = "math" in split_name

    # 1. <answer>...</answer> tags (works for both numeric and text)
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip(), "tag"

    if is_math:
        # 2. Math fallback: last number in text
        numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        if numbers:
            return numbers[-1], "last_number"
        return None, "none"
    else:
        # 3. Logic fallback: scan for known categorical answers
        text_lower = text.lower()
        # Check longest matches first to avoid "a" matching inside other words
        for ans in sorted(LOGIC_ANSWERS, key=len, reverse=True):
            if ans in text_lower:
                return ans, "logic_keyword"
        return None, "none"


def normalize_answer(answer: str, split_name: str) -> str:
    """Normalize answer string for comparison."""
    answer = answer.strip()
    is_math = "math" in split_name

    if is_math:
        answer = answer.replace(",", "")
        try:
            val = float(answer)
            return str(int(val)) if val == int(val) else str(val)
        except ValueError:
            return answer
    else:
        return answer.lower().strip()


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_greedy(
    model,
    prompt_kv,
    first_logits: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    eos_token_id: int,
    stop_ids: list[int],
) -> tuple[list[int], list[float]]:
    """Greedy decode from a pre-computed prompt KV cache, collecting entropy.

    Returns (generated_token_ids, per_position_entropy).
    """
    gen_tokens: list[int] = []
    entropies: list[float] = []
    logits = first_logits.clone()
    past_kv = prompt_kv

    for _ in range(max_new_tokens):
        # Entropy from untempered logits (float32 for stability)
        logits_f32 = logits.float()
        log_p = torch.log_softmax(logits_f32, dim=-1)
        p = log_p.exp()
        entropy = -(p * log_p).sum(dim=-1).item()
        entropies.append(entropy)

        token_id = logits.argmax(dim=-1).item()
        gen_tokens.append(token_id)

        if token_id == eos_token_id or _ends_with_stop(gen_tokens, stop_ids):
            break

        token_tensor = torch.tensor([[token_id]], device=logits.device)
        outputs = model(token_tensor, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    return gen_tokens, entropies


@torch.no_grad()
def generate_batch_samples(
    model,
    prompt_kv,
    first_logits: torch.Tensor,
    *,
    n_samples: int,
    temperature: float = 0.7,
    max_new_tokens: int = 32,
    eos_token_id: int,
    stop_ids: list[int],
) -> list[list[int]]:
    """Generate n_samples completions in parallel from a shared prompt KV cache.

    Expands the prompt KV cache across the batch dimension so all samples
    share the same prompt encoding and are generated simultaneously.
    """
    device = first_logits.device
    n_stop = len(stop_ids)

    # Expand prompt KV cache: batch=1 -> batch=n_samples
    batch_kv = _clone_kv_cache(prompt_kv)
    batch_kv.batch_repeat_interleave(n_samples)
    logits = first_logits.expand(n_samples, -1)  # [B, vocab]

    generated = torch.zeros(n_samples, max_new_tokens, dtype=torch.long, device=device)
    lengths = torch.full((n_samples,), max_new_tokens, dtype=torch.long, device=device)
    done = torch.zeros(n_samples, dtype=torch.bool, device=device)

    for t in range(max_new_tokens):
        # Sample tokens
        probs = torch.softmax(logits / temperature, dim=-1)
        probs = probs.clamp(min=1e-10)
        tokens = torch.multinomial(probs, 1)  # [B, 1]
        generated[:, t] = tokens.squeeze(1)

        # Check stopping per sample
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

        # Forward pass for next position (all samples)
        if t < max_new_tokens - 1:
            outputs = model(tokens, past_key_values=batch_kv, use_cache=True)
            batch_kv = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

    return [generated[i, : lengths[i]].tolist() for i in range(n_samples)]


def decode_generated(
    gen_tokens: list[int],
    eos_token_id: int,
    stop_ids: list[int],
    tokenizer,
) -> str:
    """Decode generated tokens, truncating at EOS or </answer>."""
    # Truncate at first EOS
    try:
        eos_pos = gen_tokens.index(eos_token_id)
        gen_tokens = gen_tokens[:eos_pos]
    except ValueError:
        pass

    # Truncate after </answer (keep the stop sequence + one extra token for ">")
    n = len(stop_ids)
    for i in range(n, len(gen_tokens) + 1):
        if gen_tokens[i - n : i] == stop_ids:
            end = min(i + 1, len(gen_tokens))
            gen_tokens = gen_tokens[:end]
            break

    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# pass@k estimator
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: 1 - C(n-c, k) / C(n, k).

    n: total samples, c: number correct, k: k value.
    """
    if n < k:
        return float(c > 0)
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    # Use log-space for numerical stability
    # C(n-c, k) / C(n, k) = product_{i=0}^{k-1} (n-c-i) / (n-i)
    log_ratio = 0.0
    for i in range(k):
        if n - c - i <= 0:
            return 1.0
        log_ratio += math.log(n - c - i) - math.log(n - i)
    return 1.0 - math.exp(log_ratio)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(
    model,
    tokenizer,
    model_name: str,
    model_type: str,
    split_name: str,
    dataset,
    *,
    n_samples: int = 50,
    temperature: float = 0.7,
    max_samples: int | None = None,
    max_new_tokens: int = 32,
) -> dict:
    """Evaluate one model on one split.

    For each question:
      1. Compute prompt KV cache once
      2. Run greedy pass (collect entropy + accuracy)
      3. Run n_samples temperature samples (for pass@k)
    """
    data = dataset[split_name]
    stop_ids = _get_stop_ids(tokenizer)
    eos_id = tokenizer.eos_token_id
    device = next(model.parameters()).device

    # First N_FEW_SHOT examples are reserved as demos (excluded for both
    # base and instruct so both evaluate on the same questions).
    demos = [data[i] for i in range(N_FEW_SHOT)]
    eval_start = N_FEW_SHOT

    # Select evaluation examples
    if max_samples is not None:
        eval_end = min(eval_start + max_samples, len(data))
    else:
        eval_end = len(data)

    eval_indices = list(range(eval_start, eval_end))
    total = len(eval_indices)

    correct_greedy = 0
    per_question: list[dict] = []
    all_entropies: list[list[float]] = []

    for qi, idx in enumerate(eval_indices):
        example = data[idx]
        question = example["question"]
        gold_raw = example["answer"]
        gold = normalize_answer(gold_raw, split_name)

        # Build prompt
        if model_type == "base":
            prompt_ids = build_few_shot_prompt(question, demos, tokenizer, device)
        else:
            prompt_ids = build_instruct_prompt(question, tokenizer, device)

        # Compute prompt KV cache once
        prompt_out = model(prompt_ids, use_cache=True)
        prompt_kv = prompt_out.past_key_values
        first_logits = prompt_out.logits[:, -1, :]

        # 1. Greedy pass (with entropy collection)
        #    Deep-copy the KV cache because DynamicCache is mutated in-place
        #    by the model's forward pass — without this, the temperature
        #    samples would inherit the greedy tokens in the cache.
        greedy_tokens, entropy_profile = generate_greedy(
            model,
            _clone_kv_cache(prompt_kv),
            first_logits,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            stop_ids=stop_ids,
        )
        greedy_text = decode_generated(greedy_tokens, eos_id, stop_ids, tokenizer)
        greedy_answer, greedy_method = extract_answer(greedy_text, split_name)
        greedy_norm = normalize_answer(greedy_answer, split_name) if greedy_answer else None
        is_correct = greedy_norm == gold
        if is_correct:
            correct_greedy += 1

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
            sample_text = decode_generated(sample_tokens, eos_id, stop_ids, tokenizer)
            sample_answer, _ = extract_answer(sample_text, split_name)
            sample_norm = normalize_answer(sample_answer, split_name) if sample_answer else None
            if sample_norm == gold:
                n_correct_samples += 1

        # Compute pass@k for this question
        pk = {}
        for k in PASS_AT_K_VALUES:
            if k <= n_samples:
                pk[k] = pass_at_k(n_samples, n_correct_samples, k)

        per_question.append({
            "index": idx,
            "question": question[:120] + "..." if len(question) > 120 else question,
            "gold": gold,
            "greedy_answer": greedy_norm,
            "greedy_correct": is_correct,
            "greedy_extraction_method": greedy_method,
            "greedy_response": greedy_text,
            "n_correct_samples": n_correct_samples,
            "n_samples": n_samples,
            "pass_at_k": pk,
            "mean_entropy": float(np.mean(entropy_profile)) if entropy_profile else None,
        })

        if (qi + 1) % 10 == 0 or (qi + 1) == total:
            acc = correct_greedy / (qi + 1)
            print(f"    [{split_name}] {qi + 1}/{total} — greedy acc: {acc:.1%}")

    # Aggregate metrics
    accuracy = correct_greedy / total if total > 0 else 0.0

    # Average pass@k across questions
    avg_pass_at_k = {}
    for k in PASS_AT_K_VALUES:
        if k <= n_samples:
            vals = [q["pass_at_k"].get(k, 0.0) for q in per_question]
            avg_pass_at_k[k] = float(np.mean(vals))

    # Build entropy profile (per-position average)
    max_entropy_len = max((len(e) for e in all_entropies), default=0)
    entropy_profile_avg = []
    if max_entropy_len > 0:
        for pos in range(max_entropy_len):
            vals = [e[pos] for e in all_entropies if pos < len(e)]
            entropy_profile_avg.append(float(np.mean(vals)))

    mean_entropy = float(np.mean([
        q["mean_entropy"] for q in per_question if q["mean_entropy"] is not None
    ])) if per_question else 0.0

    return {
        "split": split_name,
        "model": model_name,
        "model_type": model_type,
        "accuracy": accuracy,
        "correct": correct_greedy,
        "total": total,
        "pass_at_k": avg_pass_at_k,
        "mean_entropy": mean_entropy,
        "entropy_profile": entropy_profile_avg,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame, all_results: dict) -> None:
    """Print formatted summary tables to stdout."""
    # 1. Per-split accuracy pivot
    print(f"\n{'=' * 70}")
    print("PER-SPLIT ACCURACY (greedy)")
    print(f"{'=' * 70}")
    pivot = df.pivot_table(
        index="split", columns="model", values="accuracy", aggfunc="first"
    )
    print(pivot.to_string(float_format="{:.1%}".format))

    # 2. Dialect gap per family
    print(f"\n{'=' * 70}")
    print("DIALECT GAP (Original - AAVE)")
    print(f"{'=' * 70}")
    families = {}
    for _, row in df.iterrows():
        name = row["model"]
        # Extract family: e.g. "llama-3.1-8b" from "llama-3.1-8b-base"
        if name.endswith("-base"):
            family = name[: -len("-base")]
            variant = "base"
        elif name.endswith("-instruct"):
            family = name[: -len("-instruct")]
            variant = "instruct"
        else:
            continue
        families.setdefault(family, {}).setdefault(variant, []).append(row)

    for family, variants in sorted(families.items()):
        print(f"\n  {family}:")
        for variant in ["base", "instruct"]:
            if variant not in variants:
                continue
            rows = variants[variant]
            orig_rows = [r for r in rows if r["split"].endswith("_original")]
            aave_rows = [r for r in rows if r["split"].endswith("_aave")]
            if orig_rows and aave_rows:
                o_correct = sum(r["correct"] for r in orig_rows)
                o_total = sum(r["total"] for r in orig_rows)
                a_correct = sum(r["correct"] for r in aave_rows)
                a_total = sum(r["total"] for r in aave_rows)
                o_acc = o_correct / o_total if o_total > 0 else 0
                a_acc = a_correct / a_total if a_total > 0 else 0
                gap = o_acc - a_acc
                print(
                    f"    {variant:<10}: Original={o_acc:.1%}  "
                    f"AAVE={a_acc:.1%}  Gap={gap:+.1%}"
                )

    # 3. pass@k table
    print(f"\n{'=' * 70}")
    print("pass@k")
    print(f"{'=' * 70}")
    pk_rows = []
    for _, row in df.iterrows():
        pk_row = {"model": row["model"], "split": row["split"]}
        for k in PASS_AT_K_VALUES:
            col = f"pass@{k}"
            if col in row and not pd.isna(row[col]):
                pk_row[col] = row[col]
        pk_rows.append(pk_row)
    if pk_rows:
        pk_df = pd.DataFrame(pk_rows).set_index(["model", "split"])
        print(pk_df.to_string(float_format="{:.3f}".format))

    # 4. Mean entropy per model
    print(f"\n{'=' * 70}")
    print("MEAN ENTROPY (greedy path)")
    print(f"{'=' * 70}")
    for name in df["model"].unique():
        mdf = df[df["model"] == name]
        me = mdf["mean_entropy"].mean()
        print(f"  {name:<30}: {me:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode collapse analysis: base vs instruct on ReDial."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Models to evaluate (default: all 4).",
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
        "--max-new-tokens-base",
        type=int,
        default=512,
        help="Maximum tokens to generate for base models (default: 512).",
    )
    parser.add_argument(
        "--max-new-tokens-inst",
        type=int,
        default=512,
        help="Maximum tokens to generate for instruct models (default: 512).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mode_collapse",
        help="Directory for result files (default: results/mode_collapse).",
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Model weight dtype (default: bf16).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
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

    # Load dataset
    if args.reasoning:
        print("Loading ReDial reasoning dataset (local) ...")
        dataset = load_reasoning_dataset(args.splits)
    else:
        print("Loading ReDial dataset ...")
        dataset = load_dataset(DATASET_ID)

    all_results: dict = {}
    summary_rows: list[dict] = []
    entropy_profiles: dict = {}

    for model_name in args.models:
        model_id, model_type = MODELS[model_name]

        print(f"\n{'=' * 60}")
        print(f"Model:  {model_name}  ({model_id})  [{model_type}]")
        print(f"Samples: {args.n_samples}  |  Temp: {args.temperature}")
        print(f"{'=' * 60}")

        model, tokenizer = load_model(model_id, dtype=dtype)
        model_results = {}

        for split_name in args.splits:
            if split_name not in dataset:
                print(f"  Skipping unknown split: {split_name}")
                continue

            n = len(dataset[split_name])
            print(f"\n  Evaluating {split_name} ({n} examples) ...")
            t0 = time.time()

            max_new_tokens = (
                args.max_new_tokens_base if model_type == "base"
                else args.max_new_tokens_inst
            )
            result = evaluate_split(
                model,
                tokenizer,
                model_name,
                model_type,
                split_name,
                dataset,
                n_samples=args.n_samples,
                temperature=args.temperature,
                max_samples=args.max_samples,
                max_new_tokens=max_new_tokens,
            )
            elapsed = time.time() - t0

            model_results[split_name] = result

            # Build summary row
            row: dict = {
                "model": model_name,
                "model_type": model_type,
                "split": split_name,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
                "mean_entropy": result["mean_entropy"],
                "time_s": round(elapsed, 1),
            }
            for k, v in result["pass_at_k"].items():
                row[f"pass@{k}"] = v
            summary_rows.append(row)

            # Store entropy profile
            entropy_profiles.setdefault(model_name, {})[split_name] = result["entropy_profile"]

            print(
                f"  => {split_name}: {result['accuracy']:.1%} "
                f"({result['correct']}/{result['total']}) "
                f"in {elapsed:.1f}s  "
                f"mean_entropy={result['mean_entropy']:.3f}"
            )

        all_results[model_name] = model_results

        # Free GPU memory before loading next model
        del model
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------

    # 1. Detailed results (per-question)
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDetailed results -> {detailed_path}")

    # 2. Summary CSV
    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"Summary -> {csv_path}")

    # 3. Entropy profiles
    entropy_path = os.path.join(output_dir, "entropy_profiles.json")
    with open(entropy_path, "w") as f:
        json.dump(entropy_profiles, f, indent=2)
    print(f"Entropy profiles -> {entropy_path}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    if not df.empty:
        print_summary(df, all_results)


if __name__ == "__main__":
    main()
