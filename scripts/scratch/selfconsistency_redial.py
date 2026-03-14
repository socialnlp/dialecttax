"""
Self-consistency evaluation on preprocessed ReDial datasets.

Generates K completions per prompt via temperature sampling, then evaluates:
  - pass@K:  is the correct answer among any of K samples?
  - maj@K:   uniform majority vote over extracted answers.
  - wmaj@K:  log-prob weighted majority vote.
  - emaj@K:  ECE-optimal temperature weighted majority vote (per dialect).

Usage:
    # Single run
    python scripts/ReDial/selfconsistency_redial.py

    # Override K or temperature
    python scripts/ReDial/selfconsistency_redial.py K=10 temperature=0.8

    # Sweep
    python scripts/ReDial/selfconsistency_redial.py -m \
      model=llama_base,qwen_base reasoning=naive,cot
"""

import gc
import json
import logging
import os
from collections import Counter
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

def _resolve_project_dir(key: str) -> str:
    return dialecttax.utils.load_config("default")["directories"][key]

OmegaConf.register_new_resolver("project", _resolve_project_dir, use_cache=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

import dialecttax

log = logging.getLogger(__name__)

# Hydra 1.3.2 + Python 3.14 compatibility patch
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
    """Load model and tokenizer, packing onto fewest GPUs possible."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        device_map = "auto"
        max_memory = None
    else:
        max_memory = {
            i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**3) - 2}GiB"
            for i in range(n_gpus)
        }
        device_map = "sequential"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="bfloat16", device_map=device_map, max_memory=max_memory,
    )
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

    Returns (prompts, demo_indices).
    """
    if demo_indices is None:
        demo_indices = dialecttax.prompts.get_demo_indices(len(ds), n_few_shot, rng=rng)

    if model_type == "base":
        format_demos = dialecttax.prompts.format_prompts_math(dialecttax.prompts.MATH_DEMO[reasoning][dialect])
        demos = dialecttax.prompts.get_demos(ds, format_demos, demo_indices)
    else:
        demos = None

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

def _generate_on_gpu(rank, model_id, tokenizer_id, prompts, K, max_new_tokens,
                     batch_size, temperature, return_dict):
    """Worker function for mp.spawn: load model on one GPU and generate."""
    torch.cuda.set_device(rank)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="bfloat16", device_map=f"cuda:{rank}",
    )
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    n = len(prompts)
    completions = [[] for _ in range(n)]
    seq_lps = [[] for _ in range(n)]
    n_batches = (n + batch_size - 1) // batch_size

    for k in tqdm(range(K), desc=f"GPU {rank} samples", unit="sample", position=rank):
        for start in tqdm(range(0, n, batch_size), total=n_batches,
                          desc=f"  GPU {rank} k={k+1}/{K}", unit="batch",
                          leave=False, position=rank):
            batch = prompts[start:start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(f"cuda:{rank}")
            input_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            gen_ids = outputs.sequences[:, input_len:]
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            scores = torch.stack(outputs.scores, dim=1)
            log_probs = F.log_softmax(scores, dim=-1)
            gen_len = scores.shape[1]

            for j in range(len(batch)):
                ids = gen_ids[j]
                eos_pos = (ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                end = eos_pos[0].item() if len(eos_pos) > 0 else gen_len
                if end == 0:
                    lp = 0.0
                else:
                    lp = log_probs[j, torch.arange(end), ids[:end]].mean().item()
                completions[start + j].append(decoded[j])
                seq_lps[start + j].append(lp)

            del inputs, outputs, scores, log_probs
            torch.cuda.empty_cache()

    return_dict[rank] = {"completions": completions, "seq_log_probs": seq_lps}
    del model
    torch.cuda.empty_cache()


def generate_k_samples(
    model,
    tokenizer,
    prompts: list[str],
    *,
    K: int,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
) -> dict:
    """Generate K completions per prompt, parallelized across GPUs.

    Uses multiprocessing with spawn to run one worker per GPU.
    Each worker loads its own model copy to avoid CUDA threading issues.

    Returns dict with:
        completions: list[list[str]] — completions[i][k] is k-th completion
            for i-th prompt.
        seq_log_probs: list[list[float]] — seq_log_probs[i][k] is the
            mean per-token log-prob of the k-th completion for i-th prompt.
    """
    import torch.multiprocessing as mp

    tokenizer.padding_side = "left"
    model_id = model.config._name_or_path
    tokenizer_id = tokenizer.name_or_path
    n_prompts = len(prompts)

    # Determine how many GPUs to use (cap at n_prompts)
    n_gpus = min(torch.cuda.device_count(), n_prompts)

    # Single GPU: run directly, no multiprocessing overhead
    if n_gpus <= 1:
        n_batches = (n_prompts + batch_size - 1) // batch_size
        all_completions = [[] for _ in range(n_prompts)]
        all_seq_lps = [[] for _ in range(n_prompts)]

        for k in tqdm(range(K), desc="Samples", unit="sample"):
            pbar = tqdm(total=n_batches, desc=f"  Sample {k + 1}/{K}", unit="batch", leave=False)
            for start in range(0, n_prompts, batch_size):
                batch_prompts = prompts[start:start + batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                input_len = inputs.input_ids.shape[1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )

                gen_ids = outputs.sequences[:, input_len:]
                completions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
                scores = torch.stack(outputs.scores, dim=1)
                log_probs = F.log_softmax(scores, dim=-1)
                gen_len = scores.shape[1]

                for j in range(len(batch_prompts)):
                    ids = gen_ids[j]
                    eos_pos = (ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                    end = eos_pos[0].item() if len(eos_pos) > 0 else gen_len
                    if end == 0:
                        seq_lp = 0.0
                    else:
                        token_lps = log_probs[j, torch.arange(end), ids[:end]]
                        seq_lp = token_lps.mean().item()
                    all_completions[start + j].append(completions[j])
                    all_seq_lps[start + j].append(seq_lp)

                del inputs, outputs, scores, log_probs
                torch.cuda.empty_cache()
                pbar.update(1)

            pbar.close()

        return {"completions": all_completions, "seq_log_probs": all_seq_lps}

    # Multi-GPU: free main model, spawn workers that each load their own copy
    model.cpu()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    shard_size = (n_prompts + n_gpus - 1) // n_gpus
    shards = [prompts[i * shard_size:(i + 1) * shard_size] for i in range(n_gpus)]
    # Drop empty trailing shards
    shards = [s for s in shards if s]
    n_workers = len(shards)

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    return_dict = manager.dict()

    processes = []
    for rank in range(n_workers):
        p = mp.Process(
            target=_generate_on_gpu,
            args=(rank, model_id, tokenizer_id, shards[rank], K,
                  max_new_tokens, batch_size, temperature, return_dict),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Merge results in order
    all_completions, all_seq_lps = [], []
    for rank in range(n_workers):
        result = return_dict[rank]
        all_completions.extend(result["completions"])
        all_seq_lps.extend(result["seq_log_probs"])

    # Reload model on GPU 0 for the caller (needed for subsequent dialects)
    new_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype="bfloat16", device_map="cuda:0",
    )
    new_model.generation_config.pad_token_id = tokenizer.eos_token_id
    new_model.eval()

    return {"completions": all_completions, "seq_log_probs": all_seq_lps,
            "_model": new_model}


##########
# VOTING #
##########

def _uniform_majority_vote(extracted: list[str | None], gold: str) -> dict:
    """Standard majority vote with uniform weights."""
    norm = dialecttax.data.graders.math.normalize_answer
    valid = [(norm(e), 1.0) for e in extracted if e is not None]
    return _weighted_vote(valid, norm(gold))


def _logprob_weighted_vote(
    extracted: list[str | None], seq_lps: list[float], gold: str,
) -> dict:
    """Majority vote weighted by exp(seq_log_prob)."""
    norm = dialecttax.data.graders.math.normalize_answer
    valid = [
        (norm(e), np.exp(lp))
        for e, lp in zip(extracted, seq_lps) if e is not None
    ]
    return _weighted_vote(valid, norm(gold))


def _tempered_weighted_vote(
    extracted: list[str | None], seq_lps: list[float], gold: str, T: float,
) -> dict:
    """Majority vote weighted by exp(seq_log_prob / T)."""
    norm = dialecttax.data.graders.math.normalize_answer
    valid = [
        (norm(e), np.exp(lp / T))
        for e, lp in zip(extracted, seq_lps) if e is not None
    ]
    return _weighted_vote(valid, norm(gold))


def _weighted_vote(
    answer_weight_pairs: list[tuple[str, float]], gold_norm: str,
) -> dict:
    """Aggregate weighted votes and return result dict.

    Args:
        answer_weight_pairs: list of (normalized_answer, weight).
        gold_norm: normalized gold answer.

    Returns:
        Dict with winner, correct, confidence, n_valid.
    """
    if not answer_weight_pairs:
        return {"winner": None, "correct": False, "confidence": 0.0, "n_valid": 0}

    totals = {}
    for ans, w in answer_weight_pairs:
        totals[ans] = totals.get(ans, 0.0) + w

    total_weight = sum(totals.values())
    winner = max(totals, key=totals.get)
    confidence = totals[winner] / total_weight if total_weight > 0 else 0.0

    return {
        "winner": winner,
        "correct": winner == gold_norm,
        "confidence": confidence,
        "n_valid": len(answer_weight_pairs),
    }


###############
# ECE SEARCH #
###############

def _compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences >= lo) & (confidences < hi if i < n_bins - 1 else confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return ece


def find_ece_optimal_temperature(
    all_extracted: list[list[str | None]],
    all_seq_lps: list[list[float]],
    gold_answers: list[str],
    T_grid: np.ndarray | None = None,
) -> float:
    """Find temperature T that minimizes ECE for tempered weighted voting.

    Searches over a grid of T values. For each T, runs tempered weighted
    vote on all samples, then computes ECE from the vote confidences.

    Args:
        all_extracted: extracted answers per sample per k.
        all_seq_lps: sequence log-probs per sample per k.
        gold_answers: gold answers per sample.
        T_grid: temperatures to search over.

    Returns:
        Optimal temperature.
    """
    if T_grid is None:
        T_grid = np.concatenate([
            np.arange(0.1, 1.0, 0.1),
            np.arange(1.0, 5.5, 0.5),
        ])

    norm = dialecttax.data.graders.math.normalize_answer
    gold_norms = [norm(g) for g in gold_answers]

    best_T, best_ece = 1.0, float("inf")
    for T in T_grid:
        confidences = []
        corrects = []
        for extracted, seq_lps, gold_n in zip(all_extracted, all_seq_lps, gold_norms):
            result = _tempered_weighted_vote(extracted, seq_lps, gold_n, T)
            confidences.append(result["confidence"])
            corrects.append(result["correct"])

        ece = _compute_ece(np.array(confidences), np.array(corrects, dtype=float))
        if ece < best_ece:
            best_ece = ece
            best_T = T

    return best_T


###########
# GRADING #
###########

def grade_k_samples(
    completions_k: list[list[str]],
    seq_lps_k: list[list[float]],
    gold_answers: list[str],
    ece_T: float = 1.0,
) -> list[dict]:
    """Grade K completions per sample with all voting methods.

    Args:
        completions_k: completions per sample per k.
        seq_lps_k: sequence log-probs per sample per k.
        gold_answers: gold answers per sample.
        ece_T: ECE-optimal temperature for tempered weighted voting.

    Returns:
        List of per-sample result dicts.
    """
    results = []
    for i, (comps, lps, gold) in enumerate(zip(completions_k, seq_lps_k, gold_answers)):
        K = len(comps)
        extracted = [dialecttax.data.graders.math.extract_answer(c) for c in comps]
        correct = [dialecttax.data.graders.math.grade(e, gold) for e in extracted]

        # pass@K
        pass_at_k = any(correct)

        # Voting methods
        maj = _uniform_majority_vote(extracted, gold)
        wmaj = _logprob_weighted_vote(extracted, lps, gold)
        emaj = _tempered_weighted_vote(extracted, lps, gold, ece_T)

        # Answer diversity
        norm = dialecttax.data.graders.math.normalize_answer
        valid_answers = [norm(e) for e in extracted if e is not None]
        unique_answers = len(set(valid_answers)) if valid_answers else 0

        results.append({
            "gold": gold,
            "K": K,
            "n_correct": sum(correct),
            "n_accepted": len(valid_answers),
            "pass_at_k": pass_at_k,
            # Uniform majority vote
            "maj_correct": maj["correct"],
            "maj_confidence": maj["confidence"],
            # Log-prob weighted vote
            "wmaj_correct": wmaj["correct"],
            "wmaj_confidence": wmaj["confidence"],
            # ECE-tempered weighted vote
            "emaj_correct": emaj["correct"],
            "emaj_confidence": emaj["confidence"],
            "emaj_T": ece_T,
            # Diversity
            "agreement_rate": maj["confidence"],
            "unique_answers": unique_answers,
            "extracted_answers": extracted,
            "seq_log_probs": lps,
            "per_sample_correct": correct,
        })

    return results


#######
# CLI #
#######

@hydra.main(version_base=None, config_path="../../configs/selfconsistency_redial", config_name="config")
def main(cfg: DictConfig) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Skip if outputs already exist (unless rerun=true)
    output_dir = HydraConfig.get().runtime.output_dir
    expected_files = [f"results_{d.name}.jsonl" for d in cfg.dialects]
    if not cfg.rerun and all(
        os.path.exists(os.path.join(output_dir, f)) for f in expected_files
    ):
        log.info("Skipping %s/%s — outputs already exist in %s",
                 cfg.model.name, cfg.reasoning.name, output_dir)
        return

    rng = np.random.default_rng(cfg.seed)
    project_config = dialecttax.utils.load_config(cfg.project_config)
    reasoning = cfg.reasoning.name

    log.info("Loading model: %s (%s)", cfg.model.name, cfg.model.hf_id)
    tokenizer, model = load_model_and_tokenizer(cfg.model.hf_id)

    os.makedirs(output_dir, exist_ok=True)
    log.info("Output directory: %s", output_dir)

    demo_indices = None
    for dialect_cfg in cfg.dialects:
        dialect = dialect_cfg.name
        log.info("Processing dialect: %s", dialect)
        ds = dialecttax.data.redial.load_dataset(project_config["directories"]["preprocessed"], dialect_cfg.path_file)
        log.info("Loaded %d samples", len(ds))

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
        log.info("Evaluating %d samples × K=%d (%d excluded as demos)",
                 len(ds), cfg.K, len(demo_indices))

        # Generate K samples per prompt
        gen_output = generate_k_samples(
            model,
            tokenizer,
            prompts,
            K=cfg.K,
            max_new_tokens=cfg.reasoning.max_new_tokens,
            batch_size=cfg.batch_size,
            temperature=cfg.temperature,
        )
        completions_k = gen_output["completions"]
        seq_lps_k = gen_output["seq_log_probs"]
        # Multi-GPU path deletes the original model and returns a new one
        if "_model" in gen_output:
            model = gen_output["_model"]

        # Find ECE-optimal temperature for this dialect
        gold_answers = [sample["answer"] for sample in ds]
        all_extracted = [
            [dialecttax.data.graders.math.extract_answer(c) for c in comps]
            for comps in completions_k
        ]
        ece_T = find_ece_optimal_temperature(all_extracted, seq_lps_k, gold_answers)
        log.info("  ECE-optimal T for %s: %.2f", dialect, ece_T)

        # Grade with all voting methods
        sample_results = grade_k_samples(completions_k, seq_lps_k, gold_answers, ece_T=ece_T)

        # Run-level summary
        n = len(sample_results)
        run_summary = {
            "model": cfg.model.name,
            "dialect": dialect,
            "reasoning": reasoning,
            "K": cfg.K,
            "temperature": cfg.temperature,
            "n_samples": n,
            "greedy_accuracy": float(np.mean([s["per_sample_correct"][0] for s in sample_results])),
            "pass_at_k": float(np.mean([s["pass_at_k"] for s in sample_results])),
            "maj_at_k": float(np.mean([s["maj_correct"] for s in sample_results])),
            "wmaj_at_k": float(np.mean([s["wmaj_correct"] for s in sample_results])),
            "emaj_at_k": float(np.mean([s["emaj_correct"] for s in sample_results])),
            "ece_T": float(ece_T),
            "mean_agreement_rate": float(np.mean([s["agreement_rate"] for s in sample_results])),
            "mean_unique_answers": float(np.mean([s["unique_answers"] for s in sample_results])),
        }

        # Save per-sample JSONL
        sample_path = os.path.join(output_dir, f"results_{dialect}.jsonl")
        with open(sample_path, "w") as f:
            for s in sample_results:
                f.write(json.dumps(s, default=str) + "\n")

        # Save summary
        summary_path = os.path.join(output_dir, f"summary_{dialect}.json")
        with open(summary_path, "w") as f:
            json.dump(run_summary, f, indent=2, default=str)

        log.info("[%s] %s / %s / K=%d", dialect, cfg.model.name, reasoning, cfg.K)
        log.info("  greedy_accuracy:  %.3f", run_summary["greedy_accuracy"])
        log.info("  pass@%d:          %.3f", cfg.K, run_summary["pass_at_k"])
        log.info("  maj@%d:           %.3f", cfg.K, run_summary["maj_at_k"])
        log.info("  wmaj@%d:          %.3f", cfg.K, run_summary["wmaj_at_k"])
        log.info("  emaj@%d (T=%.2f): %.3f", cfg.K, ece_T, run_summary["emaj_at_k"])
        log.info("  mean_agreement:   %.3f", run_summary["mean_agreement_rate"])
        log.info("  mean_unique_ans:  %.1f", run_summary["mean_unique_answers"])
        log.info("  Saved: %s, %s", sample_path, summary_path)

        del ds, prompts, gen_output, completions_k, seq_lps_k, sample_results
        torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
