"""
Surprisingly Popular (SP) evaluation on preprocessed ReDial datasets.

Performs a single greedy decode per prompt, then computes SP scores from two
genuinely different signals:
  - vote_share    = P(answer | prompt + reasoning + ####)
  - predicted_share = P(answer | prompt + ####)
  - sp_score      = vote_share - predicted_share

The SP algorithm surfaces answers that reasoning makes more likely: the
"surprisingly popular" answer is the one whose post-reasoning probability
most exceeds its prompt-only probability.

Answers are extracted as multi-token spans: all generated tokens after the
#### marker until whitespace, newline, or EOS.  Vote and predicted shares
are geometric means of per-token softmax probabilities across the span.

Usage:
    # Single run
    python scripts/ReDial/surprisingly_popular.py

    # Sweep
    python scripts/ReDial/surprisingly_popular.py -m \
      model=llama_base,qwen_base reasoning=naive,cot
"""

import gc
import json
import logging
import os

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

def greedy_decode(model, tokenizer, prompts, *, max_new_tokens, batch_size):
    """Greedy decode all prompts.

    Args:
        model: HuggingFace causal LM.
        tokenizer: corresponding tokenizer.
        prompts: list of prompt strings.
        max_new_tokens: max tokens to generate per prompt.
        batch_size: batch size for generation.

    Returns:
        Dict with completions (list[str]), scores (list[tuple[Tensor]]),
        and generated_ids (list[Tensor]).
    """
    tokenizer.padding_side = "left"
    n = len(prompts)
    n_batches = (n + batch_size - 1) // batch_size

    all_completions = []
    all_scores = []
    all_gen_ids = []

    for start in tqdm(range(0, n, batch_size), total=n_batches,
                      desc="Greedy decode", unit="batch"):
        batch_prompts = prompts[start:start + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = outputs.sequences[:, input_len:]
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for j in range(len(batch_prompts)):
            all_completions.append(decoded[j])
            all_scores.append(tuple(s[j].cpu() for s in outputs.scores))
            all_gen_ids.append(gen_ids[j].cpu())

        del inputs, outputs
        torch.cuda.empty_cache()

    return {
        "completions": all_completions,
        "scores": all_scores,
        "generated_ids": all_gen_ids,
    }


######################
# CANDIDATE EXTRACT  #
######################

def _find_answer_spans(generated_ids, tokenizer):
    """Find the token span of the answer (after ####) for each sample.

    The answer span extends from the first non-whitespace token after
    the #### marker until a whitespace/newline token or end of sequence.

    Args:
        generated_ids: list of 1-D token ID tensors (one per sample).
        tokenizer: tokenizer for decoding marker and whitespace detection.

    Returns:
        List of (start, end) tuples or None if #### marker was not found.
    """
    marker_ids = tokenizer.encode("####", add_special_tokens=False)
    marker_len = len(marker_ids)
    marker_t = torch.tensor(marker_ids, dtype=generated_ids[0].dtype)

    spans = []
    for ids in generated_ids:
        found = None
        for pos in range(len(ids) - marker_len + 1):
            if torch.equal(ids[pos:pos + marker_len], marker_t):
                # Skip marker + optional whitespace token
                start = pos + marker_len
                if start < len(ids):
                    if tokenizer.decode([ids[start]]).strip() == "":
                        start += 1
                if start >= len(ids):
                    break
                # Read until whitespace, newline, or EOS
                end = start
                while end < len(ids):
                    if ids[end].item() == tokenizer.eos_token_id:
                        break
                    decoded = tokenizer.decode([ids[end]])
                    if decoded.strip() == "" or "\n" in decoded:
                        break
                    end += 1
                if end > start:
                    found = (start, end)
                break
        spans.append(found)
    return spans


def _build_candidates(
    model, tokenizer, prompts, generated_ids, scores, answer_spans, m, batch_size,
):
    """Build up to m candidate answers per sample.

    The first candidate is the greedy answer extracted from the generated
    tokens.  The remaining m-1 come from alternative first tokens at the
    answer position, each continued greedily until whitespace.

    Args:
        model: HuggingFace causal LM.
        tokenizer: corresponding tokenizer.
        prompts: list of prompt strings.
        generated_ids: list of 1-D token ID tensors (generated portion only).
        scores: list of tuples of per-step score tensors.
        answer_spans: list of (start, end) tuples or None.
        m: maximum number of candidates per sample.
        batch_size: batch size for alternative generation.

    Returns:
        Tuple of (candidates, vote_shares) where:
          candidates: list[list[str]] — up to m candidate answer strings.
          vote_shares: list[dict[str, float]] — answer → geometric mean prob.
    """
    n = len(prompts)
    all_candidates = [[] for _ in range(n)]
    all_vote_shares = [{} for _ in range(n)]
    alt_jobs = []

    # Phase 1: greedy answers + queue alternative first tokens
    for i, span in enumerate(answer_spans):
        if span is None:
            continue
        start, end = span
        greedy_str = tokenizer.decode(generated_ids[i][start:end]).strip()
        if not greedy_str:
            continue

        # Greedy vote share (geometric mean of per-token softmax probs)
        log_probs = []
        for t in range(start, min(end, len(scores[i]))):
            logits = scores[i][t]
            probs = F.softmax(logits, dim=-1)
            log_probs.append(torch.log(probs[generated_ids[i][t]]).item())
        greedy_vs = float(np.exp(np.mean(log_probs))) if log_probs else 0.0

        all_candidates[i].append(greedy_str)
        all_vote_shares[i][greedy_str] = greedy_vs

        # Top-m alternative first tokens
        if m > 1 and start < len(scores[i]):
            first_probs = F.softmax(scores[i][start], dim=-1)
            topk = torch.topk(first_probs, min(m, len(first_probs)))
            greedy_first = generated_ids[i][start].item()

            for tok_id, prob in zip(topk.indices, topk.values):
                tok_val = tok_id.item()
                if tok_val == greedy_first or len(all_candidates[i]) >= m:
                    continue
                prefix = (
                    tokenizer.encode(prompts[i])
                    + generated_ids[i][:start].tolist()
                    + [tok_val]
                )
                alt_jobs.append({
                    "sample_idx": i,
                    "prefix_ids": prefix,
                    "first_tok_prob": prob.item(),
                })
                all_candidates[i].append(None)  # placeholder

    # Phase 2: batch-generate continuations for alternatives
    tokenizer.padding_side = "left"
    for batch_start in tqdm(
        range(0, len(alt_jobs), batch_size),
        desc="Alt candidates", unit="batch", disable=not alt_jobs,
    ):
        batch = alt_jobs[batch_start : batch_start + batch_size]
        max_len = max(len(j["prefix_ids"]) for j in batch)
        padded, attn = [], []
        for j in batch:
            pad = max_len - len(j["prefix_ids"])
            padded.append([tokenizer.pad_token_id] * pad + j["prefix_ids"])
            attn.append([0] * pad + [1] * len(j["prefix_ids"]))

        input_ids = torch.tensor(padded, device=model.device)
        attention_mask = torch.tensor(attn, device=model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=10,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        cont_ids = outputs.sequences[:, max_len:]
        for k, job in enumerate(batch):
            i = job["sample_idx"]
            first_tok = job["prefix_ids"][-1]

            # Read continuation until whitespace / newline / EOS
            tokens = []
            for t in range(cont_ids.shape[1]):
                tok = cont_ids[k, t].item()
                if tok == tokenizer.eos_token_id:
                    break
                decoded = tokenizer.decode([tok])
                if decoded.strip() == "" or "\n" in decoded:
                    break
                tokens.append(tok)

            answer_tokens = [first_tok] + tokens
            answer_str = tokenizer.decode(answer_tokens).strip()
            if not answer_str:
                answer_str = tokenizer.decode([first_tok]).strip()

            # Vote share: geometric mean across all answer tokens
            lp = [np.log(job["first_tok_prob"])]
            for t_idx, tok in enumerate(tokens):
                if t_idx < len(outputs.scores):
                    p = F.softmax(outputs.scores[t_idx][k], dim=-1)
                    lp.append(torch.log(p[tok]).item())
            vote_share = float(np.exp(np.mean(lp)))

            # Fill first remaining placeholder for this sample
            try:
                ph = all_candidates[i].index(None)
                all_candidates[i][ph] = answer_str
            except ValueError:
                pass
            if answer_str not in all_vote_shares[i]:
                all_vote_shares[i][answer_str] = vote_share

        del input_ids, attention_mask, outputs, cont_ids
        torch.cuda.empty_cache()

    # Phase 3: deduplicate, drop remaining Nones
    for i in range(n):
        seen = set()
        deduped = []
        for c in all_candidates[i]:
            if c is not None and c not in seen:
                seen.add(c)
                deduped.append(c)
        all_candidates[i] = deduped

    return all_candidates, all_vote_shares


def _prompt_only_scores(model, tokenizer, prompts, candidates_per_sample, batch_size):
    """Compute predicted shares via prompt-only forward pass with teacher forcing.

    For each (sample, candidate) pair, constructs prompt + "####" + " answer",
    runs a single forward pass, and reads per-token probabilities at the
    answer positions.  The predicted share is the geometric mean.

    Args:
        model: HuggingFace causal LM.
        tokenizer: corresponding tokenizer.
        prompts: list of original prompt strings.
        candidates_per_sample: list[list[str]] — candidate answers per sample.
        batch_size: batch size for forward pass.

    Returns:
        predicted_shares: list[dict[str, float]] — answer → geometric mean
        prob per sample (from prompt-only logits).
    """
    tokenizer.padding_side = "left"
    n = len(prompts)

    # Flatten (sample, candidate) pairs into jobs.
    # Tokenize prefix and answer separately to avoid BPE boundary issues.
    jobs = []
    for i in range(n):
        prefix_ids = tokenizer.encode(prompts[i] + "####")
        for c in candidates_per_sample[i]:
            answer_ids = tokenizer.encode(" " + c, add_special_tokens=False)
            jobs.append({
                "sample_idx": i,
                "candidate": c,
                "full_ids": prefix_ids + answer_ids,
                "prefix_len": len(prefix_ids),
                "answer_ids": answer_ids,
            })

    all_predicted_shares = [{} for _ in range(n)]

    for batch_start in tqdm(
        range(0, len(jobs), batch_size),
        desc="Prompt-only pass", unit="batch", disable=not jobs,
    ):
        batch = jobs[batch_start : batch_start + batch_size]
        max_len = max(len(j["full_ids"]) for j in batch)

        padded, attn = [], []
        for j in batch:
            pad = max_len - len(j["full_ids"])
            padded.append([tokenizer.pad_token_id] * pad + j["full_ids"])
            attn.append([0] * pad + [1] * len(j["full_ids"]))

        input_ids = torch.tensor(padded, device=model.device)
        attention_mask = torch.tensor(attn, device=model.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        for k, j in enumerate(batch):
            ans_ids = j["answer_ids"]
            plen = j["prefix_len"]
            pad_len = max_len - len(j["full_ids"])

            if not ans_ids:
                all_predicted_shares[j["sample_idx"]][j["candidate"]] = 0.0
                continue

            # Logits predicting ans_ids[t] are at position pad_len + plen - 1 + t
            # (causal LM: logits[pos] predicts the token at pos+1)
            log_probs = []
            for t, tok_id in enumerate(ans_ids):
                pos = pad_len + plen - 1 + t
                if 0 <= pos < outputs.logits.shape[1]:
                    p = F.softmax(outputs.logits[k, pos], dim=-1)
                    log_probs.append(torch.log(p[tok_id]).item())

            predicted = float(np.exp(np.mean(log_probs))) if log_probs else 0.0
            all_predicted_shares[j["sample_idx"]][j["candidate"]] = predicted

        del input_ids, attention_mask, outputs
        torch.cuda.empty_cache()

    return all_predicted_shares


##########
# VOTING #
##########

def _sp_vote(candidates, vote_shares, predicted_shares, gold):
    """SP vote from pre-computed vote and predicted shares.

    Args:
        candidates: list of candidate answer strings.
        vote_shares: dict mapping candidate → vote share (from reasoning logits).
        predicted_shares: dict mapping candidate → predicted share (prompt-only).
        gold: gold answer string.

    Returns:
        Dict with winner, correct, sp_score, vote_shares, predicted_shares,
        sp_scores.
    """
    norm = dialecttax.data.graders.math.normalize_answer
    gold_norm = norm(gold)

    if not candidates:
        return {
            "winner": None,
            "correct": False,
            "sp_score": 0.0,
            "vote_shares": {},
            "predicted_shares": {},
            "sp_scores": {},
        }

    sp_scores = {}
    for c in candidates:
        vs = vote_shares.get(c, 0.0)
        ps = predicted_shares.get(c, 0.0)
        sp_scores[c] = vs - ps

    winner = max(sp_scores, key=sp_scores.get)

    return {
        "winner": winner,
        "correct": norm(winner) == gold_norm,
        "sp_score": sp_scores[winner],
        "vote_shares": vote_shares,
        "predicted_shares": predicted_shares,
        "sp_scores": sp_scores,
    }


###########
# GRADING #
###########

def grade_samples(
    completions,
    candidates_per_sample,
    vote_shares_per_sample,
    predicted_shares_per_sample,
    gold_answers,
    m,
):
    """Grade each sample with greedy accuracy + SP voting.

    Args:
        completions: list of greedy-decoded completion strings.
        candidates_per_sample: list[list[str]] — candidates per sample.
        vote_shares_per_sample: list[dict[str, float]] — vote shares per sample.
        predicted_shares_per_sample: list[dict[str, float]] — predicted shares.
        gold_answers: list of gold answer strings.
        m: number of candidate answers.

    Returns:
        List of per-sample result dicts.
    """
    results = []
    for i, (comp, cands, vs, ps, gold) in enumerate(zip(
        completions, candidates_per_sample, vote_shares_per_sample,
        predicted_shares_per_sample, gold_answers,
    )):
        greedy_answer = dialecttax.data.graders.math.extract_answer(comp)
        greedy_correct = dialecttax.data.graders.math.grade(greedy_answer, gold)

        sp = _sp_vote(cands, vs, ps, gold)

        results.append({
            "gold": gold,
            "m": m,
            "greedy_answer": greedy_answer,
            "greedy_correct": greedy_correct,
            "sp_winner": sp["winner"],
            "sp_correct": sp["correct"],
            "sp_score": sp["sp_score"],
            "candidates": cands,
            "vote_shares": sp["vote_shares"],
            "predicted_shares": sp["predicted_shares"],
            "sp_scores": sp["sp_scores"],
        })

    return results


#######
# CLI #
#######

@hydra.main(version_base=None, config_path="../../configs/surprisingly_popular", config_name="config")
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
        log.info("Evaluating %d samples, m=%d (%d excluded as demos)",
                 len(ds), cfg.m, len(demo_indices))

        # 1. Greedy decode
        gen_output = greedy_decode(
            model,
            tokenizer,
            prompts,
            max_new_tokens=cfg.reasoning.max_new_tokens,
            batch_size=cfg.batch_size,
        )
        completions = gen_output["completions"]
        scores = gen_output["scores"]
        generated_ids = gen_output["generated_ids"]

        # 2. Find answer spans (multi-token, until whitespace/newline)
        answer_spans = _find_answer_spans(generated_ids, tokenizer)

        # 3. Build m candidates per sample (greedy + alternatives)
        candidates, vote_shares = _build_candidates(
            model, tokenizer, prompts, generated_ids, scores,
            answer_spans, cfg.m, cfg.batch_size,
        )

        # 4. Prompt-only forward pass for predicted shares
        predicted_shares = _prompt_only_scores(
            model, tokenizer, prompts, candidates, cfg.batch_size,
        )

        # 5. Grade samples
        gold_answers = [sample["answer"] for sample in ds]
        sample_results = grade_samples(
            completions, candidates, vote_shares,
            predicted_shares, gold_answers, cfg.m,
        )

        # Run-level summary
        n = len(sample_results)
        run_summary = {
            "model": cfg.model.name,
            "dialect": dialect,
            "reasoning": reasoning,
            "m": cfg.m,
            "n_samples": n,
            "greedy_accuracy": float(np.mean([s["greedy_correct"] for s in sample_results])),
            "sp_accuracy": float(np.mean([s["sp_correct"] for s in sample_results])),
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

        log.info("[%s] %s / %s / m=%d", dialect, cfg.model.name, reasoning, cfg.m)
        log.info("  greedy_accuracy: %.3f", run_summary["greedy_accuracy"])
        log.info("  sp_accuracy:     %.3f", run_summary["sp_accuracy"])
        log.info("  Saved: %s, %s", sample_path, summary_path)

        del ds, prompts, gen_output, completions, scores, generated_ids
        del candidates, vote_shares, predicted_shares, sample_results
        torch.cuda.empty_cache()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
