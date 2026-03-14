"""
Run ReDial benchmarks with forced character-level tokenization.

For each ReDial sample, encodes the prompt character-by-character (bypassing
subword merges) and generates an answer. Compares accuracy against canonical
tokenization to measure model robustness to non-canonical tokenization.

Follows the methodology of Zheng et al. (2025), "Broken Tokens? Your Language
Model can Secretly Handle Non-Canonical Tokenizations."

Usage:
    python scripts/characters/generate_characters.py
    python scripts/characters/generate_characters.py model=llama_8b_base
    python scripts/characters/generate_characters.py --multirun model=llama_8b_base,llama_8b_instruct
"""

import argparse
import gc
import json
import logging
import os
import re
import warnings

warnings.filterwarnings("ignore", message="resource_tracker:.*leaked semaphore")

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import dialecttax
import dialecttax.characters
import dialecttax.data.graders.mqa as mqa_grader
import dialecttax.logits

_ATTN_IMPL = "sdpa"


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


def _load_model(name, model_id, device="auto"):
    """Load a language model, reusing across multirun iterations.

    Args:
        name: Short model name (e.g. "llama_8b_base").
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
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device, attn_implementation=_ATTN_IMPL,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    _loaded_model = (model, tokenizer)
    _loaded_model_name = name
    return model, tokenizer


########
# DATA #
########

def _build_redial_prompt(ds, i, task, dialect, reasoning="naive", family=None):
    """Build system and user prompt for a ReDial MQA sample.

    Args:
        ds: Dataset list.
        i: Sample index.
        task: Task name.
        dialect: Dialect name.
        reasoning: Reasoning strategy.
        family: Model family (e.g. "qwen") — used for system-prompt addenda.

    Returns:
        Tuple of (system_prompt, user_prompt).
    """
    instructions = dialecttax.prompts.INSTS_MQA[task][reasoning][dialect]
    choices = ds[i]["choices"]
    if task == "algorithm":
        choices_str = "\n\n".join(f"{k}.\n```\n{v}\n```" for k, v in choices.items())
    else:
        choices_str = "\n".join(f"{k}. {v}" for k, v in choices.items())
    instructions = instructions.format(choices=choices_str)

    system = dialecttax.prompts.get_system_prompt(dialect, reasoning=reasoning, family=family)
    formatter = dialecttax.prompts.FORMAT_PROMPTS_REGISTRY[task]
    template = dialecttax.prompts.PROMPTS[task][reasoning][dialect]
    body = formatter(template)(ds, i)
    user_prompt = dialecttax.prompts.get_prompt(body, instructions=instructions)
    return system, user_prompt


def _build_samples(ds, task, dialect, reasoning="naive", family=None):
    """Build sample dicts for ReDial MQA entries.

    Args:
        ds: Dataset list of dicts.
        task: Task name.
        dialect: Dialect name.
        reasoning: Reasoning strategy.
        family: Model family (e.g. "qwen") — used for system-prompt addenda.

    Returns:
        List of dicts with keys: unique_id, system, prompt, answer.
    """
    samples = []
    for i, sample in enumerate(ds):
        system, prompt = _build_redial_prompt(ds, i, task, dialect, reasoning=reasoning, family=family)
        samples.append({
            "unique_id": sample["unique_id"],
            "system": system,
            "prompt": prompt,
            "answer": str(sample["answer"]),
        })
    return samples


##############
# GENERATION #
##############

def _find_answer_step(predicted, gen_ids, tokenizer):
    """Find the generation step whose token emits the predicted answer.

    Prefers the '#### <answer>' marker (case-insensitive); falls back to the
    last standalone occurrence of <answer> as a word. The fallback covers
    fuzzy-extracted answers where the model never wrote the '####' prefix,
    which is common under character-level tokenization.

    Decodes the full sequence once and maps the match's character offset back
    to a step index via binary search over cumulative decodes — preserves the
    decode-from-start semantics (which avoids the subtle single-token-decode
    concatenation issues that can drop leading spaces) while cutting the
    decode count from O(n) to O(log n).

    Args:
        predicted: The extracted answer string, or None.
        gen_ids: List of generated token IDs.
        tokenizer: Tokenizer with a decode method.

    Returns:
        The step index, or None.
    """
    if predicted is None or not gen_ids:
        return None
    full_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    pattern = re.escape(predicted)
    marker = re.search(r"####\s*" + pattern, full_text, re.IGNORECASE)
    if marker is not None:
        target = marker.end()
    else:
        fallbacks = list(re.finditer(r"\b" + pattern + r"\b", full_text, re.IGNORECASE))
        if not fallbacks:
            return None
        target = fallbacks[-1].end()
    # Binary search: smallest k with len(decode(gen_ids[:k+1])) >= target.
    # Cumulative-decode length is monotone non-decreasing in k, and target
    # was extracted from `full_text = decode(gen_ids)`, so it is bounded by
    # the length at k = len(gen_ids) - 1 — a satisfying k always exists.
    lo, hi = 0, len(gen_ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(gen_ids[:mid + 1], skip_special_tokens=True)) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _build_sample_input_ids(sample, tokenizer, instruct):
    """Tokenize a single sample's prompt with character-level user content.

    Args:
        sample: Dict with unique_id, system, prompt, answer.
        tokenizer: HuggingFace tokenizer.
        instruct: If True, wrap with chat template; only user content is
            char-tokenized.

    Returns:
        Tuple of (input_ids: list[int], n_canonical: int).
    """
    user_text = sample["prompt"].strip()
    if instruct:
        messages = dialecttax.models.get_message(user_text, system=sample["system"].strip())
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        canonical_full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        n_canonical = len(canonical_full_ids)
        user_start = full_text.find(user_text)
        user_end = user_start + len(user_text)
        prefix_ids = tokenizer.encode(full_text[:user_start], add_special_tokens=False)
        user_char_ids = dialecttax.characters.text_to_char_ids(user_text, tokenizer)
        suffix_ids = tokenizer.encode(full_text[user_end:], add_special_tokens=False)
        all_ids = prefix_ids + user_char_ids + suffix_ids
    else:
        text = f"{sample['system']}\n\n{sample['prompt']}\n"
        n_canonical = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
        all_ids = dialecttax.characters.char_tokenize(text, tokenizer).squeeze(0).tolist()
    return all_ids, n_canonical


def _pad_left(input_id_lists, pad_id, device):
    """Left-pad a list of variable-length input id sequences.

    Args:
        input_id_lists: List of list[int].
        pad_id: Padding token id.
        device: Target device.

    Returns:
        Tuple of (input_ids[B, T], attention_mask[B, T], real_lens, max_len).
    """
    real_lens = [len(ids) for ids in input_id_lists]
    max_len = max(real_lens)
    bsz = len(input_id_lists)
    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, ids in enumerate(input_id_lists):
        pad_len = max_len - len(ids)
        input_ids[i, pad_len:] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, pad_len:] = 1
    return input_ids.to(device), attention_mask.to(device), real_lens, max_len


def compute_all_char_generate(model, tokenizer, samples, instruct=False, max_tokens_new=256, batch_size=1, label="", answer_only=False, compute_hidden=True):
    """Generate answers using character-level tokenized prompts (batched).

    Pipeline per batch:
      1. Left-pad batch's input_ids; build attention mask.
      2. `model.generate(do_sample=False, output_scores=True, ...)` — fused
         greedy decode. Captures per-step logits via `output.scores` and
         (unless `answer_only`) per-step last-layer hidden states via
         `output.hidden_states`.
      3. (Skipped if `answer_only`) Re-apply `lm_head` to step 0's last-layer
         prompt hidden states to recover per-position input logits — avoids
         a second full prompt forward pass.
      4. Per sample: trim at first EOS, decode, grade, locate answer step,
         pull answer-step log_prob/entropy from `scores` and hidden state
         from `hidden_states`.

    Wrapped in `torch.inference_mode()` (skips autograd version-counter
    bookkeeping → faster than `no_grad` in tight Python-driven loops).

    `answer_only=True` skips the prompt forward and disables hidden-state
    collection — used for the big models where input-side outputs and the
    layer-stack materialization dominate runtime on long char sequences.

    `compute_hidden=False` keeps input metrics but skips per-step hidden
    states during generation, recovering input logits from a separate
    prompt-only forward pass instead. Use this for big models that need
    input entropy without paying the per-step hidden-state materialization
    cost. The answer-step hidden vector is not collected in this mode.

    Args:
        model: CausalLM in eval mode.
        tokenizer: Corresponding tokenizer (with `padding_side="left"`).
        samples: List of dicts with unique_id, system, prompt, answer.
        instruct: If True, format with chat template.
        max_tokens_new: Maximum new tokens to generate.
        batch_size: Samples per batch (default 1).
        label: Description for progress display.
        answer_only: Skip input/hidden side-outputs (default False).
        compute_hidden: If False (and not answer_only), use a separate
            prompt forward pass for input metrics and skip the answer-step
            hidden vector (default True).

    Returns:
        Dict with keys: metadata, and (unless answer_only) input_log_probs,
        input_entropy, and (if compute_hidden) hidden.
    """
    input_device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    sample_ids = []
    sample_n_canonical = []
    for s in samples:
        ids, n_can = _build_sample_input_ids(s, tokenizer, instruct)
        sample_ids.append(ids)
        sample_n_canonical.append(n_can)

    n = len(samples)
    all_input_lp = [None] * n
    all_input_ent = [None] * n
    all_hidden = [None] * n
    metadata = [None] * n
    n_correct = 0

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        bsz = batch_end - batch_start
        bs_samples = samples[batch_start:batch_end]
        bs_id_lists = sample_ids[batch_start:batch_end]
        input_ids, attention_mask, real_lens, max_len = _pad_left(
            bs_id_lists, pad_id, input_device,
        )

        # Per-batch mean log-prob / entropy buffer, written by either the
        # compute_hidden=False prompt-only path (below, pre-generation) or
        # the default compute_hidden=True path (post-generation), and read
        # by the post-process loop.
        batch_mean_lp = [0.0] * bsz
        batch_mean_ent = [0.0] * bsz

        ###########################
        # PROMPT-ONLY FORWARD     #
        # (compute_hidden=False)  #
        ###########################

        # For big models where per-step hidden states during generation
        # dominate runtime/memory, recover input metrics from a dedicated
        # prompt-only forward pass and free prompt_logits before generate()
        # so it doesn't sit in memory across the full decode. Per-token
        # arrays are not kept in this mode (only means are needed).
        if (not answer_only) and not compute_hidden:
            with torch.inference_mode():
                prompt_out = model(input_ids=input_ids, attention_mask=attention_mask)
                prompt_logits = prompt_out.logits  # (bsz, max_len, vocab)
            del prompt_out
            for i in range(bsz):
                pad_len = max_len - real_lens[i]
                if real_lens[i] >= 2:
                    logits = prompt_logits[i, pad_len:max_len - 1].cpu()
                    target = torch.tensor(bs_id_lists[i][1:], dtype=torch.long)
                    batch_mean_lp[i] = float(dialecttax.logits.compute_log_probs(logits, target).float().mean())
                    batch_mean_ent[i] = float(dialecttax.logits.compute_entropy(logits).float().mean())
            del prompt_logits
            torch.cuda.empty_cache()

        ##############
        # GENERATION #
        ##############

        with torch.inference_mode():
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens_new,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
                output_scores=True,
                output_hidden_states=(not answer_only) and compute_hidden,
                return_dict_in_generate=True,
            )

        sequences = gen_out.sequences  # (bsz, max_len + n_steps_taken)
        n_steps_taken = sequences.shape[1] - max_len

        ##################
        # INPUT METRICS  #
        # (default path) #
        ##################

        # Recover input logits from step-0 prompt hidden states + lm_head,
        # reusing the prompt forward done by generate() — keeps per-token
        # arrays for downstream npz outputs (bits-per-byte analyses, etc.).
        #
        # lm_head is applied one sample at a time. It is a position-wise linear
        # map, so per-sample application computes the identical quantity (the
        # batch and per-sample paths are bit-identical on CPU; on GPU, kernel
        # tiling can differ by shape, bounded at <=1 bf16 ULP on isolated
        # logits — measured entropy movement 6.6e-6 at 100x the realistic flip
        # rate, vs the 3.9e-4 the verify probe tolerates). The win: the loop
        # below only ever consumes one sample's logits, so materializing the
        # (bsz, max_len, vocab) batch tensor — ~14 GB at bsz=8 on the longest
        # algorithm prompt with gemma's 262k vocab — bought nothing and was
        # what capped compute_batch_size at 4.
        if (not answer_only) and compute_hidden:
            prompt_hidden = gen_out.hidden_states[0][-1]  # (bsz, max_len, H)
            for i in range(bsz):
                pad_len = max_len - real_lens[i]
                if real_lens[i] >= 2:
                    with torch.inference_mode():
                        logits = model.lm_head(prompt_hidden[i, pad_len:max_len - 1]).cpu()
                    target = torch.tensor(bs_id_lists[i][1:], dtype=torch.long)
                    inp_lp = dialecttax.logits.compute_log_probs(logits, target).float().numpy()
                    inp_ent = dialecttax.logits.compute_entropy(logits).float().numpy()
                else:
                    inp_lp = np.zeros(0, dtype=np.float32)
                    inp_ent = np.zeros(0, dtype=np.float32)
                all_input_lp[batch_start + i] = inp_lp
                all_input_ent[batch_start + i] = inp_ent
                batch_mean_lp[i] = float(inp_lp.mean()) if len(inp_lp) > 0 else 0.0
                batch_mean_ent[i] = float(inp_ent.mean()) if len(inp_ent) > 0 else 0.0

        ################
        # POST-PROCESS #
        ################

        for i in range(bsz):
            sample = bs_samples[i]
            real_len = real_lens[i]

            # Trim at first EOS (per-sample; batched generate may have run
            # extra steps for other samples in the batch).
            gen_ids = []
            for tid in sequences[i, max_len:].tolist():
                if eos_id is not None and tid == eos_id:
                    break
                gen_ids.append(tid)

            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            predicted = mqa_grader.extract_answer(gen_text)
            if predicted is None:
                predicted = mqa_grader.extract_answer_fuzzy(gen_text)
            gold = sample["answer"]
            correct = mqa_grader.grade(predicted, gold)
            if correct:
                n_correct += 1

            # Answer-step metrics: log-prob/entropy at the step whose token
            # is the predicted answer letter (A/B/C/D). Computed only for
            # the answer step (not all steps) to avoid O(n_gen) overhead.
            answer_step = _find_answer_step(predicted, gen_ids, tokenizer)
            answer_lp = None
            answer_ent = None
            if answer_step is not None and answer_step < len(gen_ids):
                step_logits = gen_out.scores[answer_step][i:i + 1].cpu()
                tid = gen_ids[answer_step]
                answer_lp = float(dialecttax.logits.compute_log_probs(step_logits, torch.tensor([tid]))[0].item())
                answer_ent = float(dialecttax.logits.compute_entropy(step_logits)[0].item())

            if (not answer_only) and compute_hidden:
                emb = None
                if answer_step is not None and answer_step < n_steps_taken:
                    # Step 0's hidden state lives at the prompt's last real
                    # position (with left padding, that's `-1`); subsequent
                    # steps have a length-1 sequence dim.
                    if answer_step == 0:
                        h = gen_out.hidden_states[0][-1][i, -1, :]
                    else:
                        h = gen_out.hidden_states[answer_step][-1][i, 0, :]
                    emb = h.cpu().float().numpy()
                all_hidden[batch_start + i] = emb

            meta = {
                "unique_id": sample["unique_id"],
                "n_char_tokens": real_len,
                "n_canonical_tokens": sample_n_canonical[batch_start + i],
                "char_expansion": real_len / max(sample_n_canonical[batch_start + i], 1),
                "n_gen_tokens": len(gen_ids),
                "gold_answer": gold,
                "predicted_answer": predicted,
                "correct": correct,
                "completion": gen_text,
                "answer_log_prob": answer_lp,
                "answer_entropy": answer_ent,
            }
            if not answer_only:
                meta["input_mean_log_prob"] = batch_mean_lp[i]
                meta["input_mean_entropy"] = batch_mean_ent[i]
            metadata[batch_start + i] = meta

        del gen_out
        torch.cuda.empty_cache()

        idx = batch_end - 1
        acc = n_correct / (idx + 1) * 100
        print(
            f"\r  ({label}) {idx + 1}/{n} acc={acc:.1f}% bsz={bsz} max_len={max_len}",
            end="", flush=True,
        )

    print()

    result = {"metadata": metadata}
    if (not answer_only) and compute_hidden:
        result["input_log_probs"] = all_input_lp
        result["input_entropy"] = all_input_ent
        result["hidden"] = all_hidden
    return result


########
# MAIN #
########

ALL_NONE_MARKER = ".all_none_answer_entropy"


def _is_config_done(metadata_path, expected_n):
    """True if a config's metadata.jsonl is fully populated.

    "Done" means one of:
      (a) file exists, has exactly `expected_n` rows, every row carries
          the `answer_entropy` key, and at least one row has a non-None
          value; or
      (b) the `.all_none_answer_entropy` marker exists (a prior run
          completed but every row's answer_entropy was None — rerunning
          would produce the same result).

    A None value on an individual row is a legitimate outcome (the answer
    letter was extracted via fuzzy text match but couldn't be located as a
    discrete generated token). A row missing the key entirely predates the
    answer-step pipeline and needs regeneration.

    Args:
        metadata_path: Path to metadata.jsonl.
        expected_n: Required number of rows.

    Returns:
        True if config is fully done; False otherwise.
    """
    out_dir = os.path.dirname(metadata_path)
    if os.path.exists(os.path.join(out_dir, ALL_NONE_MARKER)):
        return True
    if not os.path.exists(metadata_path):
        return False
    n = 0
    any_non_none = False
    try:
        with open(metadata_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                if not isinstance(d, dict):
                    return False
                n += 1
                if "answer_entropy" not in d:
                    return False
                if d["answer_entropy"] is not None:
                    any_non_none = True
    except (OSError, json.JSONDecodeError):
        return False
    return n == expected_n and any_non_none


def _compute_and_save(cfg: DictConfig, model, tokenizer, samples, out_dir: str) -> None:
    """Run compute_all_char_generate and save outputs for one config.

    Shared between generate_characters.main and verify_characters re-runs so
    the verifier can re-run inline on mismatch without spawning a subprocess
    that would fight the parent for GPU memory.

    Args:
        cfg: Hydra config (uses model.name, dataset.name, task.name,
            reasoning.name, dialect.name, model.max_tokens_new,
            reasoning.max_tokens_new, batch_size, answer_only).
        model: Loaded HF model.
        tokenizer: Loaded HF tokenizer.
        samples: List of sample dicts from _build_samples.
        out_dir: Hydra runtime output dir.
    """
    dataset_name = cfg.dataset.name
    task = cfg.task.name
    reasoning = cfg.reasoning.name
    dialect = cfg.dialect.name
    answer_only = bool(cfg.get("answer_only", False))
    compute_hidden = bool(cfg.get("compute_hidden", True))

    label = f"{dataset_name}/{task}/{reasoning}/{dialect}/char"
    instruct = cfg.model.name.endswith("_instruct")
    variant = "instruct" if instruct else "base"
    max_tokens_new = int(cfg.reasoning.max_tokens_new[variant])
    if "max_tokens_new" in cfg.model and reasoning in cfg.model.max_tokens_new:
        max_tokens_new = int(cfg.model.max_tokens_new[reasoning])

    # A full recompute is unconstrained by verify's probe (which must reuse the
    # batch generate wrote with), and its cost is 1024 sequential decode steps
    # per batch -- so it may run wider. Falls back to batch_size when unset.
    compute_batch_size = cfg.get("compute_batch_size", None)
    batch_size = int(compute_batch_size) if compute_batch_size else int(cfg.get("batch_size", 1))

    results = compute_all_char_generate(
        model, tokenizer, samples,
        instruct=instruct, max_tokens_new=max_tokens_new,
        batch_size=batch_size, label=label, answer_only=answer_only,
        compute_hidden=compute_hidden,
    )

    os.makedirs(out_dir, exist_ok=True)
    if (not answer_only) and compute_hidden:
        np.savez(os.path.join(out_dir, "input_log_probs.npz"), *results["input_log_probs"])
        np.savez(os.path.join(out_dir, "input_entropy.npz"), *results["input_entropy"])
        valid_hidden = {str(i): h for i, h in enumerate(results["hidden"]) if h is not None}
        np.savez(os.path.join(out_dir, "hidden.npz"), **valid_hidden)

    metadata = results["metadata"]
    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    n_correct = sum(1 for m in metadata if m["correct"])
    mean_expansion = np.mean([m["char_expansion"] for m in metadata])
    if (not answer_only) and compute_hidden:
        log.info(f"Saved {len(valid_hidden)}/{len(samples)} hidden vectors to: {out_dir}/hidden.npz")
    log.info(f"Saved {len(samples)} metadata entries to: {out_dir}/metadata.jsonl")
    log.info(f"Accuracy: {n_correct}/{len(metadata)} ({n_correct / len(metadata) * 100:.1f}%)")
    log.info(f"Mean char/canonical expansion: {mean_expansion:.2f}x")


@hydra.main(version_base=None, config_path="../../configs/generate_characters", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    task = cfg.task.name
    dialect = cfg.dialect.name
    reasoning = cfg.reasoning.name

    # Skip invalid combos
    valid_dialects = list(cfg.dataset.dialects)
    if dialect not in valid_dialects:
        log.info(f"Skipping (dialect '{dialect}' not in {valid_dialects} for dataset '{dataset_name}')")
        return
    if "tasks" in cfg.dataset and task not in list(cfg.dataset.tasks):
        log.info(f"Skipping (task '{task}' not in {list(cfg.dataset.tasks)} for dataset '{dataset_name}')")
        return

    out_dir = HydraConfig.get().runtime.output_dir

    log.info(
        f"\n{'=' * 60}\n"
        f"  Model:          {cfg.model.name} ({cfg.model.model_id})\n"
        f"  Dataset:        {dataset_name}\n"
        f"  Task:           {task}\n"
        f"  Reasoning:      {reasoning}\n"
        f"  Dialect:        {dialect}\n"
        f"  Tokenization:   character-level\n"
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

    metadata_path = os.path.join(out_dir, "metadata.jsonl")
    done = _is_config_done(metadata_path, expected_n=len(ds))
    if done and not cfg.rerun:
        log.info(f"Skipping (already done): {out_dir}")
        return
    if done:
        # _compute_and_save truncates metadata.jsonl and overwrites the .npz
        # files, so this replaces the config's outputs rather than appending.
        log.info(f"rerun=true, so recomputing: {out_dir}")

    # Build samples
    family = cfg.model.name.split("_")[0]
    samples = _build_samples(ds, task, dialect, reasoning=reasoning, family=family)
    log.info(f"Built {len(samples)} samples for character-level generation")

    # Load model
    try:
        model, tokenizer = _load_model(cfg.model.name, cfg.model.model_id, device=cfg.device)
    except OSError as e:
        log.warning(f"Skipping model '{cfg.model.name}' ({cfg.model.model_id}): {e}")
        return

    _compute_and_save(cfg, model, tokenizer, samples, out_dir)

    if not _is_config_done(metadata_path, expected_n=len(ds)):
        log.warning(f"All answer_entropy values are None; writing {ALL_NONE_MARKER} to prevent infinite rerun: {out_dir}")
        with open(os.path.join(out_dir, ALL_NONE_MARKER), "w") as f:
            f.write("")


if __name__ == "__main__":
    main()
