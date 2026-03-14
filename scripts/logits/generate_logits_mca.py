"""
Compute answer-letter log-probabilities for ReDial MQA samples.

Teacher-forces the response prefix "####" onto each prompt and reads off
the log-probability of every multiple-choice answer letter from the next-
token distribution. One prompt-only forward pass per sample — no
autoregressive loop, no hidden-state capture — so the cost is a small
fraction of generate_logits.py.

For each sample and each letter L, we compute P(" L" | body + "####") via
the tokenizer's canonical tokenisation of " L" in that context. For BPE
tokenizers used by Llama / Gemma / Qwen the canonical form is a single
merged token (e.g. Llama id 362 decodes to " A"), so the whole letter
log-probability is a direct lookup in the last-position log-softmax.

Robustness: we probe per-sample whether the tokenisation of "#### L" is a
strict extension of "####" (i.e. the tokenizer doesn't reshuffle across
the boundary). If so, the letter's continuation tokens are known and we
lookup/sum them. If not (rare: would indicate an unusual tokenizer
boundary merge), we fall back to an independent teacher-forced forward
pass on the full "#### L" continuation.

We do NOT marginalise over alternative tokenisations (e.g. " " + "A"):
for every tokenizer we use the canonical merged form dominates probability
mass by ~6+ orders of magnitude, and the relative ordering / restricted
softmax / per-letter margins are effectively unchanged. If you ever need
the rigorous sum, build a TokenTrie and use find_segmentations — see
~/contracts/src/contracts/tokenizers.py.

Answer choice letters per sample are taken from the dataset and can vary
in count (ReDial 'logic' has both 3- and 4-choice items; other tasks are
uniformly 4-choice).

For CoT prompts this measures the model's forced-commit prior — the
probability it would assign the answer if not allowed to reason first.
That is meaningful but different from the generate-then-extract accuracy
reported by generate_logits.py.

Usage:
    python scripts/logits/generate_logits_mca.py
    python scripts/logits/generate_logits_mca.py model=llama_8b_base
    python scripts/logits/generate_logits_mca.py --multirun model=llama_8b_base,llama_3b_base \\
        task=math,logic dialect=sae,aave
    python scripts/logits/generate_logits_mca.py --multirun model=llama_8b_base \\
        task=math,logic dialect=sae perturbation=swap,translate_french
"""

import argparse
import gc
import json
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
import dialecttax.logits


#########
# SETUP #
#########

def setup_hydra():
    """Register OmegaConf resolver and apply Hydra compatibility patches."""
    project_config = dialecttax.utils.load_config(os.environ.get("DIALECTTAX_CONFIG", "default"))
    OmegaConf.register_new_resolver(
        "project", lambda key: project_config["directories"][key],
        replace=True,
    )

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

# "####" (no trailing space) is the teacher-forced response prefix. The
# grader regex r"####\s*[A-E]" tolerates any whitespace; the model, trained
# on "#### A"-style completions, emits a merged " A" token next — which
# contains its leading space and gives us a single-token letter lookup.
ANSWER_PREFIX = "####"


#########
# MODEL #
#########

_loaded_model = None
_loaded_model_name = None


def _load_model(name, model_id, device="auto"):
    """Load a language model, reusing across multirun iterations."""
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

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
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


def _build_redial_prompt(ds, i, task, dialect, reasoning="naive"):
    """Same prompt construction used by generate_logits / generate_characters."""
    instructions = dialecttax.prompts.INSTS_MQA[task][reasoning][dialect]

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


def _build_samples(ds, task, dialect, reasoning="naive"):
    """Build per-sample dicts. Includes the per-sample choice list because
    ReDial 'logic' has mixed 3- and 4-choice items."""
    samples = []
    for i, sample in enumerate(ds):
        system, prompt = _build_redial_prompt(ds, i, task, dialect, reasoning=reasoning)
        samples.append({
            "unique_id": sample["unique_id"],
            "system": system,
            "prompt": prompt,
            "answer": str(sample["answer"]),
            "choices": list(sample["choices"].keys()),
        })
    return samples


###############
# COMPUTATION #
###############

def _format_body_text(sample, tokenizer, instruct):
    """Assemble the prompt body without any response-prefix suffix."""
    if instruct:
        messages = dialecttax.models.get_message(
            sample["prompt"].strip(),
            system=sample["system"].strip(),
        )
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    return f"{sample['system']}\n\n{sample['prompt']}\n"


def _format_input_text(sample, tokenizer, instruct):
    """Body text plus the ANSWER_PREFIX that naive teacher-forcing pins to."""
    return _format_body_text(sample, tokenizer, instruct) + ANSWER_PREFIX


def _letter_continuation_ids(tokenizer, prefix_text, letters):
    """For each answer letter, find the canonical token id sequence that
    tokenises "{prefix_text} {letter}" beyond the tokenisation of prefix_text.

    Returns:
        (cont_ids, clean) — cont_ids maps letter -> list[int], and clean is
        True iff every letter's tokenisation extends the prefix as a strict
        prefix (i.e. no boundary reshuffle). When clean is False, individual
        letter entries may be None, indicating the caller should fall back
        to a separate full-text forward pass for that letter.
    """
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    cont_ids = {}
    clean = True
    for letter in letters:
        full_ids = tokenizer.encode(prefix_text + " " + letter, add_special_tokens=False)
        if full_ids[: len(prefix_ids)] == prefix_ids:
            cont_ids[letter] = full_ids[len(prefix_ids):]
        else:
            clean = False
            cont_ids[letter] = None
    return cont_ids, clean


def _teacher_force_letter_logp(model, tokenizer, full_text, prefix_ids, device):
    """Fallback: compute logp of everything after len(prefix_ids) in
    tokenize(full_text), as a sum of per-position log-softmax values.

    Used only when the tokeniser reshuffles the "####|letter" boundary
    and the continuation isn't a strict tail of the prefix encoding.
    """
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    # Find the divergence point — where full_ids first differs from prefix_ids
    j = 0
    while j < min(len(prefix_ids), len(full_ids)) and prefix_ids[j] == full_ids[j]:
        j += 1
    # Only positions start..len-2 are read, so compute lm_head for just that tail;
    # a full-sequence lm_head over a 256k vocab OOMs large models.
    start = max(j - 1, 0)
    keep = len(full_ids) - start
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False, logits_to_keep=keep)
    # Kept rows are the last `keep` positions; row r == absolute position start + r.
    lp = torch.log_softmax(out.logits[0].float(), dim=-1).cpu()
    # Sum logp at positions start, ..., len(full_ids)-2 (each predicts the next token)
    total = 0.0
    for pos in range(start, len(full_ids) - 1):
        total += float(lp[pos - start, full_ids[pos + 1]])
    return total


#######################
# CoT: SENTINEL FLOW  #
#######################

SENTINEL_TOKEN = "<|mca_answer|>"


def _register_sentinel(tokenizer):
    """Register SENTINEL_TOKEN as an additional special token and return its id.

    Idempotent — if the token was added earlier (e.g. multirun), we just look
    up the existing id. We never feed the sentinel to the model, so there is
    no need to resize model embeddings.
    """
    if SENTINEL_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens(
            {"additional_special_tokens": [SENTINEL_TOKEN]},
        )
    return tokenizer.convert_tokens_to_ids(SENTINEL_TOKEN)


def _align_template_to_generation(template_ids, generated_ids):
    """Walk both sequences and map every template position to a generated
    position.

    We handle the specific structure of CoT generation where the template
    is the *pinned* form "[body, ####, <|answer|>]" and the generation is
    "[body, reasoning..., ####, letter...]". That is: the body prefix
    matches token-for-token, then insertions appear (the CoT reasoning),
    then the anchor "####" matches, then a one-position "hole" where the
    sentinel sits in the template and the answer letter token sits in the
    generation.

    Args:
        template_ids: Tokenisation of "body + #### + <|answer|>".
        generated_ids: Tokenisation of "body + <generated content>".

    Returns:
        Dict template_pos -> generated_pos, or None if alignment fails
        (e.g. "####" never appears in the generation).
    """
    n_t, n_g = len(template_ids), len(generated_ids)
    mapping = {}
    i, j = 0, 0
    while i < n_t:
        # Sentinel is always the last template token. It has no counterpart
        # in the generation, so we pin it to the current generation index
        # (the position where a real answer token sits, or len(gen) if the
        # generation ended at "####").
        if i == n_t - 1:
            mapping[i] = j
            return mapping
        if j >= n_g:
            # Ran out of generation before exhausting non-sentinel template
            return None
        if template_ids[i] == generated_ids[j]:
            mapping[i] = j
            i += 1
            j += 1
            continue
        # Scan forward in the generation for the current template token
        # (skipping CoT reasoning insertions).
        while j < n_g and generated_ids[j] != template_ids[i]:
            j += 1
        if j >= n_g:
            return None
        mapping[i] = j
        i += 1
        j += 1
    return mapping


def compute_answer_logprobs_cot(
    model, tokenizer, samples, instruct=False, max_new_tokens=1024, label="",
):
    """Generate CoT reasoning, then extract answer-letter log-probs at the
    position of the sentinel in the template-generation alignment.

    Implementation:
      1. Register a sentinel as a tokenizer special token (for position
         accounting only — never fed to the model).
      2. For each sample, build two tokenisations:
          - body_ids: the prompt (no suffix) — what we feed the model.
          - template_ids: body + "####" + sentinel — the alignment target.
      3. Greedy-generate with output_scores=True.
      4. Align template_ids to (body_ids + generated_ids); the sentinel
         position maps to the generated position where the answer letter
         sits — and the prior step's logits are the answer distribution.
      5. If the model never emits "####" (alignment fails), record a
         no-answer fallback and skip.
    """
    device = next(model.parameters()).device
    sentinel_id = _register_sentinel(tokenizer)

    # The "####" anchor is a single token in Llama/Gemma/Qwen; if not,
    # alignment still works but is marginally less selective.
    hash_ids = tokenizer.encode("####", add_special_tokens=False)
    log.info(f"Sentinel id: {sentinel_id}, '####' token ids: {hash_ids}")

    # Probe letter continuation tokens once using a canonical prefix. We
    # reuse _letter_continuation_ids, which handles the '####|letter'
    # boundary via strict-prefix subtraction.
    all_letters = sorted({l for s in samples for l in s["choices"]})
    probe_prefix = "placeholder text\n\n" + ANSWER_PREFIX
    probe_cont, probe_clean = _letter_continuation_ids(tokenizer, probe_prefix, all_letters)
    if not probe_clean:
        log.warning(f"Letter probe reshuffle — falling back per-sample where needed.")
    letter_token_probe = {L: probe_cont[L] for L in all_letters if probe_cont[L] is not None}

    metadata = []
    n_correct = 0
    n_no_anchor = 0

    for idx, sample in enumerate(samples):
        body_text = _format_body_text(sample, tokenizer, instruct)
        body_ids = tokenizer(body_text, return_tensors="pt")["input_ids"].to(device)
        prompt_len = body_ids.shape[1]

        # Template: body + "####" + sentinel. Use the SAME add_special_tokens
        # behaviour as the model-input encoding so body tokens align 1:1 with
        # the front of full_ids (otherwise BOS / chat-template markers will
        # be present on one side and missing on the other).
        template_text = body_text + ANSWER_PREFIX
        template_prefix_ids = tokenizer(template_text)["input_ids"]
        template_ids = template_prefix_ids + [sentinel_id]

        # Generate greedy with scores retained
        with torch.no_grad():
            gen = model.generate(
                input_ids=body_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_ids = gen.sequences[0].tolist()
        scores = gen.scores  # tuple of (1, V) tensors, one per new token
        del gen

        mapping = _align_template_to_generation(template_ids, full_ids)
        sentinel_pos = len(template_ids) - 1
        gen_answer_pos = mapping.get(sentinel_pos) if mapping is not None else None

        fallback_forward = False
        if gen_answer_pos is None:
            # No "####" in generation — use logits at the end of generation
            # as a fallback. These represent "if you had to commit now…"
            n_no_anchor += 1
            gen_answer_pos = len(full_ids)  # past-last: triggers the forward pass fallback
            fallback_forward = True

        # Logits at gen_answer_pos: predict the token at gen_answer_pos.
        # scores[t] predicts the token at position (prompt_len + t).
        # So for gen_answer_pos we want scores[gen_answer_pos - prompt_len].
        score_idx = gen_answer_pos - prompt_len
        if 0 <= score_idx < len(scores):
            logits = scores[score_idx][0].float().cpu()
        else:
            # Answer position isn't covered by generation scores (e.g. model
            # stopped before emitting "####", or "####" was the last token).
            # Fall back to a teacher-forced forward pass on
            #     body + "####"   (guaranteed to end exactly where we want).
            tf_text = body_text + ANSWER_PREFIX
            tf_ids = tokenizer(tf_text, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                out = model(input_ids=tf_ids, use_cache=False, logits_to_keep=1)
            logits = out.logits[0, -1, :].float().cpu()
            del out
            fallback_forward = True

        last_lp = torch.log_softmax(logits, dim=-1)

        answer_lp = {}
        for letter in sample["choices"]:
            tids = letter_token_probe.get(letter)
            if tids is None or len(tids) == 0:
                answer_lp[letter] = float("-inf")
                continue
            answer_lp[letter] = float(last_lp[tids[0]])

        letters = list(answer_lp.keys())
        lp_vec = np.array([answer_lp[l] for l in letters], dtype=np.float64)
        lp_vec -= lp_vec.max()
        probs = np.exp(lp_vec)
        probs /= probs.sum()
        restricted = {l: float(p) for l, p in zip(letters, probs)}

        pred = max(answer_lp, key=answer_lp.get)
        sorted_lp = sorted(answer_lp.values(), reverse=True)
        margin = float(sorted_lp[0] - sorted_lp[1]) if len(sorted_lp) > 1 else float("inf")
        gold = sample["answer"]
        correct = pred == gold
        if correct:
            n_correct += 1

        n_gen = len(full_ids) - prompt_len
        metadata.append({
            "unique_id": sample["unique_id"],
            "gold_answer": gold,
            "predicted_answer": pred,
            "correct": correct,
            "margin": margin,
            "restricted_prob_gold": restricted.get(gold, 0.0),
            "answer_logprobs": {l: float(answer_lp[l]) for l in sample["choices"]},
            "restricted_softmax": restricted,
            "choices": sample["choices"],
            "n_prompt_tokens": int(prompt_len),
            "n_gen_tokens": int(n_gen),
            "anchor_found_in_gen": not fallback_forward,
            "completion": tokenizer.decode(full_ids[prompt_len:], skip_special_tokens=True),
        })

        acc = n_correct / (idx + 1) * 100
        anchor_flag = "" if not fallback_forward else " [no-anchor]"
        print(
            f"\r  ({label}) {idx + 1}/{len(samples)} acc={acc:.1f}% "
            f"pred={pred} gold={gold} margin={margin:.2f} n_gen={n_gen}{anchor_flag}",
            end="", flush=True,
        )

    print()
    info = {
        "n_samples": len(samples),
        "n_no_anchor_fallback": n_no_anchor,
    }
    log.info(f"CoT diagnostics: {info}")
    return metadata, info


def compute_answer_logprobs(model, tokenizer, samples, instruct=False, label=""):
    """One forward pass per sample. Returns per-sample metadata dicts."""
    device = next(model.parameters()).device

    metadata = []
    n_correct = 0
    n_multitoken = 0
    n_fallback = 0

    for idx, sample in enumerate(samples):
        text = _format_input_text(sample, tokenizer, instruct)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)

        # Probe letter token ids in this sample's context
        cont_ids, clean = _letter_continuation_ids(tokenizer, text, sample["choices"])
        if not clean:
            n_fallback += 1
            if idx < 3:
                log.warning(f"Tokeniser reshuffled '####|letter' boundary for sample {idx}; "
                            f"using per-letter fallback forward pass.")

        # One forward pass on the prefix. Only the last position's logits are read,
        # so compute lm_head for that single position; a full-sequence lm_head over a
        # 256k vocab OOMs large models (e.g. gemma_12b).
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False, logits_to_keep=1)
        last_lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1).cpu()
        del out

        answer_lp = {}
        letter_token_ids = {}
        for letter in sample["choices"]:
            tids = cont_ids[letter]
            if tids is None:
                # Fallback for boundary reshuffle
                full_text = text + " " + letter
                prefix_ids = tokenizer.encode(text, add_special_tokens=False)
                answer_lp[letter] = _teacher_force_letter_logp(
                    model, tokenizer, full_text, prefix_ids, device,
                )
                letter_token_ids[letter] = []
                continue

            if len(tids) == 0:
                log.error(f"Empty continuation for letter '{letter}' (sample {idx})")
                answer_lp[letter] = float("-inf")
                letter_token_ids[letter] = []
                continue

            # First token: direct lookup in last_lp
            lp = float(last_lp[tids[0]])
            # Extra tokens (rare): independent forward pass to avoid DynamicCache mutation
            if len(tids) > 1:
                n_multitoken += 1
                ext_ids = torch.tensor(
                    [input_ids[0].tolist() + list(tids[:-1])],
                    dtype=torch.long, device=device,
                )
                # Only the last len(tids)-1 positions are read below; keep just those.
                keep = len(tids) - 1
                with torch.no_grad():
                    ext_out = model(input_ids=ext_ids, use_cache=False, logits_to_keep=keep)
                ext_lp = torch.log_softmax(ext_out.logits[0].float(), dim=-1).cpu()
                del ext_out
                # Kept rows are the last `keep` positions of ext_ids (absolute positions
                # |input_ids| .. |ext_ids|-1), which predict tids[1] .. tids[-1].
                for i, tok in enumerate(tids[1:], start=1):
                    lp += float(ext_lp[i - 1, tok])
            answer_lp[letter] = lp
            letter_token_ids[letter] = tids

        # Restricted softmax over the answer choices
        letters = list(answer_lp.keys())
        lp_vec = np.array([answer_lp[l] for l in letters], dtype=np.float64)
        lp_vec -= lp_vec.max()
        probs = np.exp(lp_vec)
        probs /= probs.sum()
        restricted = {l: float(p) for l, p in zip(letters, probs)}

        pred = max(answer_lp, key=answer_lp.get)
        sorted_lp = sorted(answer_lp.values(), reverse=True)
        margin = float(sorted_lp[0] - sorted_lp[1]) if len(sorted_lp) > 1 else float("inf")
        gold = sample["answer"]
        correct = pred == gold
        if correct:
            n_correct += 1

        metadata.append({
            "unique_id": sample["unique_id"],
            "gold_answer": gold,
            "predicted_answer": pred,
            "correct": correct,
            "margin": margin,
            "restricted_prob_gold": restricted.get(gold, 0.0),
            "answer_logprobs": {l: float(answer_lp[l]) for l in sample["choices"]},
            "restricted_softmax": restricted,
            "choices": sample["choices"],
            "n_prompt_tokens": int(input_ids.shape[1]),
            "letter_token_ids": {l: list(map(int, ids)) for l, ids in letter_token_ids.items()},
        })

        del last_lp
        torch.cuda.empty_cache()

        acc = n_correct / (idx + 1) * 100
        print(
            f"\r  ({label}) {idx + 1}/{len(samples)} acc={acc:.1f}% "
            f"pred={pred} gold={gold} margin={margin:.2f}",
            end="", flush=True,
        )

    print()
    info = {
        "n_samples": len(samples),
        "n_multitoken_letters": n_multitoken,
        "n_boundary_fallbacks": n_fallback,
    }
    if n_multitoken or n_fallback:
        log.info(f"Diagnostics: {info}")
    return metadata, info


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_logits", config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    if dataset_name != "redial":
        log.info(f"Skipping: generate_logits_mca only runs on ReDial (got '{dataset_name}')")
        return

    task = cfg.task.name
    dialect = cfg.dialect.name
    reasoning = cfg.reasoning.name
    perturbation_fn = cfg.perturbation.fn
    perturbation_name = cfg.perturbation.name

    valid_dialects = list(cfg.dataset.dialects)
    if dialect not in valid_dialects:
        log.info(f"Skipping (dialect '{dialect}' not in {valid_dialects})")
        return
    if "tasks" in cfg.dataset and task not in list(cfg.dataset.tasks):
        log.info(f"Skipping (task '{task}' not in {list(cfg.dataset.tasks)})")
        return

    # Perturbations only apply to SAE text
    if perturbation_fn is not None and dialect != "sae":
        log.info(f"Skipping (perturbation '{perturbation_name}' only applies to dialect 'sae', got '{dialect}')")
        return

    if perturbation_fn is not None:
        # Mirror generate_logits.py: nest perturbed outputs under the SAE arm so the
        # Hydra-derived run dir never overwrites the plain SAE MCA results.
        experiments_dir = _project_config["directories"]["experiments"]
        out_dir = os.path.join(
            experiments_dir, "generate_logits_mca", cfg.model.name, dataset_name,
            task, reasoning, "sae", "perturbed", perturbation_name,
        )
    else:
        out_dir = HydraConfig.get().runtime.output_dir.replace(
            "/generate_logits/", "/generate_logits_mca/", 1,
        )

    expected_files = ["metadata.jsonl"]
    if not cfg.rerun and all(os.path.exists(os.path.join(out_dir, f)) for f in expected_files):
        log.info(f"Skipping (outputs exist, rerun=false): {out_dir}")
        return

    log.info(
        f"\n{'=' * 60}\n"
        f"  Model:     {cfg.model.name} ({cfg.model.model_id})\n"
        f"  Dataset:   {dataset_name}\n"
        f"  Task:      {task}\n"
        f"  Reasoning: {reasoning}\n"
        f"  Dialect:   {dialect}\n"
        f"  Perturb:   {perturbation_name}\n"
        f"  Out dir:   {out_dir}\n"
        f"{'=' * 60}"
    )

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

    # Load and apply perturbation if specified (substitutes the text field in-place)
    if perturbation_fn is not None:
        preprocessed_dir = _project_config["directories"]["preprocessed"]
        perturbed_texts = _load_perturbation_texts(
            preprocessed_dir, perturbation_name, dataset_name, task, dialect,
        )
        if perturbed_texts is None:
            log.error(f"Perturbation file not found for {perturbation_name}/{dataset_name}/{task}/{dialect}")
            return
        row_text = cfg.task.row_text
        for i, text in enumerate(perturbed_texts):
            ds[i][row_text] = text
        log.info(f"Loaded {len(perturbed_texts)} perturbed texts ({perturbation_name})")

    samples = _build_samples(ds, task, dialect, reasoning=reasoning)
    log.info(f"Built {len(samples)} MQA samples")

    try:
        model, tokenizer = _load_model(cfg.model.name, cfg.model.model_id, device=cfg.device)
    except OSError as e:
        log.warning(f"Skipping model '{cfg.model.name}': {e}")
        return

    instruct = cfg.model.name.endswith("_instruct")
    label = f"{dataset_name}/{task}/{reasoning}/{dialect}/mca"
    if reasoning == "cot":
        variant = "instruct" if instruct else "base"
        max_new_tokens = int(cfg.reasoning.max_tokens_new[variant])
        if "max_tokens_new" in cfg.model and reasoning in cfg.model.max_tokens_new:
            max_new_tokens = int(cfg.model.max_tokens_new[reasoning])
        metadata, info = compute_answer_logprobs_cot(
            model, tokenizer, samples, instruct=instruct,
            max_new_tokens=max_new_tokens, label=label,
        )
    else:
        metadata, info = compute_answer_logprobs(
            model, tokenizer, samples, instruct=instruct, label=label,
        )

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metadata.jsonl"), "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    n_correct = sum(1 for m in metadata if m["correct"])
    mean_margin = float(np.mean([
        m["margin"] for m in metadata if np.isfinite(m["margin"])
    ])) if metadata else 0.0
    mean_gold_prob = float(np.mean([m["restricted_prob_gold"] for m in metadata])) \
        if metadata else 0.0
    log.info(f"Saved {len(metadata)} entries to: {out_dir}/metadata.jsonl")
    log.info(f"Restricted-softmax accuracy: {n_correct}/{len(metadata)} "
             f"({n_correct / max(len(metadata), 1) * 100:.1f}%)")
    log.info(f"Mean margin (nats): {mean_margin:.3f}")
    log.info(f"Mean P(gold | restricted): {mean_gold_prob:.3f}")


if __name__ == "__main__":
    main()
