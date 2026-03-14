"""
Convert preprocessed ReDial data into multiple-choice QA format.

For each task/dialect:
- Logic: already has choices and letter answer — passes through.
- Math: generates 3 numeric distractors via OpenRouter.
- Planning: generates 3 time-value distractors via OpenRouter.
- Algorithm: generates 3 wrong code distractors via OpenRouter.

Output: {preprocessed_dir}/ReDial/{task}_{dialect}_qa.jsonl

Usage:
    python scripts/ReDial/preprocess_redial_qa.py
    python scripts/ReDial/preprocess_redial_qa.py --task math --rewrite
    python scripts/ReDial/preprocess_redial_qa.py --config tucana
"""

import argparse
import json
import logging
import os
import random

import dialecttax

log = logging.getLogger(__name__)


################
# DISTRACTORS #
################

DISTRACTOR_SYSTEM = "Please generate plausible but incorrect answer choices for multiple-choice questions. Return ONLY valid JSON."

DISTRACTOR_PROMPT_MATH = (
    "Given this math problem and the correct numerical answer, generate exactly 3 plausible "
    "but INCORRECT numerical answers. They should reflect common mistakes (e.g. off-by-one, "
    "wrong operation, partial computation). All 3 values must be different from each other "
    "and different from the correct answer. All values must be integers.\n\n"
    "Problem: {problem}\n"
    "Correct answer: {answer}\n\n"
    "Return ONLY a JSON array of 3 different integers, e.g.: [150, 200, 50]"
)

DISTRACTOR_PROMPT_PLANNING = (
    "Given this planning/scheduling problem and the correct answer in seconds, generate exactly 3 "
    "plausible but INCORRECT time values in seconds. They should reflect common scheduling "
    "mistakes (e.g. forgetting parallelism, double-counting, off-by-one-step). "
    "All 3 values must be different from each other and different from the correct answer.\n\n"
    "Problem: {problem}\n"
    "Correct answer: {answer} seconds\n\n"
    "Return ONLY a JSON array of 3 different numbers (seconds), e.g.: [3600, 5400, 7200]"
)

DISTRACTOR_PROMPT_ALGORITHM = (
    "Given this programming problem, context, and correct solution, generate exactly 3 plausible "
    "but INCORRECT Python function bodies. Each should contain a subtle bug (e.g. off-by-one, "
    "wrong comparison, missing edge case, wrong return value). All 3 must be different from "
    "each other and different from the correct solution. Do NOT include the function "
    "signature — only the body (matching the indentation of the correct solution).\n\n"
    "Problem: {problem}\n"
    "Context:\n```\n{context}\n```\n"
    "Correct solution:\n```\n{answer}\n```\n\n"
    'Return ONLY a JSON array of 3 strings, each being a different wrong function body, e.g.:\n'
    '["return x + 1", "return x - 1", "return x"]'
)


def _build_distractor_messages(samples, task):
    """Build OpenRouter messages for distractor generation.

    Args:
        samples: List of preprocessed sample dicts.
        task: Task name.

    Returns:
        List of message lists for dialecttax.endpoints.generate().
    """
    messages = []
    for sample in samples:
        if task == "math":
            prompt = DISTRACTOR_PROMPT_MATH.format(problem=sample["problem"], answer=sample["answer"])
        elif task == "planning":
            answer_val = sample["answer"][0] if isinstance(sample["answer"], list) else sample["answer"]
            prompt = DISTRACTOR_PROMPT_PLANNING.format(problem=sample["problem"], answer=int(answer_val))
        elif task == "algorithm":
            prompt = DISTRACTOR_PROMPT_ALGORITHM.format(
                problem=sample["problem"], context=sample.get("context", ""), answer=sample["answer"],
            )
        else:
            raise ValueError(f"No distractor generation for task: {task}")
        messages.append([
            {"role": "system", "content": DISTRACTOR_SYSTEM},
            {"role": "user", "content": prompt},
        ])
    return messages


def _parse_distractors(completion, task):
    """Parse distractor values from LLM completion.

    Args:
        completion: Raw completion string from OpenRouter.
        task: Task name.

    Returns:
        List of 3 distractor values, or None on parse failure.
    """
    if completion is None:
        return None
    # Strip markdown fences if present
    text = completion.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == 3:
            return parsed
    except json.JSONDecodeError:
        pass
    return None


###########
# CHOICES #
###########

def _normalize_indent(code):
    """Normalize code indentation so first line is at column 0.

    Subtracts the minimum indent of non-first lines from all non-first
    lines, so relative indentation is preserved starting from 0.

    Args:
        code: Code string with newlines.

    Returns:
        Re-indented code string.
    """
    lines = code.split("\n")
    non_first_indents = [len(l) - len(l.lstrip()) for l in lines[1:] if l.strip()]
    if not non_first_indents:
        return code
    min_indent = min(non_first_indents)
    if min_indent == 0:
        return code
    result = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            result.append("")
        else:
            indent = len(line) - len(line.lstrip())
            new_indent = max(0, indent - min_indent)
            result.append(" " * new_indent + line.lstrip())
    return "\n".join(result)


def _make_choices(correct_answer, distractors, task, rng):
    """Assemble and shuffle choices from correct answer + distractors.

    Args:
        correct_answer: The correct answer value.
        distractors: List of 3 distractor values.
        task: Task name.
        rng: random.Random instance for shuffling.

    Returns:
        Tuple of (choices_dict, answer_letter), or (None, None) if choices
        are not all unique.
    """
    if task == "math":
        options = [str(int(correct_answer))] + [str(int(d)) for d in distractors]
    elif task == "planning":
        # LLM generates single values; wrap into [val, val] pairs to match answer format
        correct_pair = correct_answer if isinstance(correct_answer, list) else [correct_answer, correct_answer]
        options = [correct_pair] + [[int(d), int(d)] for d in distractors]
    elif task == "algorithm":
        options = [_normalize_indent(s) for s in [correct_answer] + list(distractors)]
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Check all choices are unique (use tuples for lists)
    hashable = [tuple(o) if isinstance(o, list) else o for o in options]
    if len(set(hashable)) != len(options):
        return None, None

    # Shuffle via index permutation to handle non-hashable types (lists)
    letters = ["A", "B", "C", "D"]
    indices = list(range(len(options)))
    rng.shuffle(indices)
    correct_letter = letters[indices.index(0)]
    choices = {letters[i]: options[indices[i]] for i in range(4)}
    return choices, correct_letter


##########
# RESUME #
##########

def _load_existing_responses(path_save, n_total):
    """Load responses from a previous complete or crashed run.

    Checks the final responses file first, then any .tmp crash recovery
    file for additional responses not yet in the final file.

    Args:
        path_save: Path to the responses JSONL file.
        n_total: Total number of expected responses.

    Returns:
        Tuple of (responses list of length n_total, set of done indices).
    """
    responses = [None] * n_total
    done = set()

    if not path_save:
        return responses, done

    # Load from complete responses file
    if os.path.exists(path_save):
        with open(path_save) as f:
            for line in f:
                r = json.loads(line)
                idx = r.get("_idx")
                if idx is not None and 0 <= idx < n_total:
                    responses[idx] = r
                    done.add(idx)

    # Load from .tmp crash recovery (may have responses not yet in final file)
    tmp_path = path_save + ".tmp"
    idx_path = path_save + ".idx"
    if os.path.exists(tmp_path) and os.path.exists(idx_path):
        with open(idx_path) as f:
            indices = json.load(f)
        with open(tmp_path) as f:
            for line_i, line in enumerate(f):
                if line_i < len(indices):
                    orig_idx = indices[line_i]
                    if 0 <= orig_idx < n_total and orig_idx not in done:
                        responses[orig_idx] = json.loads(line)
                        done.add(orig_idx)

    return responses, done


def _save_responses(path_save, responses):
    """Write complete merged responses to the final file.

    Args:
        path_save: Path to write the responses JSONL file.
        responses: List of response dicts (may contain None for missing).
    """
    if not path_save:
        return
    with open(path_save, "w") as f:
        for i, r in enumerate(responses):
            if r is not None:
                r["_idx"] = i
                f.write(json.dumps(r) + "\n")
    # Clean up any leftover crash recovery files
    for suffix in (".tmp", ".idx"):
        p = path_save + suffix
        if os.path.exists(p):
            os.remove(p)


################
# CONVERT TASK #
################

def convert_logic(samples):
    """Logic samples already have choices and letter answers — pass through.

    Args:
        samples: List of preprocessed logic sample dicts.

    Returns:
        List of QA-format sample dicts.
    """
    qa_samples = []
    for sample in samples:
        qa_samples.append({
            **sample,
            "choices": dict(sample["choices"]),
            "answer": sample["answer"],
        })
    return qa_samples


def convert_with_distractors(samples, task, api_key, rng, path_save=None, **kwargs):
    """Convert samples by generating distractors via OpenRouter.

    Resumes from existing responses if path_save or its .tmp crash recovery
    file exist, only generating for missing indices.

    Args:
        samples: List of preprocessed sample dicts.
        task: Task name (math, planning, or algorithm).
        api_key: OpenRouter API key.
        rng: random.Random instance for shuffling.
        path_save: Path to save raw OpenRouter responses as JSONL.
        **kwargs: Extra kwargs passed to dialecttax.endpoints.generate().

    Returns:
        List of QA-format sample dicts.
    """
    messages = _build_distractor_messages(samples, task)
    n_total = len(messages)

    # Resume from existing responses (complete file or .tmp crash recovery)
    responses, done_indices = _load_existing_responses(path_save, n_total)
    missing_indices = sorted(set(range(n_total)) - done_indices)

    if not missing_indices:
        log.info(f"All {n_total} distractor responses already exist, skipping generation")
    else:
        if done_indices:
            log.info(f"Resuming: {len(done_indices)}/{n_total} done, generating {len(missing_indices)} remaining")
        else:
            log.info(f"Generating distractors for {n_total} {task} samples...")

        missing_messages = [messages[i] for i in missing_indices]
        new_responses = dialecttax.endpoints.generate(
            missing_messages, api_key=api_key,
            path_save=path_save, save_indices=missing_indices,
            **kwargs,
        )
        for idx, resp in zip(missing_indices, new_responses):
            responses[idx] = resp

        # Save complete merged responses (overwrites generate()'s partial file)
        _save_responses(path_save, responses)

    # Build QA samples, retrying failed/duplicate distractors
    max_retries = 3
    qa_results = [None] * n_total  # indexed by sample position

    for attempt in range(max_retries + 1):
        retry_indices = []
        completions = dialecttax.endpoints.get_completions(responses)

        for i, (sample, completion) in enumerate(zip(samples, completions)):
            if qa_results[i] is not None:
                continue
            distractors = _parse_distractors(completion, task)
            if distractors is None:
                retry_indices.append(i)
                continue
            choices, answer_letter = _make_choices(sample["answer"], distractors, task, rng)
            if choices is None:
                retry_indices.append(i)
                continue
            qa_sample = {k: v for k, v in sample.items() if k != "answer"}
            qa_sample["choices"] = choices
            qa_sample["answer"] = answer_letter
            qa_results[i] = qa_sample

        if not retry_indices or attempt == max_retries:
            break

        log.info(f"Retrying {len(retry_indices)} samples with bad distractors (attempt {attempt + 1}/{max_retries})")
        retry_messages = [messages[i] for i in retry_indices]
        retry_responses = dialecttax.endpoints.generate(retry_messages, api_key=api_key, **kwargs)
        for idx, resp in zip(retry_indices, retry_responses):
            responses[idx] = resp
        _save_responses(path_save, responses)

    n_failed = sum(1 for r in qa_results if r is None)
    if n_failed > 0:
        log.warning(f"Failed to generate distractors for {n_failed}/{len(samples)} samples after {max_retries} retries")
    return qa_results


#########
# CHECK #
#########

def _normalize_answer(answer, task):
    """Normalize an original answer for comparison with a QA choice value.

    Args:
        answer: Original answer value from the preprocessed sample.
        task: Task name.

    Returns:
        Normalized value comparable to a QA choice value.
    """
    if task == "math":
        return str(int(answer))
    if task == "planning":
        return [float(v) for v in answer] if isinstance(answer, list) else [float(answer), float(answer)]
    if task == "algorithm":
        # Choices are stored with normalized indentation (see _make_choices), so
        # dedent the original the same way before comparing.
        return _normalize_indent(answer)
    return answer


def check_qa_alignment(path_original, path_qa, task):
    """Verify that QA answers match original answers.

    Loads both files, matches by unique_id, and checks that the selected
    choice in the QA file corresponds to the original answer.

    Args:
        path_original: Path to the preprocessed JSONL file.
        path_qa: Path to the QA JSONL file.
        task: Task name.

    Returns:
        Number of mismatched samples (0 means all correct).
    """
    with open(path_original) as f:
        originals = {s["unique_id"]: s for s in (json.loads(line) for line in f)}
    with open(path_qa) as f:
        qa_samples = [json.loads(line) for line in f]

    n_mismatch = 0
    for qa in qa_samples:
        uid = qa["unique_id"]
        orig = originals.get(uid)
        if orig is None:
            log.warning(f"  {uid}: FAIL - not found in original file")
            n_mismatch += 1
            continue

        if task == "logic":
            correct = qa["answer"] == orig["answer"]
            if not correct:
                n_mismatch += 1
            status = "OK" if correct else "FAIL"
            log.info(f"  {uid}: {status} - answer={qa['answer']}, original={orig['answer']}")
        else:
            selected = qa["choices"][qa["answer"]]
            expected = _normalize_answer(orig["answer"], task)
            correct = selected == expected
            if not correct:
                n_mismatch += 1
            status = "OK" if correct else "FAIL"
            log.info(f"  {uid}: {status} - answer={qa['answer']}, selected={selected!r}, original={expected!r}")

    log.info(f"  Checked {len(qa_samples)} samples: {len(qa_samples) - n_mismatch}/{len(qa_samples)} correct")
    return n_mismatch


########
# MAIN #
########

def parse_args():
    parser = argparse.ArgumentParser(description="Convert preprocessed ReDial to multiple-choice QA")
    parser.add_argument("--config", default="default", help="Config file name")
    parser.add_argument("--rewrite", action=argparse.BooleanOptionalAction, default=False, help="Overwrite existing files")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for choice shuffling")
    parser.add_argument("--model", default="openai/gpt-5-mini", help="OpenRouter model for distractor generation")
    parser.add_argument("--task", nargs="+", default=["algorithm", "logic", "math", "planning"], help="Tasks to process")
    parser.add_argument("--max-workers", type=int, default=16, help="Max concurrent API requests")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = dialecttax.utils.load_config(args.config)

    dir_redial = os.path.join(config["directories"]["preprocessed"], "ReDial")
    api_key = dialecttax.utils.get_api_key(config["keys"]["openrouter"])
    rng = random.Random(args.seed)

    kwargs_generate = {
        "model": args.model,
        "temperature": 0.0,
        "max_tokens_new": 4096,
        "max_workers": args.max_workers,
        "reasoning_effort": "medium",
    }

    dialects = ["sae", "aave"]

    for task in args.task:
        # Generate distractors once per task (from SAE), reuse for all dialects
        choices_map = None
        if task != "logic":
            path_sae = os.path.join(dir_redial, f"{task}_sae.jsonl")
            if not os.path.exists(path_sae):
                log.warning(f"SAE input not found, skipping task: {path_sae}")
                continue
            with open(path_sae) as f:
                sae_samples = [json.loads(line) for line in f]

            path_responses = os.path.join(dir_redial, f"{task}_qa_responses.jsonl")
            qa_results = convert_with_distractors(
                sae_samples, task, api_key, rng, path_save=path_responses, **kwargs_generate,
            )
            # Map by original index: same ordering across dialects
            choices_map = {i: (s["choices"], s["answer"]) for i, s in enumerate(qa_results) if s is not None}

        for dialect in dialects:
            path_in = os.path.join(dir_redial, f"{task}_{dialect}.jsonl")
            path_out = os.path.join(dir_redial, f"{task}_{dialect}_qa.jsonl")

            if not os.path.exists(path_in):
                log.warning(f"Input not found, skipping: {path_in}")
                continue
            if not args.rewrite and os.path.exists(path_out):
                log.info(f"Skipping {task}_{dialect}_qa (already exists): {path_out}")
                check_qa_alignment(path_in, path_out, task)
                continue

            with open(path_in) as f:
                samples = [json.loads(line) for line in f]
            log.info(f"Loaded {len(samples)} samples from {path_in}")

            if task == "logic":
                qa_samples = convert_logic(samples)
            else:
                qa_samples = []
                for i, sample in enumerate(samples):
                    if i not in choices_map:
                        continue
                    choices, answer_letter = choices_map[i]
                    qa_sample = {k: v for k, v in sample.items() if k != "answer"}
                    qa_sample["choices"] = choices
                    qa_sample["answer"] = answer_letter
                    qa_samples.append(qa_sample)

            with open(path_out, "w") as f:
                for item in qa_samples:
                    f.write(json.dumps(item) + "\n")
            log.info(f"Saved {len(qa_samples)} QA samples to: {path_out}")

            log.info(f"Checking {task}_{dialect}_qa alignment...")
            check_qa_alignment(path_in, path_out, task)

    log.info("Done!")


if __name__ == "__main__":
    main()
