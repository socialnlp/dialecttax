"""
Patch or continue a benchmark_redial run.

Two actions:
  continue – Resume generation from unfinished sample indices.
  patch    – Re-generate samples that returned errors or empty completions.

Uses the same Hydra configs as benchmark_redial.py.

Usage:
    python scripts/ReDial/patch_benchmark_redial.py +action=continue model=llama_8b task=math dialect=sae
    python scripts/ReDial/patch_benchmark_redial.py +action=patch model=llama_8b task=math dialect=sae
    python scripts/ReDial/patch_benchmark_redial.py +action=patch --multirun model=llama_8b,gemma_12b task=math,logic dialect=sae,aave
"""

import json
import logging
import os
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import dialecttax
from benchmark_redial import (
    _project_config,
    resolve_generation_config,
    load_redial_dataset,
    build_messages,
    grade_task,
    save_results,
)
from dialecttax.endpoints import is_error

log = logging.getLogger(__name__)


###########
# HELPERS #
###########

def _load_indexed_responses(path: str) -> dict[int, dict]:
    """Load JSONL responses keyed by sample index.

    Uses ``_idx`` field if present, falling back to line number.

    Args:
        path: Path to responses JSONL.

    Returns:
        Dict mapping sample index to response dict.
    """
    if not os.path.exists(path):
        return {}
    responses: dict[int, dict] = {}
    with open(path, "r") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            idx = r.get("_idx", line_num)
            responses[idx] = r
    return responses


def _save_all_responses(responses: dict[int, dict], path: str) -> None:
    """Write all responses as JSONL, ordered by sample index with ``_idx``.

    Args:
        responses: Dict mapping sample index to response dict.
        path: Output JSONL path (parent dirs created if needed).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for idx in sorted(responses.keys()):
            r = responses[idx]
            r["_idx"] = idx
            f.write(json.dumps(r) + "\n")


def _recover_temp(resp_path: str, existing: dict[int, dict]) -> None:
    """Recover partial responses from a crashed generate() call.

    Uses the sidecar ``.idx`` file to map temp file lines back to global
    dataset indices, merges into *existing*, and cleans up both files.
    No-op if either ``.tmp`` or ``.idx`` is missing.

    Args:
        resp_path: Path to the main responses.jsonl.
        existing: Dict of idx -> response (mutated in place).
    """
    tmp_path = resp_path + ".tmp"
    idx_path = resp_path + ".idx"
    if not os.path.exists(tmp_path) or not os.path.exists(idx_path):
        return
    with open(idx_path, "r") as f:
        target_indices: list[int] = json.load(f)
    with open(tmp_path, "r") as f:
        temp_responses: list[dict] = [json.loads(line) for line in f if line.strip()]
    # Only map lines that were actually written (crash may truncate)
    for j, idx in enumerate(target_indices[:len(temp_responses)]):
        existing[idx] = temp_responses[j]
    log.info(f"Recovered {len(temp_responses)} responses from previous run")
    os.remove(tmp_path)
    os.remove(idx_path)


def _generate_with_recovery(
    existing: dict[int, dict],
    target_indices: list[int],
    ds: list[dict],
    resp_path: str,
    task: str,
    reasoning: str,
    dialect: str,
    family: str,
    api_key: str,
    gen_kwargs: dict,
) -> None:
    """Generate responses with crash recovery via temp + sidecar files.

    Delegates ``.tmp`` / ``.idx`` management to ``generate()``, which
    writes a sidecar ``.idx`` with *target_indices* and flushes partial
    results to a ``.tmp`` file.  On success, merges new responses into
    *existing* and rewrites ``responses.jsonl`` with correct ``_idx``
    fields.

    If the process crashes mid-generation, a subsequent
    ``_recover_temp()`` call will pick up whatever was flushed.

    Args:
        existing: Dict of idx -> response (mutated in place).
        target_indices: Sorted global dataset indices to generate.
        ds: Full dataset (list of sample dicts).
        resp_path: Path to the main responses.jsonl.
        task: Task name (e.g. ``"math"``).
        reasoning: Reasoning strategy name (e.g. ``"cot"``).
        dialect: Dialect name (e.g. ``"aave"``).
        family: Model family name (e.g. ``"llama"``).
        api_key: OpenRouter API key.
        gen_kwargs: Keyword args forwarded to ``generate()``.
    """
    messages = build_messages(ds, task, reasoning, dialect, family, indices=target_indices)
    new_responses: list[dict] = dialecttax.endpoints.generate(
        messages, api_key,
        path_save=resp_path,
        save_indices=target_indices,
        **gen_kwargs,
    )

    # Map returned responses (ordered by target_indices) back to global indices
    for j, idx in enumerate(target_indices):
        existing[idx] = new_responses[j]

    # Rewrite main file with correct _idx
    _save_all_responses(existing, resp_path)


def _grade_and_save(
    task: str,
    responses_dict: dict[int, dict],
    ds: list[dict],
    out_path: str,
    print_tag: str,
) -> None:
    """Grade all responses and save results.jsonl.

    Args:
        task: Task name (e.g. ``"math"``).
        responses_dict: Dict of idx -> response covering all dataset indices.
        ds: Full dataset (list of sample dicts).
        out_path: Path to write results.jsonl.
        print_tag: Logging prefix (e.g. ``"[llama_8b/math/sae/cot]"``).
    """
    ordered: list[dict] = [responses_dict[i] for i in range(len(ds))]
    completions: list[str | None] = dialecttax.endpoints.get_completions(ordered)
    results: list[dict] = grade_task(task, completions, ds)
    save_results(results, out_path)
    n_correct = sum(1 for r in results if r.get("correct"))
    acc = n_correct / len(results) * 100 if results else 0
    log.info(f"{print_tag} Accuracy: {n_correct}/{len(results)} ({acc:.0f}%)")
    log.info(f"Saved results to: {out_path}")


#########################
# CONTINUE / PATCH IMPL #
#########################

def _continue_generation(
    ds: list[dict],
    resp_path: str,
    out_path: str,
    task: str,
    reasoning: str,
    dialect: str,
    family: str,
    api_key: str,
    gen_kwargs: dict,
    print_tag: str,
) -> None:
    """Resume generation from the next unfinished sample.

    Loads existing responses, recovers any temp files from a previous crash,
    determines which dataset indices are missing, and generates only those.
    Re-grades the full dataset once all samples are present.

    Args:
        ds: Full dataset (list of sample dicts).
        resp_path: Path to responses.jsonl (read + rewritten).
        out_path: Path to write results.jsonl.
        task: Task name.
        reasoning: Reasoning strategy name.
        dialect: Dialect name.
        family: Model family name.
        api_key: OpenRouter API key.
        gen_kwargs: Keyword args forwarded to ``generate()``.
        print_tag: Logging prefix.
    """
    existing = _load_indexed_responses(resp_path)
    _recover_temp(resp_path, existing)
    done = {i for i in existing if i < len(ds)}
    missing = sorted(set(range(len(ds))) - done)

    if not missing:
        log.info(f"{print_tag} All {len(ds)} samples already generated, re-grading")
        _save_all_responses(existing, resp_path)
        _grade_and_save(task, existing, ds, out_path, print_tag)
        return

    log.info(
        f"{print_tag} {len(done)}/{len(ds)} done, "
        f"generating {len(missing)} remaining"
    )

    _generate_with_recovery(
        existing, missing, ds, resp_path,
        task, reasoning, dialect, family, api_key, gen_kwargs,
    )

    n_errors = sum(1 for r in existing.values() if is_error(r))
    log.info(f"TOTAL {len(existing)} RESPONSES [ERRORS: {n_errors}]")

    _grade_and_save(task, existing, ds, out_path, print_tag)


def _patch_samples(
    ds: list[dict],
    resp_path: str,
    out_path: str,
    task: str,
    reasoning: str,
    dialect: str,
    family: str,
    api_key: str,
    gen_kwargs: dict,
    print_tag: str,
) -> None:
    """Re-generate samples with errors or empty completions.

    Loads existing responses, identifies indices with errors or empty content,
    and re-generates only those.  Re-grades the full dataset afterward.

    Args:
        ds: Full dataset (list of sample dicts).
        resp_path: Path to responses.jsonl (read + rewritten).
        out_path: Path to write results.jsonl.
        task: Task name.
        reasoning: Reasoning strategy name.
        dialect: Dialect name.
        family: Model family name.
        api_key: OpenRouter API key.
        gen_kwargs: Keyword args forwarded to ``generate()``.
        print_tag: Logging prefix.
    """
    existing = _load_indexed_responses(resp_path)
    _recover_temp(resp_path, existing)
    if not existing:
        log.error(f"{print_tag} No responses found at {resp_path}")
        return

    error_indices = sorted(i for i, r in existing.items() if is_error(r) and i < len(ds))
    if not error_indices:
        log.info(f"{print_tag} No errors to patch in {len(existing)} responses")
        return

    log.info(f"{print_tag} Patching {len(error_indices)} error(s) in {len(existing)} responses")

    _generate_with_recovery(
        existing, error_indices, ds, resp_path,
        task, reasoning, dialect, family, api_key, gen_kwargs,
    )

    n_remaining = sum(1 for r in existing.values() if is_error(r))
    log.info(f"TOTAL {len(existing)} RESPONSES [REMAINING ERRORS: {n_remaining}]")

    _grade_and_save(task, existing, ds, out_path, print_tag)


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/benchmark_redial", config_name="config")
def main(cfg: DictConfig) -> None:
    action = cfg.get("action")
    if action not in ("patch", "continue"):
        raise ValueError(f"action must be 'patch' or 'continue', got: {action}")

    api_key = dialecttax.utils.get_api_key(_project_config["keys"]["openrouter"])
    dir_preprocessed = _project_config["directories"]["preprocessed"]

    rc = resolve_generation_config(cfg)
    task = rc["task"]
    reasoning = rc["reasoning"]
    dialect = rc["dialect"]
    family = rc["family"]
    print_tag = rc["print_tag"]

    # Capture stderr; only write .err file if non-empty
    out_dir = HydraConfig.get().runtime.output_dir
    hydra_cfg = HydraConfig.get()
    log_path = os.path.join(out_dir, hydra_cfg.job.name + ".log")
    err_path = log_path.replace(".log", ".err")
    os.makedirs(out_dir, exist_ok=True)
    _orig_stderr = sys.stderr
    sys.stderr = open(err_path, "w")

    # Paths
    resp_path = os.path.join(out_dir, "responses.jsonl")
    out_path = os.path.join(out_dir, "results.jsonl")

    # Load data
    ds = load_redial_dataset(dir_preprocessed, task, dialect, print_tag)
    if ds is None:
        return

    gen_kwargs = dict(
        model=rc["model_name"],
        max_tokens_new=rc["max_tokens_new"],
        max_tokens_reasoning=rc["max_tokens_reasoning"],
        reasoning_effort=rc["reasoning_effort"],
        temperature=rc["temperature"],
        max_workers=rc["max_workers"],
    )

    if action == "continue":
        _continue_generation(
            ds, resp_path, out_path, task, reasoning, dialect,
            family, api_key, gen_kwargs, print_tag,
        )
    else:
        _patch_samples(
            ds, resp_path, out_path, task, reasoning, dialect,
            family, api_key, gen_kwargs, print_tag,
        )

    # Remove .err file if empty
    sys.stderr.close()
    sys.stderr = _orig_stderr
    if os.path.getsize(err_path) == 0:
        os.remove(err_path)


if __name__ == "__main__":
    main()
