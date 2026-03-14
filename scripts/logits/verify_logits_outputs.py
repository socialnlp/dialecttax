"""Verify the small-model logits campaign left a complete, uncorrupted output tree.

CPU-only. The experiments directory comes from configs/{config}.yaml; set
DIALECTTAX_DATA_ROOT to override it with <root>/experiments. Two checks:

1. Inventory: every expected config dir for the 18 small models holds a
   complete file set (same completeness rules as the rerun=false gates in
   generate_logits.py / generate_logits_mca.py). Prints any missing dirs.
   NOTE the unperturbed redial and MCA trees nest reasoning only since
   2026-07-11: dataset.output_subdir used to pin "naive", so the sweep's
   reasoning=cot arms resolved to naive's dir and were skipped by the gate.
   The pin is now ${reasoning.name} and the cb1-cb14 lanes backfilled the
   missing cot baseline, so both arms are expected everywhere.
2. Integrity: CRC-test every .npz (zipfile.testzip) and JSON-parse every
   metadata.jsonl in the dirs that had two concurrent writers during the
   split-off overlap windows -- concurrent end-of-config writes could
   interleave and corrupt a file without any job erroring.

Usage:
    python scripts/logits/verify_logits_outputs.py
    python scripts/logits/verify_logits_outputs.py --config server
"""

import argparse
import json
import os
import zipfile

import numpy as np

import dialecttax.utils

STEMS = ["gemma_12b", "llama_3b", "llama_1b", "qwen_8b", "gemma_4b", "gemma_1b", "llama_8b", "qwen_4b", "qwen_1.7b"]
MODELS = [f"{s}_{v}" for s in STEMS for v in ("base", "instruct")]
TASKS = ["math", "algorithm", "logic", "planning"]
REASONINGS = ["naive", "cot"]
PERTURBATIONS = [
    "swap-0.05", "capitalize-random", "capitalize-alternating", "drop-0.15", "drop-0.05", "insert-0.05",
    "translate-french", "translate-chinese", "translate-hindi", "translate-polish",
    "translate-khmer", "translate-yoruba",
]
MULTIVALUE_DIALECTS = ["sae", "aave", "appalachian", "chicano", "indian", "singapore"]
RISK_STEMS = ("llama_3b", "llama_1b", "gemma_1b", "qwen_1.7b")
RISK_MODELS = [m for m in MODELS if any(m.startswith(s) for s in RISK_STEMS)]

FORWARD = ["input_log_probs.npz", "input_entropy.npz", "metadata.jsonl"]
GENERATE = ["input_log_probs.npz", "input_entropy.npz", "gen_log_probs.npz", "gen_entropy.npz", "hidden.npz", "metadata.jsonl"]
GENERATE_AO = ["input_log_probs.npz", "input_entropy.npz", "hidden.npz", "metadata.jsonl"]


def _complete(d):
    return any(all(os.path.exists(os.path.join(d, f)) for f in fs) for fs in (FORWARD, GENERATE, GENERATE_AO))


def _expected(root, model):
    """Yield (dir, kind) for every config dir the campaign should have written."""
    base = os.path.join(root, "generate_logits", model)
    mca = os.path.join(root, "generate_logits_mca", model)
    for task in TASKS:
        for r in REASONINGS:
            for dialect in ("sae", "aave"):
                yield os.path.join(base, "redial", task, r, dialect), "logits"
                yield os.path.join(mca, "redial", task, r, dialect), "mca"
            for p in PERTURBATIONS:
                yield os.path.join(base, "redial", task, r, "sae", "perturbed", p), "logits"
    for dialect in ("sae", "aave"):
        yield os.path.join(base, "parallelaave", dialect), "logits"
    for dialect in MULTIVALUE_DIALECTS:
        yield os.path.join(base, "multivalue", dialect), "logits"
    for ds in ("parallelaave", "multivalue"):
        for p in PERTURBATIONS:
            yield os.path.join(base, ds, "sae", "perturbed", p), "logits"


def _experiments_dir(config: str) -> str:
    """Resolve the experiments directory to verify.

    DIALECTTAX_DATA_ROOT wins when set (it was exported by the in-pod lanes);
    otherwise the path comes from the project config, like every other script.

    Args:
        config: Config file name without .yaml, e.g. "default" or "server".

    Returns:
        Absolute path to the experiments directory.
    """
    root = os.environ.get("DIALECTTAX_DATA_ROOT")
    if root:
        return os.path.join(root, "experiments")
    return dialecttax.utils.load_config(config)["directories"]["experiments"]


def parse_args():
    """Parse command-line arguments.

    Returns:
        Namespace with the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Verify the logits campaign output tree.")
    parser.add_argument("--config", default="default", help="Config file name (without .yaml)")
    return parser.parse_args()


def main():
    args = parse_args()
    root = _experiments_dir(args.config)
    print(f"[root] {root}")
    total_missing = 0
    for model in MODELS:
        expected = list(_expected(root, model))
        missing = [
            d for d, kind in expected
            if not (_complete(d) if kind == "logits" else os.path.exists(os.path.join(d, "metadata.jsonl")))
        ]
        total_missing += len(missing)
        status = "OK" if not missing else f"MISSING {len(missing)}"
        print(f"[inventory] {model:<22} {len(expected) - len(missing)}/{len(expected)} complete  {status}")
        for d in missing:
            print(f"  [missing] {os.path.relpath(d, root)}")

    bad = 0
    checked = 0
    for model in RISK_MODELS:
        for tree in ("generate_logits", "generate_logits_mca"):
            for dirpath, _, files in os.walk(os.path.join(root, tree, model)):
                for f in files:
                    path = os.path.join(dirpath, f)
                    try:
                        if f.endswith(".npz"):
                            checked += 1
                            with zipfile.ZipFile(path) as z:
                                corrupt = z.testzip()
                            if corrupt is not None:
                                raise ValueError(f"CRC failure on member {corrupt}")
                            np.load(path).files
                        elif f == "metadata.jsonl":
                            checked += 1
                            with open(path) as fh:
                                for line in fh:
                                    json.loads(line)
                    except Exception as e:
                        bad += 1
                        print(f"[CORRUPT] {os.path.relpath(path, root)}: {e}")

    print(f"[integrity] {checked} files checked across {len(RISK_MODELS)} overlap-risk models, {bad} corrupt")
    print(f"[verdict] missing={total_missing} corrupt={bad} -> {'ALL DATA PRESENT AND SOUND' if not (total_missing or bad) else 'GAPS FOUND'}")


if __name__ == "__main__":
    main()
