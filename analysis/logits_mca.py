"""
MCA (forced-choice answer-probability) accuracy and answer-entropy analysis on ReDial.

Compares the SAE baseline against AAVE and the perturbed-SAE transformation
conditions under the multiple-choice answer protocol, per (model, task) cell:

1. Per-condition accuracy and restricted-softmax answer-entropy deltas vs SAE,
   with one-sided Wilcoxon signed-rank tests against zero.
2. Paired AAVE-vs-perturbation tests (accuracy loss larger / entropy smaller),
   Bonferroni-corrected over the number of perturbations.

Usage:
    python analysis/logits_mca.py
    python analysis/logits_mca.py --experiments-dir /path/to/experiments --reasoning cot
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

import dialecttax.utils


###########
# LOADING #
###########

def load_mca_cells(experiments_dir, reasoning="naive"):
    """Load per-(model, task, condition) MCA accuracy and answer entropy.

    Conditions are the dialect arms ("sae", "aave") plus each perturbed-SAE
    transformation. Answer entropy is the entropy of the restricted softmax
    over the answer letters, averaged over samples.

    Args:
        experiments_dir: Root experiments directory.
        reasoning: Reasoning arm ("naive" or "cot").

    Returns:
        DataFrame with columns model, task, condition, accuracy, entropy, n.
    """
    root = os.path.join(experiments_dir, "generate_logits_mca")
    files = []
    for path in glob.glob(os.path.join(root, "*", "redial", "*", reasoning, "*", "metadata.jsonl")):
        model, _ds, task, _r, cond, _ = os.path.relpath(path, root).split(os.sep)
        files.append((path, model, task, cond))
    for path in glob.glob(os.path.join(root, "*", "redial", "*", reasoning, "sae", "perturbed", "*", "metadata.jsonl")):
        parts = os.path.relpath(path, root).split(os.sep)
        files.append((path, parts[0], parts[2], parts[6]))

    rows = []
    for path, model, task, cond in files:
        correct, entropy = [], []
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                probs = np.array(list(r["restricted_softmax"].values()), dtype=float)
                probs = probs[probs > 0]
                correct.append(bool(r["correct"]))
                entropy.append(float(-(probs * np.log(probs)).sum()))
        rows.append({
            "model": model, "task": task, "condition": cond,
            "accuracy": np.mean(correct) * 100, "entropy": np.mean(entropy), "n": len(correct),
        })
    return pd.DataFrame(rows)


def _delta_frame(cells, metric):
    """Pivot cells to per-condition deltas vs the SAE baseline.

    Args:
        cells: Output of load_mca_cells.
        metric: Column to compare ("accuracy" or "entropy").

    Returns:
        DataFrame indexed by (model, task) with one delta column per condition.
    """
    wide = cells.pivot_table(index=["model", "task"], columns="condition", values=metric)
    return wide.drop(columns=["sae"]).sub(wide["sae"], axis=0)


#########
# TESTS #
#########

def condition_deltas(cells):
    """Per-condition accuracy and entropy deltas vs SAE with tests against zero.

    Accuracy uses a one-sided Wilcoxon signed-rank test for a loss (delta < 0);
    entropy uses a two-sided test.

    Args:
        cells: Output of load_mca_cells.

    Returns:
        DataFrame with one row per condition.
    """
    d_acc = _delta_frame(cells, "accuracy")
    d_ent = _delta_frame(cells, "entropy")
    rows = []
    for cond in d_acc.columns:
        acc, ent = d_acc[cond].dropna(), d_ent[cond].dropna()
        _, p_acc = stats.wilcoxon(acc, alternative="less")
        _, p_ent = stats.wilcoxon(ent, alternative="two-sided")
        rows.append({
            "condition": cond,
            "n_cells": len(acc),
            "acc_delta_pp": acc.mean(),
            "acc_median_pp": acc.median(),
            "acc_p_loss": p_acc,
            "entropy_delta": ent.mean(),
            "entropy_p": p_ent,
        })
    return pd.DataFrame(rows).sort_values("acc_delta_pp")


def test_aave_vs_perturbations(cells):
    """Paired AAVE-vs-perturbation tests on accuracy and entropy deltas.

    For each perturbation, one-sided Wilcoxon signed-rank tests across the
    shared (model, task) cells ask whether the AAVE accuracy loss is larger
    and whether the AAVE entropy delta is smaller, Bonferroni-corrected over
    the number of perturbations.

    Args:
        cells: Output of load_mca_cells.

    Returns:
        DataFrame with one row per perturbation.
    """
    d_acc = _delta_frame(cells, "accuracy")
    d_ent = _delta_frame(cells, "entropy")
    perts = [c for c in d_acc.columns if c != "aave"]
    rows = []
    for cond in perts:
        pair_acc = d_acc[["aave", cond]].dropna()
        pair_ent = d_ent[["aave", cond]].dropna()
        _, p_acc = stats.wilcoxon(pair_acc["aave"], pair_acc[cond], alternative="less")
        _, p_ent = stats.wilcoxon(pair_ent["aave"], pair_ent[cond], alternative="less")
        rows.append({
            "perturbation": cond,
            "n_cells": len(pair_acc),
            "aave_acc_delta": pair_acc["aave"].mean(),
            "pert_acc_delta": pair_acc[cond].mean(),
            "acc_p_bonferroni": min(p_acc * len(perts), 1.0),
            "aave_ent_delta": pair_ent["aave"].mean(),
            "pert_ent_delta": pair_ent[cond].mean(),
            "ent_p_bonferroni": min(p_ent * len(perts), 1.0),
        })
    return pd.DataFrame(rows)


def print_table(title, df):
    """Print a formatted table with a header.

    Args:
        title: Table title string.
        df: DataFrame to print.
    """
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print(df.to_string(index=False))
    print()


########
# MAIN #
########

def main() -> None:
    parser = argparse.ArgumentParser()
    project_config = dialecttax.utils.load_config()
    default_dir = project_config["directories"]["experiments"]
    parser.add_argument("--experiments-dir", default=default_dir)
    parser.add_argument("--reasoning", default="naive", choices=["naive", "cot"])
    args = parser.parse_args()

    cells = load_mca_cells(args.experiments_dir, reasoning=args.reasoning)
    n_cond = cells["condition"].nunique()
    print(f"Loaded {len(cells)} cells across {n_cond} conditions ({args.reasoning} reasoning)")

    deltas = condition_deltas(cells)
    print_table(f"MCA Accuracy / Answer-Entropy Deltas vs SAE ({args.reasoning})", deltas.round(4))

    vs = test_aave_vs_perturbations(cells)
    print_table(f"AAVE vs Each Perturbation (paired Wilcoxon, Bonferroni; {args.reasoning})", vs.round(4))


if __name__ == "__main__":
    main()
