"""
W2 experiment: Correlate hidden-state dialect separability with accuracy gap.

For each model, computes:
  1. Mean hidden-state separability (5-fold CV logistic regression accuracy)
  2. Mean SAE-AAVE accuracy gap (canonical tokenization)

Then reports Pearson and Spearman correlations across models.

Usage:
    python scripts/scratch/w2_separability_vs_accuracy_gap.py
    python scripts/scratch/w2_separability_vs_accuracy_gap.py --experiments-dir /data/gemini/ellang/dialecttax/experiments
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import dialecttax.utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "analysis"))
from characters import (
    _load_metadata,
    _drop_broken_combos,
    hidden_dialect_separability,
    load_merged,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default")
    parser.add_argument("--experiments-dir", default=None)
    parser.add_argument("--out-dir", default="analysis/plots/characters")
    args = parser.parse_args()

    if args.experiments_dir is None:
        cfg = dialecttax.utils.load_config(args.config)
        args.experiments_dir = cfg["directories"]["experiments"]

    ###############
    # SEPARABILITY
    ###############

    print("Computing hidden-state dialect separability (canonical tokenization)...")
    sep_df = hidden_dialect_separability(args.experiments_dir)
    sep_canonical = sep_df[sep_df["tok"] == "canonical"]
    sep_by_model = sep_canonical.groupby("model")["acc"].mean().reset_index()
    sep_by_model.columns = ["model", "separability"]
    print(f"  Models with separability data: {len(sep_by_model)}")
    print(sep_by_model.to_string(index=False))

    ################
    # ACCURACY GAP
    ################

    print("\nComputing SAE-AAVE accuracy gap (canonical tokenization)...")
    merged = load_merged(args.experiments_dir)
    clean, dropped = _drop_broken_combos(merged, min_acc=1.0)

    pivot = (
        clean.groupby(["model", "task", "reasoning", "dialect"])["correct_can"]
        .mean().mul(100).unstack("dialect")
    )
    if "sae" not in pivot.columns or "aave" not in pivot.columns:
        print("ERROR: Missing SAE or AAVE in data.")
        return
    gap = (pivot["sae"] - pivot["aave"]).reset_index()
    gap.columns = ["model", "task", "reasoning", "accuracy_gap"]
    gap_by_model = gap.groupby("model")["accuracy_gap"].mean().reset_index()
    gap_by_model.columns = ["model", "accuracy_gap"]
    print(f"  Models with accuracy data: {len(gap_by_model)}")
    print(gap_by_model.to_string(index=False))

    ###############
    # CORRELATION
    ###############

    df = sep_by_model.merge(gap_by_model, on="model")
    print(f"\n{'=' * 60}")
    print(f" SEPARABILITY vs ACCURACY GAP ({len(df)} models)")
    print(f"{'=' * 60}")
    print(df.to_string(index=False))

    if len(df) < 3:
        print("\nToo few models for correlation. Need at least 3.")
        return

    x = df["separability"].values
    y = df["accuracy_gap"].values

    r_pearson, p_pearson = stats.pearsonr(x, y)
    r_spearman, p_spearman = stats.spearmanr(x, y)

    print(f"\nPearson:  r = {r_pearson:.3f}, p = {p_pearson:.4f}")
    print(f"Spearman: ρ = {r_spearman:.3f}, p = {p_spearman:.4f}")

    ##########
    # PLOT
    ##########

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(x, y, s=60, edgecolor="black", linewidth=0.7, zorder=3)

    for i, row in df.iterrows():
        label = row["model"].replace("_instruct", "").replace("_", " ")
        ax.annotate(label, (row["separability"], row["accuracy_gap"]),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points")

    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color="gray", alpha=0.7)

    ax.set_xlabel("Hidden-state dialect separability (% accuracy)")
    ax.set_ylabel("SAE − AAVE accuracy gap (pp)")
    ax.set_title(
        f"Separability vs. accuracy gap\n"
        f"Pearson r={r_pearson:.2f} (p={p_pearson:.3f}), "
        f"Spearman ρ={r_spearman:.2f} (p={p_spearman:.3f})",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "separability_vs_accuracy_gap")
    fig.savefig(f"{out_path}.png", dpi=150)
    fig.savefig(f"{out_path}.pdf")
    print(f"\nSaved plot to {out_path}.png/.pdf")
    plt.close(fig)

    #######################
    # ALSO BY TASK/REASON
    #######################

    print(f"\n{'=' * 60}")
    print(" SEPARABILITY vs ACCURACY GAP (per model × task × reasoning)")
    print(f"{'=' * 60}")
    sep_fine = sep_canonical.rename(columns={"acc": "separability"})
    gap_fine = gap.rename(columns={"accuracy_gap": "accuracy_gap"})
    fine = sep_fine.merge(gap_fine, on=["model", "task", "reasoning"])
    if len(fine) > 3:
        r2, p2 = stats.pearsonr(fine["separability"], fine["accuracy_gap"])
        rho2, prho2 = stats.spearmanr(fine["separability"], fine["accuracy_gap"])
        print(f"  N = {len(fine)} (model × task × reasoning) combos")
        print(f"  Pearson:  r = {r2:.3f}, p = {p2:.4e}")
        print(f"  Spearman: ρ = {rho2:.3f}, p = {prho2:.4e}")


if __name__ == "__main__":
    main()
