"""
Section 5.3: Inference-Time Hidden States.

Analyzes per-layer cosine similarity between dialect and SAE hidden states
to measure where semantically paired representations remain aligned or
diverge across the network.

Plots:
1. Layer-wise similarity curves per dialect (one plot per model/dataset)
2. Base vs instruct comparison (same model family, same dialect)
3. Summary table of layer-profile statistics

Usage:
    python analysis/layers.py
    python analysis/layers.py --experiments-dir /path/to/experiments
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import dialecttax.utils


plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.constrained_layout.use": True,
})

_SET2 = sns.color_palette("Set2", 6)
DIALECT_COLORS = {
    "aave": _SET2[1],
    "appalachian": _SET2[2],
    "chicano": _SET2[3],
    "indian": _SET2[4],
    "singapore": _SET2[5],
}
DIALECT_LABELS = {
    "aave": "AAVE",
    "appalachian": "Appalachian",
    "chicano": "Chicano",
    "indian": "Indian",
    "singapore": "Singapore",
}
DIALECT_ORDER = ["aave", "appalachian", "chicano", "indian", "singapore"]

MODEL_LABELS = {
    "llama_1b_base": "Llama 3.2 1B",
    "llama_1b_instruct": "Llama 3.2 1B Instruct",
    "llama_3b_base": "Llama 3.2 3B",
    "llama_3b_instruct": "Llama 3.2 3B Instruct",
    "llama_8b_base": "Llama 3.1 8B",
    "llama_8b_instruct": "Llama 3.1 8B Instruct",
    "gemma_1b_base": "Gemma 3 1B",
    "gemma_1b_instruct": "Gemma 3 1B Instruct",
    "gemma_4b_base": "Gemma 3 4B",
    "gemma_4b_instruct": "Gemma 3 4B Instruct",
    "gemma_12b_base": "Gemma 3 12B",
    "gemma_12b_instruct": "Gemma 3 12B Instruct",
    "qwen_1.7b_base": "Qwen 3 1.7B",
    "qwen_1.7b_instruct": "Qwen 3 1.7B Instruct",
    "qwen_4b_base": "Qwen 3 4B",
    "qwen_4b_instruct": "Qwen 3 4B Instruct",
    "qwen_8b_base": "Qwen 3 8B",
    "qwen_8b_instruct": "Qwen 3 8B Instruct",
}

DATASETS = ["multivalue", "parallelaave"]
DATASET_LABELS = {"multivalue": "MultiVALUE", "parallelaave": "ParallelAAVE"}


###########
# LOADING #
###########

def load_layer_similarity(experiments_dir, model, dataset, dialect):
    """Load layer similarity array for a model/dataset/dialect combo.

    Args:
        experiments_dir: Root experiments directory.
        model: Model name (e.g. "llama_8b_base").
        dataset: Dataset name (e.g. "multivalue").
        dialect: Dialect name (e.g. "aave").

    Returns:
        ndarray of shape (n_samples, n_layers) or None if not found.
    """
    path = os.path.join(experiments_dir, "generate_layers", model, dataset, dialect, "layer_similarity.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def discover_models(experiments_dir):
    """Find all models with layer similarity results.

    Args:
        experiments_dir: Root experiments directory.

    Returns:
        List of model name strings.
    """
    layers_dir = os.path.join(experiments_dir, "generate_layers")
    if not os.path.isdir(layers_dir):
        return []
    return sorted([
        d for d in os.listdir(layers_dir)
        if os.path.isdir(os.path.join(layers_dir, d)) and d != "multirun.yaml"
    ])


def discover_dialects(experiments_dir, model, dataset):
    """Find all dialects with results for a model/dataset.

    Args:
        experiments_dir: Root experiments directory.
        model: Model name.
        dataset: Dataset name.

    Returns:
        List of dialect name strings.
    """
    dialect_dir = os.path.join(experiments_dir, "generate_layers", model, dataset)
    if not os.path.isdir(dialect_dir):
        return []
    return sorted([
        d for d in os.listdir(dialect_dir)
        if os.path.exists(os.path.join(dialect_dir, d, "layer_similarity.npy"))
    ])


############
# PLOTTING #
############

# Base hues per family from Set2; size variants darken/lighten
_SET2_FAMILIES = sns.color_palette("Set2", 8)
_LLAMA_SHADES = sns.light_palette(_SET2_FAMILIES[0], n_colors=5, reverse=True)
_GEMMA_SHADES = sns.light_palette(_SET2_FAMILIES[1], n_colors=5, reverse=True)
_QWEN_SHADES = sns.light_palette(_SET2_FAMILIES[2], n_colors=5, reverse=True)

# shade index: 0 = darkest (largest), 2 = medium, 4 = lightest (smallest)
MODEL_COLORS = {
    "llama_8b": _LLAMA_SHADES[0], "llama_3b": _LLAMA_SHADES[2], "llama_1b": _LLAMA_SHADES[3],
    "gemma_12b": _GEMMA_SHADES[0], "gemma_4b": _GEMMA_SHADES[2], "gemma_1b": _GEMMA_SHADES[3],
    "qwen_8b": _QWEN_SHADES[0], "qwen_4b": _QWEN_SHADES[2], "qwen_1.7b": _QWEN_SHADES[3],
}


def _model_style(model_name):
    """Return (color, linewidth, linestyle) for a model.

    Color encodes family + size (darker = larger). Linestyle encodes base/instruct.
    """
    is_instruct = model_name.endswith("_instruct")
    stem = model_name.replace("_base", "").replace("_instruct", "")
    color = MODEL_COLORS.get(stem, "gray")
    ls = "--" if is_instruct else "-"
    return color, 1.5, ls


def plot_multivalue_aggregate(experiments_dir, models, output_dir):
    """Plot per-model aggregated MultiVALUE layer similarity across all dialects.

    For each model: 5 thin dashed grey curves (one per MultiVALUE dialect) and
    one colored curve = mean across those 5 dialects. Color/linestyle
    convention follows :func:`_model_style` so base vs instruct stays
    distinguishable.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.
        output_dir: Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = False

    for model in models:
        per_dialect_means = []
        n_layers_ref = None
        for dialect in DIALECT_ORDER:
            sim = load_layer_similarity(experiments_dir, model, "multivalue", dialect)
            if sim is None:
                continue
            mean_sim = sim.mean(axis=0)
            if n_layers_ref is None:
                n_layers_ref = len(mean_sim)
            per_dialect_means.append(mean_sim)
            layers_norm = np.linspace(0, 1, len(mean_sim))
            ax.plot(layers_norm, mean_sim, color="#A8A8A8", linewidth=0.7,
                    linestyle="--", alpha=0.45, zorder=1)
        if not per_dialect_means:
            continue
        agg = np.mean(np.stack(per_dialect_means, axis=0), axis=0)
        layers_norm = np.linspace(0, 1, len(agg))
        color, lw, ls = _model_style(model)
        label = MODEL_LABELS.get(model, model)
        ax.plot(layers_norm, agg, color=color, linewidth=lw + 0.4, linestyle=ls,
                label=label, zorder=3)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    FONT = 20
    AXIS_FONT = 20
    ax.set_xlabel("Layer depth (normalized)", fontsize=AXIS_FONT)
    ax.set_ylabel("Cosine similarity", fontsize=AXIS_FONT)
    ax.tick_params(axis="both", labelsize=AXIS_FONT)

    family_handles = [
        plt.Line2D([0], [0], color=_LLAMA_SHADES[0], linewidth=2, label="Llama"),
        plt.Line2D([0], [0], color=_GEMMA_SHADES[0], linewidth=2, label="Gemma"),
        plt.Line2D([0], [0], color=_QWEN_SHADES[0], linewidth=2, label="Qwen"),
    ]
    type_handles = [
        plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-", label="Base (mean)"),
        plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Instruct (mean)"),
        plt.Line2D([0], [0], color="#A8A8A8", linewidth=0.7, linestyle="--",
                   alpha=0.6, label="Per-dialect"),
    ]
    leg1 = ax.legend(handles=family_handles, title="Family", fontsize=FONT,
                     title_fontsize=FONT, loc="lower left", frameon=True, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=type_handles, title="Line", fontsize=FONT, title_fontsize=FONT,
              loc="lower center", ncols=1, frameon=True, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "multivalue_aggregate_all_models.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "multivalue_aggregate_all_models.png"), bbox_inches="tight")
    plt.close(fig)


def plot_all_models(experiments_dir, dataset, dialect, models, output_dir):
    """Plot layer-wise similarity curves for all models on one graph.

    Color = model family, linewidth = model size, linestyle = base (solid) / instruct (dashed).

    Args:
        experiments_dir: Root experiments directory.
        dataset: Dataset name.
        dialect: Dialect name.
        models: List of model name strings.
        output_dir: Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(7, 5.5))
    plotted = False

    for model in models:
        sim = load_layer_similarity(experiments_dir, model, dataset, dialect)
        if sim is None:
            continue
        mean_sim = sim.mean(axis=0)
        # Normalize layers to [0, 1] so models with different depths are comparable
        layers_norm = np.linspace(0, 1, len(mean_sim))
        color, lw, ls = _model_style(model)
        label = MODEL_LABELS.get(model, model)
        ax.plot(layers_norm, mean_sim, color=color, linewidth=lw, linestyle=ls, label=label)
        plotted = True

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel("Layer depth (normalized)")
    ax.set_ylabel("Cosine similarity")

    # Legend 1: family (darkest shade per family)
    family_handles = [
        plt.Line2D([0], [0], color=_LLAMA_SHADES[0], linewidth=2, label="Llama"),
        plt.Line2D([0], [0], color=_GEMMA_SHADES[0], linewidth=2, label="Gemma"),
        plt.Line2D([0], [0], color=_QWEN_SHADES[0], linewidth=2, label="Qwen"),
    ]
    leg1 = ax.legend(handles=family_handles, title="Family", fontsize=11, title_fontsize=12,
                     loc="lower left", frameon=True, framealpha=0.9)
    ax.add_artist(leg1)

    # Legend 2: base vs instruct
    type_handles = [
        plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-", label="Base"),
        plt.Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--", label="Instruct"),
    ]
    ax.legend(handles=type_handles, title="Type", fontsize=11, title_fontsize=12,
              loc="lower center", ncols=2, frameon=True, framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{dataset}_{dialect}_all_models.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{dataset}_{dialect}_all_models.png"), bbox_inches="tight")
    plt.close(fig)


############
# SUMMARY  #
############

def compute_summary_table(experiments_dir):
    """Compute layer-profile summary statistics for all model/dataset/dialect combos.

    For each combo, reports:
    - Layer 0 similarity (embedding layer)
    - Minimum similarity and its layer
    - Final layer similarity
    - Mean similarity across all layers

    Args:
        experiments_dir: Root experiments directory.

    Returns:
        DataFrame with summary statistics.
    """
    rows = []
    models = discover_models(experiments_dir)

    for model in models:
        for dataset in DATASETS:
            dialects = discover_dialects(experiments_dir, model, dataset)
            for dialect in dialects:
                sim = load_layer_similarity(experiments_dir, model, dataset, dialect)
                if sim is None:
                    continue
                mean_sim = sim.mean(axis=0)
                rows.append({
                    "model": MODEL_LABELS.get(model, model),
                    "dataset": DATASET_LABELS.get(dataset, dataset),
                    "dialect": DIALECT_LABELS.get(dialect, dialect),
                    "sim_layer0": mean_sim[0],
                    "sim_min": mean_sim.min(),
                    "min_layer": int(mean_sim.argmin()),
                    "sim_final": mean_sim[-1],
                    "sim_mean": mean_sim.mean(),
                    "n_layers": len(mean_sim),
                    "n_samples": sim.shape[0],
                })

    return pd.DataFrame(rows)


#####################
# STATISTICAL TESTS #
#####################

def test_layer_differences(experiments_dir):
    """Test whether cosine similarity varies significantly across layers.

    For each model/dataset/dialect, runs a Friedman test across all layers
    (repeated measures over samples). A significant result means the similarity
    profile is not flat — layers differ.

    Args:
        experiments_dir: Root experiments directory.

    Returns:
        DataFrame with test results.
    """
    rows = []
    models = discover_models(experiments_dir)

    for model in models:
        for dataset in DATASETS:
            dialects = discover_dialects(experiments_dir, model, dataset)
            for dialect in dialects:
                sim = load_layer_similarity(experiments_dir, model, dataset, dialect)
                if sim is None or sim.shape[1] < 3:
                    continue
                # Friedman test: are layer distributions different?
                # sim shape: (n_samples, n_layers) — each row is one sample across layers
                stat, p = stats.friedmanchisquare(*[sim[:, l] for l in range(sim.shape[1])])
                rows.append({
                    "model": MODEL_LABELS.get(model, model),
                    "dataset": DATASET_LABELS.get(dataset, dataset),
                    "dialect": DIALECT_LABELS.get(dialect, dialect),
                    "friedman_stat": stat,
                    "p_value": p,
                    "n_layers": sim.shape[1],
                    "n_samples": sim.shape[0],
                })

    return pd.DataFrame(rows)


def test_final_layer_drop(experiments_dir):
    """Test whether the final layer similarity is lower than the mean of other layers.

    For each model/dataset/dialect, runs a paired Wilcoxon signed-rank test
    comparing each sample's final-layer similarity to its mean similarity across
    all non-final layers. A significant result means the final layer has lower
    similarity than the rest of the network.

    Args:
        experiments_dir: Root experiments directory.

    Returns:
        DataFrame with test results.
    """
    rows = []
    models = discover_models(experiments_dir)

    for model in models:
        for dataset in DATASETS:
            dialects = discover_dialects(experiments_dir, model, dataset)
            for dialect in dialects:
                sim = load_layer_similarity(experiments_dir, model, dataset, dialect)
                if sim is None or sim.shape[1] < 2:
                    continue
                # Per-sample: final layer vs mean of all other layers
                final_sim = sim[:, -1]                       # (n_samples,)
                other_mean = sim[:, :-1].mean(axis=1)        # (n_samples,)
                diff = other_mean - final_sim                # positive = final is lower

                stat, p = stats.wilcoxon(final_sim, other_mean, alternative="less")
                rows.append({
                    "model": MODEL_LABELS.get(model, model),
                    "dataset": DATASET_LABELS.get(dataset, dataset),
                    "dialect": DIALECT_LABELS.get(dialect, dialect),
                    "mean_final": final_sim.mean(),
                    "mean_other": other_mean.mean(),
                    "mean_drop": diff.mean(),
                    "wilcoxon_stat": stat,
                    "p_value": p,
                    "n_samples": sim.shape[0],
                })

    return pd.DataFrame(rows)


def test_each_layer_vs_final(experiments_dir):
    """Test whether each layer's similarity is significantly higher than the final layer.

    For each model/dataset/dialect, runs a paired Wilcoxon signed-rank test
    comparing each layer to the final layer across samples (with Bonferroni
    correction). A significant result means that layer is more similar than
    the final layer.

    Args:
        experiments_dir: Root experiments directory.

    Returns:
        DataFrame with per-layer test results.
    """
    rows = []
    models = discover_models(experiments_dir)

    for model in models:
        for dataset in DATASETS:
            dialects = discover_dialects(experiments_dir, model, dataset)
            for dialect in dialects:
                sim = load_layer_similarity(experiments_dir, model, dataset, dialect)
                if sim is None or sim.shape[1] < 3:
                    continue
                n_layers = sim.shape[1]
                final_sim = sim[:, -1]
                n_tests = n_layers - 1  # excluding final layer itself

                for layer in range(n_layers - 1):
                    layer_sim = sim[:, layer]
                    stat, p_raw = stats.wilcoxon(layer_sim, final_sim, alternative="greater")
                    p_corrected = min(p_raw * n_tests, 1.0)  # Bonferroni
                    rows.append({
                        "model": MODEL_LABELS.get(model, model),
                        "dataset": DATASET_LABELS.get(dataset, dataset),
                        "dialect": DIALECT_LABELS.get(dialect, dialect),
                        "layer": layer,
                        "n_layers": n_layers,
                        "mean_layer": layer_sim.mean(),
                        "mean_final": final_sim.mean(),
                        "mean_diff": (layer_sim - final_sim).mean(),
                        "wilcoxon_stat": stat,
                        "p_raw": p_raw,
                        "p_corrected": p_corrected,
                        "significant": p_corrected < 0.05,
                    })

    return pd.DataFrame(rows)


def summarize_layer_vs_final(df):
    """Summarize per-layer tests for each model/dataset/dialect.

    The main statistic is the share of preceding layers that are significantly
    more similar than the final layer. ``first_non_sig_layer`` is diagnostic
    only: early embedding layers can also be less similar than the final layer,
    so it should not be interpreted as the onset of a late-layer drop.

    Args:
        df: Output of test_each_layer_vs_final.

    Returns:
        DataFrame with one row per model/dataset/dialect.
    """
    rows = []
    for (model, dataset, dialect), group in df.groupby(["model", "dataset", "dialect"]):
        n_sig = group["significant"].sum()
        n_layers = group["n_layers"].iloc[0]
        # Diagnostic only: non-significance can occur in early layers when the
        # embedding layer is less similar than the final layer.
        non_sig = group[~group["significant"]]
        first_non_sig = int(non_sig["layer"].iloc[0]) if len(non_sig) > 0 else n_layers - 1
        # Number of layers significantly higher than final.
        rows.append({
            "model": model,
            "dataset": dataset,
            "dialect": dialect,
            "n_layers": n_layers,
            "n_sig_layers": int(n_sig),
            "first_non_sig_layer": first_non_sig,
            "pct_sig": n_sig / (n_layers - 1) * 100,
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
    parser.add_argument("--output-dir", default="analysis/plots/layers")
    args = parser.parse_args()

    models = discover_models(args.experiments_dir)
    print(f"Found models: {models}")

    ########################################
    # All models on one graph per dialect
    ########################################

    for dataset in DATASETS:
        # Collect all dialects across all models
        all_dialects = set()
        for model in models:
            all_dialects.update(discover_dialects(args.experiments_dir, model, dataset))
        for dialect in sorted(all_dialects):
            print(f"\nPlotting all models: {dataset} / {dialect}")
            plot_all_models(args.experiments_dir, dataset, dialect, models, args.output_dir)

    ########################################
    # MultiVALUE: aggregated across dialects, one curve per model
    ########################################

    print("\nPlotting MultiVALUE aggregate across dialects (all models)")
    plot_multivalue_aggregate(args.experiments_dir, models, args.output_dir)

    ########################################
    # Summary table
    ########################################

    print("\nComputing summary table...")
    summary = compute_summary_table(args.experiments_dir)
    if len(summary) > 0:
        print_table("Layer Similarity Summary", summary.round(4))

    ########################################
    # Statistical tests
    ########################################

    print("\nTest 1: Do layers differ? (Friedman test)")
    friedman_df = test_layer_differences(args.experiments_dir)
    if len(friedman_df) > 0:
        print_table("Friedman Test: Similarity Varies Across Layers", friedman_df.round(4))

    print("\nTest 2: Is final layer similarity lower? (Wilcoxon signed-rank)")
    drop_df = test_final_layer_drop(args.experiments_dir)
    if len(drop_df) > 0:
        print_table("Final Layer Drop: Final vs Mean(Other Layers)", drop_df.round(4))

    print("\nTest 3: Each layer vs final layer (Wilcoxon, Bonferroni-corrected)")
    layer_vs_final_df = test_each_layer_vs_final(args.experiments_dir)
    if len(layer_vs_final_df) > 0:
        summary_lf = summarize_layer_vs_final(layer_vs_final_df)
        print_table("Layer vs Final: Share of Layers Above Final", summary_lf.round(2))

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
