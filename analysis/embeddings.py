"""
Section 3: Same Meaning, Higher Taxes.

Analyzes whether models treat dialect variation as surface-level or semantic,
and whether vocabulary coverage explains the tokenization tax.

Plots:
1. multivalue_transformation_similarity — cross-dialect similarity vs
   perturbation/translation baselines across embedding dimensions
2. vocab_coverage_heatmap — vocabulary coverage by dialect and tokenizer
3. multivalue_meaning_vs_tax — per-sample scatter of semantic similarity
   vs token length ratio (the "same meaning, higher taxes" plot)

Usage:
    python analysis/embeddings.py
    python analysis/embeddings.py --experiments-dir /path/to/experiments
"""

import argparse
import json
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
    "sae": _SET2[0],
    "aave": _SET2[1],
    "appalachian": _SET2[2],
    "chicano": _SET2[3],
    "indian": _SET2[4],
    "singapore": _SET2[5],
}
DIALECT_LABELS = {
    "sae": "SAE",
    "aave": "AAVE",
    "appalachian": "Appalachian",
    "chicano": "Chicano",
    "indian": "Indian",
    "singapore": "Singapore",
}
DIALECT_ORDER = ["sae", "aave", "appalachian", "chicano", "indian", "singapore"]

TOKENIZER_LABELS = {
    "bpe": "BPE (GPT-5)",
    "gpt2": "BPE (GPT-2)",
    "gemma": "BPE (Gemma)",
    "llama": "BPE (Llama)",
    "qwen": "BPE (Qwen)",
    "unigram": "Unigram (T5)",
    "wordpiece": "WordPiece (BERT)",
}
TOKENIZER_ORDER = ["bpe", "gpt2", "gemma", "llama", "qwen", "unigram", "wordpiece"]
TOKENIZER_FAMILY = {
    "bpe": "BPE", "gpt2": "BPE", "gemma": "BPE", "llama": "BPE", "qwen": "BPE",
    "unigram": "Unigram", "wordpiece": "WordPiece",
}

PERTURBATION_LABELS = {
    "swap-0.05": "Swap",
    "drop-0.15": "Drop",
    "insert-0.05": "Insert",
    "capitalize-random": "Capitalize (random)",
    "capitalize-alternating": "Capitalize (alternating)",
}
PERTURBATION_ORDER = ["swap-0.05", "drop-0.15", "insert-0.05", "capitalize-random", "capitalize-alternating"]

TRANSLATION_LABELS = {
    "translate-french": "Translate (French)",
    "translate-chinese": "Translate (Chinese)",
    "translate-hindi": "Translate (Hindi)",
    "translate-polish": "Translate (Polish)",
    "translate-khmer": "Translate (Khmer)",
    "translate-yoruba": "Translate (Yoruba)",
}
TRANSLATION_ORDER = [
    "translate-french", "translate-chinese",
    "translate-hindi", "translate-polish",
    "translate-khmer", "translate-yoruba",
]

EMB_DIM = 768
EMB_DIMS = [128, 256, 512, 768]


###########
# LOADING #
###########

def load_token_results(experiments_dir: str, dataset: str) -> pd.DataFrame:
    """Load tokenization results from generate_tokens outputs.

    Args:
        experiments_dir: Root experiments directory.
        dataset: One of "parallelaave", "multivalue".

    Returns:
        DataFrame with tokenization metrics and metadata columns.
    """
    rows = []
    token_dir = os.path.join(experiments_dir, "generate_tokens", dataset)
    if not os.path.isdir(token_dir):
        return pd.DataFrame()

    for dialect in sorted(os.listdir(token_dir)):
        dialect_dir = os.path.join(token_dir, dialect)
        if not os.path.isdir(dialect_dir):
            continue
        for tokenizer in sorted(os.listdir(dialect_dir)):
            path = os.path.join(dialect_dir, tokenizer, "tokens.jsonl")
            if not os.path.isfile(path):
                continue
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    sample["dataset"] = dataset
                    sample["dialect"] = dialect
                    sample["tokenizer"] = tokenizer
                    rows.append(sample)

    df = pd.DataFrame(rows)
    df.drop(columns=["tokens", "encoded"], errors="ignore", inplace=True)
    return df


def _load_embeddings_dim(emb_dir: str, dataset: str, dialect: str, dim: int, perturbation: str | None = None) -> np.ndarray | None:
    """Load embeddings at a specific dimension."""
    parts = [emb_dir, dataset, dialect]
    if perturbation:
        parts.append(perturbation)
    path = os.path.join(*parts, f"embeddings-{dim}.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path)


##########################
# 3.1 VOCAB COMPOSITION #
##########################

def table_vocab_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Vocabulary coverage (p_in_vocab) per dialect and tokenizer.

    Args:
        df: DataFrame from load_token_results.

    Returns:
        Pivot table: dialect rows, tokenizer columns, values = mean p_in_vocab.
    """
    dialects = [d for d in DIALECT_ORDER if d in df["dialect"].unique()]
    tokenizers = [t for t in TOKENIZER_ORDER if t in df["tokenizer"].unique()]

    pivot = df.groupby(["dialect", "tokenizer"])["p_in_vocab"].mean().unstack("tokenizer")
    pivot = pivot.reindex(index=dialects, columns=tokenizers)
    pivot.index = pivot.index.map(DIALECT_LABELS)
    pivot.columns = pivot.columns.map(TOKENIZER_LABELS)
    return pivot.round(3)


def plot_vocab_coverage(df: pd.DataFrame, output_dir: str) -> None:
    """Heatmap of vocabulary coverage by dialect and tokenizer.

    Args:
        df: DataFrame from load_token_results.
        output_dir: Directory to save the plot.
    """
    dialects = [d for d in DIALECT_ORDER if d in df["dialect"].unique()]
    tokenizers = [t for t in TOKENIZER_ORDER if t in df["tokenizer"].unique()]

    pivot = df.groupby(["dialect", "tokenizer"])["p_in_vocab"].mean().unstack("tokenizer")
    pivot = pivot.reindex(index=dialects, columns=tokenizers)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0.6, vmax=1.0)

    ax.set_xticks(range(len(tokenizers)))
    ax.set_xticklabels([TOKENIZER_LABELS[t] for t in tokenizers], fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(dialects)))
    ax.set_yticklabels([DIALECT_LABELS[d] for d in dialects])

    for i in range(len(dialects)):
        for j in range(len(tokenizers)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if val < 0.75 else "black")

    fig.colorbar(im, ax=ax, label="Vocab Coverage (p_in_vocab)", shrink=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "vocab_coverage_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "vocab_coverage_heatmap.png"), bbox_inches="tight")
    plt.close(fig)


def plot_token_overlap(df: pd.DataFrame, experiments_dir: str, output_dir: str) -> None:
    """Heatmap: fraction of subword tokens shared between each dialect and SAE.

    For each (dialect, tokenizer), computes the Jaccard index of the token
    sets used to encode dialect text vs SAE text.

    Args:
        df: DataFrame from load_token_results (needs 'tokens' column — reload with tokens).
        experiments_dir: Root experiments directory (to reload with tokens).
        output_dir: Directory to save the plot.
    """
    import json as _json

    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df["dialect"].unique()]
    tokenizers = [t for t in TOKENIZER_ORDER if t in df["tokenizer"].unique()]
    dataset = df["dataset"].iloc[0]
    token_dir = os.path.join(experiments_dir, "generate_tokens", dataset)

    # Collect token sets per (dialect, tokenizer)
    def _get_token_set(dialect, tokenizer):
        path = os.path.join(token_dir, dialect, tokenizer, "tokens.jsonl")
        if not os.path.isfile(path):
            return set()
        tokens = set()
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = _json.loads(line)
                if sample.get("tokens"):
                    tokens.update(sample["tokens"])
        return tokens

    # Compute Jaccard overlap
    records = []
    for tokenizer in tokenizers:
        sae_tokens = _get_token_set("sae", tokenizer)
        if not sae_tokens:
            continue
        for dialect in dialects:
            dial_tokens = _get_token_set(dialect, tokenizer)
            if not dial_tokens:
                continue
            intersection = sae_tokens & dial_tokens
            union = sae_tokens | dial_tokens
            jaccard = len(intersection) / len(union) if union else 0
            # Also compute: what fraction of dialect tokens are in SAE's set
            coverage = len(intersection) / len(dial_tokens) if dial_tokens else 0
            records.append({
                "dialect": dialect,
                "tokenizer": tokenizer,
                "jaccard": jaccard,
                "dialect_in_sae": coverage,
                "n_shared": len(intersection),
                "n_dialect_only": len(dial_tokens - sae_tokens),
                "n_sae_only": len(sae_tokens - dial_tokens),
            })

    overlap_df = pd.DataFrame(records)

    # Heatmap of Jaccard overlap
    pivot = overlap_df.pivot_table(index="dialect", columns="tokenizer", values="jaccard")
    pivot = pivot.reindex(index=dialects, columns=tokenizers)

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(tokenizers)))
    ax.set_xticklabels([TOKENIZER_LABELS[t] for t in tokenizers], fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(len(dialects)))
    ax.set_yticklabels([DIALECT_LABELS[d] for d in dialects])

    for i in range(len(dialects)):
        for j in range(len(tokenizers)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if val < 0.65 else "black")

    fig.colorbar(im, ax=ax, label="Token Jaccard Overlap with SAE", shrink=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "token_overlap_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "token_overlap_heatmap.png"), bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print("\n  Token overlap summary (dialect-only tokens):")
    for _, row in overlap_df.iterrows():
        print(f"    {DIALECT_LABELS[row['dialect']]:>12s} x {TOKENIZER_LABELS[row['tokenizer']]:>18s}: "
              f"Jaccard={row['jaccard']:.3f}  dialect-only={row['n_dialect_only']}  SAE-only={row['n_sae_only']}")


##########################################
# 3.2 TRANSFORMATION SIMILARITY         #
##########################################

def compute_null_baseline(emb_dir: str, dataset: str) -> pd.DataFrame:
    """Compute null baseline similarity from unrelated pairs.

    Args:
        emb_dir: Root generate_embeddings directory.
        dataset: Dataset name.

    Returns:
        DataFrame with dialect, mu_null, sigma_null.
    """
    dialects = [d for d in DIALECT_ORDER if os.path.isdir(os.path.join(emb_dir, dataset, d))]
    records = []
    for dialect in dialects:
        emb = _load_embeddings_dim(emb_dir, dataset, dialect, EMB_DIM)
        if emb is None:
            continue
        sim = emb @ emb.T
        n = sim.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = sim[mask]
        records.append({
            "dialect": dialect,
            "mu_null": off_diag.mean(),
            "sigma_null": off_diag.std(),
        })
    return pd.DataFrame(records)


def compute_cross_dialect_similarity(emb_dir: str, dataset: str) -> pd.DataFrame:
    """Compute sim(SAE, dialect) across all embedding dimensions.

    Args:
        emb_dir: Root generate_embeddings directory.
        dataset: Dataset name.

    Returns:
        DataFrame with label, type, dim, mean_sim, std_sim, n.
    """
    dialects = [d for d in DIALECT_ORDER if d != "sae" and os.path.isdir(os.path.join(emb_dir, dataset, d))]
    records = []
    for dim in EMB_DIMS:
        emb_sae = _load_embeddings_dim(emb_dir, dataset, "sae", dim)
        if emb_sae is None:
            continue
        for dialect in dialects:
            emb_d = _load_embeddings_dim(emb_dir, dataset, dialect, dim)
            if emb_d is None:
                continue
            sims = np.sum(emb_sae * emb_d, axis=1)
            records.append({
                "label": f"SAE \u2192 {DIALECT_LABELS[dialect]}",
                "type": "Dialect",
                "dim": dim,
                "mean_sim": sims.mean(),
                "std_sim": sims.std(),
                "n": len(sims),
            })
    return pd.DataFrame(records)


def compute_perturbation_similarity_sae(emb_dir: str, dataset: str) -> pd.DataFrame:
    """Compute sim(SAE_original, SAE_perturbed) across all embedding dimensions.

    Args:
        emb_dir: Root generate_embeddings directory.
        dataset: Dataset name.

    Returns:
        DataFrame with label, type, dim, mean_sim, std_sim, n.
    """
    all_perturbations = PERTURBATION_ORDER + TRANSLATION_ORDER
    records = []
    for dim in EMB_DIMS:
        emb_sae = _load_embeddings_dim(emb_dir, dataset, "sae", dim)
        if emb_sae is None:
            continue
        for pert in all_perturbations:
            emb_pert = _load_embeddings_dim(emb_dir, dataset, "sae", dim, pert)
            if emb_pert is None:
                # Warn rather than skip silently: a partially failed embedding
                # sweep would otherwise drop rows from the plot unnoticed.
                print(f"  WARNING: missing embeddings for {dataset}/sae/{pert} at dim={dim}")
                continue
            sims = np.sum(emb_sae * emb_pert, axis=1)
            label = PERTURBATION_LABELS.get(pert, TRANSLATION_LABELS.get(pert, pert))
            pert_type = "Translation" if pert.startswith("translate") else "Perturbation"
            records.append({
                "label": label,
                "type": pert_type,
                "dim": dim,
                "mean_sim": sims.mean(),
                "std_sim": sims.std(),
                "n": len(sims),
            })
    return pd.DataFrame(records)


def plot_transformation_similarity(cross_df: pd.DataFrame, pert_df: pd.DataFrame,
                                    null_df: pd.DataFrame, dataset: str, output_dir: str) -> None:
    """Dot plot comparing dialect, perturbation, and translation similarities.

    Colors encode embedding dimension (128, 256, 512, 768).
    Markers: dialect = circle, perturbation = square, translation = diamond.
    Includes a zoomed inset on the top entries.

    Args:
        cross_df: Output of compute_cross_dialect_similarity.
        pert_df: Output of compute_perturbation_similarity_sae.
        null_df: Output of compute_null_baseline.
        dataset: Dataset name (for output filename).
        output_dir: Directory to save the plot.
    """
    combined = pd.concat([pert_df, cross_df], ignore_index=True)

    # Sort by mean_sim at dim=768 for consistent y ordering
    order_df = combined[combined["dim"] == 768].sort_values("mean_sim", ascending=True)
    label_order = order_df["label"].tolist()
    for label in combined["label"].unique():
        if label not in label_order:
            label_order.insert(0, label)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    dim_colors = {
        128: sns.color_palette("Set2")[0],
        256: sns.color_palette("Set2")[1],
        512: sns.color_palette("Set2")[2],
        768: sns.color_palette("Set2")[3],
    }
    type_markers = {
        "Dialect": "o",
        "Perturbation": "s",
        "Translation": "D",
    }

    for _, row in combined.iterrows():
        y_pos = label_order.index(row["label"])
        ax.scatter(row["mean_sim"], y_pos, color=dim_colors[row["dim"]],
                   marker=type_markers[row["type"]], s=60, edgecolors="black",
                   linewidths=0.4, zorder=3, alpha=0.85)

    # Null baseline
    mu_null = null_df["mu_null"].mean()
    ax.axvline(mu_null, color="gray", ls="--", lw=0.8)

    ax.set_yticks(range(len(label_order)))
    ax.set_yticklabels(label_order, fontsize=12)
    ax.set_xlabel("Semantic equivalence")
    # Lower bound follows mu_null so the baseline can never fall outside the axes
    # (multivalue sits at 0.376, just left of the old hard-coded 0.4).
    ax.set_xlim(min(0.4, mu_null - 0.03), 1.02)

    # Inset: zoom into top rows (only for multivalue where dialects cluster tightly)
    if dataset == "multivalue":
        top_n = 7
        top_labels = label_order[-top_n:]
        top_data = combined[combined["label"].isin(top_labels)]
        x_lo = 0.97
        x_hi = 1.0

        axins = ax.inset_axes([0.33, 0.60, 0.55, 0.38])
        for _, row in top_data.iterrows():
            y_pos = top_labels.index(row["label"])
            axins.scatter(row["mean_sim"], y_pos, color=dim_colors[row["dim"]],
                          marker=type_markers[row["type"]], s=50, edgecolors="black",
                          linewidths=0.4, zorder=3, alpha=0.85)

        axins.set_yticks(range(len(top_labels)))
        axins.set_yticklabels(top_labels, fontsize=10)
        axins.set_xlim(x_lo, x_hi)
        axins.set_ylim(-0.5, top_n - 0.5)
        axins.tick_params(axis="x", labelsize=6)
        axins.set_facecolor("white")
        axins.patch.set_alpha(0.9)

        # Dashed rectangle on main plot + connector lines to inset
        from matplotlib.patches import ConnectionPatch, Rectangle

        main_y_lo = len(label_order) - top_n - 0.5
        main_y_hi = len(label_order) - 0.5
        rect = Rectangle((x_lo, main_y_lo), x_hi - x_lo, main_y_hi - main_y_lo,
                          fill=False, edgecolor="gray", linewidth=0.8, alpha=0.5, ls="--")
        ax.add_patch(rect)

        for (xy_main, xy_inset) in [
            ((x_lo, main_y_hi), (x_hi, top_n - 0.5)),
            ((x_lo, main_y_lo), (x_hi, -0.5)),
        ]:
            con = ConnectionPatch(xyA=xy_inset, xyB=xy_main, coordsA="data", coordsB="data",
                                  axesA=axins, axesB=ax, color="gray", linewidth=0.6, alpha=0.4)
            fig.add_artist(con)

    # Legend — dimensions (colors)
    dim_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=dim_colors[d],
                   markeredgecolor="black", markersize=7, label=str(d))
                   for d in EMB_DIMS]
    leg1 = ax.legend(handles=dim_handles, title="Dim", fontsize=11, title_fontsize=12,
                     loc="lower right", frameon=True, framealpha=0.9)
    ax.add_artist(leg1)

    # Legend — types (markers), with the null-baseline line as the last entry
    type_handles = [plt.Line2D([0], [0], marker=type_markers[t], color="w", markerfacecolor="gray",
                    markeredgecolor="black", markersize=10, label=t)
                    for t in ["Dialect", "Perturbation", "Translation"]]
    type_handles.append(plt.Line2D([0], [0], color="gray", ls="--", lw=0.8, label="Unrelated pairs"))
    if dataset == "multivalue":
        ax.legend(handles=type_handles, title="Transformation", fontsize=11, title_fontsize=12,
                  loc="lower left", ncols=1, frameon=True, framealpha=0.9, markerscale=0.85,
                  bbox_to_anchor=(0.08, 0.0))
    else:
        ax.legend(handles=type_handles, title="Transformation", fontsize=11, title_fontsize=12,
                  loc="upper left", ncols=1, frameon=True, framealpha=0.9, markerscale=0.85,
                  bbox_to_anchor=(0.08, 1.0))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{dataset}_transformation_similarity.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{dataset}_transformation_similarity.png"), bbox_inches="tight")
    plt.close(fig)


#################################
# 3.3 MEANING vs TAX SCATTER   #
#################################

def plot_meaning_vs_tax(emb_dir: str, df: pd.DataFrame, dataset: str, output_dir: str) -> None:
    """Scatter plot: per-sample semantic similarity vs token length ratio.

    X-axis: cosine similarity between SAE and dialect embeddings (meaning preserved).
    Y-axis: token length ratio dialect/SAE (tax applied).
    Each dot is one sample, colored by dialect.

    Args:
        emb_dir: Root generate_embeddings directory.
        df: Tokenization DataFrame from load_token_results.
        dataset: Dataset name (for embedding paths and output filename).
        output_dir: Directory to save the plot.
    """
    emb_sae = _load_embeddings_dim(emb_dir, dataset, "sae", EMB_DIM)
    if emb_sae is None:
        print(f"  SAE embeddings not found for {dataset}, skipping meaning vs tax plot")
        return

    sae_tokens = df[df["dialect"] == "sae"]
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df["dialect"].unique()]

    # Average fertility across all tokenizers per sample
    sae_fert = sae_tokens.groupby("RID")["fertility"].mean()

    fig, ax = plt.subplots(figsize=(6, 4))

    for dialect in dialects:
        emb_d = _load_embeddings_dim(emb_dir, dataset, dialect, EMB_DIM)
        if emb_d is None:
            continue

        # Per-sample cosine similarity
        sims = np.sum(emb_sae * emb_d, axis=1)

        # Per-sample fertility (averaged across tokenizers)
        dial_tokens = df[df["dialect"] == dialect]
        dial_fert = dial_tokens.groupby("RID")["fertility"].mean()

        # Align by RID (both are 0-indexed, same length)
        common_rids = sae_fert.index.intersection(dial_fert.index)
        fert_delta = dial_fert.loc[common_rids].values - sae_fert.loc[common_rids].values

        # Embeddings are indexed 0..N-1, matching RID order
        n = min(len(sims), len(fert_delta))
        # For parallelaave, clip outliers
        if dataset == "parallelaave":
            mask = (fert_delta[:n] > -1.5) & (fert_delta[:n] < 1.5)
            ax.scatter(sims[:n][mask], fert_delta[:n][mask], color=DIALECT_COLORS[dialect],
                       s=15, alpha=0.7, edgecolors="none", label=DIALECT_LABELS[dialect])
        else:
            ax.scatter(sims[:n], fert_delta[:n], color=DIALECT_COLORS[dialect],
                       s=15, alpha=0.7, edgecolors="none", label=DIALECT_LABELS[dialect])

    ax.axhline(0.0, color="gray", ls="--", lw=0.8, zorder=0)

    ax.set_xlabel("Semantic equivalence")
    ax.set_ylabel("\u0394fertility")
    if dataset == "parallelaave":
        ax.legend(title="Dialect", fontsize=11, title_fontsize=12, markerscale=2, loc="lower left")
    else:
        ax.legend(title="Dialect", fontsize=11, title_fontsize=12, markerscale=2, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{dataset}_meaning_vs_tax.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{dataset}_meaning_vs_tax.png"), bbox_inches="tight")
    plt.close(fig)


######################
# STATISTICAL TESTS  #
######################

def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values while preserving the input order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    if np.any(~np.isfinite(p_values)) or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be finite and lie in [0, 1]")

    n_tests = len(p_values)
    if n_tests == 0:
        return p_values.copy()

    order = np.argsort(p_values)
    adjusted_sorted = np.maximum.accumulate(
        (n_tests - np.arange(n_tests)) * p_values[order]
    )
    adjusted = np.empty(n_tests, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def test_dialect_vs_perturbation(emb_dir: str, dataset: str) -> pd.DataFrame:
    """Test whether dialect pairs are more similar than transformation controls.

    Each dialect and control similarity shares the same SAE source sentence, so
    comparisons use a one-sided paired Wilcoxon signed-rank test rather than an
    independent-sample test. Holm adjustment controls the family-wise error
    rate across all dialect-control comparisons within the dataset.

    Args:
        emb_dir: Root generate_embeddings directory.
        dataset: Dataset name.

    Returns:
        DataFrame with condition means, paired differences, Wilcoxon statistic,
        raw and Holm-adjusted p-values, and the Holm-corrected rejection decision.
    """
    emb_sae = _load_embeddings_dim(emb_dir, dataset, "sae", EMB_DIM)
    if emb_sae is None:
        return pd.DataFrame()

    dialects = [d for d in DIALECT_ORDER if d != "sae" and os.path.isdir(os.path.join(emb_dir, dataset, d))]
    all_perturbations = PERTURBATION_ORDER + TRANSLATION_ORDER

    records = []
    for dialect in dialects:
        emb_d = _load_embeddings_dim(emb_dir, dataset, dialect, EMB_DIM)
        if emb_d is None:
            continue
        dialect_sims = np.asarray(np.sum(emb_sae * emb_d, axis=1), dtype=np.float64)

        for pert in all_perturbations:
            emb_pert = _load_embeddings_dim(emb_dir, dataset, "sae", EMB_DIM, pert)
            if emb_pert is None:
                continue
            pert_sims = np.asarray(np.sum(emb_sae * emb_pert, axis=1), dtype=np.float64)
            if len(dialect_sims) != len(pert_sims):
                raise ValueError(
                    f"Paired similarity arrays differ in length for {dataset}/{dialect}/{pert}: "
                    f"{len(dialect_sims)} != {len(pert_sims)}"
                )

            valid = np.isfinite(dialect_sims) & np.isfinite(pert_sims)
            dialect_valid = dialect_sims[valid]
            pert_valid = pert_sims[valid]
            difference = dialect_valid - pert_valid
            if len(difference) == 0:
                continue
            if np.all(difference == 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = stats.wilcoxon(
                    dialect_valid,
                    pert_valid,
                    alternative="greater",
                )
            label = PERTURBATION_LABELS.get(pert, TRANSLATION_LABELS.get(pert, pert))
            records.append({
                "dialect": DIALECT_LABELS[dialect],
                "perturbation": label,
                "n_pairs": len(difference),
                "dialect_mean": dialect_valid.mean(),
                "pert_mean": pert_valid.mean(),
                "mean_difference": difference.mean(),
                "median_difference": np.median(difference),
                "wilcoxon_stat": float(stat),
                "p_raw": float(p),
            })

    result = pd.DataFrame(records)
    if len(result) == 0:
        return result
    result["p_holm"] = _holm_adjust(result["p_raw"].to_numpy())
    result["reject_holm_0.05"] = result["p_holm"] < 0.05
    return result


def test_fertility_delta(emb_dir: str, df: pd.DataFrame, dataset: str, sim_threshold: float = 0.9) -> pd.DataFrame:
    """Wilcoxon signed-rank: is Δfertility significantly > 0 for high-similarity pairs?

    Args:
        emb_dir: Root generate_embeddings directory.
        df: Tokenization DataFrame from load_token_results.
        dataset: Dataset name.
        sim_threshold: Minimum cosine similarity to include a pair.

    Returns:
        DataFrame with dialect, n_all, n_high, mean_delta_all, mean_delta_high,
        p_all, p_high.
    """
    emb_sae = _load_embeddings_dim(emb_dir, dataset, "sae", EMB_DIM)
    if emb_sae is None:
        return pd.DataFrame()

    sae_fert = df[df["dialect"] == "sae"].groupby("RID")["fertility"].mean()
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df["dialect"].unique()]

    records = []
    for dialect in dialects:
        emb_d = _load_embeddings_dim(emb_dir, dataset, dialect, EMB_DIM)
        if emb_d is None:
            continue
        sims = np.sum(emb_sae * emb_d, axis=1)

        dial_fert = df[df["dialect"] == dialect].groupby("RID")["fertility"].mean()
        common_rids = sae_fert.index.intersection(dial_fert.index)
        fert_delta = dial_fert.loc[common_rids].values - sae_fert.loc[common_rids].values

        n = min(len(sims), len(fert_delta))
        sims = sims[:n]
        fert_delta = fert_delta[:n]

        # All pairs
        stat_all, p_all = stats.wilcoxon(fert_delta, alternative="greater")

        # High-similarity pairs
        mask = sims > sim_threshold
        if mask.sum() > 10:
            stat_hi, p_hi = stats.wilcoxon(fert_delta[mask], alternative="greater")
        else:
            stat_hi, p_hi = np.nan, np.nan

        records.append({
            "dialect": DIALECT_LABELS[dialect],
            "n_all": n,
            "mean_delta_all": fert_delta.mean(),
            "p_all": p_all,
            "n_high_sim": int(mask.sum()),
            "mean_delta_high": fert_delta[mask].mean() if mask.sum() > 0 else np.nan,
            "p_high": p_hi,
        })
    return pd.DataFrame(records)


def test_similarity_fertility_correlation(emb_dir: str, df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Spearman correlation between semantic similarity and Δfertility.

    Args:
        emb_dir: Root generate_embeddings directory.
        df: Tokenization DataFrame from load_token_results.
        dataset: Dataset name.

    Returns:
        DataFrame with dialect, rho, p_value, n.
    """
    emb_sae = _load_embeddings_dim(emb_dir, dataset, "sae", EMB_DIM)
    if emb_sae is None:
        return pd.DataFrame()

    sae_fert = df[df["dialect"] == "sae"].groupby("RID")["fertility"].mean()
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df["dialect"].unique()]

    records = []
    for dialect in dialects:
        emb_d = _load_embeddings_dim(emb_dir, dataset, dialect, EMB_DIM)
        if emb_d is None:
            continue
        sims = np.sum(emb_sae * emb_d, axis=1)

        dial_fert = df[df["dialect"] == dialect].groupby("RID")["fertility"].mean()
        common_rids = sae_fert.index.intersection(dial_fert.index)
        fert_delta = dial_fert.loc[common_rids].values - sae_fert.loc[common_rids].values

        n = min(len(sims), len(fert_delta))
        rho, p = stats.spearmanr(sims[:n], fert_delta[:n])
        records.append({
            "dialect": DIALECT_LABELS[dialect],
            "rho": rho,
            "p_value": p,
            "n": n,
        })
    return pd.DataFrame(records)


################
# PRINT UTILS  #
################

def print_table(title: str, df: pd.DataFrame) -> None:
    """Print a table with a title header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(df.to_string())
    print()


########
# MAIN #
########

def main() -> None:
    parser = argparse.ArgumentParser()
    project_config = dialecttax.utils.load_config()
    default_dir = project_config["directories"]["experiments"]
    parser.add_argument("--experiments-dir", default=default_dir)
    parser.add_argument("--output-dir", default="analysis/plots/embeddings")
    args = parser.parse_args()

    emb_dir = os.path.join(args.experiments_dir, "generate_embeddings")

    ########################################
    # 3.1 Vocabulary composition
    ########################################

    print("Loading MultiValue tokenization results...")
    df_mv = load_token_results(args.experiments_dir, "multivalue")
    if len(df_mv) > 0:
        print(f"  {len(df_mv)} samples")
        print_table("Vocab Coverage by Dialect and Tokenizer (MultiValue)", table_vocab_coverage(df_mv))
        plot_vocab_coverage(df_mv, args.output_dir)
        plot_token_overlap(df_mv, args.experiments_dir, args.output_dir)
    else:
        print("  No tokenization results found for MultiValue")

    ########################################
    # 3.2 Transformation similarity
    # 3.3 Meaning vs tax scatter
    ########################################

    print("\nLoading ParallelAAVE tokenization results...")
    df_pa = load_token_results(args.experiments_dir, "parallelaave")
    print(f"  {len(df_pa)} samples")

    for dataset, df in [("multivalue", df_mv), ("parallelaave", df_pa)]:
        print(f"\n--- {dataset} ---")

        # 3.2 Transformation similarity
        print(f"Computing transformation similarities ({dataset})...")
        cross_df = compute_cross_dialect_similarity(emb_dir, dataset)
        pert_sae_df = compute_perturbation_similarity_sae(emb_dir, dataset)
        null_ds = compute_null_baseline(emb_dir, dataset)
        if len(cross_df) > 0 and len(pert_sae_df) > 0:
            print_table(f"Cross-Dialect Similarity ({dataset})", cross_df[cross_df["dim"] == 768].round(4))
            plot_transformation_similarity(cross_df, pert_sae_df, null_ds, dataset, args.output_dir)
        else:
            print("  No embeddings found")

        # 3.3 Meaning vs tax
        print(f"Generating meaning vs tax scatter ({dataset})...")
        if len(df) > 0:
            plot_meaning_vs_tax(emb_dir, df, dataset, args.output_dir)

        ########################################
        # Statistical tests
        ########################################

        # Test 1: paired dialect vs transformation-control similarity
        print(f"\nStatistical tests ({dataset})...")
        test1 = test_dialect_vs_perturbation(emb_dir, dataset)
        if len(test1) > 0:
            print_table(
                f"Test 1: Paired Wilcoxon, Dialect vs Transformation Controls ({dataset})",
                test1.round(4),
            )

        # Test 2: Δfertility > 0 for high-sim pairs
        if len(df) > 0:
            test2 = test_fertility_delta(emb_dir, df, dataset)
            if len(test2) > 0:
                print_table(f"Test 2: Δfertility > 0 ({dataset})", test2.round(4))

        # Test 3: correlation between similarity and Δfertility
        if len(df) > 0:
            test3 = test_similarity_fertility_correlation(emb_dir, df, dataset)
            if len(test3) > 0:
                print_table(f"Test 3: Similarity-Fertility Correlation ({dataset})", test3.round(4))

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
