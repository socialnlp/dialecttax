"""
Section 2: Language Models (Still) Have Tokenization Biases.

Loads per-sample tokenization results from generate_tokens.py outputs and
produces tables and plots comparing fertility, vocabulary coverage, and
token overhead across dialects and tokenizer families.

Usage:
    python analysis/tokenization.py
    python analysis/tokenization.py --experiments-dir /path/to/experiments/generate_tokens
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
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
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
    "bpe": "BPE",
    "gpt2": "BPE",
    "gemma": "BPE",
    "llama": "BPE",
    "qwen": "BPE",
    "unigram": "Unigram",
    "wordpiece": "WordPiece",
}
# Release date of the tokenizer itself (not necessarily the wrapping model).
# o200k_base (gpt-5) was introduced with GPT-4o (2024-05) and inherited by GPT-5.
# Qwen-3 inherits the Qwen-2.5 tokenizer (2024-09). Llama-3 introduced a new
# 128K tokenizer (2024-04); Llama-3.3 reuses it. Gemma-3 introduced a new 262K
# tokenizer (2025-03).
TOKENIZER_RELEASE = {
    "wordpiece": "2018-10",
    "gpt2": "2019-02",
    "unigram": "2019-10",
    "llama": "2024-04",
    "bpe": "2024-05",
    "gemma": "2025-03",
    "qwen": "2025-04",
}

METRICS = ["fertility", "p_in_vocab", "avg_tokens_per_word", "avg_types_per_word", "n_tokens", "n_words"]
METRIC_LABELS = {
    "fertility": "Fertility",
    "p_in_vocab": "Vocabulary coverage",
    "avg_tokens_per_word": "Average tokens per word",
    "avg_types_per_word": "Average types per word",
    "n_tokens": "Token count",
    "n_words": "Word count",
}


###########
# LOADING #
###########

def load_dataset_results(experiments_dir: str, dataset: str) -> pd.DataFrame:
    """Load all tokens.jsonl files for a dataset into a single DataFrame.

    Walks experiments_dir/{dataset}/.../{tokenizer}/tokens.jsonl and adds
    dialect, tokenizer, and (for redial) task columns from the path.

    Args:
        experiments_dir: Root directory containing generate_tokens outputs.
        dataset: One of "parallelaave", "multivalue", "redial".

    Returns:
        DataFrame with all samples and metadata columns.
    """
    rows = []
    dataset_dir = os.path.join(experiments_dir, dataset)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"No directory found at {dataset_dir}")

    if dataset == "redial":
        # redial/{task}/{reasoning}/{dialect}/{tokenizer}/tokens.jsonl
        for task in sorted(os.listdir(dataset_dir)):
            task_dir = os.path.join(dataset_dir, task)
            if not os.path.isdir(task_dir):
                continue
            for reasoning in sorted(os.listdir(task_dir)):
                reasoning_dir = os.path.join(task_dir, reasoning)
                if not os.path.isdir(reasoning_dir):
                    continue
                for dialect in sorted(os.listdir(reasoning_dir)):
                    dialect_dir = os.path.join(reasoning_dir, dialect)
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
                                sample["task"] = task
                                sample["reasoning"] = reasoning
                                sample["dialect"] = dialect
                                sample["tokenizer"] = tokenizer
                                rows.append(sample)
    else:
        # {dataset}/{dialect}/{tokenizer}/tokens.jsonl
        for dialect in sorted(os.listdir(dataset_dir)):
            dialect_dir = os.path.join(dataset_dir, dialect)
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


#############################
# 2.1 MEASURING THE TAX    #
#############################

def table_sae_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1: Mean tokenization metrics per tokenizer on SAE.

    Args:
        df: DataFrame from load_dataset_results (any dataset).

    Returns:
        Pivot table with tokenizers as rows and metrics as columns.
    """
    sae = df[df["dialect"] == "sae"]
    agg = sae.groupby("tokenizer")[METRICS].mean()
    agg = agg.reindex([t for t in TOKENIZER_ORDER if t in agg.index])
    agg.index = agg.index.map(TOKENIZER_LABELS)
    agg.columns = agg.columns.map(METRIC_LABELS)
    return agg.round(3)


####################################
# 2.2 DISPARITIES ACROSS DIALECTS #
####################################

def paired_fertility_delta(df_pa: pd.DataFrame) -> pd.DataFrame:
    """Compute per-sample fertility delta (AAVE - SAE) on ParallelAAVE.

    Samples are paired by RID within each tokenizer.

    Args:
        df_pa: ParallelAAVE DataFrame from load_dataset_results.

    Returns:
        DataFrame with columns: tokenizer, RID, delta_fertility.
    """
    sae = df_pa[df_pa["dialect"] == "sae"].set_index(["tokenizer", "RID"])
    aave = df_pa[df_pa["dialect"] == "aave"].set_index(["tokenizer", "RID"])
    joined = aave[["fertility"]].join(sae[["fertility"]], lsuffix="_aave", rsuffix="_sae")
    joined["delta_fertility"] = joined["fertility_aave"] - joined["fertility_sae"]
    return joined.reset_index()[["tokenizer", "RID", "delta_fertility"]]


def plot_paired_deltas(deltas: pd.DataFrame, df_pa: pd.DataFrame, output_dir: str) -> None:
    """Figure 1: Dumbbell chart of paired fertility on ParallelAAVE.

    Each row is a tokenizer. SAE and AAVE mean fertility shown as paired
    dots connected by a line segment. The gap is the delta; annotated
    with the mean delta value.

    Args:
        deltas: Output of paired_fertility_delta.
        df_pa: ParallelAAVE DataFrame from load_dataset_results.
        output_dir: Directory to save the plot.
    """
    tokenizers = [t for t in TOKENIZER_ORDER if t in deltas["tokenizer"].unique()]
    means = df_pa.groupby(["tokenizer", "dialect"])["fertility"].mean().unstack("dialect")
    means = means.reindex([t for t in tokenizers if t in means.index])

    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = np.arange(len(tokenizers))

    marker_size = 90
    marker_radius_pts = np.sqrt(marker_size) / 2  # scatter size is area in pts^2

    for i, t in enumerate(tokenizers):
        sae_val = means.loc[t, "sae"]
        aave_val = means.loc[t, "aave"]

        ax.scatter(sae_val, i, color=DIALECT_COLORS["sae"], s=marker_size, zorder=3, edgecolors="black", linewidths=0.7)
        ax.scatter(aave_val, i, color=DIALECT_COLORS["aave"], s=marker_size, zorder=3, edgecolors="black", linewidths=0.7)

        # Arrow from right edge of SAE marker to left edge of AAVE marker
        ax.annotate("", xy=(aave_val, i), xytext=(sae_val, i),
                     arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=0.8,
                                     shrinkA=marker_radius_pts, shrinkB=marker_radius_pts),
                     zorder=1)

        # Annotate delta below the line; push lower if gap is narrow
        delta = aave_val - sae_val
        mid = (sae_val + aave_val) / 2
        y_offset = 0.55 if delta < 0.05 else 0.45
        ax.text(mid, i + y_offset, f"+{delta:.3f}", ha="center", va="bottom", fontsize=9, color="#555555")


    ax.set_yticks(y)
    ax.set_yticklabels([TOKENIZER_LABELS[t] for t in tokenizers], fontsize=13)
    ax.set_xlabel("Mean fertility (tokens / words)", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.invert_yaxis()
    ax.set_ylim(len(tokenizers) - 0.3, -0.5)  # extra padding at bottom for label

    # Legend
    ax.scatter([], [], color=DIALECT_COLORS["sae"], s=90, label="SAE", edgecolors="black", linewidths=0.7)
    ax.scatter([], [], color=DIALECT_COLORS["aave"], s=90, label="AAVE", edgecolors="black", linewidths=0.7)
    ax.legend(title="Dialect", loc="upper right", frameon=True, framealpha=0.9,
              fontsize=13, title_fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "parallelaave_fertility.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "parallelaave_fertility.png"), bbox_inches="tight")
    plt.close(fig)


def test_paired_deltas(deltas: pd.DataFrame) -> pd.DataFrame:
    """Statistical tests for paired fertility deltas.

    Runs Wilcoxon signed-rank test per tokenizer (non-parametric,
    accounts for non-normal delta distributions).

    Args:
        deltas: Output of paired_fertility_delta.

    Returns:
        DataFrame with tokenizer, mean_delta, median_delta, statistic, p_value.
    """
    results = []
    for tokenizer in TOKENIZER_ORDER:
        d = deltas[deltas["tokenizer"] == tokenizer]["delta_fertility"].dropna()
        if len(d) < 10:
            continue
        stat, p = stats.wilcoxon(d, alternative="two-sided")
        results.append({
            "tokenizer": TOKENIZER_LABELS[tokenizer],
            "n": len(d),
            "mean_delta": d.mean(),
            "median_delta": d.median(),
            "std_delta": d.std(),
            "wilcoxon_stat": stat,
            "p_value": p,
        })
    return pd.DataFrame(results)


def _plot_ratio_boxplots(df_mv: pd.DataFrame, tokenizers: list[str], ax, title: str, label_fontsize: float = 9) -> None:
    """Plot boxplots of per-sample n_tokens ratio (dialect / SAE) for given tokenizers.

    Args:
        df_mv: MultiValue DataFrame.
        tokenizers: List of tokenizer keys to include (averaged across).
        ax: Matplotlib axes to plot on.
        title: Panel title.
    """
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df_mv["dialect"].unique()]

    # Compute per-sample ratio: dialect n_tokens / SAE n_tokens (paired by RID)
    sae = df_mv[df_mv["dialect"] == "sae"]
    ratios_by_dialect = {}
    for dialect in dialects:
        dial = df_mv[df_mv["dialect"] == dialect]
        dialect_ratios = []
        for t in tokenizers:
            sae_t = sae[sae["tokenizer"] == t].set_index("RID")["n_tokens"]
            dial_t = dial[dial["tokenizer"] == t].set_index("RID")["n_tokens"]
            joined = dial_t / sae_t
            dialect_ratios.append(joined.dropna().values)
        ratios_by_dialect[dialect] = np.concatenate(dialect_ratios)

    data = [ratios_by_dialect[d] for d in dialects]
    colors = [DIALECT_COLORS[d] for d in dialects]

    bp = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("#555555")
            line.set_linewidth(0.8)
    for line in bp["medians"]:
        line.set_color("black")
        line.set_linewidth(1.0)

    ax.axhline(1.0, color="black", ls=":", lw=0.8, zorder=0)
    ax.set_xticks(range(1, len(dialects) + 1))
    ax.set_xticklabels([DIALECT_LABELS[d] for d in dialects], fontsize=label_fontsize)
    ax.set_xlabel("Dialect", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    if title:
        ax.set_title(title, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_multivalue_ratio_bpe(df_mv: pd.DataFrame, output_dir: str) -> None:
    """Ratio of dialect to SAE token lengths, BPE tokenizers only.

    Args:
        df_mv: MultiValue DataFrame from load_dataset_results.
        output_dir: Directory to save the plot.
    """
    bpe_tokenizers = [t for t in TOKENIZER_ORDER if TOKENIZER_FAMILY.get(t) == "BPE" and t in df_mv["tokenizer"].unique()]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _plot_ratio_boxplots(df_mv, bpe_tokenizers, ax, title="", label_fontsize=13)
    ax.set_ylabel("Token length ratio to SAE", fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "multivalue_boxplot_ratio_n_tokens_bpe.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "multivalue_boxplot_ratio_n_tokens_bpe.png"), bbox_inches="tight")
    plt.close(fig)


def plot_multivalue_ratio_panels(df_mv: pd.DataFrame, output_dir: str) -> None:
    """Ratio of dialect to SAE token lengths, one panel per tokenizer family.

    Args:
        df_mv: MultiValue DataFrame from load_dataset_results.
        output_dir: Directory to save the plot.
    """
    families = ["BPE", "Unigram", "WordPiece"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

    for ax, family in zip(axes, families):
        tokenizers = [t for t in TOKENIZER_ORDER if TOKENIZER_FAMILY.get(t) == family and t in df_mv["tokenizer"].unique()]
        _plot_ratio_boxplots(df_mv, tokenizers, ax, family, label_fontsize=8)

    axes[0].set_ylabel("Token length ratio to SAE")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "multivalue_boxplot_ratio_n_tokens.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "multivalue_boxplot_ratio_n_tokens.png"), bbox_inches="tight")
    plt.close(fig)


#######################################
# 2.3 ACROSS TOKENIZER FAMILIES      #
#######################################

def plot_dialect_gap_by_family(df_mv: pd.DataFrame, output_dir: str) -> None:
    """Figure 4: Dialect fertility gap (vs SAE) grouped by tokenizer family.

    Shows that the gap persists across BPE, Unigram, and WordPiece.

    Args:
        df_mv: MultiValue DataFrame from load_dataset_results.
        output_dir: Directory to save the plot.
    """
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df_mv["dialect"].unique()]
    tokenizers = [t for t in TOKENIZER_ORDER if t in df_mv["tokenizer"].unique()]

    # Compute mean fertility per (dialect, tokenizer), then subtract SAE
    pivot = df_mv.groupby(["dialect", "tokenizer"])["fertility"].mean().unstack("tokenizer")
    sae_row = pivot.loc["sae"]
    gaps = pivot.loc[dialects].subtract(sae_row, axis="columns")
    gaps["family"] = gaps.index  # placeholder for melt

    # Restructure: for each tokenizer, get its family and the gap per dialect
    records = []
    for tokenizer in tokenizers:
        family = TOKENIZER_FAMILY[tokenizer]
        for dialect in dialects:
            if tokenizer in gaps.columns:
                records.append({
                    "family": family,
                    "tokenizer": TOKENIZER_LABELS[tokenizer],
                    "dialect": DIALECT_LABELS[dialect],
                    "gap": gaps.loc[dialect, tokenizer],
                })
    gap_df = pd.DataFrame(records)

    families = ["BPE", "Unigram", "WordPiece"]
    fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 5), sharey=True)

    for ax, family in zip(axes, families):
        sub = gap_df[gap_df["family"] == family]
        family_tokenizers = sub["tokenizer"].unique()
        n_tok = len(family_tokenizers)
        n_dial = len(dialects)
        width = 0.8 / n_tok

        for j, tokenizer in enumerate(family_tokenizers):
            tok_sub = sub[sub["tokenizer"] == tokenizer]
            x = np.arange(n_dial)
            offset = (j - n_tok / 2 + 0.5) * width
            ax.bar(x + offset, tok_sub["gap"].values, width, label=tokenizer, alpha=0.8)

        ax.axhline(0, color="gray", ls="--", lw=0.8)
        ax.set_xticks(range(n_dial))
        ax.set_xticklabels([DIALECT_LABELS[d] for d in dialects], rotation=45, ha="right")
        ax.set_title(family)
        ax.legend(fontsize=7)

    axes[0].set_ylabel("Fertility Gap (dialect - SAE)")
    fig.suptitle("Tokenization Tax by Family: Fertility Gap Relative to SAE", fontsize=14)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "dialect_gap_by_family.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "dialect_gap_by_family.png"), bbox_inches="tight")
    plt.close(fig)


def table_family_summary(df_mv: pd.DataFrame) -> pd.DataFrame:
    """Summary table: mean fertility gap per tokenizer family.

    Args:
        df_mv: MultiValue DataFrame from load_dataset_results.

    Returns:
        DataFrame with family, mean dialect gap, and per-dialect gaps.
    """
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df_mv["dialect"].unique()]
    tokenizers = [t for t in TOKENIZER_ORDER if t in df_mv["tokenizer"].unique()]

    pivot = df_mv.groupby(["dialect", "tokenizer"])["fertility"].mean().unstack("tokenizer")
    sae_row = pivot.loc["sae"]
    gaps = pivot.loc[dialects].subtract(sae_row, axis="columns")

    records = []
    for family in ["BPE", "Unigram", "WordPiece"]:
        family_tokenizers = [t for t in tokenizers if TOKENIZER_FAMILY[t] == family]
        family_gaps = gaps[family_tokenizers].mean(axis=1)
        row = {"family": family, "mean_gap": family_gaps.mean()}
        for dialect in dialects:
            row[DIALECT_LABELS[dialect]] = family_gaps.loc[dialect]
        records.append(row)
    return pd.DataFrame(records).round(4)


################
# INCOME PLOT  #
################

DIALECT_MEDIAN_INCOME = {
    "sae": 80610,
    "aave": 56490,
    "appalachian": 61688,
    "chicano": 65540,
    "indian": 166200,
    "singapore": 134818,
}

def plot_income_dialect(df_mv: pd.DataFrame, output_dir: str) -> None:
    """Scatter plot of token length ratio to SAE vs median income by dialect.

    Reproduces the original bts/notebooks/text/multivalue.ipynb income plot.

    Args:
        df_mv: MultiValue DataFrame from load_dataset_results.
        output_dir: Directory to save the plot.
    """
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df_mv["dialect"].unique()]
    sae = df_mv[df_mv["dialect"] == "sae"]

    # Compute per-family mean ratio
    records = []
    for family in ["BPE", "Unigram", "WordPiece"]:
        family_tokenizers = [t for t in TOKENIZER_ORDER if TOKENIZER_FAMILY.get(t) == family and t in df_mv["tokenizer"].unique()]
        for dialect in dialects:
            dial = df_mv[df_mv["dialect"] == dialect]
            ratios = []
            for t in family_tokenizers:
                sae_t = sae[sae["tokenizer"] == t].set_index("RID")["n_tokens"]
                dial_t = dial[dial["tokenizer"] == t].set_index("RID")["n_tokens"]
                ratio = (dial_t / sae_t).dropna()
                ratios.append(ratio.mean())
            records.append({
                "Tokenizer": family,
                "Dialect": DIALECT_LABELS[dialect],
                "income": DIALECT_MEDIAN_INCOME[dialect],
                "ratio": np.mean(ratios),
            })
    df_ratios_mean = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(4, 3))
    x_min = np.min(df_ratios_mean["income"]) - 8000
    x_max = np.max(df_ratios_mean["income"]) + 8000
    ax.set_xlim(x_min, x_max)
    y_min = np.min(df_ratios_mean["ratio"]) - 0.01
    y_max = np.max(df_ratios_mean["ratio"]) + 0.01
    ax.set_ylim(y_min, y_max)

    median_income = DIALECT_MEDIAN_INCOME["sae"]
    plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0)
    plt.axvline(x=median_income, color="gray", linestyle="--", linewidth=1.0)
    plt.fill_between(x=np.linspace(x_min, median_income, 100), y1=1.0, y2=y_max, alpha=0.25, color="red")
    plt.fill_between(x=np.linspace(median_income, x_max, 100), y1=1.0, y2=y_max, alpha=0.1, color="red")
    plt.fill_betweenx(y=np.linspace(y_min, 1.0, 100), x1=x_min, x2=median_income, alpha=0.1, color="green")
    plt.fill_betweenx(y=np.linspace(y_min, 1.0, 100), x1=median_income, x2=x_max, alpha=0.25, color="green")

    hue_column = "Tokenizer"
    g = sns.scatterplot(df_ratios_mean, x="income", y="ratio", hue=hue_column, style="Dialect", s=100, alpha=0.8, edgecolor="black", ax=ax)

    # Split legend into tokenizer and dialect
    legend_hue_values = df_ratios_mean[hue_column].nunique() + 1
    handles, labels = ax.get_legend_handles_labels()
    legend_hue_handles = handles[:legend_hue_values]
    legend_hue_labels = labels[:legend_hue_values]
    legend_style_handles = handles[legend_hue_values:]
    legend_style_labels = labels[legend_hue_values:]

    legend_hue = ax.legend(legend_hue_handles, legend_hue_labels, ncols=1, fontsize=7.5, frameon=True, labelspacing=0.5, markerscale=0.7)
    legend_style = ax.legend(legend_style_handles, legend_style_labels, ncols=1, fontsize=7.5, frameon=True, labelspacing=0.5, markerscale=0.7)
    ax.add_artist(legend_hue)

    legend_hue.set_bbox_to_anchor((.28, .83))
    legend_style.set_bbox_to_anchor((.28, .5))

    xlabels = ["${:,.0f}k".format(x) for x in g.get_xticks() // 1000]
    g.set_xticklabels(xlabels)
    ax.set_xlabel("Income (USD)")
    ax.set_ylabel("Token length ratio to SAE")
    sns.set_style("ticks")
    sns.despine(top=True, right=True, left=False, bottom=False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "multivalue_income_dialect_n_tokens.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "multivalue_income_dialect_n_tokens.png"), bbox_inches="tight")
    plt.close(fig)


############
# TIMELINE #
############

def _timeline_positions(tokenizers: list[str], gap_years: float = 2.0, extra_units: float = 1.5) -> list[float]:
    """Evenly-spaced x positions with extra space inserted across multi-year gaps.

    Args:
        tokenizers: Tokenizer keys in chronological order.
        gap_years: Minimum inter-release gap that triggers extra space.
        extra_units: Additional x units added on top of the unit step for wide gaps.
    """
    xs = [0.0]
    for i in range(1, len(tokenizers)):
        prev = pd.to_datetime(TOKENIZER_RELEASE[tokenizers[i - 1]])
        curr = pd.to_datetime(TOKENIZER_RELEASE[tokenizers[i]])
        step = 1.0 + (extra_units if (curr - prev).days / 365.25 > gap_years else 0.0)
        xs.append(xs[-1] + step)
    return xs


def _per_sample_ratios(df_mv: pd.DataFrame, tokenizers: list[str], dialects: list[str],
                        metric: str = "n_tokens") -> dict[tuple[str, str], "np.ndarray"]:
    """Paired per-sample dialect/SAE ratio for a metric, keyed by (tokenizer, dialect)."""
    sae = df_mv[df_mv["dialect"] == "sae"]
    out: dict[tuple[str, str], "np.ndarray"] = {}
    for t in tokenizers:
        sae_t = sae[sae["tokenizer"] == t].set_index("RID")[metric].replace(0, np.nan)
        for d in dialects:
            dial_t = df_mv[(df_mv["dialect"] == d) & (df_mv["tokenizer"] == t)].set_index("RID")[metric]
            out[(t, d)] = (dial_t / sae_t).dropna().values
    return out


def plot_tokenizer_timeline(df_mv: pd.DataFrame, output_dir: str) -> None:
    """Timeline of dialect/SAE ratio across all token metrics on MultiValue.

    Grid of panels (one per metric) sharing the release-date x-axis. Each
    panel draws one line per non-SAE dialect with markers coded by tokenizer
    algorithm family.

    Args:
        df_mv: MultiValue DataFrame from load_dataset_results.
        output_dir: Directory to save the plot.
    """
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df_mv["dialect"].unique()]
    tokenizers = [t for t in TOKENIZER_ORDER if t in df_mv["tokenizer"].unique() and t in TOKENIZER_RELEASE]
    tokenizers.sort(key=lambda t: TOKENIZER_RELEASE[t])

    xs = _timeline_positions(tokenizers)
    family_marker = {"BPE": "o", "Unigram": "s", "WordPiece": "^"}

    ordered = ["p_in_vocab"] + sorted([m for m in METRICS if m != "p_in_vocab"], key=lambda m: METRIC_LABELS[m])

    ncols = 3
    nrows = (len(ordered) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5 * ncols, 7.8 * nrows), sharex=True)
    axes = axes.flatten()

    for idx, metric in enumerate(ordered):
        ax = axes[idx]
        per_sample = _per_sample_ratios(df_mv, tokenizers, dialects, metric=metric)
        for d in dialects:
            y = [per_sample[(t, d)].mean() for t in tokenizers]
            ax.plot(xs, y, color=DIALECT_COLORS[d], lw=2.8, alpha=0.6, zorder=1)
            for x_val, t, y_val in zip(xs, tokenizers, y):
                ax.scatter(x_val, y_val, color=DIALECT_COLORS[d],
                           marker=family_marker[TOKENIZER_FAMILY[t]],
                           s=440, edgecolors="black", linewidths=1.6, zorder=2)

        ax.axhline(1.0, color="black", ls=":", lw=1.1, zorder=0)
        ax.set_title(METRIC_LABELS[metric], fontsize=36)
        ax.tick_params(axis="y", labelsize=30)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for idx in range(len(ordered), nrows * ncols):
        axes[idx].set_visible(False)

    for ax in axes[:len(ordered)]:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{TOKENIZER_LABELS[t]} ({TOKENIZER_RELEASE[t]})" for t in tokenizers],
                           rotation=90, ha="center", fontsize=30)

    dialect_handles = [plt.Line2D([0], [0], marker="o", color=DIALECT_COLORS[d], linestyle="-",
                                   label=DIALECT_LABELS[d], markeredgecolor="black", markeredgewidth=1.5,
                                   markersize=24, lw=3.2)
                       for d in dialects]
    family_handles = [plt.Line2D([0], [0], marker=m, color="gray", linestyle="", label=f,
                                  markeredgecolor="black", markeredgewidth=1.5, markersize=24)
                      for f, m in family_marker.items()]
    fig.legend(handles=dialect_handles + family_handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.08), ncol=len(dialects) + len(family_marker),
               fontsize=34, frameon=True)

    fig.supylabel("Mean ratio to SAE", fontsize=42)
    fig.supxlabel("Tokenizer (chronological, not to scale)", fontsize=36)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "tokenizer_timeline.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "tokenizer_timeline.png"), bbox_inches="tight")
    plt.close(fig)


def test_tokenizer_timeline(df_mv: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation between tokenizer release date and dialect/SAE ratio.

    Per dialect, correlates each tokenizer's release date (ordinal) against
    its mean n_tokens ratio to SAE. A negative rho means the tax shrinks
    over time; rho near zero means the tax is stable.

    Args:
        df_mv: MultiValue DataFrame from load_dataset_results.

    Returns:
        DataFrame with dialect, spearman_rho, p_value, n_tokenizers.
    """
    dialects = [d for d in DIALECT_ORDER if d != "sae" and d in df_mv["dialect"].unique()]
    tokenizers = [t for t in TOKENIZER_ORDER if t in df_mv["tokenizer"].unique() and t in TOKENIZER_RELEASE]
    tokenizers.sort(key=lambda t: TOKENIZER_RELEASE[t])

    per_sample = _per_sample_ratios(df_mv, tokenizers, dialects)
    date_ord = [pd.to_datetime(TOKENIZER_RELEASE[t]).toordinal() for t in tokenizers]

    records = []
    for d in dialects:
        mean_ratios = [per_sample[(t, d)].mean() for t in tokenizers]
        rho, p = stats.spearmanr(date_ord, mean_ratios)
        records.append({
            "dialect": DIALECT_LABELS[d],
            "n_tokenizers": len(tokenizers),
            "spearman_rho": rho,
            "p_value": p,
        })
    return pd.DataFrame(records).round(4)


###############
# REDIAL VIEW #
###############


##############
# PRINT UTILS #
##############

def print_table(title: str, df: pd.DataFrame) -> None:
    """Print a table with a title header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    print(df.to_string())
    print()


########
# MAIN #
########

def main() -> None:
    parser = argparse.ArgumentParser()
    project_config = dialecttax.utils.load_config()
    default_dir = project_config["directories"]["experiments"]
    parser.add_argument("--experiments-dir", default=os.path.join(default_dir, "generate_tokens"))
    parser.add_argument("--output-dir", default="analysis/plots/tokenization")
    args = parser.parse_args()

    # Load all three datasets
    print("Loading ParallelAAVE...")
    df_pa = load_dataset_results(args.experiments_dir, "parallelaave")
    print(f"  {len(df_pa)} samples")

    print("Loading MultiValue...")
    df_mv = load_dataset_results(args.experiments_dir, "multivalue")
    print(f"  {len(df_mv)} samples")

    print("Loading ReDial...")
    df_redial = load_dataset_results(args.experiments_dir, "redial")
    print(f"  {len(df_redial)} samples")

    # 2.1 SAE baseline table
    print_table("Table 1: SAE Baseline Tokenization Metrics (MultiValue)", table_sae_baseline(df_mv))
    print_table("Table 1b: SAE Baseline Tokenization Metrics (ParallelAAVE)", table_sae_baseline(df_pa))

    # 2.2 Dialect disparities — paired test on ParallelAAVE
    deltas = paired_fertility_delta(df_pa)
    print_table("Paired Fertility Delta (AAVE - SAE) on ParallelAAVE", test_paired_deltas(deltas))
    plot_paired_deltas(deltas, df_pa, args.output_dir)

    # 2.2 Dialect disparities — MultiValue breadth
    plot_multivalue_ratio_bpe(df_mv, args.output_dir)
    plot_multivalue_ratio_panels(df_mv, args.output_dir)
    plot_income_dialect(df_mv, args.output_dir)

    # 2.3 Across tokenizer families
    print_table("Fertility Gap by Tokenizer Family (MultiValue)", table_family_summary(df_mv))
    plot_dialect_gap_by_family(df_mv, args.output_dir)

    # Temporal: has the tax improved as tokenizers have advanced?
    print_table("Spearman (release date vs dialect/SAE ratio)", test_tokenizer_timeline(df_mv))
    plot_tokenizer_timeline(df_mv, args.output_dir)

    # Bonus: ReDial task view

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
