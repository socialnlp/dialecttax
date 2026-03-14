"""
Section 5.3 reviewer response: dialect vs. transformation baselines.

Addresses the reviewer question of whether the per-layer divergence between
dialect and SAE hidden states is normal LLM behavior or specific to dialect
variation. Runs the identical per-layer cosine-similarity analysis on the
non-dialect transformations (orthographic noise and translation), where the
"dialect" arm is replaced by a dialect-agnostic perturbation of the SAE text
and SAE stays the unperturbed control.

Two families of baseline transformation:
- Orthographic noise (meaning-preserving surface corruption): character
  swap/drop/insert and random/alternating capitalization.
- Translation (meaning-preserving full surface change): SAE translated into
  six languages spanning Latin and non-Latin scripts.

Plots:
1. Aggregate layer-similarity curves: dialect vs. noise vs. translation
   (one curve per group, pooled across models; thin lines per transformation).
2. Final-layer similarity ranking: every dialect and transformation, sorted,
   colored by group.
3. Per-transformation layer-similarity curves: one curve per perturbation type
   (two panels: noise, translation), direct-labeled, dialect mean as reference.
4. Summary + late-layer-drop tables.

Usage:
    python analysis/layers_baselines.py
    python analysis/layers_baselines.py --experiments-dir /path/to/experiments
"""

import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import layers  # noqa: E402  reuse loading + style conventions from the §5.3 analysis

import dialecttax.utils  # noqa: E402


############
# GROUPS   #
############

NOISE = ["swap-0.05", "drop-0.05", "drop-0.15", "insert-0.05", "capitalize-random", "capitalize-alternating"]
TRANSLATE = ["translate-french", "translate-polish", "translate-yoruba",
             "translate-chinese", "translate-hindi", "translate-khmer"]

TRANSFORM_LABELS = {
    "swap-0.05": "Swap 5%",
    "drop-0.05": "Drop 5%",
    "drop-0.15": "Drop 15%",
    "insert-0.05": "Insert 5%",
    "capitalize-random": "Capitalize (random)",
    "capitalize-alternating": "Capitalize (alternating)",
    "translate-french": "Translate (French)",
    "translate-polish": "Translate (Polish)",
    "translate-yoruba": "Translate (Yoruba)",
    "translate-chinese": "Translate (Chinese)",
    "translate-hindi": "Translate (Hindi)",
    "translate-khmer": "Translate (Khmer)",
}

# Dark2: colorblind-safe qualitative palette
GROUP_COLORS = {"dialect": "#1b9e77", "noise": "#d95f02", "translate": "#7570b3"}
GROUP_LABELS = {"dialect": "Dialect", "noise": "Perturbation", "translate": "Translation"}

N_GRID = 50  # resampled normalized-depth points for cross-model pooling


###########
# LOADING #
###########

def load_transform_similarity(experiments_dir, model, dataset, transform):
    """Load layer similarity for a perturbed-SAE vs SAE transformation.

    Args:
        experiments_dir: Root experiments directory.
        model: Model name (e.g. "llama_8b_base").
        dataset: Dataset name (e.g. "multivalue").
        transform: Transformation name (e.g. "swap-0.05", "translate-french").

    Returns:
        ndarray of shape (n_samples, n_layers) or None if not found.
    """
    path = os.path.join(experiments_dir, "generate_layers", model, dataset,
                        "perturbed", transform, "layer_similarity.npy")
    if not os.path.exists(path):
        return None
    return np.load(path)


def _resample(curve, n=N_GRID):
    """Resample a per-layer curve onto ``n`` normalized-depth points.

    Args:
        curve: 1-D per-layer mean-similarity array.
        n: Number of output points on [0, 1].

    Returns:
        Length-``n`` ndarray interpolated onto a common depth grid.
    """
    x = np.linspace(0, 1, len(curve))
    return np.interp(np.linspace(0, 1, n), x, curve)


def _dialect_group_curve(experiments_dir, model, dataset):
    """Mean resampled similarity curve across all dialects for a model/dataset.

    Args:
        experiments_dir: Root experiments directory.
        model: Model name.
        dataset: Dataset name.

    Returns:
        Length-N_GRID ndarray, or None if no dialects present.
    """
    curves = []
    for dialect in layers.DIALECT_ORDER:
        sim = layers.load_layer_similarity(experiments_dir, model, dataset, dialect)
        if sim is not None:
            curves.append(_resample(sim.mean(axis=0)))
    return np.mean(curves, axis=0) if curves else None


def _transform_group_curve(experiments_dir, model, dataset, transforms):
    """Mean resampled similarity curve across a set of transformations.

    Args:
        experiments_dir: Root experiments directory.
        model: Model name.
        dataset: Dataset name.
        transforms: List of transformation names.

    Returns:
        Length-N_GRID ndarray, or None if none present.
    """
    curves = []
    for t in transforms:
        sim = load_transform_similarity(experiments_dir, model, dataset, t)
        if sim is not None:
            curves.append(_resample(sim.mean(axis=0)))
    return np.mean(curves, axis=0) if curves else None


############
# PLOTTING #
############

def plot_group_comparison(experiments_dir, models, output_dir):
    """Plot pooled dialect/noise/translation layer-similarity curves.

    Each model/dataset contributes one curve per group; the bold line is the
    mean across those combos and the band is +/-1 std. Thin lines show the
    individual transformations (pooled across models) so the within-group
    spread is visible.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.
        output_dir: Directory to save the plot.
    """
    depth = np.linspace(0, 1, N_GRID)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Thin per-transformation lines (pooled across models/datasets)
    thin_specs = [("noise", NOISE), ("translate", TRANSLATE)]
    for group, transforms in thin_specs:
        for t in transforms:
            per_combo = []
            for model in models:
                for dataset in layers.DATASETS:
                    sim = load_transform_similarity(experiments_dir, model, dataset, t)
                    if sim is not None:
                        per_combo.append(_resample(sim.mean(axis=0)))
            if per_combo:
                ax.plot(depth, np.mean(per_combo, axis=0), color=GROUP_COLORS[group],
                        linewidth=2.0, alpha=0.30, zorder=1)
    # Thin per-dialect lines
    for dialect in layers.DIALECT_ORDER:
        per_combo = []
        for model in models:
            for dataset in layers.DATASETS:
                sim = layers.load_layer_similarity(experiments_dir, model, dataset, dialect)
                if sim is not None:
                    per_combo.append(_resample(sim.mean(axis=0)))
        if per_combo:
            ax.plot(depth, np.mean(per_combo, axis=0), color=GROUP_COLORS["dialect"],
                    linewidth=2.0, alpha=0.30, zorder=1)

    # Bold group means with +/-1 std bands
    group_curves = {
        "dialect": [], "noise": [], "translate": [],
    }
    for model in models:
        for dataset in layers.DATASETS:
            d = _dialect_group_curve(experiments_dir, model, dataset)
            if d is not None:
                group_curves["dialect"].append(d)
            n = _transform_group_curve(experiments_dir, model, dataset, NOISE)
            if n is not None:
                group_curves["noise"].append(n)
            t = _transform_group_curve(experiments_dir, model, dataset, TRANSLATE)
            if t is not None:
                group_curves["translate"].append(t)

    for group in ["dialect", "noise", "translate"]:
        stack = np.stack(group_curves[group], axis=0)
        mean = stack.mean(axis=0)
        std = stack.std(axis=0)
        ax.fill_between(depth, mean - std, mean + std, color=GROUP_COLORS[group], alpha=0.15, zorder=2)
        ax.plot(depth, mean, color=GROUP_COLORS[group], linewidth=2.3,
                label=GROUP_LABELS[group], zorder=4)

    ax.set_xlabel("Layer depth (normalized)", fontsize=15)
    ax.set_ylabel("Cosine similarity", fontsize=15)
    ax.tick_params(axis="both", labelsize=13)
    ax.set_ylim(0, 1.02)
    ax.legend(title="Transformation", fontsize=13, title_fontsize=13,
              loc="lower center", frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "hidden_state_group_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "hidden_state_group_comparison.png"), bbox_inches="tight")
    plt.close(fig)


def plot_final_layer_ranking(experiments_dir, models, output_dir):
    """Horizontal bar chart of final-layer similarity per dialect/transformation.

    Values are pooled (mean) across all models and datasets, sorted descending,
    and colored by group so the position of dialects relative to the
    transformation baselines is immediate.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.
        output_dir: Directory to save the plot.
    """
    items = []  # (label, group, mean_final)
    for dialect in layers.DIALECT_ORDER:
        vals = []
        for model in models:
            for dataset in layers.DATASETS:
                sim = layers.load_layer_similarity(experiments_dir, model, dataset, dialect)
                if sim is not None:
                    vals.append(sim.mean(axis=0)[-1])
        if vals:
            items.append((f"SAE → {layers.DIALECT_LABELS[dialect]}", "dialect", float(np.mean(vals))))
    # Match the embeddings transformation-similarity figure: one representative
    # drop rate, percentage-free labels. Display-only — pooled group stats keep
    # the full NOISE set.
    skip = {"drop-0.05"}
    short = {"swap-0.05": "Swap", "drop-0.15": "Drop", "insert-0.05": "Insert"}
    for group, transforms in [("noise", NOISE), ("translate", TRANSLATE)]:
        for t in transforms:
            if t in skip:
                continue
            vals = []
            for model in models:
                for dataset in layers.DATASETS:
                    sim = load_transform_similarity(experiments_dir, model, dataset, t)
                    if sim is not None:
                        vals.append(sim.mean(axis=0)[-1])
            if vals:
                items.append((short.get(t, TRANSFORM_LABELS[t]), group, float(np.mean(vals))))

    items.sort(key=lambda r: r[2])
    labels = [r[0] for r in items]
    colors = [GROUP_COLORS[r[1]] for r in items]
    vals = [r[2] for r in items]

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    y = np.arange(len(items))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Cosine similarity", fontsize=14)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.5, len(items) - 0.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in ["dialect", "noise", "translate"]]
    ax.legend(handles, [GROUP_LABELS[g] for g in ["dialect", "noise", "translate"]],
              title="Transformation", fontsize=12, title_fontsize=12,
              loc="lower right", frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "hidden_state_final_layer_ranking.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "hidden_state_final_layer_ranking.png"), bbox_inches="tight")
    plt.close(fig)


def _mca_accuracy_deltas(experiments_dir):
    """Mean MCA accuracy delta vs SAE per condition on ReDial (naive reasoning).

    Reads generate_logits_mca metadata for the dialect arms and the perturbed-SAE
    arms, computes per-(model, task) cell accuracy, and averages the per-cell
    delta against the SAE baseline.

    Args:
        experiments_dir: Root experiments directory.

    Returns:
        Dict mapping condition name ("aave", "swap-0.05", ...) to mean delta in
        percentage points, or None if no MCA data is present.
    """
    root = os.path.join(experiments_dir, "generate_logits_mca")
    rows = []
    for path in glob.glob(os.path.join(root, "*", "redial", "*", "naive", "*", "metadata.jsonl")):
        model, _ds, task, _r, cond, _ = os.path.relpath(path, root).split(os.sep)
        rows.append((path, model, task, cond))
    for path in glob.glob(os.path.join(root, "*", "redial", "*", "naive", "sae", "perturbed", "*", "metadata.jsonl")):
        parts = os.path.relpath(path, root).split(os.sep)
        rows.append((path, parts[0], parts[2], parts[6]))
    if not rows:
        return None

    records = []
    for path, model, task, cond in rows:
        with open(path) as f:
            correct = [json.loads(line)["correct"] for line in f if line.strip()]
        records.append({"model": model, "task": task, "cond": cond, "acc": np.mean(correct) * 100})
    acc = pd.DataFrame(records).pivot_table(index=["model", "task"], columns="cond", values="acc")
    return {c: (acc[c] - acc["sae"]).mean() for c in acc.columns if c != "sae"}


def plot_similarity_vs_accuracy(experiments_dir, models, output_dir):
    """Scatter final-layer similarity against MCA accuracy loss per condition.

    x = final-layer cosine similarity to SAE (pooled across models/datasets),
    y = mean ReDial MCA accuracy delta vs SAE (naive). Shows the dissociation:
    transformations span the whole similarity range at near-zero accuracy cost,
    while AAVE pairs the highest similarity with the largest loss. Dialects
    other than AAVE have no ReDial accuracy data, so this covers 13 conditions.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.
        output_dir: Directory to save the plot.
    """
    deltas = _mca_accuracy_deltas(experiments_dir)
    if deltas is None:
        print("No MCA data found — skipping similarity-vs-accuracy plot")
        return

    def _final_sim(loader):
        vals = []
        for model in models:
            for dataset in layers.DATASETS:
                sim = loader(model, dataset)
                if sim is not None:
                    vals.append(sim.mean(axis=0)[-1])
        return float(np.mean(vals)) if vals else None

    # Short display labels: one representative drop rate, bare perturbation and
    # language names, bare dialect name.
    skip = {"drop-0.05"}
    short = {
        "swap-0.05": "Swap", "drop-0.15": "Drop", "insert-0.05": "Insert",
        "capitalize-random": "Capitalize (random)", "capitalize-alternating": "Capitalize (alternating)",
        "translate-french": "French", "translate-polish": "Polish", "translate-yoruba": "Yoruba",
        "translate-chinese": "Chinese", "translate-hindi": "Hindi", "translate-khmer": "Khmer",
    }

    # (label, group, marker, final similarity, accuracy delta)
    points = []
    sim_aave = _final_sim(lambda m, ds: layers.load_layer_similarity(experiments_dir, m, ds, "aave"))
    if sim_aave is not None and "aave" in deltas:
        points.append(("AAVE", "dialect", "o", sim_aave, deltas["aave"]))
    for group, transforms, marker in [("noise", NOISE, "s"), ("translate", TRANSLATE, "D")]:
        for t in transforms:
            if t in skip:
                continue
            sim = _final_sim(lambda m, ds, t=t: load_transform_similarity(experiments_dir, m, ds, t))
            if sim is not None and t in deltas:
                points.append((short.get(t, TRANSFORM_LABELS[t]), group, marker, sim, deltas[t]))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axhline(0, color="#999999", linewidth=0.8, linestyle="--", zorder=1)
    # Per-label (dx, dy, ha) overrides where the default above-center placement collides
    offsets = {
        "French": (0, -24, "center"),
        "Capitalize (alternating)": (-26, -20, "left"),
        "Polish": (0, -24, "center"),
        "Insert": (-12, -4, "right"),
        "Hindi": (0, -24, "center"),
        "Chinese": (0, -24, "center"),
        "Khmer": (16, -4, "left"),
        "AAVE": (-12, -4, "right"),
        "Swap": (0, -20, "center"),
        "Capitalize (random)": (38, 12, "center"),
        "Yoruba": (0, 16, "center"),
    }
    for label, group, marker, x, y in points:
        ax.scatter(x, y, color=GROUP_COLORS[group], marker=marker, s=250,
                   edgecolors="black", linewidths=0.7, zorder=3)
        dx, dy, ha = offsets.get(label, (0, 12, "center"))
        ax.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, fontsize=10, color="#333333")

    ax.set_xlabel("Cosine similarity", fontsize=14)
    ax.set_ylabel("Δ Accuracy from SAE (percentage points)", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    handles = [
        plt.Line2D([0], [0], color=GROUP_COLORS["dialect"], marker="o", linestyle="",
                   markersize=10, markeredgecolor="black"),
        plt.Line2D([0], [0], color=GROUP_COLORS["noise"], marker="s", linestyle="",
                   markersize=10, markeredgecolor="black"),
        plt.Line2D([0], [0], color=GROUP_COLORS["translate"], marker="D", linestyle="",
                   markersize=10, markeredgecolor="black"),
        plt.Line2D([0], [0], color="#999999", linestyle="--", linewidth=1.2),
    ]
    ax.legend(handles, [GROUP_LABELS[g] for g in ["dialect", "noise", "translate"]] + ["Baseline"],
              title="Transformation", fontsize=12, title_fontsize=12,
              loc="lower left", frameon=True, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, "hidden_state_similarity_vs_accuracy.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, "hidden_state_similarity_vs_accuracy.png"), bbox_inches="tight")
    plt.close(fig)


###########
# SUMMARY #
###########

def compute_group_summary(experiments_dir, models):
    """Summarize embedding/peak/final/drop similarity per transformation group.

    For each (model, dataset), the per-group mean curve is computed, then the
    embedding-layer (depth 0), peak, and final-layer similarity are pooled
    across combos. ``drop`` is peak minus final: the size of the late-layer
    divergence that §5.3 discusses.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.

    Returns:
        DataFrame with one row per group.
    """
    per_group = {"dialect": [], "noise": [], "translate": []}
    for model in models:
        for dataset in layers.DATASETS:
            for group, curve in [
                ("dialect", _dialect_group_curve(experiments_dir, model, dataset)),
                ("noise", _transform_group_curve(experiments_dir, model, dataset, NOISE)),
                ("translate", _transform_group_curve(experiments_dir, model, dataset, TRANSLATE)),
            ]:
                if curve is not None:
                    per_group[group].append(curve)

    rows = []
    for group in ["dialect", "noise", "translate"]:
        stack = np.stack(per_group[group], axis=0)
        emb = stack[:, 0]
        peak = stack.max(axis=1)
        final = stack[:, -1]
        rows.append({
            "group": GROUP_LABELS[group],
            "n_combos": stack.shape[0],
            "sim_embedding": emb.mean(),
            "sim_peak": peak.mean(),
            "sim_final": final.mean(),
            "late_drop": (peak - final).mean(),
        })
    return pd.DataFrame(rows)


def compute_transform_summary(experiments_dir, models):
    """Per-transformation embedding/peak/final/drop, pooled across models.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.

    Returns:
        DataFrame with one row per dialect/transformation.
    """
    rows = []

    def _row(label, group, loader):
        curves = []
        for model in models:
            for dataset in layers.DATASETS:
                sim = loader(model, dataset)
                if sim is not None:
                    curves.append(_resample(sim.mean(axis=0)))
        if not curves:
            return
        stack = np.stack(curves, axis=0)
        rows.append({
            "group": GROUP_LABELS[group],
            "transform": label,
            "sim_embedding": stack[:, 0].mean(),
            "sim_peak": stack.max(axis=1).mean(),
            "sim_final": stack[:, -1].mean(),
            "late_drop": (stack.max(axis=1) - stack[:, -1]).mean(),
            "n_combos": stack.shape[0],
        })

    for dialect in layers.DIALECT_ORDER:
        _row(layers.DIALECT_LABELS[dialect], "dialect",
             lambda m, ds, d=dialect: layers.load_layer_similarity(experiments_dir, m, ds, d))
    for group, transforms in [("noise", NOISE), ("translate", TRANSLATE)]:
        for t in transforms:
            _row(TRANSFORM_LABELS[t], group,
                 lambda m, ds, t=t: load_transform_similarity(experiments_dir, m, ds, t))

    return pd.DataFrame(rows).sort_values("sim_final", ascending=False)


def test_late_drop_by_group(experiments_dir, models):
    """Test that the final layer is below the peak layer within every group.

    For each (model, dataset, group) the per-group mean curve gives one paired
    (peak, final) observation. A one-sided Wilcoxon signed-rank test over combos
    asks whether final < peak. A significant result in every group means the
    late-layer drop is not specific to dialects.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.

    Returns:
        DataFrame with one row per group.
    """
    rows = []
    for group, transforms in [("dialect", None), ("noise", NOISE), ("translate", TRANSLATE)]:
        peaks, finals = [], []
        for model in models:
            for dataset in layers.DATASETS:
                if group == "dialect":
                    curve = _dialect_group_curve(experiments_dir, model, dataset)
                else:
                    curve = _transform_group_curve(experiments_dir, model, dataset, transforms)
                if curve is not None:
                    peaks.append(curve.max())
                    finals.append(curve[-1])
        peaks, finals = np.array(peaks), np.array(finals)
        stat, p = stats.wilcoxon(finals, peaks, alternative="less")
        rows.append({
            "group": GROUP_LABELS[group],
            "n_combos": len(peaks),
            "mean_peak": peaks.mean(),
            "mean_final": finals.mean(),
            "mean_drop": (peaks - finals).mean(),
            "wilcoxon_stat": stat,
            "p_value": p,
        })
    return pd.DataFrame(rows)


# drop-0.05 is redundant with drop-0.15 and excluded from the headline figures,
# so the per-cell battery and dialect-vs-transformation tests exclude it too.
EXCLUDED_TRANSFORMS = {"drop-0.05"}


def _cell_battery_row(sim):
    """Run the per-cell battery on one similarity matrix.

    Args:
        sim: ndarray of shape (n_samples, n_layers).

    Returns:
        Dict of test statistics for the cell.
    """
    n_samples, n_layers = sim.shape
    mean_curve = sim.mean(axis=0)
    _, fried_p = stats.friedmanchisquare(*[sim[:, l] for l in range(n_layers)])

    peak_layer = int(mean_curve.argmax())
    final = sim[:, -1]
    if peak_layer == n_layers - 1:
        drop_p = np.nan  # peak is the final layer: no drop to test
    else:
        _, drop_p = stats.wilcoxon(final, sim[:, peak_layer], alternative="less")

    n_tests = n_layers - 1
    n_sig = sum(
        min(stats.wilcoxon(sim[:, l], final, alternative="greater")[1] * n_tests, 1.0) < 0.001
        for l in range(n_layers - 1)
    )
    return {
        "n_samples": n_samples,
        "n_layers": n_layers,
        "friedman_p": fried_p,
        "peak_layer": peak_layer,
        "late_drop": mean_curve[peak_layer] - mean_curve[-1],
        "drop_p": drop_p,
        "pct_sig": n_sig / n_tests * 100,
    }


def test_layer_profile_battery(experiments_dir, models):
    """Per-cell statistical battery over every condition: dialects and transformations.

    Mirrors the dialect-cell battery of §5.3 across all conditions: for each
    (model, dataset, condition) cell, a Friedman test that the layer profile is
    not flat, a paired Wilcoxon test that the final layer sits below the peak
    layer (peak taken from the mean curve), and the share of earlier layers
    significantly more similar than the final layer (Bonferroni-corrected
    p < 0.001). Transformations in EXCLUDED_TRANSFORMS are skipped.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.

    Returns:
        DataFrame with one row per (model, dataset, condition) cell.
    """
    rows = []
    for model in models:
        for dataset in layers.DATASETS:
            for dialect in layers.discover_dialects(experiments_dir, model, dataset):
                sim = layers.load_layer_similarity(experiments_dir, model, dataset, dialect)
                if sim is None or sim.shape[1] < 3:
                    continue
                rows.append({
                    "model": model, "dataset": dataset, "condition": dialect,
                    "group": GROUP_LABELS["dialect"], **_cell_battery_row(sim),
                })
            for t in NOISE + TRANSLATE:
                if t in EXCLUDED_TRANSFORMS:
                    continue
                sim = load_transform_similarity(experiments_dir, model, dataset, t)
                if sim is None or sim.shape[1] < 3:
                    continue
                rows.append({
                    "model": model, "dataset": dataset, "condition": t,
                    "group": GROUP_LABELS["noise" if t in NOISE else "translate"], **_cell_battery_row(sim),
                })
    return pd.DataFrame(rows)


def test_dialect_vs_transforms(experiments_dir, models):
    """Paired comparison of final-layer similarity: dialect against each transformation.

    For each (model, dataset) combo, the dialect group's final-layer similarity
    (mean-over-dialects curve) is paired with each transformation's final-layer
    similarity on the same combo. A one-sided Wilcoxon signed-rank test asks
    whether the dialect similarity is higher, Bonferroni-corrected over the
    number of transformations. Transformations in EXCLUDED_TRANSFORMS are skipped.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model name strings.

    Returns:
        DataFrame with one row per transformation.
    """
    transforms = [t for t in NOISE + TRANSLATE if t not in EXCLUDED_TRANSFORMS]
    dialect_final, transform_final = [], {t: [] for t in transforms}
    for model in models:
        for dataset in layers.DATASETS:
            curve = _dialect_group_curve(experiments_dir, model, dataset)
            if curve is None:
                continue
            dialect_final.append(curve[-1])
            for t in transforms:
                sim = load_transform_similarity(experiments_dir, model, dataset, t)
                transform_final[t].append(sim.mean(axis=0)[-1] if sim is not None else np.nan)

    dialect_final = np.array(dialect_final)
    rows = []
    for t in transforms:
        finals = np.array(transform_final[t])
        ok = ~np.isnan(finals)
        _, p = stats.wilcoxon(dialect_final[ok], finals[ok], alternative="greater")
        rows.append({
            "transformation": t,
            "n_combos": int(ok.sum()),
            "mean_dialect_final": dialect_final[ok].mean(),
            "mean_transform_final": finals[ok].mean(),
            "p_raw": p,
            "p_bonferroni": min(p * len(transforms), 1.0),
        })
    return pd.DataFrame(rows)


########
# MAIN #
########

def main() -> None:
    parser = argparse.ArgumentParser()
    project_config = dialecttax.utils.load_config()
    default_dir = project_config["directories"]["experiments"]
    parser.add_argument("--experiments-dir", default=default_dir)
    parser.add_argument("--output-dir", default="analysis/plots/layers/baselines")
    args = parser.parse_args()

    models = layers.discover_models(args.experiments_dir)
    print(f"Found models: {models}")

    ########################################
    # Figures
    ########################################

    print("\nPlotting group comparison (dialect vs noise vs translation)...")
    plot_group_comparison(args.experiments_dir, models, args.output_dir)

    print("Plotting final-layer ranking...")
    plot_final_layer_ranking(args.experiments_dir, models, args.output_dir)

    print("Plotting per-transformation curves...")

    print("Plotting similarity vs accuracy...")
    plot_similarity_vs_accuracy(args.experiments_dir, models, args.output_dir)

    ########################################
    # Tables
    ########################################

    print("\nComputing group summary...")
    group_summary = compute_group_summary(args.experiments_dir, models)
    layers.print_table("Layer Similarity by Transformation Group", group_summary.round(4))

    print("Computing per-transformation summary...")
    transform_summary = compute_transform_summary(args.experiments_dir, models)
    layers.print_table("Final-Layer Similarity by Transformation", transform_summary.round(4))

    print("Testing late-layer drop within each group...")
    drop_df = test_late_drop_by_group(args.experiments_dir, models)
    layers.print_table("Late-Layer Drop: Final < Peak (Wilcoxon)", drop_df.round(4))

    print("Running per-cell battery over (model, dataset, condition)...")
    battery = test_layer_profile_battery(args.experiments_dir, models)
    n = len(battery)
    n_friedman = int((battery["friedman_p"] < 0.001).sum())
    n_drop = int((battery["drop_p"] < 0.001).sum())
    print(f"  Friedman rejects flat profile (p<0.001): {n_friedman}/{n} cells")
    print(f"  Final < peak (Wilcoxon p<0.001): {n_drop}/{n} cells")
    summary = battery.groupby(["group", "condition"]).agg(
        drop_mean=("late_drop", "mean"), drop_min=("late_drop", "min"), drop_max=("late_drop", "max"),
        pct_sig_min=("pct_sig", "min"),
    ).reset_index()
    layers.print_table("Per-Condition Late Drop (per-cell battery)", summary.round(3))

    print("Testing dialect final-layer similarity vs each transformation (paired Wilcoxon)...")
    vs_df = test_dialect_vs_transforms(args.experiments_dir, models)
    layers.print_table("Dialect Final-Layer Similarity > Transformation (Bonferroni)", vs_df.round(6))

    print(f"\nPlots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
