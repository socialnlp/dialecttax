"""
Gradient projection analysis for dialect fairness.

Loads per-document CountSketch-projected gradients from generate_gradients
outputs and computes pairwise cosine similarities between SAE and AAVE
dialect pairs. Produces summary statistics and figures for Section 4.1.

Usage:
    python analysis/gradients.py
    python analysis/gradients.py --experiments-dir /path/to/experiments
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
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 15,
    "legend.title_fontsize": 16,
    "figure.constrained_layout.use": True,
})

_SET2 = sns.color_palette("Set2", 6)
DIALECT_COLORS = {"sae": _SET2[0], "aave": _SET2[1]}
DIALECT_LABELS = {"sae": "SAE", "aave": "AAVE"}

TASKS = ["math", "algorithm", "logic", "planning"]
DIALECTS = ["sae", "aave"]
PERTURBATIONS = ["swap-0.05", "drop-0.15", "insert-0.05", "capitalize-random", "capitalize-alternating"]
PROJECTION_DIM = 8192


###########
# LOADING #
###########

def _load_projections(experiments_dir, model, task, dialect):
    """Load projections and metadata for a single (model, task, dialect).

    Args:
        experiments_dir: Root experiments directory.
        model: Model name (e.g. "gemma_1b_base").
        task: Task name (e.g. "math").
        dialect: Dialect name ("sae" or "aave").

    Returns:
        Tuple of (projections ndarray, metadata list of dicts), or (None, None) if missing.
    """
    # Layout nests a reasoning level between task and dialect; gradients are naive-only.
    base = os.path.join(experiments_dir, "generate_gradients", model, "redial", task, "naive", dialect)
    proj_path = os.path.join(base, f"projections-{PROJECTION_DIM}.npy")
    meta_path = os.path.join(base, "metadata.jsonl")
    if not os.path.exists(proj_path) or not os.path.exists(meta_path):
        return None, None
    projections = np.load(proj_path)
    with open(meta_path) as f:
        metadata = [json.loads(line) for line in f]
    return projections, metadata


def load_all_projections(experiments_dir):
    """Load all gradient projections into a DataFrame.

    Returns:
        DataFrame with columns: model, task, dialect, sample_idx, unique_id,
        loss, grad_norm, n_tokens, and the projection vector stored separately.
        Also returns a dict mapping (model, task, dialect) -> projections ndarray.
    """
    rows = []
    proj_dict = {}
    grad_dir = os.path.join(experiments_dir, "generate_gradients")
    if not os.path.isdir(grad_dir):
        raise FileNotFoundError(f"No generate_gradients directory at {grad_dir}")

    for model in sorted(os.listdir(grad_dir)):
        model_dir = os.path.join(grad_dir, model)
        if not os.path.isdir(model_dir) or model.startswith(".") or model.endswith(".yaml"):
            continue
        for task in TASKS:
            for dialect in DIALECTS:
                projections, metadata = _load_projections(experiments_dir, model, task, dialect)
                if projections is None:
                    continue
                proj_dict[(model, task, dialect)] = projections
                for i, meta in enumerate(metadata):
                    rows.append({
                        "model": model,
                        "task": task,
                        "dialect": dialect,
                        "sample_idx": i,
                        **meta,
                    })

    df = pd.DataFrame(rows)
    df["family"] = df["model"].apply(lambda m: m.rsplit("_", 1)[0])
    print(f"Loaded {len(df)} gradient projections: "
          f"{df['model'].nunique()} models, "
          f"{df['task'].nunique()} tasks, "
          f"{df['dialect'].nunique()} dialects")
    return df, proj_dict


################
# PERTURBATIONS #
################

def _load_perturbed_projections(experiments_dir, model, task, perturbation):
    """Load perturbed SAE projections for a single (model, task, perturbation).

    Args:
        experiments_dir: Root experiments directory.
        model: Model name.
        task: Task name.
        perturbation: Perturbation name (e.g. "swap-0.05").

    Returns:
        Projections ndarray, or None if missing.
    """
    base = os.path.join(
        experiments_dir, "generate_gradients", model, "redial", task, "sae", "perturbed", perturbation,
    )
    proj_path = os.path.join(base, f"projections-{PROJECTION_DIM}.npy")
    if not os.path.exists(proj_path):
        return None
    return np.load(proj_path)


def load_all_perturbed_projections(experiments_dir):
    """Load perturbed SAE gradient projections for all models/tasks/perturbations.

    Returns:
        Dict mapping (model, task, perturbation) -> projections ndarray.
    """
    perturbed_dict = {}
    grad_dir = os.path.join(experiments_dir, "generate_gradients")
    if not os.path.isdir(grad_dir):
        return perturbed_dict

    for model in sorted(os.listdir(grad_dir)):
        model_dir = os.path.join(grad_dir, model)
        if not os.path.isdir(model_dir) or model.startswith(".") or model.endswith(".yaml"):
            continue
        for task in TASKS:
            for perturbation in PERTURBATIONS:
                proj = _load_perturbed_projections(experiments_dir, model, task, perturbation)
                if proj is not None:
                    perturbed_dict[(model, task, perturbation)] = proj

    print(f"Loaded {len(perturbed_dict)} perturbed projection sets")
    return perturbed_dict


def compute_perturbation_similarities(proj_dict, perturbed_dict):
    """Compute per-sample cosine similarity between SAE and perturbed SAE gradients.

    Args:
        proj_dict: Dict mapping (model, task, dialect) -> projections ndarray.
        perturbed_dict: Dict mapping (model, task, perturbation) -> projections ndarray.

    Returns:
        DataFrame with columns: model, task, perturbation, sample_idx, sim_perturbed.
    """
    rows = []
    for (model, task, perturbation), proj_perturbed in perturbed_dict.items():
        key_sae = (model, task, "sae")
        if key_sae not in proj_dict:
            continue
        proj_sae = proj_dict[key_sae]
        n = min(proj_sae.shape[0], proj_perturbed.shape[0])
        sims = _cosine_sim_paired(proj_sae[:n], proj_perturbed[:n])
        for i, s in enumerate(sims):
            rows.append({
                "model": model,
                "task": task,
                "perturbation": perturbation,
                "sample_idx": i,
                "sim_perturbed": s,
            })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["family"] = df["model"].apply(lambda m: m.rsplit("_", 1)[0])
    return df


def compute_perturbation_z_scores(perturbation_df, baseline_df):
    """Compute z-scores for perturbation similarities against unrelated baseline.

    Args:
        perturbation_df: DataFrame from compute_perturbation_similarities.
        baseline_df: DataFrame from compute_unrelated_baseline.

    Returns:
        DataFrame with columns: model, task, perturbation, mu_perturbed,
        mu_unrelated, sigma_unrelated, z_score.
    """
    rows = []
    for (model, task, perturbation), grp in perturbation_df.groupby(["model", "task", "perturbation"]):
        mu_perturbed = grp["sim_perturbed"].mean()
        bl = baseline_df[(baseline_df["model"] == model) & (baseline_df["task"] == task)]
        if len(bl) == 0:
            continue
        mu_unrelated = bl["sim_unrelated"].mean()
        sigma_unrelated = bl["sim_unrelated"].std()
        z = (mu_perturbed - mu_unrelated) / max(sigma_unrelated, 1e-8)
        rows.append({
            "model": model,
            "task": task,
            "perturbation": perturbation,
            "mu_perturbed": mu_perturbed,
            "mu_unrelated": mu_unrelated,
            "sigma_unrelated": sigma_unrelated,
            "z_score": z,
        })
    df = pd.DataFrame(rows)
    if len(df) > 0:
        df["family"] = df["model"].apply(lambda m: m.rsplit("_", 1)[0])
    return df


def print_perturbation_summary(perturbation_df, perturbation_z_df, paired_df):
    """Print summary comparing perturbation vs dialect gradient divergence."""
    print("\n=== Perturbation Baseline: sim(SAE, perturbed_SAE) ===")

    # Mean similarity per perturbation
    print("\nMean cosine similarity by perturbation (across all models/tasks):")
    for perturb, grp in perturbation_df.groupby("perturbation"):
        print(f"  {perturb:25s}: mean={grp['sim_perturbed'].mean():.4f}, "
              f"std={grp['sim_perturbed'].std():.4f}, n={len(grp)}")

    # Compare to dialect paired similarity
    mu_dialect = paired_df["sim_paired"].mean()
    mu_perturbed_all = perturbation_df["sim_perturbed"].mean()
    print(f"\n  sim(SAE, AAVE) mean:          {mu_dialect:.4f}")
    print(f"  sim(SAE, perturbed_SAE) mean: {mu_perturbed_all:.4f}")

    # Z-scores
    print("\nMean z-score by perturbation (positive = more similar than unrelated):")
    for perturb, grp in perturbation_z_df.groupby("perturbation"):
        print(f"  {perturb:25s}: z={grp['z_score'].mean():+.2f}")

    # Mann-Whitney U: are perturbation similarities significantly different from dialect?
    print("\nMann-Whitney U: sim(SAE, perturbed_SAE) vs sim(SAE, AAVE):")
    u, p = stats.mannwhitneyu(
        perturbation_df["sim_perturbed"], paired_df["sim_paired"], alternative="two-sided",
    )
    print(f"  U={u:.0f}, p={p:.3e}")
    direction = "perturbed > dialect" if perturbation_df["sim_perturbed"].mean() > mu_dialect else "dialect > perturbed"
    print(f"  Direction: {direction}")


######################
# COSINE SIMILARITY  #
######################

def _cosine_sim_paired(proj_a, proj_b):
    """Compute per-sample cosine similarity between paired projection matrices.

    Args:
        proj_a: ndarray of shape (n, d).
        proj_b: ndarray of shape (n, d).

    Returns:
        1-D array of cosine similarities, length n.
    """
    dot = np.sum(proj_a * proj_b, axis=1)
    norm_a = np.linalg.norm(proj_a, axis=1)
    norm_b = np.linalg.norm(proj_b, axis=1)
    return dot / np.maximum(norm_a * norm_b, 1e-8)


def _cosine_sim_unrelated(proj, n_pairs=5000, rng=None):
    """Compute cosine similarity between random unrelated pairs.

    Args:
        proj: ndarray of shape (n, d).
        n_pairs: Number of random pairs to sample.
        rng: numpy random Generator.

    Returns:
        1-D array of cosine similarities.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = proj.shape[0]
    idx_a = rng.integers(0, n, size=n_pairs)
    idx_b = rng.integers(0, n, size=n_pairs)
    # Resample until no self-pairs
    mask = idx_a == idx_b
    while mask.any():
        idx_b[mask] = rng.integers(0, n, size=mask.sum())
        mask = idx_a == idx_b
    return _cosine_sim_paired(proj[idx_a], proj[idx_b])


def compute_paired_similarities(proj_dict):
    """Compute paired cosine similarities between SAE and AAVE for each (model, task).

    Args:
        proj_dict: Dict mapping (model, task, dialect) -> projections ndarray.

    Returns:
        DataFrame with columns: model, task, sample_idx, sim_paired.
    """
    rows = []
    for (model, task, dialect), _ in proj_dict.items():
        if dialect != "sae":
            continue
        key_aave = (model, task, "aave")
        if key_aave not in proj_dict:
            continue
        proj_sae = proj_dict[(model, task, "sae")]
        proj_aave = proj_dict[key_aave]
        n = min(proj_sae.shape[0], proj_aave.shape[0])
        sims = _cosine_sim_paired(proj_sae[:n], proj_aave[:n])
        for i, s in enumerate(sims):
            rows.append({"model": model, "task": task, "sample_idx": i, "sim_paired": s})
    df = pd.DataFrame(rows)
    df["family"] = df["model"].apply(lambda m: m.rsplit("_", 1)[0])
    return df


def compute_unrelated_baseline(proj_dict, n_pairs=5000):
    """Compute unrelated-pair cosine similarity baseline per (model, task).

    Args:
        proj_dict: Dict mapping (model, task, dialect) -> projections ndarray.
        n_pairs: Number of random pairs per (model, task).

    Returns:
        DataFrame with columns: model, task, sim_unrelated.
    """
    rng = np.random.default_rng(42)
    rows = []
    for (model, task, dialect), proj in proj_dict.items():
        if dialect != "sae":
            continue
        sims = _cosine_sim_unrelated(proj, n_pairs=n_pairs, rng=rng)
        for s in sims:
            rows.append({"model": model, "task": task, "sim_unrelated": s})
    return pd.DataFrame(rows)


def compute_z_scores(paired_df, baseline_df):
    """Compute z-scores: z = (mu_paired - mu_unrelated) / sigma_unrelated.

    Args:
        paired_df: DataFrame from compute_paired_similarities.
        baseline_df: DataFrame from compute_unrelated_baseline.

    Returns:
        DataFrame with columns: model, task, mu_paired, mu_unrelated, sigma_unrelated, z_score.
    """
    rows = []
    for (model, task), grp in paired_df.groupby(["model", "task"]):
        mu_paired = grp["sim_paired"].mean()
        bl = baseline_df[(baseline_df["model"] == model) & (baseline_df["task"] == task)]
        mu_unrelated = bl["sim_unrelated"].mean()
        sigma_unrelated = bl["sim_unrelated"].std()
        z = (mu_paired - mu_unrelated) / max(sigma_unrelated, 1e-8)
        rows.append({
            "model": model,
            "task": task,
            "mu_paired": mu_paired,
            "mu_unrelated": mu_unrelated,
            "sigma_unrelated": sigma_unrelated,
            "z_score": z,
        })
    df = pd.DataFrame(rows)
    df["family"] = df["model"].apply(lambda m: m.rsplit("_", 1)[0])
    return df


#####################
# SUMMARY STATISTICS #
#####################

def print_summary(paired_df, z_df):
    """Print summary tables of paired similarity and z-scores."""
    print("\n=== Paired Cosine Similarity (SAE <-> AAVE) ===")
    summary = paired_df.groupby(["model", "task"])["sim_paired"].agg(["mean", "std", "count"])
    print(summary.to_string())

    print("\n=== Z-Scores (paired vs unrelated) ===")
    pivot = z_df.pivot_table(index="model", columns="task", values="z_score")
    print(pivot.to_string(float_format="%.2f"))

    print("\n=== Aggregate by model ===")
    agg = z_df.groupby("model")["z_score"].agg(["mean", "std"])
    print(agg.to_string(float_format="%.2f"))


##########
# PLOTS  #
##########

MODEL_ORDER = [
    "llama_1b_base", "gemma_1b_base", "llama_3b_base",
    "llama_8b_base", "qwen_1.7b_base", "gemma_4b_base",
    "qwen_4b_base", "qwen_8b_base", "gemma_12b_base",
]

MODEL_LABELS = {
    "llama_1b_base": "L-1B", "llama_3b_base": "L-3B", "llama_8b_base": "L-8B",
    "gemma_1b_base": "G-1B", "gemma_4b_base": "G-4B", "gemma_12b_base": "G-12B",
    "qwen_1.7b_base": "Q-1.7B", "qwen_4b_base": "Q-4B", "qwen_8b_base": "Q-8B",
}

FAMILY_COLORS = {"llama": _SET2[0], "gemma": _SET2[1], "qwen": _SET2[2]}
TASK_COLORS = {task: _SET2[i] for i, task in enumerate(TASKS)}


def plot_paired_vs_unrelated(paired_df, baseline_df, output_dir):
    """Split violin plot comparing paired (SAE<->AAVE) vs unrelated cosine similarities."""
    FAMILY_HATCHES = {"llama": "//", "gemma": "..", "qwen": ""}

    # Order models by increasing minimum paired cosine similarity
    min_paired = paired_df.groupby("model")["sim_paired"].min().sort_values()
    models = [m for m in min_paired.index if m in MODEL_ORDER]
    model_label_order = [MODEL_LABELS.get(m, m) for m in models]

    # Build long-form DataFrame
    rows = []
    for model in models:
        label = MODEL_LABELS.get(model, model)
        family = model.rsplit("_", 2)[0]
        for v in paired_df[paired_df["model"] == model]["sim_paired"].values:
            rows.append({"Model": label, "Type": "SAE \u2194 AAVE (paired)",
                         "Cosine similarity": v, "family": family})
        for v in baseline_df[baseline_df["model"] == model]["sim_unrelated"].values:
            rows.append({"Model": label, "Type": "Unrelated",
                         "Cosine similarity": v, "family": family})
    plot_df = pd.DataFrame(rows)
    plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=model_label_order, ordered=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.violinplot(
        data=plot_df, x="Model", y="Cosine similarity", hue="Type",
        palette={"SAE \u2194 AAVE (paired)": _SET2[1], "Unrelated": "#B0B0B0"},
        split=True, inner="quart", cut=0, ax=ax,
    )

    # Apply family hatching to violin bodies
    # Bodies alternate: body 2*i = model i paired, body 2*i+1 = model i unrelated
    violin_bodies = [c for c in ax.collections if hasattr(c, "get_paths")]
    for idx, body in enumerate(violin_bodies):
        model_idx = idx // 2
        family = models[model_idx].rsplit("_", 2)[0]
        body.set_hatch(FAMILY_HATCHES[family])
        body.set_edgecolor("black")
        body.set_linewidth(0.6)

    for line in ax.lines:
        line.set_color("black")

    ax.set_xticklabels(model_label_order, fontsize=16)
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Model size")
    ax.set_ylim(top=1.15)

    # Two separate legends
    from matplotlib.patches import Patch
    type_handles = [
        Patch(facecolor=_SET2[1], edgecolor="black", linewidth=0.6, label="Paired"),
        Patch(facecolor="#B0B0B0", edgecolor="black", linewidth=0.6, label="Unrelated"),
    ]
    family_handles = [
        Patch(facecolor="#B0B0B0", hatch="//", edgecolor="black", linewidth=0.6, label="Llama"),
        Patch(facecolor="#B0B0B0", hatch="..", edgecolor="black", linewidth=0.6, label="Gemma"),
        Patch(facecolor="#B0B0B0", hatch="",   edgecolor="black", linewidth=0.6, label="Qwen"),
    ]
    leg1 = ax.legend(handles=type_handles, loc="upper right",
                     ncols=2, fontsize=12)
    leg2 = ax.legend(handles=family_handles, loc="lower right",
                     ncols=3, fontsize=12)
    ax.add_artist(leg1)
    sns.despine(ax=ax)

    fig.savefig(os.path.join(output_dir, "gradient_paired_vs_unrelated.pdf"))
    fig.savefig(os.path.join(output_dir, "gradient_paired_vs_unrelated.png"))
    plt.close(fig)
    print("  Saved gradient_paired_vs_unrelated.pdf")


def plot_z_scores_by_model(z_df, output_dir):
    """Grouped bar chart of z-scores per model, colored by task, hatched by family."""
    FAMILY_HATCHES = {"llama": "//", "gemma": "..", "qwen": ""}
    FULL_LABELS = {
        "llama_1b_base": "Llama 1B", "llama_3b_base": "Llama 3B", "llama_8b_base": "Llama 8B",
        "gemma_1b_base": "Gemma 1B", "gemma_4b_base": "Gemma 4B", "gemma_12b_base": "Gemma 12B",
        "qwen_1.7b_base": "Qwen 1.7B", "qwen_4b_base": "Qwen 4B", "qwen_8b_base": "Qwen 8B",
    }
    Z_MODEL_ORDER = [
        "gemma_1b_base", "gemma_4b_base", "gemma_12b_base",
        "llama_1b_base", "llama_3b_base", "llama_8b_base",
        "qwen_1.7b_base", "qwen_4b_base", "qwen_8b_base",
    ]
    models = [m for m in Z_MODEL_ORDER if m in z_df["model"].unique()]
    model_label_order = [FULL_LABELS[m] for m in models]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(models))
    n_tasks = len(TASKS)
    width = 0.8 / n_tasks

    for j, task in enumerate(TASKS):
        vals = []
        hatches = []
        for model in models:
            row = z_df[(z_df["model"] == model) & (z_df["task"] == task)]
            vals.append(row["z_score"].values[0] if len(row) > 0 else 0)
            family = model.rsplit("_", 2)[0]
            hatches.append(FAMILY_HATCHES[family])
        bars = ax.bar(
            x + j * width, vals, width,
            label=task.capitalize(), color=_SET2[j],
            edgecolor="white", linewidth=0.5,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")

    ax.set_xticks(x + width * (n_tasks - 1) / 2)
    ax.set_xticklabels(model_label_order, fontsize=15.5)
    ax.set_ylabel("z-score")
    ax.set_xlabel("")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    # Two separate legends: task and family
    from matplotlib.patches import Patch
    task_handles = [Patch(facecolor=_SET2[j], label=task.capitalize()) for j, task in enumerate(TASKS)]
    family_handles = [
        Patch(facecolor="#CCCCCC", hatch="//", edgecolor="white", label="Llama"),
        Patch(facecolor="#CCCCCC", hatch="..", edgecolor="white", label="Gemma"),
        Patch(facecolor="#CCCCCC", hatch="", edgecolor="white", label="Qwen"),
    ]
    leg1 = ax.legend(handles=task_handles, title="Task", loc="lower left",
                     ncols=1, fontsize=15, title_fontsize=16)
    leg2 = ax.legend(handles=family_handles, title="Family", loc="lower right",
                     ncols=1, fontsize=15, title_fontsize=16)
    ax.add_artist(leg1)
    sns.despine(ax=ax)

    fig.savefig(os.path.join(output_dir, "gradient_z_scores.pdf"))
    fig.savefig(os.path.join(output_dir, "gradient_z_scores.png"))
    plt.close(fig)
    print("  Saved gradient_z_scores.pdf")


def plot_loss_comparison(meta_df, output_dir):
    """Grouped bar chart of mean cross-entropy loss, hatching by family, color by dialect."""
    models = [m for m in MODEL_ORDER if m in meta_df["model"].unique()]
    FAMILY_HATCHES = {"llama": "//", "gemma": "..", "qwen": ""}

    # Aggregate per (model, dialect)
    agg = meta_df[meta_df["model"].isin(models)].groupby(["model", "dialect"])["loss"].agg(["mean", "std"]).reset_index()
    agg["family"] = agg["model"].apply(lambda m: m.rsplit("_", 2)[0])

    model_label_order = [MODEL_LABELS[m] for m in models]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for j, (dialect_key, dialect_label) in enumerate(DIALECT_LABELS.items()):
        means = []
        stds = []
        hatches = []
        for model in models:
            row = agg[(agg["model"] == model) & (agg["dialect"] == dialect_key)].iloc[0]
            means.append(row["mean"])
            stds.append(row["std"])
            hatches.append(FAMILY_HATCHES[row["family"]])
        bars = ax.bar(
            x + j * width, means, width, yerr=stds,
            label=dialect_label, color=DIALECT_COLORS[dialect_key],
            edgecolor="black", linewidth=0.6,
            capsize=3, error_kw={"linewidth": 1.2},
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            bar.set_edgecolor("black")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(model_label_order, rotation=0, ha="center", fontsize=16)
    ax.set_ylabel("Mean cross-entropy loss")
    ax.set_xlabel("Model size")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    # Two separate legends
    from matplotlib.patches import Patch
    dialect_handles = [
        Patch(facecolor=DIALECT_COLORS["sae"], edgecolor="black", linewidth=0.6, label="SAE"),
        Patch(facecolor=DIALECT_COLORS["aave"], edgecolor="black", linewidth=0.6, label="AAVE"),
    ]
    family_handles = [
        Patch(facecolor="#B0B0B0", hatch="//", edgecolor="black", linewidth=0.6, label="Llama"),
        Patch(facecolor="#B0B0B0", hatch="..", edgecolor="black", linewidth=0.6, label="Gemma"),
        Patch(facecolor="#B0B0B0", hatch="",   edgecolor="black", linewidth=0.6, label="Qwen"),
    ]
    leg1 = ax.legend(handles=family_handles, loc="upper right",
                     ncols=3, fontsize=14)
    leg2 = ax.legend(handles=dialect_handles, loc="upper left",
                     bbox_to_anchor=(0.02, 1.0), ncols=2, fontsize=14)
    ax.add_artist(leg1)
    sns.despine(ax=ax)

    fig.savefig(os.path.join(output_dir, "gradients_cross_entropy_loss.pdf"))
    fig.savefig(os.path.join(output_dir, "gradients_cross_entropy_loss.png"))
    plt.close(fig)
    print("  Saved gradients_cross_entropy_loss.pdf")


############################
# PERTURBATION COMPARISON  #
############################

PERTURBATION_LABELS = {
    "swap-0.05": "Swap",
    "drop-0.15": "Drop",
    "insert-0.05": "Insert",
    "capitalize-random": "Cap (rand)",
    "capitalize-alternating": "Cap (alt)",
}


def plot_perturbation_by_type(perturbation_df, paired_df, output_dir):
    """Box plot of per-perturbation gradient similarity vs dialect similarity.

    One box per perturbation type (aggregated over models/tasks), plus
    the dialect paired distribution for comparison.
    """
    rows = []
    for v in paired_df["sim_paired"].values:
        rows.append({"Type": "AAVE (dialect)", "Cosine similarity": v})
    for perturb, grp in perturbation_df.groupby("perturbation"):
        label = PERTURBATION_LABELS.get(perturb, perturb)
        for v in grp["sim_perturbed"].values:
            rows.append({"Type": label, "Cosine similarity": v})
    plot_df = pd.DataFrame(rows)

    type_order = ["AAVE (dialect)"] + [PERTURBATION_LABELS.get(p, p) for p in PERTURBATIONS]
    plot_df["Type"] = pd.Categorical(plot_df["Type"], categories=type_order, ordered=True)

    colors = [_SET2[1]] + [_SET2[2]] * len(PERTURBATIONS)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=plot_df, x="Type", y="Cosine similarity",
        palette=colors, ax=ax, fliersize=2,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cosine similarity")
    ax.axhline(y=paired_df["sim_paired"].mean(), color=_SET2[1], linestyle="--",
               linewidth=1, alpha=0.7, label="Dialect mean")
    sns.despine(ax=ax)

    fig.savefig(os.path.join(output_dir, "gradient_perturbation_by_type.pdf"))
    fig.savefig(os.path.join(output_dir, "gradient_perturbation_by_type.png"))
    plt.close(fig)
    print("  Saved gradient_perturbation_by_type.pdf")


#################
# LOSS GAP TEST #
#################

def test_loss_gap(meta_df, dialect_high="aave", dialect_low="sae", family_alpha=0.05):
    """Paired Wilcoxon test that loss is higher for one dialect than the other.

    Pairs per-sample losses by (model, task, sample_idx) across the two
    dialects and runs a one-sided Wilcoxon signed-rank test per model
    (H1: loss[dialect_high] > loss[dialect_low]). Bonferroni-corrects
    across models. Also reports Cohen's d_z on the paired delta.

    Args:
        meta_df: Output of load_all_projections.
        dialect_high: Dialect hypothesized to have higher loss.
        dialect_low: Reference dialect.
        family_alpha: Family-wise error rate for Bonferroni correction.

    Returns:
        DataFrame indexed by model with columns n, mean_delta, median_delta,
        d_z, wilcoxon_stat, p_raw, p_bonferroni, reject.
    """
    piv = meta_df.pivot_table(
        index=["model", "task", "sample_idx"],
        columns="dialect",
        values="loss",
        aggfunc="first",
    ).dropna(subset=[dialect_high, dialect_low])
    piv["delta"] = piv[dialect_high] - piv[dialect_low]

    rows = []
    for model, g in piv.groupby("model"):
        stat, p = stats.wilcoxon(g[dialect_high], g[dialect_low], alternative="greater")
        delta = g["delta"]
        rows.append({
            "model": model,
            "n": len(g),
            "mean_delta": float(delta.mean()),
            "median_delta": float(delta.median()),
            "d_z": float(delta.mean() / delta.std(ddof=1)),
            "wilcoxon_stat": float(stat),
            "p_raw": float(p),
        })
    out = pd.DataFrame(rows).set_index("model")
    out["p_bonferroni"] = np.minimum(1.0, out["p_raw"] * len(out))
    out["reject"] = out["p_bonferroni"] < family_alpha
    return out


def print_loss_gap_summary(loss_gap_df, dialect_high="AAVE", dialect_low="SAE"):
    """Print per-model paired loss-gap test with Bonferroni-corrected p-values."""
    print(f"\n=== Paired Wilcoxon: loss[{dialect_high}] > loss[{dialect_low}] ===")
    print(f"  (one-sided signed-rank, Bonferroni over {len(loss_gap_df)} models)\n")
    for model, row in loss_gap_df.iterrows():
        tag = "**" if row["reject"] else "  "
        print(f"  {tag} {model:18s} n={int(row['n']):4d}  "
              f"mean_delta={row['mean_delta']:+.4f} nats  "
              f"d_z={row['d_z']:.2f}  "
              f"p_raw={row['p_raw']:.3e}  "
              f"p_bonf={row['p_bonferroni']:.3e}")
    print(f"\n  Overall: mean_delta={loss_gap_df['mean_delta'].mean():+.4f} nats "
          f"(range {loss_gap_df['mean_delta'].min():+.4f} to "
          f"{loss_gap_df['mean_delta'].max():+.4f}),  "
          f"d_z in [{loss_gap_df['d_z'].min():.2f}, {loss_gap_df['d_z'].max():.2f}]")


def summarize_loss_by_dialect(meta_df):
    """Per-(model, dialect) mean/std cross-entropy loss — the bar-chart data.

    Returns a DataFrame indexed by model with columns
    [n, sae_mean, sae_std, aave_mean, aave_std, gap].
    """
    models = [m for m in MODEL_ORDER if m in meta_df["model"].unique()]
    sub = meta_df[meta_df["model"].isin(models)]
    agg = sub.groupby(["model", "dialect"])["loss"].agg(["mean", "std", "count"]).reset_index()
    piv_mean = agg.pivot(index="model", columns="dialect", values="mean")
    piv_std = agg.pivot(index="model", columns="dialect", values="std")
    piv_n = agg.pivot(index="model", columns="dialect", values="count")
    out = pd.DataFrame({
        "n": piv_n["sae"].astype(int),
        "sae_mean": piv_mean["sae"], "sae_std": piv_std["sae"],
        "aave_mean": piv_mean["aave"], "aave_std": piv_std["aave"],
    })
    out["gap"] = out["aave_mean"] - out["sae_mean"]
    return out.loc[models]


def print_loss_by_dialect_summary(loss_summary_df):
    """Print per-(model, dialect) mean/std cross-entropy loss — bar-chart data."""
    print("\n=== Mean cross-entropy loss by (model, dialect) ===")
    print(f"  (n per cell shown; underlies figure:gradient_analysis-cross_entropy_loss)\n")
    for model, row in loss_summary_df.iterrows():
        print(f"  {model:18s} n={int(row['n']):4d}  "
              f"SAE={row['sae_mean']:.4f} ({row['sae_std']:.4f})  "
              f"AAVE={row['aave_mean']:.4f} ({row['aave_std']:.4f})  "
              f"gap={row['gap']:+.4f}")


########################
# CORRECTNESS ANALYSIS #
########################

def load_correctness(experiments_dir, models, tasks, dialects):
    """Load correctness labels from generate_logits (naive reasoning) metadata.

    Args:
        experiments_dir: Root experiments directory.
        models: List of model names.
        tasks: List of task names.
        dialects: List of dialect names.

    Returns:
        Dict mapping (model, task, dialect, unique_id) -> bool.
    """
    correctness = {}
    for model in models:
        for task in tasks:
            for dialect in dialects:
                path = os.path.join(
                    experiments_dir, "generate_logits", model,
                    "redial", task, "naive", dialect, "metadata.jsonl",
                )
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    for line in f:
                        row = json.loads(line)
                        correctness[(model, task, dialect, row["unique_id"])] = row["correct"]
    return correctness


def compute_similarity_correctness(proj_dict, experiments_dir):
    """Join paired gradient similarity with correctness from both dialects.

    For each (model, task, sample), computes:
    - sim_paired: cosine similarity between SAE and AAVE gradient projections
    - correct_sae: whether the model answered correctly on SAE
    - correct_aave: whether the model answered correctly on AAVE
    - correct_both: correct on both dialects

    Args:
        proj_dict: Dict mapping (model, task, dialect) -> projections ndarray.
        experiments_dir: Root experiments directory.

    Returns:
        DataFrame with columns: model, task, sample_idx, sim_paired,
        correct_sae, correct_aave, correct_both.
    """
    models = sorted({m for m, t, d in proj_dict.keys()})
    correctness = load_correctness(experiments_dir, models, TASKS, DIALECTS)

    rows = []
    for (model, task, dialect), _ in proj_dict.items():
        if dialect != "sae":
            continue
        key_aave = (model, task, "aave")
        if key_aave not in proj_dict:
            continue

        proj_sae = proj_dict[(model, task, "sae")]
        proj_aave = proj_dict[key_aave]
        n = min(proj_sae.shape[0], proj_aave.shape[0])
        sims = _cosine_sim_paired(proj_sae[:n], proj_aave[:n])

        # Load unique_ids from gradient metadata to join (naive reasoning level)
        grad_meta_path = os.path.join(
            experiments_dir, "generate_gradients", model,
            "redial", task, "naive", "sae", "metadata.jsonl",
        )
        with open(grad_meta_path) as f:
            grad_meta = [json.loads(line) for line in f]

        aave_meta_path = os.path.join(
            experiments_dir, "generate_gradients", model,
            "redial", task, "naive", "aave", "metadata.jsonl",
        )
        with open(aave_meta_path) as f:
            aave_meta = [json.loads(line) for line in f]

        for i in range(n):
            uid_sae = grad_meta[i]["unique_id"]
            uid_aave = aave_meta[i]["unique_id"]
            c_sae = correctness.get((model, task, "sae", uid_sae))
            c_aave = correctness.get((model, task, "aave", uid_aave))
            if c_sae is None or c_aave is None:
                continue
            rows.append({
                "model": model,
                "task": task,
                "sample_idx": i,
                "sim_paired": sims[i],
                "correct_sae": c_sae,
                "correct_aave": c_aave,
                "correct_both": c_sae and c_aave,
            })

    df = pd.DataFrame(rows)
    df["family"] = df["model"].apply(lambda m: m.rsplit("_", 1)[0])
    return df


def plot_similarity_vs_correctness(sim_corr_df, output_dir):
    """Split violin: gradient similarity grouped by whether both dialects are correct."""
    FAMILY_HATCHES = {"llama": "//", "gemma": "..", "qwen": ""}

    # Order by increasing mean similarity (same as paired_vs_unrelated)
    mean_sim = sim_corr_df.groupby("model")["sim_paired"].min().sort_values()
    models = list(mean_sim.index)
    model_label_order = [MODEL_LABELS.get(m, m) for m in models]

    plot_df = sim_corr_df[sim_corr_df["model"].isin(models)].copy()
    plot_df["Model"] = plot_df["model"].map(MODEL_LABELS)
    plot_df["Model"] = pd.Categorical(plot_df["Model"], categories=model_label_order, ordered=True)
    plot_df["Correct (both)"] = plot_df["correct_both"].map({True: "Both correct", False: "At least one wrong"})

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.violinplot(
        data=plot_df, x="Model", y="sim_paired", hue="Correct (both)",
        palette={"Both correct": _SET2[0], "At least one wrong": _SET2[3]},
        split=True, inner="quart", cut=0, ax=ax,
    )

    # Apply family hatching
    violin_bodies = [c for c in ax.collections if hasattr(c, "get_paths")]
    for idx, body in enumerate(violin_bodies):
        model_idx = idx // 2
        if model_idx < len(models):
            family = models[model_idx].rsplit("_", 2)[0]
            body.set_hatch(FAMILY_HATCHES[family])
            body.set_edgecolor("white")
            body.set_linewidth(0.5)

    # Color inner lines
    n = len(models)
    for idx, line in enumerate(ax.lines):
        line_within = idx % 6
        if line_within < 3:
            line.set_color("black")
        else:
            line.set_color(_SET2[3])

    ax.set_xticklabels(model_label_order, fontsize=16)
    ax.set_ylabel("Cosine similarity")
    ax.set_xlabel("Model size")
    sns.despine(ax=ax)

    fig.savefig(os.path.join(output_dir, "gradient_similarity_vs_correctness.pdf"))
    fig.savefig(os.path.join(output_dir, "gradient_similarity_vs_correctness.png"))
    plt.close(fig)
    print("  Saved gradient_similarity_vs_correctness.pdf")


def print_correctness_summary(sim_corr_df):
    """Print correlation between gradient similarity and correctness."""
    print("\n=== Gradient Similarity vs Correctness ===")

    # Per-model point-biserial correlation
    print("\nPoint-biserial r (sim_paired vs correct_both):")
    for model in sorted(sim_corr_df["model"].unique()):
        sub = sim_corr_df[sim_corr_df["model"] == model]
        if sub["correct_both"].nunique() < 2:
            print(f"  {model}: insufficient variance")
            continue
        r, p = stats.pointbiserialr(sub["correct_both"].astype(int), sub["sim_paired"])
        print(f"  {model}: r={r:.3f}, p={p:.3e}, n={len(sub)}")

    # Aggregate
    r, p = stats.pointbiserialr(
        sim_corr_df["correct_both"].astype(int), sim_corr_df["sim_paired"],
    )
    print(f"  OVERALL: r={r:.3f}, p={p:.3e}, n={len(sim_corr_df)}")

    # Mean similarity by correctness outcome
    print("\nMean similarity by outcome:")
    for outcome, label in [(True, "Both correct"), (False, "At least one wrong")]:
        sub = sim_corr_df[sim_corr_df["correct_both"] == outcome]
        print(f"  {label}: mean={sub['sim_paired'].mean():.4f}, n={len(sub)}")


################################
# ACCURACY × GRADIENTS BRIDGE  #
################################
# Caveat: gradient projections exist for *base* models, perturbation accuracy
# is computed on *instruct* models. We bridge the two by matching on
# (family, size). The interpretation is: does a perturbation that pushes
# the base-model gradient further from SAE also cause the corresponding
# instruct model to lose more accuracy?

def _family_size(model):
    """'llama_8b_instruct' -> ('llama', '8b').  'qwen_1.7b_base' -> ('qwen', '1.7b')."""
    parts = model.split("_")
    return parts[0], parts[1]


def _read_accuracy(path):
    """Read a metadata.jsonl. Returns (accuracy %, n) or (None, n) if corrupted."""
    if not os.path.exists(path):
        return None, 0
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return None, 0
    bad = sum(1 for r in rows
              if r.get("completion", "") == ""
              or str(r.get("input_mean_log_prob", "")) == "nan")
    if bad > len(rows) * 0.5:
        return None, len(rows)
    n_correct = sum(1 for r in rows if r.get("correct", False))
    return n_correct / len(rows) * 100, len(rows)


def load_perturbation_accuracy(experiments_dir, reasoning="naive"):
    """Per-(model, task, condition) accuracy from generate_logits.

    Conditions: 'SAE', 'AAVE', and 'SAE+{perturbation}'.
    """
    rows = []
    logits_dir = os.path.join(experiments_dir, "generate_logits")
    if not os.path.isdir(logits_dir):
        return pd.DataFrame()
    for model in sorted(os.listdir(logits_dir)):
        mdir = os.path.join(logits_dir, model)
        if not os.path.isdir(mdir) or model.startswith(".") or model.endswith(".yaml"):
            continue
        for task in TASKS:
            for dialect in DIALECTS:
                path = os.path.join(mdir, "redial", task, reasoning, dialect, "metadata.jsonl")
                acc, n = _read_accuracy(path)
                if acc is not None:
                    rows.append({
                        "model": model, "task": task,
                        "condition": "SAE" if dialect == "sae" else "AAVE",
                        "accuracy": acc, "n": n,
                    })
            for pert in PERTURBATIONS:
                path = os.path.join(
                    mdir, "redial", task, reasoning, "sae", "perturbed", pert, "metadata.jsonl",
                )
                acc, n = _read_accuracy(path)
                if acc is not None:
                    rows.append({
                        "model": model, "task": task,
                        "condition": f"SAE+{pert}", "accuracy": acc, "n": n,
                    })
    df = pd.DataFrame(rows)
    if len(df):
        df["family"], df["size"] = zip(*df["model"].map(_family_size))
        df["variant"] = df["model"].apply(lambda m: "instruct" if "_instruct" in m else "base")
    return df


def correlate_accuracy_vs_gradient(acc_df, paired_df, perturbation_df):
    """Match instruct accuracy gap to base gradient similarity by (family, size, task).

    Returns:
        DataFrame with columns: family, size, task, condition, accuracy_gap, grad_sim.
        accuracy_gap is (cond_acc - SAE_acc) in percentage points (instruct).
        grad_sim is mean cosine similarity between SAE and {AAVE | perturbed} on
        the matching base model.
    """
    if len(acc_df) == 0:
        return pd.DataFrame()

    instruct = acc_df[acc_df["variant"] == "instruct"]
    sae = (instruct[instruct["condition"] == "SAE"]
           .set_index(["family", "size", "task"])["accuracy"])

    paired = paired_df.copy()
    paired["family"], paired["size"] = zip(*paired["model"].map(_family_size))
    grad_aave = paired.groupby(["family", "size", "task"])["sim_paired"].mean()

    if len(perturbation_df):
        pert = perturbation_df.copy()
        pert["family"], pert["size"] = zip(*pert["model"].map(_family_size))
        grad_pert = pert.groupby(["family", "size", "task", "perturbation"])["sim_perturbed"].mean()
    else:
        grad_pert = pd.Series(dtype=float)

    rows = []
    for _, row in instruct.iterrows():
        if row["condition"] == "SAE":
            continue
        key = (row["family"], row["size"], row["task"])
        if key not in sae.index:
            continue
        gap = row["accuracy"] - sae[key]
        if row["condition"] == "AAVE":
            sim = grad_aave.get(key)
            cond = "AAVE"
        elif row["condition"].startswith("SAE+"):
            pert_name = row["condition"][4:]
            sim = grad_pert.get(key + (pert_name,))
            cond = pert_name
        else:
            continue
        if sim is None or pd.isna(sim):
            continue
        rows.append({
            "family": row["family"], "size": row["size"], "task": row["task"],
            "condition": cond, "accuracy_gap": gap, "grad_sim": float(sim),
        })
    return pd.DataFrame(rows)


def print_accuracy_vs_gradient_summary(corr_df):
    """Per-condition aggregates and Spearman/Pearson correlations."""
    if len(corr_df) == 0:
        print("\n(no overlapping accuracy × gradient data to correlate)")
        return

    print("\n=== ACCURACY (instruct, naive) vs GRADIENT (base) — matched on (family, size) ===")
    cond_order = ["AAVE"] + [p for p in PERTURBATIONS if p in corr_df["condition"].unique()]
    grp = (corr_df.groupby("condition")
                  .agg(mean_acc_gap=("accuracy_gap", "mean"),
                       std_acc_gap=("accuracy_gap", "std"),
                       mean_grad_sim=("grad_sim", "mean"),
                       n=("accuracy_gap", "count"))
                  .reindex([c for c in cond_order if c in corr_df["condition"].unique()])
                  .round(3))
    print("\n  Per-condition aggregates:")
    print(grp.to_string())

    if len(corr_df) >= 6:
        r_p, p_p = stats.pearsonr(corr_df["grad_sim"], corr_df["accuracy_gap"])
        r_s, p_s = stats.spearmanr(corr_df["grad_sim"], corr_df["accuracy_gap"])
        print(f"\n  Pooled (n={len(corr_df)}):")
        print(f"    Pearson  r = {r_p:+.3f}  p = {p_p:.3e}")
        print(f"    Spearman ρ = {r_s:+.3f}  p = {p_s:.3e}")

    print("\n  Per-condition Spearman ρ (grad_sim vs accuracy_gap):")
    for cond in cond_order:
        g = corr_df[corr_df["condition"] == cond]
        if len(g) < 4 or g["grad_sim"].nunique() < 3:
            continue
        r, p = stats.spearmanr(g["grad_sim"], g["accuracy_gap"])
        print(f"    {cond:25s}  ρ = {r:+.3f}  p = {p:.3e}  n = {len(g)}")

    # Per-condition aggregate (one point per condition): does the perturbation
    # ranking by gradient sim match the ranking by accuracy gap?
    cond_agg = (corr_df.groupby("condition")
                       .agg(mean_grad_sim=("grad_sim", "mean"),
                            mean_acc_gap=("accuracy_gap", "mean")))
    if len(cond_agg) >= 4:
        r, p = stats.spearmanr(cond_agg["mean_grad_sim"], cond_agg["mean_acc_gap"])
        print(f"\n  Across-condition rank correlation (n={len(cond_agg)} conditions):")
        print(f"    Spearman ρ(mean_grad_sim, mean_acc_gap) = {r:+.3f}  p = {p:.3e}")
        print(f"    (positive ⇒ conditions that disturb gradients more also drop accuracy more)")


def plot_accuracy_vs_gradient(corr_df, output_dir):
    """Scatter: gradient cosine sim (base) vs accuracy gap (instruct), one point
    per (family, size, task, condition). Color by condition.
    """
    if len(corr_df) == 0:
        return
    cond_order = ["AAVE"] + [p for p in PERTURBATIONS if p in corr_df["condition"].unique()]
    cond_order = [c for c in cond_order if c in corr_df["condition"].unique()]
    palette = {"AAVE": _SET2[1]}
    for i, p in enumerate([c for c in cond_order if c != "AAVE"]):
        palette[p] = _SET2[(i + 2) % len(_SET2)]

    fig, ax = plt.subplots(figsize=(9, 6))
    for cond in cond_order:
        sub = corr_df[corr_df["condition"] == cond]
        if len(sub) == 0:
            continue
        marker = "o" if cond == "AAVE" else "s"
        size_pt = 70 if cond == "AAVE" else 35
        ax.scatter(sub["grad_sim"], sub["accuracy_gap"],
                   color=palette[cond], label=cond,
                   s=size_pt, alpha=0.75, marker=marker,
                   edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("mean cosine sim (SAE ↔ condition), base model")
    ax.set_ylabel("Δ accuracy vs SAE (instruct, p.p.)")
    ax.set_title("Gradient divergence (base) vs accuracy drop (instruct)")
    ax.legend(fontsize=12, loc="best")
    sns.despine(ax=ax)
    fig.savefig(os.path.join(output_dir, "accuracy_vs_gradient.pdf"))
    fig.savefig(os.path.join(output_dir, "accuracy_vs_gradient.png"))
    plt.close(fig)
    print("  Saved accuracy_vs_gradient.pdf")


########
# MAIN #
########

def main():
    parser = argparse.ArgumentParser(description="Gradient projection analysis")
    parser.add_argument("--config", default=os.environ.get("DIALECTTAX_CONFIG", "default"))
    parser.add_argument("--experiments-dir", default=None)
    parser.add_argument(
        "--output-dir", default="analysis/plots/gradients",
        help="Directory for plot outputs (default: analysis/plots/gradients).",
    )
    args = parser.parse_args()

    if args.experiments_dir:
        experiments_dir = args.experiments_dir
    else:
        project_config = dialecttax.utils.load_config(args.config)
        experiments_dir = project_config["directories"]["experiments"]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load
    meta_df, proj_dict = load_all_projections(experiments_dir)

    # Paired similarities
    paired_df = compute_paired_similarities(proj_dict)
    baseline_df = compute_unrelated_baseline(proj_dict)
    z_df = compute_z_scores(paired_df, baseline_df)

    # Summary
    print_summary(paired_df, z_df)

    # Perturbation baseline: sim(SAE, perturbed_SAE)
    perturbed_dict = load_all_perturbed_projections(experiments_dir)
    perturbation_df = compute_perturbation_similarities(proj_dict, perturbed_dict)
    perturbation_z_df = compute_perturbation_z_scores(perturbation_df, baseline_df)
    print_perturbation_summary(perturbation_df, perturbation_z_df, paired_df)

    # Loss-gap test: restrict to base models, per the Section 4.1 figure
    base_meta = meta_df[meta_df["model"].str.endswith("_base")]
    loss_gap_df = test_loss_gap(base_meta)
    print_loss_gap_summary(loss_gap_df)
    loss_gap_df.to_csv(os.path.join(output_dir, "loss_gap_aave_vs_sae.csv"))
    loss_by_dialect_df = summarize_loss_by_dialect(base_meta)
    print_loss_by_dialect_summary(loss_by_dialect_df)
    loss_by_dialect_df.to_csv(os.path.join(output_dir, "loss_by_dialect.csv"))

    # Correctness analysis
    sim_corr_df = compute_similarity_correctness(proj_dict, experiments_dir)
    print_correctness_summary(sim_corr_df)

    # Cross-modality: accuracy gap (instruct) vs gradient sim (base)
    acc_df = load_perturbation_accuracy(experiments_dir)
    acc_grad_df = correlate_accuracy_vs_gradient(acc_df, paired_df, perturbation_df)
    print_accuracy_vs_gradient_summary(acc_grad_df)

    # Plots
    plot_paired_vs_unrelated(paired_df, baseline_df, output_dir)
    plot_z_scores_by_model(z_df, output_dir)
    plot_loss_comparison(meta_df, output_dir)
    plot_similarity_vs_correctness(sim_corr_df, output_dir)
    if len(perturbation_df) > 0:
        plot_perturbation_by_type(perturbation_df, paired_df, output_dir)
    plot_accuracy_vs_gradient(acc_grad_df, output_dir)

    # Save tables for LaTeX
    z_pivot = z_df.pivot_table(index="model", columns="task", values="z_score")
    z_pivot.to_csv(os.path.join(output_dir, "gradient_z_scores.csv"))
    paired_summary = paired_df.groupby(["model", "task"])["sim_paired"].agg(["mean", "std"])
    paired_summary.to_csv(os.path.join(output_dir, "gradient_paired_similarity.csv"))
    if len(perturbation_df) > 0:
        perturbation_summary = perturbation_df.groupby(
            ["model", "task", "perturbation"]
        )["sim_perturbed"].agg(["mean", "std"])
        perturbation_summary.to_csv(os.path.join(output_dir, "gradient_perturbation_similarity.csv"))
        perturbation_z_df.to_csv(os.path.join(output_dir, "gradient_perturbation_z_scores.csv"), index=False)
    if len(acc_grad_df) > 0:
        acc_grad_df.to_csv(os.path.join(output_dir, "accuracy_vs_gradient.csv"), index=False)
    print(f"\n  Saved CSV tables to {output_dir}")


if __name__ == "__main__":
    main()
