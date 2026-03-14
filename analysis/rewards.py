"""
Reward model analysis for dialect fairness.

Loads per-sample reward scores from benchmark_rewards outputs and computes
score disparities between SAE and AAVE dialect pairs. Produces summary
statistics and figures for Section 4.2.

Usage:
    python analysis/rewards.py
    python analysis/rewards.py --experiments-dir /path/to/experiments
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

_SET2 = sns.color_palette("Set2", 8)
DIALECT_COLORS = {"sae": _SET2[0], "aave": _SET2[1]}
DIALECT_LABELS = {"sae": "SAE", "aave": "AAVE"}

TASKS = ["math", "algorithm", "logic", "planning"]
DIALECTS = ["sae", "aave"]
MULTIVALUE_DIALECTS = ["aave", "appalachian", "chicano", "indian", "singapore"]

REWARD_MODEL_ORDER = [
    "skywork_llama_3b",
    "skywork_qwen_4b",
    "skywork_llama_8b",
    "skywork_qwen_8b",
    "skywork_gemma_27b",
    "qrm_llama_8b",
    "qrm_gemma_27b",
    "ai2_llama_8b_base",
    "ai2_llama_8b",
    "ai2_llama_70b",
]

REWARD_MODEL_LABELS = {
    "skywork_llama_3b": "Skywork\nLlama 3B",
    "skywork_qwen_4b": "Skywork\nQwen 4B",
    "skywork_llama_8b": "Skywork\nLlama 8B",
    "skywork_qwen_8b": "Skywork\nQwen 8B",
    "skywork_gemma_27b": "Skywork\nGemma 27B",
    "qrm_llama_8b": "QRM\nLlama 8B",
    "qrm_gemma_27b": "QRM\nGemma 27B",
    "ai2_llama_8b_base": "Ai2\nLlama 8B\n(base)",
    "ai2_llama_8b": "Ai2\nLlama 8B",
    "ai2_llama_70b": "Ai2\nLlama 70B",
}

PROVIDER_HATCHES = {"skywork": "//", "qrm": "..", "ai2": ""}
PROVIDER_COLORS = {"skywork": _SET2[2], "qrm": _SET2[3], "ai2": _SET2[4]}


###########
# LOADING #
###########

def load_all_rewards(experiments_dir):
    """Load all reward score samples into a DataFrame.

    Returns:
        DataFrame with columns: reward_model, provider, task, dialect, unique_id,
        score, sample_idx.
    """
    rows = []
    base = os.path.join(experiments_dir, "benchmark_rewards")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No benchmark_rewards directory at {base}")

    for rm in sorted(os.listdir(base)):
        rm_dir = os.path.join(base, rm)
        if not os.path.isdir(rm_dir) or rm.startswith(".") or rm.endswith(".yaml"):
            continue
        redial_dir = os.path.join(rm_dir, "redial")
        if not os.path.isdir(redial_dir):
            continue
        provider = rm.split("_")[0]
        for task in TASKS:
            for dialect in DIALECTS:
                path = os.path.join(redial_dir, task, "naive", dialect, "samples.jsonl")
                if not os.path.exists(path):
                    continue
                with open(path) as f:
                    for i, line in enumerate(f):
                        sample = json.loads(line)
                        rows.append({
                            "reward_model": rm,
                            "provider": provider,
                            "task": task,
                            "dialect": dialect,
                            "unique_id": sample["unique_id"],
                            "score": sample["score"],
                            "sample_idx": i,
                        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} reward samples: "
          f"{df['reward_model'].nunique()} reward models, "
          f"{df['task'].nunique()} tasks, "
          f"{df['dialect'].nunique()} dialects")
    return df


######################
# PAIRED SCORE GAPS  #
######################

def compute_paired_gaps(df):
    """Compute per-sample score gap (SAE - AAVE) for each (reward_model, task, sample).

    Args:
        df: DataFrame from load_all_rewards.

    Returns:
        DataFrame with columns: reward_model, provider, task, sample_idx, score_sae,
        score_aave, gap (SAE - AAVE).
    """
    sae = df[df["dialect"] == "sae"].copy()
    aave = df[df["dialect"] == "aave"].copy()

    merged = sae.merge(
        aave[["reward_model", "task", "sample_idx", "score"]],
        on=["reward_model", "task", "sample_idx"],
        suffixes=("_sae", "_aave"),
    )
    merged["gap"] = merged["score_sae"] - merged["score_aave"]
    merged["provider"] = merged["reward_model"].apply(lambda x: x.split("_")[0])
    return merged


#####################
# SUMMARY STATISTICS #
#####################

def print_summary(df, gaps_df):
    """Print summary tables of reward scores and gaps."""
    print("\n=== Mean reward score by dialect ===")
    summary = df.groupby(["reward_model", "dialect"])["score"].agg(["mean", "std"]).round(4)
    print(summary.to_string())

    print("\n=== Score gap (SAE - AAVE) by reward model ===")
    gap_summary = gaps_df.groupby("reward_model")["gap"].agg(["mean", "std", "count"])
    for rm, row in gap_summary.iterrows():
        sub = gaps_df[gaps_df["reward_model"] == rm]["gap"]
        t, p = stats.ttest_1samp(sub, 0)
        print(f"  {rm:25s}: gap={row['mean']:+.4f} (std={row['std']:.4f}), t={t:.2f}, p={p:.2e}, n={int(row['count'])}")

    print("\n=== Score gap by task ===")
    for task in TASKS:
        sub = gaps_df[gaps_df["task"] == task]["gap"]
        t, p = stats.ttest_1samp(sub, 0)
        print(f"  {task:12s}: gap={sub.mean():+.4f} (std={sub.std():.4f}), t={t:.2f}, p={p:.2e}, n={len(sub)}")


##########
# PLOTS  #
##########

def plot_score_by_dialect(df, output_dir):
    """Grouped bar chart of mean reward score by dialect, hatched by provider."""
    models = [m for m in REWARD_MODEL_ORDER if m in df["reward_model"].unique()]
    model_labels = [REWARD_MODEL_LABELS[m] for m in models]

    agg = df[df["reward_model"].isin(models)].groupby(
        ["reward_model", "dialect"]
    )["score"].agg(["mean", "std"]).reset_index()
    agg["provider"] = agg["reward_model"].apply(lambda x: x.split("_")[0])

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for j, (dialect_key, dialect_label) in enumerate(DIALECT_LABELS.items()):
        means = []
        stds = []
        hatches = []
        for model in models:
            row = agg[(agg["reward_model"] == model) & (agg["dialect"] == dialect_key)].iloc[0]
            means.append(row["mean"])
            stds.append(row["std"])
            hatches.append(PROVIDER_HATCHES[row["provider"]])
        bars = ax.bar(
            x + j * width, means, width, yerr=stds,
            label=dialect_label, color=DIALECT_COLORS[dialect_key],
            edgecolor="white", linewidth=0.5,
            capsize=3, error_kw={"linewidth": 1.2},
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(model_labels, fontsize=13)
    ax.set_ylabel("Mean reward score")
    ax.set_xlabel("Reward model")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

    # Two separate legends
    from matplotlib.patches import Patch
    dialect_handles = [
        Patch(facecolor=DIALECT_COLORS["sae"], label="SAE"),
        Patch(facecolor=DIALECT_COLORS["aave"], label="AAVE"),
    ]
    provider_handles = [
        Patch(facecolor="#CCCCCC", hatch="//", edgecolor="white", label="Skywork"),
        Patch(facecolor="#CCCCCC", hatch="", edgecolor="white", label="Ai2"),
    ]
    leg1 = ax.legend(handles=provider_handles, title="Provider", loc="upper right",
                     fontsize=15, title_fontsize=16)
    leg2 = ax.legend(handles=dialect_handles, title="Dialect", loc="upper right",
                     bbox_to_anchor=(0.72, 1.0), ncols=2, fontsize=15, title_fontsize=16)
    ax.add_artist(leg1)
    sns.despine(ax=ax)

    fig.savefig(os.path.join(output_dir, "rewards_score_by_dialect.pdf"))
    fig.savefig(os.path.join(output_dir, "rewards_score_by_dialect.png"))
    plt.close(fig)
    print("  Saved rewards_score_by_dialect.pdf")


def plot_gap_by_task(gaps_df, output_dir):
    """Bar chart of mean score gap (SAE - AAVE) by reward model, grouped by task."""
    models = [m for m in REWARD_MODEL_ORDER if m in gaps_df["reward_model"].unique()]
    model_labels = [REWARD_MODEL_LABELS[m] for m in models]

    plot_df = gaps_df[gaps_df["reward_model"].isin(models)].copy()
    plot_df["Reward model"] = plot_df["reward_model"].map(REWARD_MODEL_LABELS)
    plot_df["Reward model"] = pd.Categorical(
        plot_df["Reward model"], categories=model_labels, ordered=True,
    )
    plot_df["Task"] = plot_df["task"].str.capitalize()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Manual bars for hatching
    x = np.arange(len(models))
    n_tasks = len(TASKS)
    width = 0.8 / n_tasks

    for j, task in enumerate(TASKS):
        vals = []
        hatches = []
        for model in models:
            sub = plot_df[(plot_df["reward_model"] == model) & (plot_df["task"] == task)]
            vals.append(sub["gap"].mean())
            hatches.append(PROVIDER_HATCHES[model.split("_")[0]])
        bars = ax.bar(
            x + j * width, vals, width,
            label=task.capitalize(), color=_SET2[j],
            edgecolor="white", linewidth=0.5,
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            bar.set_edgecolor("white")

    ax.set_xticks(x + width * (n_tasks - 1) / 2)
    ax.set_xticklabels(model_labels, fontsize=13)
    ax.set_ylabel("Score gap (SAE \u2212 AAVE)")
    ax.set_xlabel("Reward model")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

    # Two separate legends
    from matplotlib.patches import Patch
    task_handles = [Patch(facecolor=_SET2[j], label=task.capitalize()) for j, task in enumerate(TASKS)]
    provider_handles = [
        Patch(facecolor="#CCCCCC", hatch="//", edgecolor="white", label="Skywork"),
        Patch(facecolor="#CCCCCC", hatch="", edgecolor="white", label="Ai2"),
    ]
    leg1 = ax.legend(handles=task_handles, title="Task", loc="upper left",
                     ncols=1, fontsize=15, title_fontsize=16)
    leg2 = ax.legend(handles=provider_handles, title="Provider", loc="upper right",
                     ncols=1, fontsize=15, title_fontsize=16)
    ax.add_artist(leg1)
    sns.despine(ax=ax)

    fig.savefig(os.path.join(output_dir, "rewards_gap_by_task.pdf"))
    fig.savefig(os.path.join(output_dir, "rewards_gap_by_task.png"))
    plt.close(fig)
    print("  Saved rewards_gap_by_task.pdf")


def plot_gap_significance(gaps_df, output_dir):
    """Dot plot of mean gap with 95% CI per reward model, showing significance."""
    models = [m for m in REWARD_MODEL_ORDER if m in gaps_df["reward_model"].unique()]

    rows = []
    for model in models:
        sub = gaps_df[gaps_df["reward_model"] == model]["gap"]
        mean = sub.mean()
        se = sub.std() / np.sqrt(len(sub))
        ci95 = 1.96 * se
        t, p = stats.ttest_1samp(sub, 0)
        rows.append({
            "model": model,
            "label": REWARD_MODEL_LABELS[model].replace("\n", " "),
            "mean": mean,
            "ci95": ci95,
            "p": p,
            "significant": p < 0.05,
            "provider": model.split("_")[0],
        })
    sig_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(sig_df))
    colors = [_SET2[2] if s else _SET2[5] for s in sig_df["significant"]]
    markers = ["o" if p == "skywork" else "D" for p in sig_df["provider"]]

    for i, (_, row) in enumerate(sig_df.iterrows()):
        marker = "o" if row["provider"] == "skywork" else "D"
        ax.errorbar(
            row["mean"], i, xerr=row["ci95"],
            fmt=marker, color=colors[i], markersize=10,
            capsize=5, capthick=1.5, linewidth=1.5,
        )

    ax.axvline(x=0, color="black", linewidth=0.5, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sig_df["label"])
    ax.set_xlabel("Score gap (SAE \u2212 AAVE)")
    ax.invert_yaxis()

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Line2D([], [], color=_SET2[2], marker="o", linestyle="", markersize=8, label="p < 0.05"),
        Line2D([], [], color=_SET2[5], marker="o", linestyle="", markersize=8, label="p \u2265 0.05"),
        Line2D([], [], color="gray", marker="o", linestyle="", markersize=8, label="Skywork"),
        Line2D([], [], color="gray", marker="D", linestyle="", markersize=8, label="Ai2"),
    ]
    ax.legend(handles=handles, ncols=2, fontsize=13)
    sns.despine(ax=ax)

    fig.savefig(os.path.join(output_dir, "rewards_gap_significance.pdf"))
    fig.savefig(os.path.join(output_dir, "rewards_gap_significance.png"))
    plt.close(fig)
    print("  Saved rewards_gap_significance.pdf")


##############################
# DIALECT-EXCLUSIVE TOKENS  #
##############################

TOKENIZERS = ["bpe", "gemma", "gpt2", "llama", "qwen", "unigram", "wordpiece"]


def load_token_scores(experiments_dir):
    """Load per-token reward scores for AAVE-vs-SAE exclusive tokens across
    all three dialect corpora.

    Sources:
        - redial: per-task SAE/AAVE vocabularies
        - parallelaave: single SAE/AAVE pair
        - multivalue: SAE/AAVE pair (ignores other MultiVALUE dialects)

    Returns:
        DataFrame with columns: reward_model, provider, dataset, task,
        tokenizer, dialect, token, score. For non-redial datasets, task is None.
    """
    rows = []
    base = os.path.join(experiments_dir, "benchmark_rewards")

    def _add_exclusive(sae_scores, aave_scores, *, rm, provider, dataset, task, tokenizer):
        sae_only = set(sae_scores.keys()) - set(aave_scores.keys())
        aave_only = set(aave_scores.keys()) - set(sae_scores.keys())
        for token in sae_only:
            rows.append({"reward_model": rm, "provider": provider, "dataset": dataset,
                         "task": task, "tokenizer": tokenizer, "dialect": "sae",
                         "dialect_compared": "aave", "token": token, "score": sae_scores[token]})
        for token in aave_only:
            rows.append({"reward_model": rm, "provider": provider, "dataset": dataset,
                         "task": task, "tokenizer": tokenizer, "dialect": "aave",
                         "dialect_compared": "aave", "token": token, "score": aave_scores[token]})

    for rm in sorted(os.listdir(base)):
        rm_dir = os.path.join(base, rm)
        if not os.path.isdir(rm_dir) or rm.startswith(".") or rm.endswith(".yaml") or rm.endswith(".zip"):
            continue
        provider = rm.split("_")[0]

        # redial: per-task
        redial_dir = os.path.join(rm_dir, "redial")
        if os.path.isdir(redial_dir):
            for task in TASKS:
                for tokenizer in TOKENIZERS:
                    sae_path = os.path.join(redial_dir, task, "naive", "sae", f"tokens_{tokenizer}.json")
                    aave_path = os.path.join(redial_dir, task, "naive", "aave", f"tokens_{tokenizer}.json")
                    if not os.path.exists(sae_path) or not os.path.exists(aave_path):
                        continue
                    with open(sae_path) as f:
                        sae_scores = json.load(f)
                    with open(aave_path) as f:
                        aave_scores = json.load(f)
                    _add_exclusive(sae_scores, aave_scores, rm=rm, provider=provider,
                                   dataset="redial", task=task, tokenizer=tokenizer)

        # parallelaave: SAE vs AAVE only. multivalue: SAE vs each of 5 non-SAE dialects.
        for ds, non_sae_dialects in (("parallelaave", ["aave"]),
                                     ("multivalue", ["aave", "appalachian", "chicano", "indian", "singapore"])):
            ds_dir = os.path.join(rm_dir, ds)
            if not os.path.isdir(ds_dir):
                continue
            for tokenizer in TOKENIZERS:
                sae_path = os.path.join(ds_dir, "sae", f"tokens_{tokenizer}.json")
                if not os.path.exists(sae_path):
                    continue
                with open(sae_path) as f:
                    sae_scores = json.load(f)
                for dialect in non_sae_dialects:
                    d_path = os.path.join(ds_dir, dialect, f"tokens_{tokenizer}.json")
                    if not os.path.exists(d_path):
                        continue
                    with open(d_path) as f:
                        d_scores = json.load(f)
                    sae_only = set(sae_scores.keys()) - set(d_scores.keys())
                    dialect_only = set(d_scores.keys()) - set(sae_scores.keys())
                    for token in sae_only:
                        rows.append({"reward_model": rm, "provider": provider, "dataset": ds,
                                     "task": None, "tokenizer": tokenizer, "dialect": "sae",
                                     "dialect_compared": dialect, "token": token, "score": sae_scores[token]})
                    for token in dialect_only:
                        rows.append({"reward_model": rm, "provider": provider, "dataset": ds,
                                     "task": None, "tokenizer": tokenizer, "dialect": dialect,
                                     "dialect_compared": dialect, "token": token, "score": d_scores[token]})

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} dialect-exclusive token scores: "
          f"{df['reward_model'].nunique()} reward models, "
          f"{df['tokenizer'].nunique()} tokenizers, "
          f"{df['dataset'].nunique()} datasets")
    return df


def print_token_summary(token_df):
    """Print summary of dialect-exclusive token reward scores."""
    print("\n=== Dialect-exclusive token scores ===")
    for rm in sorted(token_df["reward_model"].unique()):
        sub = token_df[token_df["reward_model"] == rm]
        for dialect in DIALECTS:
            d = sub[sub["dialect"] == dialect]
            print(f"  {rm:25s} {dialect}: mean={d['score'].mean():+.4f}, "
                  f"std={d['score'].std():.4f}, n={len(d)}")

    print("\n=== Mean exclusive token score by dialect (all models) ===")
    for dialect in DIALECTS:
        d = token_df[token_df["dialect"] == dialect]
        print(f"  {dialect}: mean={d['score'].mean():+.4f}, std={d['score'].std():.4f}, n={len(d)}")

    sae_scores = token_df[token_df["dialect"] == "sae"]["score"]
    aave_scores = token_df[token_df["dialect"] == "aave"]["score"]
    t, p = stats.ttest_ind(sae_scores, aave_scores)
    print(f"  t-test (SAE-exclusive vs AAVE-exclusive): t={t:.2f}, p={p:.2e}")


def _gap_d_p(sae_scores, dialect_scores, sigma=None):
    """Compute raw gap, Cohen's d, and t-test p for two score arrays.

    Args:
        sae_scores: SAE-exclusive token scores.
        dialect_scores: dialect-exclusive token scores.
        sigma: optional pre-computed std for the denominator of Cohen's d
            (e.g. the RM's overall pooled std). If None, uses the
            within-pair pooled std.

    Returns:
        Tuple (gap, cohen_d, p). Returns (nan, nan, nan) if either array is empty.
    """
    if len(sae_scores) == 0 or len(dialect_scores) == 0:
        return float("nan"), float("nan"), float("nan")
    gap = sae_scores.mean() - dialect_scores.mean()
    if sigma is None:
        sigma = pd.concat([sae_scores, dialect_scores]).std()
    cohen = gap / sigma if sigma > 0 else 0.0
    _, p = stats.ttest_ind(sae_scores, dialect_scores)
    return gap, cohen, p


def compute_token_breakdown(token_df):
    """Per-(reward_model, dialect_compared) gap + Cohen's d + p.

    For the per-RM Pooled column, gap aggregates across all dialect_compared
    values and Cohen's d uses the RM's overall score std (so cells in the
    Pooled column are comparable to each other across RMs).

    Returns:
        DataFrame with columns: reward_model, dialect, gap, cohen_d, p,
        n_sae, n_dialect. The dialect column takes values in
        MULTIVALUE_DIALECTS plus "pooled".
    """
    rows = []
    for rm in sorted(token_df["reward_model"].unique()):
        rm_df = token_df[token_df["reward_model"] == rm]
        for dialect in MULTIVALUE_DIALECTS:
            pair = rm_df[rm_df["dialect_compared"] == dialect]
            s = pair[pair["dialect"] == "sae"]["score"]
            d = pair[pair["dialect"] == dialect]["score"]
            gap, cohen, p = _gap_d_p(s, d)
            rows.append({"reward_model": rm, "dialect": dialect,
                         "gap": gap, "cohen_d": cohen, "p": p,
                         "n_sae": len(s), "n_dialect": len(d)})
        # Per-RM pooled: all dialect_compared values, normalized by RM's overall std
        s = rm_df[rm_df["dialect"] == "sae"]["score"]
        d = rm_df[rm_df["dialect"] != "sae"]["score"]
        gap, cohen, p = _gap_d_p(s, d, sigma=rm_df["score"].std())
        rows.append({"reward_model": rm, "dialect": "pooled",
                     "gap": gap, "cohen_d": cohen, "p": p,
                     "n_sae": len(s), "n_dialect": len(d)})
    return pd.DataFrame(rows)


def compute_token_pooled_summary(token_df, breakdown_df=None):
    """Pooled aggregates across the full token-level dataset.

    Returns a dict with:
        - grand_pooled: dict with gap, n_sae, n_dialect, p (across-RM std)
        - per_corpus: dict mapping dataset -> dict with gap, n, p
        - per_dialect_mean_d: dict mapping dialect -> mean Cohen's d across RMs
        - mean_d_overall: float, mean of per-RM pooled Cohen's d
        - sigma_min, sigma_max: tuple (rm, sigma) for the per-RM std extremes
    """
    out = {}

    s = token_df[token_df["dialect"] == "sae"]["score"]
    d = token_df[token_df["dialect"] != "sae"]["score"]
    gap, cohen, p = _gap_d_p(s, d)
    out["grand_pooled"] = {"gap": gap, "cohen_d": cohen, "p": p,
                           "n_sae": len(s), "n_dialect": len(d), "n_total": len(s) + len(d)}

    out["per_corpus"] = {}
    for ds in sorted(token_df["dataset"].unique()):
        sub = token_df[token_df["dataset"] == ds]
        s = sub[sub["dialect"] == "sae"]["score"]
        d = sub[sub["dialect"] != "sae"]["score"]
        gap, cohen, p = _gap_d_p(s, d)
        out["per_corpus"][ds] = {"gap": gap, "cohen_d": cohen, "p": p,
                                 "n_sae": len(s), "n_dialect": len(d)}

    if breakdown_df is None:
        breakdown_df = compute_token_breakdown(token_df)

    out["per_dialect_mean_d"] = {}
    for dialect in MULTIVALUE_DIALECTS:
        ds = breakdown_df[breakdown_df["dialect"] == dialect]["cohen_d"].dropna()
        out["per_dialect_mean_d"][dialect] = float(ds.mean()) if len(ds) else float("nan")
    out["mean_d_overall"] = float(
        breakdown_df[breakdown_df["dialect"] == "pooled"]["cohen_d"].dropna().mean()
    )

    sigmas = token_df.groupby("reward_model")["score"].std()
    out["sigma_min"] = (sigmas.idxmin(), float(sigmas.min()))
    out["sigma_max"] = (sigmas.idxmax(), float(sigmas.max()))
    return out


def print_token_breakdown_summary(token_df):
    """Print the per-RM × per-dialect breakdown and pooled aggregates.

    Mirrors the numbers cited in the "Rewards are contextual, not lexical"
    paragraph and the appendix table.
    """
    breakdown = compute_token_breakdown(token_df)
    summary = compute_token_pooled_summary(token_df, breakdown_df=breakdown)

    g = summary["grand_pooled"]
    print("\n=== Token-level dialect gap (pooled across all 3 corpora, all dialects) ===")
    print(f"  pooled raw gap = {g['gap']:+.4f}")
    print(f"  mean Cohen's d across RMs = {summary['mean_d_overall']:+.4f}")
    print(f"  n = {g['n_total']:,}  (SAE-excl {g['n_sae']:,}, dialect-excl {g['n_dialect']:,})")
    print(f"  per-RM sigma range: {summary['sigma_min'][1]:.4f} ({summary['sigma_min'][0]}) "
          f"to {summary['sigma_max'][1]:.4f} ({summary['sigma_max'][0]})")

    print("\n=== Per-corpus raw gap ===")
    for ds_name, stats_d in summary["per_corpus"].items():
        print(f"  {ds_name:15s} gap = {stats_d['gap']:+.4f}, "
              f"p = {stats_d['p']:.2e}, n = ({stats_d['n_sae']:,}, {stats_d['n_dialect']:,})")

    print("\n=== Mean Cohen's d across RMs, per dialect ===")
    for dialect in MULTIVALUE_DIALECTS:
        print(f"  {dialect:15s} mean d = {summary['per_dialect_mean_d'][dialect]:+.4f}")

    print("\n=== Per-(RM, dialect) breakdown ===")
    print(f"  {'RM':25s} {'dialect':15s} {'gap':>9s} {'d':>8s} {'p':>10s}")
    for rm in REWARD_MODEL_ORDER:
        for dialect in MULTIVALUE_DIALECTS + ["pooled"]:
            row = breakdown[(breakdown["reward_model"] == rm) & (breakdown["dialect"] == dialect)]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            sig = "***" if r["p"] < 0.001 else ("**" if r["p"] < 0.01 else ("*" if r["p"] < 0.05 else ""))
            print(f"  {rm:25s} {dialect:15s} {r['gap']:>+9.4f} {r['cohen_d']:>+8.4f} {r['p']:>10.2e} {sig}")

    print("\n=== Token-level direction per task ===")
    for task in TASKS:
        sub = token_df[token_df["task"] == task]
        sae_t = sub[sub["dialect"] == "sae"]["score"]
        aave_t = sub[sub["dialect"] == "aave"]["score"]
        if len(sae_t) == 0 or len(aave_t) == 0:
            continue
        t_stat, t_p = stats.ttest_ind(sae_t, aave_t)
        print(f"  {task:12s}: SAE-excl={sae_t.mean():+.4f}, AAVE-excl={aave_t.mean():+.4f}, "
              f"gap={sae_t.mean() - aave_t.mean():+.4f}, p={t_p:.2e}")


###########################
# TASK-LEVEL DIAGNOSTICS  #
###########################

def dialect_token_exclusivity_by_task(experiments_dir):
    """Compute the fraction of dialect-exclusive tokens per task.

    For each (task, tokenizer), loads SAE and AAVE token vocabularies from
    generate_words outputs and computes the Jaccard distance (1 - |intersection|/|union|)
    and the exclusive fraction (|exclusive| / |union|).

    Returns:
        DataFrame with per-(task, tokenizer) exclusivity metrics.
    """
    rows = []
    gw_dir = os.path.join(experiments_dir, "generate_words", "redial")

    for task in TASKS:
        for tokenizer in TOKENIZERS:
            sae_path = os.path.join(gw_dir, task, "sae", f"tokens_{tokenizer}.json")
            aave_path = os.path.join(gw_dir, task, "aave", f"tokens_{tokenizer}.json")
            if not os.path.exists(sae_path) or not os.path.exists(aave_path):
                continue
            with open(sae_path) as f:
                sae_tokens = set(json.load(f).keys())
            with open(aave_path) as f:
                aave_tokens = set(json.load(f).keys())

            union = sae_tokens | aave_tokens
            intersection = sae_tokens & aave_tokens
            sae_only = sae_tokens - aave_tokens
            aave_only = aave_tokens - sae_tokens
            exclusive = sae_only | aave_only

            rows.append({
                "task": task,
                "tokenizer": tokenizer,
                "n_sae": len(sae_tokens),
                "n_aave": len(aave_tokens),
                "n_union": len(union),
                "n_shared": len(intersection),
                "n_exclusive": len(exclusive),
                "jaccard": len(intersection) / len(union) if union else 0,
                "exclusive_frac": len(exclusive) / len(union) if union else 0,
            })

    return pd.DataFrame(rows)


def task_exclusivity_vs_reward_gap(experiments_dir, gaps_df):
    """Compare per-task dialect token exclusivity with reward score gap.

    Prints a summary table showing, for each task, the mean fraction of
    dialect-exclusive tokens alongside the mean |Δr| across reward models.
    """
    excl = dialect_token_exclusivity_by_task(experiments_dir)
    if len(excl) == 0:
        print("No token exclusivity data found.")
        return

    task_excl = excl.groupby("task")["exclusive_frac"].mean()
    task_gap = gaps_df.groupby("task")["gap"].agg(["mean", lambda x: np.mean(np.abs(x))])
    task_gap.columns = ["mean_gap", "mean_abs_gap"]

    combined = pd.DataFrame({
        "exclusive_frac": task_excl,
        "mean_gap": task_gap["mean_gap"],
        "mean_abs_gap": task_gap["mean_abs_gap"],
    }).reindex(TASKS)

    print("\n=== Task-level: dialect exclusivity vs reward gap ===")
    print(combined.round(4).to_string())

    if len(combined.dropna()) >= 3:
        r, p = stats.pearsonr(combined["exclusive_frac"].values, combined["mean_abs_gap"].values)
        print(f"\n  Pearson r(exclusive_frac, |Δr|) = {r:.3f}, p = {p:.3f} (n={len(combined)})")


########
# MAIN #
###################
# RM SEPARABILITY #
###################

def load_rm_hidden_states(experiments_dir):
    """Load RM last-token hidden states from generate_rewards_hidden_states.

    Returns:
        Dict keyed by (reward_model, task, dialect) -> ndarray (n_samples, hidden_dim).
    """
    root = os.path.join(experiments_dir, "generate_rewards_hidden_states")
    if not os.path.isdir(root):
        return {}
    out = {}
    for rm in sorted(os.listdir(root)):
        rm_dir = os.path.join(root, rm, "redial")
        if not os.path.isdir(rm_dir):
            continue
        for task in sorted(os.listdir(rm_dir)):
            task_dir = os.path.join(rm_dir, task, "naive")
            if not os.path.isdir(task_dir):
                continue
            for dialect in sorted(os.listdir(task_dir)):
                hidden_path = os.path.join(task_dir, dialect, "hidden.npz")
                if not os.path.exists(hidden_path):
                    continue
                with np.load(hidden_path) as z:
                    arrs = [z[k] for k in z.files]
                if arrs:
                    out[(rm, task, dialect)] = np.stack(arrs).astype(np.float32)
    return out


def rm_hidden_dialect_separability(experiments_dir):
    """5-fold CV accuracy of logistic regression predicting dialect from the
    last-token hidden state of each reward model, aggregated across tasks.

    Mirrors :func:`analysis.characters.hidden_dialect_separability`. A score
    above 50% (chance) means dialect identity is linearly decodable from the
    representation the RM's score head reads from.

    Returns:
        DataFrame with columns: reward_model, task, acc, n.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    hiddens = load_rm_hidden_states(experiments_dir)
    if not hiddens:
        print("No RM hidden states found at experiments/generate_rewards_hidden_states/.")
        print("Run: bash scripts/rewards/generate_rewards.sh --phase hidden")
        return pd.DataFrame()

    pairs = {(rm, t) for (rm, t, _) in hiddens}
    rows = []
    for rm, t in sorted(pairs):
        sae = hiddens.get((rm, t, "sae"))
        aave = hiddens.get((rm, t, "aave"))
        if sae is None or aave is None or len(sae) < 10 or len(aave) < 10:
            continue
        if sae.shape[1] != aave.shape[1]:
            continue
        # Drop NaN rows (Gemma 2 bf16 sometimes produced NaN before the eager-attn fix;
        # an occasional one-off NaN can also slip in from individual forward failures)
        sae_ok = ~np.isnan(sae).any(axis=1)
        aave_ok = ~np.isnan(aave).any(axis=1)
        sae = sae[sae_ok]
        aave = aave[aave_ok]
        n_dropped = int((~sae_ok).sum() + (~aave_ok).sum())
        if len(sae) < 10 or len(aave) < 10:
            continue
        X = np.concatenate([sae, aave])
        y = np.concatenate([np.zeros(len(sae)), np.ones(len(aave))])
        try:
            clf = LogisticRegression(max_iter=400, C=1.0, solver="liblinear")
            acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
        except Exception:
            continue
        rows.append({"reward_model": rm, "task": t, "acc": acc * 100,
                     "n": len(X), "n_dropped": n_dropped})
    return pd.DataFrame(rows)


def print_rm_separability_summary(sep_df):
    """Print per-RM CV accuracy averaged across tasks, plus the task breakdown."""
    if len(sep_df) == 0:
        return
    print("\n=== RM dialect separability (5-fold CV logistic regression) ===")
    pivot = sep_df.pivot_table(index="reward_model", columns="task", values="acc", aggfunc="mean")
    pivot = pivot.reindex([m for m in REWARD_MODEL_ORDER if m in pivot.index])
    pivot["mean"] = pivot.mean(axis=1)
    print(pivot.round(1).to_string())
    print("\n  Grand mean across RMs and tasks: "
          f"{sep_df['acc'].mean():.1f}%   (chance = 50%)")


########

def main():
    parser = argparse.ArgumentParser(description="Reward model analysis")
    parser.add_argument("--config", default=os.environ.get("DIALECTTAX_CONFIG", "default"))
    parser.add_argument("--experiments-dir", default=None)
    parser.add_argument(
        "--output-dir", default="analysis/plots/rewards",
        help="Directory for plot outputs (default: analysis/plots/rewards).",
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
    df = load_all_rewards(experiments_dir)
    gaps_df = compute_paired_gaps(df)

    # Summary
    print_summary(df, gaps_df)

    # Plots
    plot_score_by_dialect(df, output_dir)
    plot_gap_by_task(gaps_df, output_dir)
    plot_gap_significance(gaps_df, output_dir)

    # Dialect-exclusive token analysis
    token_df = load_token_scores(experiments_dir)
    print_token_summary(token_df)
    print_token_breakdown_summary(token_df)

    # Save per-RM × per-dialect breakdown (mirrors the appendix table)
    token_breakdown_df = compute_token_breakdown(token_df)
    token_breakdown_df.to_csv(os.path.join(output_dir, "rewards_token_breakdown.csv"), index=False)

    # Task-level diagnostics
    task_exclusivity_vs_reward_gap(experiments_dir, gaps_df)

    # RM hidden-state dialect separability (requires generate_rewards_hidden_states/)
    sep_df = rm_hidden_dialect_separability(experiments_dir)
    print_rm_separability_summary(sep_df)

    # Save tables
    gap_pivot = gaps_df.pivot_table(index="reward_model", columns="task", values="gap", aggfunc="mean")
    gap_pivot.to_csv(os.path.join(output_dir, "rewards_gap_by_task.csv"))
    if len(sep_df):
        sep_pivot = sep_df.pivot_table(index="reward_model", columns="task", values="acc", aggfunc="mean")
        sep_pivot["mean"] = sep_pivot.mean(axis=1)
        sep_pivot.round(1).to_csv(os.path.join(output_dir, "rewards_hidden_separability.csv"))
    print(f"\n  Saved CSV tables to {output_dir}")


if __name__ == "__main__":
    main()
