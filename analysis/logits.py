"""
Logit- and entropy-based evidence of the tokenization tax across dialects.

Organized around the claims in `Paper/paper.tex`:

    §4.1  Cross-entropy is systematically higher on AAVE than on SAE
          (the paper reports a 0.5-0.7 nat gap on ReDial).
    §4.1  The gap is independent of model competence (no interaction with
          per-sample correctness).
    §2    The dialect ranking Appalachian > AAVE > Chicano > SAE > Indian
          > Singapore that holds at the tokenizer level also shows up in
          model-internal log-probabilities on MultiVALUE.
    §3    Scale amplifies representational divergence. We ask the analogous
          question for the input cross-entropy gap.

The generate_logits experiment records, per sample:
    redial:         input_mean_log_prob, input_mean_entropy,
                    gen_mean_log_prob, gen_mean_entropy, correct, ...
    multivalue:     mean_log_prob, mean_entropy (no QA, no correctness)
    parallelaave:   mean_log_prob, mean_entropy (no QA, no correctness)

Usage:
    python analysis/logits.py
    python analysis/logits.py --experiments-dir /data/gemini/ellang/dialecttax/experiments
    python analysis/logits.py --skip-plots
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import dialecttax.utils

plt.rcParams.update({
    "figure.dpi": 140,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.constrained_layout.use": True,
})

DIALECT_ORDER = ["sae", "aave", "appalachian", "chicano", "indian", "singapore"]
FAMILY_ORDER = ["llama", "gemma", "qwen"]
MODEL_SIZE_RE = {
    "llama": [("1b", 1.0), ("3b", 3.0), ("8b", 8.0), ("70b", 70.0)],
    "gemma": [("1b", 1.0), ("4b", 4.0), ("12b", 12.0), ("27b", 27.0)],
    "qwen":  [("1.7b", 1.7), ("4b", 4.0), ("8b", 8.0), ("9b", 9.0), ("27b", 27.0), ("32b", 32.0)],
}


###########
# LOADING #
###########

def _parse_model(name):
    """Split a model dir name like 'llama_8b_instruct' into (family, size, variant)."""
    parts = name.split("_")
    family = parts[0]
    variant = "instruct" if name.endswith("_instruct") else "base"
    size = None
    for tok, val in MODEL_SIZE_RE.get(family, []):
        if f"_{tok}_" in f"_{name}_":
            size = val
            break
    return family, size, variant


def _load_redial(experiments_dir):
    """Redial rows carry correctness + separate input/gen logit metrics."""
    root = os.path.join(experiments_dir, "generate_logits")
    pattern = os.path.join(root, "*", "redial", "*", "*", "*", "metadata.jsonl")
    rows = []
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        model, _ds, task, reasoning, dialect, _ = parts
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["model"] = model
                r["task"] = task
                r["reasoning"] = reasoning
                r["dialect"] = dialect
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["family"], df["size_b"], df["variant"] = zip(*df["model"].map(_parse_model))
        df["input_ce"] = -df["input_mean_log_prob"]
        df["gen_ce"] = -df["gen_mean_log_prob"]
    return df


def _load_redial_perturbations(experiments_dir):
    """Load perturbed SAE metadata from .../sae/perturbed/{name}/metadata.jsonl."""
    root = os.path.join(experiments_dir, "generate_logits")
    pattern = os.path.join(root, "*", "redial", "*", "*", "sae", "perturbed", "*", "metadata.jsonl")
    rows = []
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        model, _ds, task, reasoning, _dialect, _perturbed, perturbation, _ = parts
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if str(r.get("input_mean_log_prob", "")) == "nan":
                    continue
                r["model"] = model
                r["task"] = task
                r["reasoning"] = reasoning
                r["dialect"] = "sae"
                r["perturbation"] = perturbation
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["family"], df["size_b"], df["variant"] = zip(*df["model"].map(_parse_model))
        df["input_ce"] = -df["input_mean_log_prob"]
        if "gen_mean_log_prob" in df.columns:
            df["gen_ce"] = -df["gen_mean_log_prob"]
    return df


def _load_flat_dataset(experiments_dir, dataset):
    """MultiVALUE / ParallelAAVE have the simpler schema (no tasks, no QA)."""
    root = os.path.join(experiments_dir, "generate_logits")
    pattern = os.path.join(root, "*", dataset, "*", "metadata.jsonl")
    rows = []
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        model, _ds, dialect, _ = parts
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r["model"] = model
                r["dialect"] = dialect
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["family"], df["size_b"], df["variant"] = zip(*df["model"].map(_parse_model))
        df["ce"] = -df["mean_log_prob"]
    return df


def _load_flat_perturbations(experiments_dir, dataset):
    """SAE-transform logits for a flat dataset (MultiVALUE / ParallelAAVE).

    Reads .../generate_logits/{model}/{dataset}/sae/perturbed/{transform}/metadata.jsonl,
    covering both character perturbations (swap-0.05, ...) and translations
    (translate-french, ...). All rows are the SAE text under one transform, paired
    to the SAE baseline by unique_id.
    """
    root = os.path.join(experiments_dir, "generate_logits")
    pattern = os.path.join(root, "*", dataset, "sae", "perturbed", "*", "metadata.jsonl")
    rows = []
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        model, _ds, _sae, _perturbed, transform, _ = parts
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if str(r.get("mean_log_prob", "")) == "nan":
                    continue
                r["model"] = model
                r["perturbation"] = transform
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["family"], df["size_b"], df["variant"] = zip(*df["model"].map(_parse_model))
        df["ce"] = -df["mean_log_prob"]
    return df


##################
# §4.1  CE GAP   #
##################

def redial_ce_gap(redial):
    """Per-model input CE gap on ReDial: mean(CE_aave) - mean(CE_sae).

    The paper's headline §4.1 number is ~0.5-0.7 nats for the 9 base models.
    """
    grp = redial.groupby(["model", "family", "size_b", "variant", "dialect"])["input_ce"].mean()
    wide = grp.unstack("dialect")
    if "aave" not in wide.columns or "sae" not in wide.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "input_ce_sae": wide["sae"],
        "input_ce_aave": wide["aave"],
        "gap_nats": wide["aave"] - wide["sae"],
    }).sort_values(["variant", "family", "size_b"])
    return out.round(4)


def redial_ce_gap_paired(redial):
    """Paired (same unique-id stripped of dialect marker) CE gap with a
    paired-sample t-test. Tests the null that aave CE == sae CE per-sample.
    """
    df = redial.copy()
    df["pair_id"] = (
        df["unique_id"].str.replace("_original-", "-", regex=False)
                       .str.replace("_aave-", "-", regex=False)
    )
    rows = []
    for (model, task, reasoning), g in df.groupby(["model", "task", "reasoning"]):
        wide = g.pivot_table(index="pair_id", columns="dialect", values="input_ce", aggfunc="first")
        if not {"sae", "aave"}.issubset(wide.columns):
            continue
        wide = wide.dropna()
        if len(wide) < 10:
            continue
        diff = (wide["aave"] - wide["sae"]).values
        t, p = stats.ttest_rel(wide["aave"], wide["sae"])
        rows.append({
            "model": model, "task": task, "reasoning": reasoning,
            "n": len(wide),
            "mean_gap_nats": diff.mean(),
            "median_gap_nats": float(np.median(diff)),
            "t": t, "p": p,
        })
    return pd.DataFrame(rows).sort_values(["model", "task", "reasoning"]).round(4)


def redial_ce_gap_overall(redial):
    """Summary: fraction of (model, task, reasoning) combos with gap > 0 and
    mean / median gap across all base models — the numbers that go in prose.
    """
    paired = redial_ce_gap_paired(redial)
    if len(paired) == 0:
        return {}
    base = paired.merge(
        redial[["model", "variant"]].drop_duplicates(), on="model",
    )
    summary = {}
    for variant, g in base.groupby("variant"):
        summary[variant] = {
            "n_combos": len(g),
            "frac_positive": float((g["mean_gap_nats"] > 0).mean()),
            "mean_gap_nats": float(g["mean_gap_nats"].mean()),
            "median_gap_nats": float(g["mean_gap_nats"].median()),
            "frac_sig_p_001": float((g["p"] < 0.001).mean()),
        }
    return summary


#######################
# §4.1  INDEPENDENCE  #
#######################

def ce_gap_vs_correctness(redial):
    """Is the AAVE-SAE CE gap bigger on items the model already struggles with?

    We stratify paired items by 'both correct' vs 'any incorrect' and compare
    the CE gap distributions. An interaction would mean competence confounds
    the tax — the paper rules this out for gradients; we check it for logits.
    """
    df = redial.copy()
    df["pair_id"] = (
        df["unique_id"].str.replace("_original-", "-", regex=False)
                       .str.replace("_aave-", "-", regex=False)
    )
    rows = []
    for model, g in df.groupby("model"):
        wide = g.pivot_table(
            index="pair_id", columns="dialect",
            values=["input_ce", "correct"], aggfunc="first",
        )
        needed = {("input_ce", "sae"), ("input_ce", "aave"),
                  ("correct", "sae"), ("correct", "aave")}
        if not needed.issubset(wide.columns):
            continue
        wide = wide.dropna()
        if len(wide) < 30:
            continue
        gap = (wide[("input_ce", "aave")] - wide[("input_ce", "sae")]).values
        both = ((wide[("correct", "sae")] == 1) & (wide[("correct", "aave")] == 1)).values
        any_wrong = ~both
        if both.sum() < 5 or any_wrong.sum() < 5:
            continue
        u, p = stats.mannwhitneyu(gap[both], gap[any_wrong], alternative="two-sided")
        # point-biserial between gap and 'both correct'
        pb_r, pb_p = stats.pointbiserialr(both.astype(int), gap)
        rows.append({
            "model": model,
            "n": len(wide),
            "gap_both_correct": float(gap[both].mean()),
            "gap_any_incorrect": float(gap[any_wrong].mean()),
            "mw_p": p,
            "pb_r": pb_r,
            "pb_p": pb_p,
        })
    return pd.DataFrame(rows).round(4)


#########################
# §2  DIALECT RANKING   #
#########################

def multivalue_dialect_ranking(mv):
    """Per-model mean CE by dialect, plus ranking stability vs token ranking.

    Paper §2 reports the token-length ranking:
        Appalachian > AAVE > Chicano > SAE > Indian > Singapore
    We report CE ranking and report how closely it matches.
    """
    grp = mv.groupby(["model", "family", "size_b", "variant", "dialect"])["ce"].mean()
    wide = grp.unstack("dialect")
    keep = [d for d in DIALECT_ORDER if d in wide.columns]
    wide = wide[keep]
    # Per-model rank of each dialect by CE (1 = lowest CE)
    ranks = wide.rank(axis=1, method="average")
    # Mean rank across models; lower mean rank == consistently easier to predict
    out = pd.DataFrame({
        "mean_ce": wide.mean(),
        "mean_rank": ranks.mean(),
        "n_models": wide.notna().all(axis=1).sum(),
    }).sort_values("mean_rank").round(4)
    return out


def multivalue_paired_gaps(mv):
    """For each non-SAE dialect, paired CE gap vs SAE across models (Wilcoxon)."""
    df = mv.copy()
    # pair id strips dialect suffix; within this dataset unique_id already
    # indexes the shared source document
    df["pair_id"] = df["unique_id"]
    rows = []
    non_sae = [d for d in DIALECT_ORDER if d != "sae" and d in df["dialect"].unique()]
    for model, g in df.groupby("model"):
        for d in non_sae:
            wide = g.pivot_table(index="pair_id", columns="dialect", values="ce", aggfunc="first")
            if not {"sae", d}.issubset(wide.columns):
                continue
            wide = wide.dropna(subset=["sae", d])
            if len(wide) < 20:
                continue
            diff = (wide[d] - wide["sae"]).values
            try:
                w, p = stats.wilcoxon(diff)
            except ValueError:
                w, p = np.nan, np.nan
            rows.append({
                "model": model,
                "dialect": d,
                "n": len(wide),
                "mean_gap_nats": float(diff.mean()),
                "median_gap_nats": float(np.median(diff)),
                "wilcoxon_p": p,
            })
    return pd.DataFrame(rows).round(4)


#########################
# §3  SCALING THE GAP   #
#########################

def ce_gap_vs_scale(redial):
    """AAVE-SAE input CE gap as a function of parameter count, per family/variant.

    Returns long-format DataFrame for easy plotting + per-family
    Spearman correlation of gap vs size.
    """
    gap = redial_ce_gap(redial).reset_index()
    if len(gap) == 0:
        return pd.DataFrame(), {}
    corr = {}
    for (family, variant), g in gap.groupby(["family", "variant"]):
        g = g.dropna(subset=["size_b", "gap_nats"])
        if g["size_b"].nunique() < 3:
            continue
        r, p = stats.spearmanr(g["size_b"], g["gap_nats"])
        corr[(family, variant)] = {"n_sizes": g["size_b"].nunique(),
                                    "spearman_r": float(r), "p": float(p)}
    return gap, corr


##############################
# PERTURBATION vs AAVE GAP  #
##############################

def perturbation_accuracy(redial, perturbations):
    """Accuracy for SAE baseline, AAVE, and each SAE perturbation condition.

    Returns:
        DataFrame with columns: model, family, size_b, variant, task, reasoning,
        condition, accuracy, n.
    """
    rows = []
    for (model, task, reasoning), g in redial.groupby(["model", "task", "reasoning"]):
        for dialect in ["sae", "aave"]:
            sub = g[g["dialect"] == dialect]
            if len(sub) == 0:
                continue
            n_correct = int(sub["correct"].sum())
            label = "SAE" if dialect == "sae" else "AAVE"
            rows.append({
                "model": model, "task": task, "reasoning": reasoning,
                "condition": label,
                "accuracy": n_correct / len(sub) * 100,
                "n": len(sub),
            })

    if len(perturbations):
        for (model, task, reasoning, pert), g in perturbations.groupby(
            ["model", "task", "reasoning", "perturbation"]
        ):
            if len(g) == 0:
                continue
            n_correct = int(g["correct"].sum())
            rows.append({
                "model": model, "task": task, "reasoning": reasoning,
                "condition": f"SAE+{pert}",
                "accuracy": n_correct / len(g) * 100,
                "n": len(g),
            })

    df = pd.DataFrame(rows)
    if len(df):
        df["family"], df["size_b"], df["variant"] = zip(*df["model"].map(_parse_model))
    return df


def perturbation_accuracy_gap(acc_df):
    """Accuracy gap relative to SAE baseline per model x task x reasoning.

    Returns:
        DataFrame with columns: model, task, reasoning, condition, delta.
    """
    sae = acc_df[acc_df["condition"] == "SAE"].set_index(["model", "task", "reasoning"])["accuracy"]
    rows = []
    for cond in acc_df["condition"].unique():
        if cond == "SAE":
            continue
        cond_acc = acc_df[acc_df["condition"] == cond].set_index(["model", "task", "reasoning"])["accuracy"]
        shared = sae.index.intersection(cond_acc.index)
        for idx in shared:
            rows.append({
                "model": idx[0], "task": idx[1], "reasoning": idx[2],
                "condition": cond,
                "delta": cond_acc[idx] - sae[idx],
            })
    return pd.DataFrame(rows)


def perturbation_accuracy_summary(acc_df, reasoning="naive"):
    """Print a concise summary table of accuracy across conditions."""
    sub = acc_df[acc_df["reasoning"] == reasoning]
    if len(sub) == 0:
        return

    pivot = sub.pivot_table(
        index=["model", "task"], columns="condition", values="accuracy", aggfunc="first",
    )
    col_order = ["SAE", "AAVE"] + sorted([c for c in pivot.columns if c.startswith("SAE+")])
    pivot = pivot[[c for c in col_order if c in pivot.columns]]

    print(f"\n  Per model × task accuracy ({reasoning} reasoning):")
    print(pivot.round(1).to_string())

    print(f"\n  Per-model mean accuracy ({reasoning} reasoning):")
    model_mean = sub.groupby(["model", "condition"])["accuracy"].mean().unstack("condition")
    model_mean = model_mean[[c for c in col_order if c in model_mean.columns]]
    print(model_mean.round(1).to_string())

    print(f"\n  Grand mean accuracy ({reasoning} reasoning):")
    grand = sub.groupby("condition")["accuracy"].agg(["mean", "std", "count"])
    grand = grand.reindex([c for c in col_order if c in grand.index])
    print(grand.round(1).to_string())

    gaps = perturbation_accuracy_gap(sub)
    if len(gaps):
        print(f"\n  Mean accuracy Δ from SAE baseline ({reasoning} reasoning):")
        gap_summary = gaps.groupby("condition")["delta"].agg(["mean", "std", "count"])
        gap_summary.columns = ["mean_delta", "std_delta", "n"]
        gap_summary = gap_summary.reindex([c for c in col_order if c in gap_summary.index])
        print(gap_summary.round(1).to_string())


############
# PLOTTING #
############

def plot_ce_gap_by_model(redial, out_dir):
    """Figure: AAVE vs SAE input CE per model (mirrors the paper's Fig. on
    cross-entropy loss). One panel per family, instruct + base overlaid.
    """
    gap = redial_ce_gap(redial).reset_index()
    if len(gap) == 0:
        return
    families = [f for f in FAMILY_ORDER if f in gap["family"].unique()]
    fig, axes = plt.subplots(1, len(families), figsize=(4 * len(families), 3.3),
                             sharey=True)
    if len(families) == 1:
        axes = [axes]
    for ax, fam in zip(axes, families):
        sub = gap[gap["family"] == fam].sort_values(["variant", "size_b"])
        for variant, col in [("base", "#4C72B0"), ("instruct", "#DD8452")]:
            s = sub[sub["variant"] == variant]
            if len(s) == 0:
                continue
            ax.plot(s["size_b"], s["input_ce_sae"], "o-", label=f"{variant} SAE",
                    color=col, alpha=0.5)
            ax.plot(s["size_b"], s["input_ce_aave"], "s--", label=f"{variant} AAVE",
                    color=col)
        ax.set_xscale("log")
        ax.set_title(fam)
        ax.set_xlabel("params (B)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("input CE (nats, per token)")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Input cross-entropy on ReDial: AAVE is consistently higher than SAE")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "redial_ce_by_model.png"))
    plt.close(fig)


def plot_gap_distribution(redial, out_dir):
    """Distribution of per-(model, task, reasoning) AAVE-SAE CE gaps."""
    paired = redial_ce_gap_paired(redial)
    if len(paired) == 0:
        return
    paired = paired.merge(
        redial[["model", "variant", "family"]].drop_duplicates(), on="model",
    )
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    for variant, col in [("base", "#4C72B0"), ("instruct", "#DD8452")]:
        v = paired[paired["variant"] == variant]["mean_gap_nats"]
        if len(v) == 0:
            continue
        ax.hist(v, bins=30, alpha=0.55, label=f"{variant} (n={len(v)})", color=col)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("mean CE gap (AAVE − SAE), nats")
    ax.set_ylabel("# (model, task, reasoning)")
    ax.set_title("Paired AAVE−SAE CE gap on ReDial — mostly positive")
    ax.legend()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "redial_gap_histogram.png"))
    plt.close(fig)


def plot_gap_vs_scale(redial, out_dir):
    """AAVE-SAE CE gap as a function of model parameters."""
    gap, corr = ce_gap_vs_scale(redial)
    if len(gap) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    markers = {"base": "o", "instruct": "s"}
    colors = {"llama": "#4C72B0", "gemma": "#DD8452", "qwen": "#55A868"}
    for (family, variant), g in gap.groupby(["family", "variant"]):
        if g["size_b"].isna().all():
            continue
        g = g.sort_values("size_b")
        label = f"{family} {variant}"
        key = (family, variant)
        if key in corr:
            label += f"  (ρ={corr[key]['spearman_r']:.2f})"
        ax.plot(g["size_b"], g["gap_nats"], markers.get(variant, "o") + "-",
                color=colors.get(family, "k"),
                alpha=0.55 if variant == "base" else 1.0,
                label=label)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("params (B, log)")
    ax.set_ylabel("AAVE − SAE CE gap (nats)")
    ax.set_title("Does the gap shrink with scale?")
    ax.legend(fontsize=7)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, "redial_gap_vs_scale.png"))
    plt.close(fig)


def plot_perturbation_accuracy(acc_df, out_dir, reasoning="naive"):
    """Bar chart: mean accuracy per condition (SAE, AAVE, perturbations)."""
    sub = acc_df[acc_df["reasoning"] == reasoning]
    if len(sub) == 0:
        return
    cond_order = ["SAE", "AAVE"] + sorted([c for c in sub["condition"].unique() if c.startswith("SAE+")])
    means = sub.groupby("condition")["accuracy"].mean().reindex(cond_order).dropna()
    sems = sub.groupby("condition")["accuracy"].sem().reindex(means.index)

    colors = ["#4C72B0", "#C44E52"] + ["#8C8C8C"] * (len(means) - 2)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(range(len(means)), means.values, yerr=sems.values, capsize=3,
                  color=colors[:len(means)], edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=25, ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_title(f"SAE vs AAVE vs SAE+perturbation accuracy ({reasoning})")
    ax.grid(axis="y", alpha=0.3)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"perturbation_accuracy_{reasoning}.png"))
    plt.close(fig)


########
# MAIN #
########

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default")
    parser.add_argument("--experiments-dir", default=None)
    parser.add_argument("--plots-dir", default="analysis/plots/logits")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.experiments_dir is None:
        cfg = dialecttax.utils.load_config(args.config)
        args.experiments_dir = cfg["directories"]["experiments"]

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    print(f"Loading from {args.experiments_dir} ...")
    redial = _load_redial(args.experiments_dir)
    multivalue = _load_flat_dataset(args.experiments_dir, "multivalue")
    parallelaave = _load_flat_dataset(args.experiments_dir, "parallelaave")
    print(f"  redial:        {len(redial):>7} rows  ({redial['model'].nunique() if len(redial) else 0} models)")
    print(f"  multivalue:    {len(multivalue):>7} rows  ({multivalue['model'].nunique() if len(multivalue) else 0} models)")
    print(f"  parallelaave:  {len(parallelaave):>7} rows  ({parallelaave['model'].nunique() if len(parallelaave) else 0} models)")

    if len(redial):
        print("\n" + "=" * 72)
        print(" §4.1  CE GAP (AAVE − SAE) PER MODEL ON ReDial")
        print("=" * 72)
        print(redial_ce_gap(redial))

        print("\n  Prose summary (for abstract / §4.1):")
        for variant, s in redial_ce_gap_overall(redial).items():
            print(f"    {variant:>8}  n={s['n_combos']:>3}  "
                  f"mean={s['mean_gap_nats']:+.3f} nats  "
                  f"median={s['median_gap_nats']:+.3f}  "
                  f"frac>0={s['frac_positive']:.0%}  "
                  f"frac p<.001={s['frac_sig_p_001']:.0%}")

        print("\n" + "=" * 72)
        print(" §4.1  INDEPENDENCE FROM COMPETENCE (gap on solved vs unsolved)")
        print("=" * 72)
        print(ce_gap_vs_correctness(redial))

        print("\n" + "=" * 72)
        print(" §3  CE GAP VS MODEL SCALE")
        print("=" * 72)
        gap, corr = ce_gap_vs_scale(redial)
        print(gap[["model", "family", "size_b", "variant", "gap_nats"]])
        print("\n  Spearman(gap, params) per family:")
        for k, v in corr.items():
            print(f"    {k[0]:>6} {k[1]:>8}  n_sizes={v['n_sizes']}  "
                  f"ρ={v['spearman_r']:+.2f}  p={v['p']:.3f}")

    if len(multivalue):
        print("\n" + "=" * 72)
        print(" §2  DIALECT RANKING ON MultiVALUE (CE, lower = easier)")
        print("    paper's token-length order: Appalachian > AAVE > Chicano > SAE > Indian > Singapore")
        print("=" * 72)
        print(multivalue_dialect_ranking(multivalue))

        print("\n  Paired per-model vs-SAE gaps (one row per model × non-SAE dialect):")
        print(multivalue_paired_gaps(multivalue).groupby("dialect")[
            ["mean_gap_nats", "median_gap_nats"]
        ].agg(["mean", "median", "count"]).round(4))

    if len(parallelaave):
        pa_rank = parallelaave.groupby("dialect")["ce"].mean()
        print("\n" + "=" * 72)
        print(" ParallelAAVE: pooled mean CE by dialect")
        print("=" * 72)
        print(pa_rank.round(4))

    perturbations = _load_redial_perturbations(args.experiments_dir)
    print(f"\n  perturbations: {len(perturbations):>7} rows  ({perturbations['model'].nunique() if len(perturbations) else 0} models)")

    if len(redial) and len(perturbations):
        print("\n" + "=" * 72)
        print(" PERTURBATION ACCURACY: SAE vs AAVE vs SAE+perturbations")
        print("=" * 72)
        acc = perturbation_accuracy(redial, perturbations)
        perturbation_accuracy_summary(acc, reasoning="naive")

    if not args.skip_plots and len(redial):
        print(f"\nSaving plots to {args.plots_dir}/ ...")
        plot_ce_gap_by_model(redial, args.plots_dir)
        plot_gap_distribution(redial, args.plots_dir)
        plot_gap_vs_scale(redial, args.plots_dir)
        print(f"Done.")


if __name__ == "__main__":
    main()
