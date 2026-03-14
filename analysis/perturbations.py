"""
Perturbation analysis for SAE-text perturbations on ReDial.

For each (model, task, reasoning, perturbation), we report:

  1. Accuracy — reuses :mod:`analysis.logits` summary, run for both
     ``naive`` and ``cot`` reasoning.
  2. Paired Δ in input/gen log-prob and entropy versus unperturbed SAE,
     joined per ``unique_id``.
  3. The unperturbed AAVE − SAE Δ as an anchor — the "tokenization tax"
     baseline against which perturbation magnitudes are compared.

Perturbations covered (driven by the run script):
    swap-0.05, capitalize-random, capitalize-alternating

Usage:
    python analysis/perturbations.py
    python analysis/perturbations.py --experiments-dir /path/to/experiments
    python analysis/perturbations.py --skip-plots
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import dialecttax.utils

# Reuse loaders / accuracy helpers from the sibling logits module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logits import (  # noqa: E402
    DIALECT_ORDER,
    _load_flat_dataset,
    _load_flat_perturbations,
    _load_redial,
    _load_redial_perturbations,
    perturbation_accuracy,
    perturbation_accuracy_summary,
    plot_perturbation_accuracy,
)
from embeddings import (  # noqa: E402
    DIALECT_LABELS,
    PERTURBATION_LABELS,
    TRANSLATION_LABELS,
    compute_cross_dialect_similarity,
    compute_perturbation_similarity_sae,
)


plt.rcParams.update({
    "figure.dpi": 150,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.constrained_layout.use": True,
})


METRICS_NAIVE = ["input_mean_log_prob", "input_mean_entropy",
                 "gen_mean_log_prob", "gen_mean_entropy"]
METRICS_COT = ["input_mean_log_prob", "input_mean_entropy", "answer_entropy"]

# Friendly labels for plotting / tables
METRIC_LABELS = {
    "d_input_mean_log_prob":  "Δ input log-prob",
    "d_input_mean_entropy":   "Δ input entropy",
    "d_gen_mean_log_prob":    "Δ gen log-prob",
    "d_gen_mean_entropy":     "Δ gen entropy",
    "d_answer_entropy":       "Δ answer entropy",
}


############
# Δ TABLES #
############

def _paired_diff(merged, col):
    """Return (mean, p) for ``perturbed - sae`` on shared rows, or (nan, nan)."""
    a = merged[f"{col}_pert"]
    b = merged[f"{col}_sae"]
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return float("nan"), float("nan")
    diff = (a[mask] - b[mask]).values
    _, p = stats.ttest_rel(a[mask], b[mask])
    return float(diff.mean()), float(p)


def metric_shifts(redial, perturbations):
    """Per (model, task, reasoning, perturbation) paired Δ vs unperturbed SAE.

    Args:
        redial: Output of :func:`logits._load_redial` (baseline rows).
        perturbations: Output of :func:`logits._load_redial_perturbations`.

    Returns:
        Long-form DataFrame with one row per group. Columns include the
        grouping keys, ``n``, and ``d_<metric>`` / ``d_<metric>_p`` pairs
        for whichever metric columns were available on both sides.
    """
    sae = redial[redial["dialect"] == "sae"]
    rows = []
    for (model, task, reasoning, pert), pg in perturbations.groupby(
        ["model", "task", "reasoning", "perturbation"]
    ):
        bs = sae[(sae["model"] == model) & (sae["task"] == task) & (sae["reasoning"] == reasoning)]
        if len(bs) == 0:
            continue
        merged = pg.merge(bs, on="unique_id", suffixes=("_pert", "_sae"))
        if len(merged) == 0:
            continue
        row = {"model": model, "task": task, "reasoning": reasoning,
               "perturbation": pert, "n": len(merged)}
        cols = METRICS_NAIVE if reasoning == "naive" else METRICS_COT
        for col in cols:
            if f"{col}_pert" in merged.columns and f"{col}_sae" in merged.columns:
                m, p = _paired_diff(merged, col)
                row[f"d_{col}"] = m
                row[f"d_{col}_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def aave_shifts(redial):
    """Anchor: paired Δ for AAVE − SAE on the same metrics.

    The ReDial unique_id convention encodes dialect as ``..._original-i`` for
    SAE and ``..._aave-i`` for AAVE; pair by stripping that marker.
    """
    sae = redial[redial["dialect"] == "sae"].copy()
    aave = redial[redial["dialect"] == "aave"].copy()
    sae["pair_id"] = sae["unique_id"].str.replace("_original-", "-", regex=False)
    aave["pair_id"] = aave["unique_id"].str.replace("_aave-", "-", regex=False)

    rows = []
    for (model, task, reasoning), ag in aave.groupby(["model", "task", "reasoning"]):
        bs = sae[(sae["model"] == model) & (sae["task"] == task) & (sae["reasoning"] == reasoning)]
        if len(bs) == 0:
            continue
        merged = ag.merge(bs, on="pair_id", suffixes=("_pert", "_sae"))
        if len(merged) == 0:
            continue
        row = {"model": model, "task": task, "reasoning": reasoning,
               "perturbation": "AAVE", "n": len(merged)}
        cols = METRICS_NAIVE if reasoning == "naive" else METRICS_COT
        for col in cols:
            if f"{col}_pert" in merged.columns and f"{col}_sae" in merged.columns:
                m, p = _paired_diff(merged, col)
                row[f"d_{col}"] = m
                row[f"d_{col}_p"] = p
        rows.append(row)
    return pd.DataFrame(rows)


def shifts_summary(shifts):
    """Mean Δ across (model × task) per (reasoning, perturbation)."""
    metric_cols = [c for c in shifts.columns if c.startswith("d_") and not c.endswith("_p")]
    out = shifts.groupby(["reasoning", "perturbation"])[metric_cols].mean()
    return out.round(4)


def shifts_summary_by_variant(shifts):
    """Same as :func:`shifts_summary` but split by base/instruct variant.

    Reproduces the per-variant means cited in the paper's calibration
    paragraphs (§5.1 base, §5.3 instruct).
    """
    metric_cols = [c for c in shifts.columns if c.startswith("d_") and not c.endswith("_p")]
    df = shifts.copy()
    df["variant"] = df["model"].str.extract(r"_(base|instruct)$", expand=False)
    out = df.groupby(["variant", "reasoning", "perturbation"])[metric_cols].mean()
    return out.round(4)


def accuracy_summary_by_variant(acc_df):
    """Per-variant accuracy Δ from SAE baseline (mean across model × task)."""
    from logits import perturbation_accuracy_gap

    gaps = perturbation_accuracy_gap(acc_df)
    if len(gaps) == 0:
        return pd.DataFrame()
    gaps = gaps.copy()
    gaps["variant"] = gaps["model"].str.extract(r"_(base|instruct)$", expand=False)
    out = gaps.groupby(["variant", "reasoning", "condition"])["delta"].agg(["mean", "std", "count"])
    return out.round(2)


##############
# PERPLEXITY #
##############

def _ppl_corpus(df):
    """Token-weighted corpus perplexity from per-sample mean log-prob.

    input_mean_log_prob is averaged over (n_prompt_tokens - 1) positions per
    sample, so the corpus mean is a token-weighted average of those means.
    """
    df = df.dropna(subset=["input_mean_log_prob", "n_prompt_tokens"])
    df = df[df["n_prompt_tokens"] > 1]
    if len(df) == 0:
        return float("nan")
    weights = df["n_prompt_tokens"].astype(float) - 1.0
    sum_lp = (df["input_mean_log_prob"] * weights).sum()
    return float(np.exp(-sum_lp / weights.sum()))


def perplexity_table(redial, perturbations):
    """Corpus perplexity per (model, reasoning, condition).

    Condition is one of: SAE, AAVE, SAE+<perturbation>. Perplexity is computed
    over input (prompt) tokens — prompts include the fixed system scaffolding
    plus the dialect-varying task text, so absolute values mix both.
    """
    rows = []
    for (model, reasoning), g in redial.groupby(["model", "reasoning"]):
        for dialect in ["sae", "aave"]:
            sub = g[g["dialect"] == dialect]
            if len(sub) == 0:
                continue
            rows.append({
                "model": model, "reasoning": reasoning,
                "condition": "SAE" if dialect == "sae" else "AAVE",
                "perplexity": _ppl_corpus(sub),
                "n_samples": len(sub),
            })
    if len(perturbations):
        for (model, reasoning, pert), g in perturbations.groupby(
            ["model", "reasoning", "perturbation"]
        ):
            rows.append({
                "model": model, "reasoning": reasoning,
                "condition": f"SAE+{pert}",
                "perplexity": _ppl_corpus(g),
                "n_samples": len(g),
            })
    return pd.DataFrame(rows)


def perplexity_summary(ppl_df, reasoning="naive"):
    """Print per-model and grand-mean perplexity for a given reasoning mode."""
    sub = ppl_df[ppl_df["reasoning"] == reasoning]
    if len(sub) == 0:
        return
    cond_order = ["SAE", "AAVE"] + sorted([c for c in sub["condition"].unique() if c.startswith("SAE+")])
    cond_order = [c for c in cond_order if c in sub["condition"].unique()]

    pivot = sub.pivot_table(index="model", columns="condition",
                            values="perplexity", aggfunc="first")[cond_order]
    print(f"\n  Per-model corpus perplexity ({reasoning}):")
    print(pivot.round(2).to_string())

    print(f"\n  Geometric-mean perplexity across models ({reasoning}):")
    # Geometric mean = exp(mean(log(ppl))) — the right way to aggregate ratios
    gm = pivot.apply(lambda c: float(np.exp(np.log(c.dropna()).mean())))
    print(gm.round(3).to_string())

    if "SAE" in pivot.columns:
        ratio = pivot.div(pivot["SAE"], axis=0)
        print(f"\n  Per-model perplexity ratio vs SAE ({reasoning}):")
        print(ratio.round(3).to_string())
        print(f"\n  Geometric-mean ratio vs SAE ({reasoning}):")
        gmr = ratio.apply(lambda c: float(np.exp(np.log(c.dropna()).mean())))
        print(gmr.round(3).to_string())


##############################
# §2  DIALECT PERPLEXITY     #
##############################
# MultiVALUE / ParallelAAVE carry parallel SAE-vs-dialect texts (no QA), so the
# dialect "tax" can be read directly as a per-token perplexity ratio. Perturbations
# and translations of these same SAE texts exist only as embeddings (for the §2
# similarity figure), not as LM log-probs, so they cannot be compared here.

def perplexity_by_dialect(flat):
    """Pooled per-token perplexity by dialect for a flat dataset (MultiVALUE / ParallelAAVE).

    Perplexity is exp of the mean per-token cross-entropy, pooled across samples
    and models. The ratio column expresses each dialect's perplexity relative to
    SAE: the "higher tax" the model levies on a semantically-equivalent surface
    form (§2). Values > 1 mean the transformed text costs more per token.

    Args:
        flat: DataFrame from logits._load_flat_dataset (carries the `ce` column).

    Returns:
        DataFrame indexed by dialect with columns: n, mean_ce_nats, ppl,
        ppl_ratio_vs_sae.
    """
    g = flat.groupby("dialect")["ce"].agg(["mean", "count"])
    g["ppl"] = np.exp(g["mean"])
    sae_ppl = g.loc["sae", "ppl"] if "sae" in g.index else np.nan
    g["ppl_ratio_vs_sae"] = g["ppl"] / sae_ppl
    g = g.rename(columns={"mean": "mean_ce_nats", "count": "n"})
    keep = [d for d in DIALECT_ORDER if d in g.index]
    return g.reindex(keep)[["n", "mean_ce_nats", "ppl", "ppl_ratio_vs_sae"]].round(4)


def paired_perplexity_ratio(flat):
    """Paired transformed-vs-SAE perplexity ratio, one row per model × dialect.

    Within each model, samples are matched by unique_id (the shared source
    document). The per-sample perplexity ratio is exp(CE_dialect - CE_sae), and a
    Wilcoxon signed-rank test is run on the per-sample CE differences. A ratio
    above 1 means the model assigns higher perplexity to the semantically-
    equivalent transformed surface form (§2).

    Args:
        flat: DataFrame from logits._load_flat_dataset.

    Returns:
        DataFrame with columns: model, dialect, n, median_ppl_ratio,
        mean_ppl_ratio, frac_higher, wilcoxon_p.
    """
    non_sae = [d for d in DIALECT_ORDER if d != "sae" and d in flat["dialect"].unique()]
    rows = []
    for model, g in flat.groupby("model"):
        wide = g.pivot_table(index="unique_id", columns="dialect", values="ce", aggfunc="first")
        if "sae" not in wide.columns:
            continue
        for d in non_sae:
            if d not in wide.columns:
                continue
            sub = wide[["sae", d]].dropna()
            if len(sub) < 20:
                continue
            diff = (sub[d] - sub["sae"]).values
            ratio = np.exp(diff)
            try:
                _, p = stats.wilcoxon(diff)
            except ValueError:
                p = np.nan
            rows.append({
                "model": model, "dialect": d, "n": len(sub),
                "median_ppl_ratio": float(np.median(ratio)),
                "mean_ppl_ratio": float(ratio.mean()),
                "frac_higher": float((ratio > 1).mean()),
                "wilcoxon_p": p,
            })
    return pd.DataFrame(rows).round(4)


def summarize_paired_perplexity(ratio_df):
    """Aggregate paired_perplexity_ratio across models, one row per dialect."""
    if len(ratio_df) == 0:
        return pd.DataFrame()
    agg = ratio_df.groupby("dialect").agg(
        median_ppl_ratio=("median_ppl_ratio", "median"),
        mean_ppl_ratio=("mean_ppl_ratio", "mean"),
        frac_higher=("frac_higher", "mean"),
        n_models=("model", "nunique"),
        frac_models_sig=("wilcoxon_p", lambda s: float((s < 0.001).mean())),
    )
    keep = [d for d in DIALECT_ORDER if d in agg.index]
    return agg.reindex(keep).round(4)


def _condition_type(cond):
    """Classify a flat-dataset condition into dialect / perturbation / translation."""
    if cond.startswith("translate-"):
        return "translation"
    if cond in DIALECT_ORDER:
        return "dialect"
    return "perturbation"


def redial_perplexity_comparison(redial, perturbations):
    """Paired per-sample input-perplexity ratio vs SAE on ReDial.

    Same construction as :func:`transform_perplexity_comparison` but on the
    QA-style ReDial corpus: AAVE pairs to SAE by stripping the dialect marker
    in ``unique_id``; each perturbed-SAE row pairs to SAE by ``unique_id``.
    Per-sample ratio = exp(input_ce_cond - input_ce_sae), pooled within a model
    across (task, reasoning), then aggregated across models.

    Returns:
        DataFrame with columns: condition, median_ppl_ratio, frac_higher,
        n_models — one row per condition, sorted by median descending.
    """
    sae = redial[redial["dialect"] == "sae"].copy()
    aave = redial[redial["dialect"] == "aave"].copy()
    sae["pair_id"] = sae["unique_id"].str.replace("_original-", "-", regex=False)
    aave["pair_id"] = aave["unique_id"].str.replace("_aave-", "-", regex=False)

    rows = []
    # AAVE — paired by stripped id
    for (model, task, reasoning), ag in aave.groupby(["model", "task", "reasoning"]):
        bs = sae[(sae["model"] == model) & (sae["task"] == task) & (sae["reasoning"] == reasoning)]
        merged = ag.merge(bs, on="pair_id", suffixes=("_a", "_s"))
        if len(merged) == 0:
            continue
        diff = merged["input_ce_a"] - merged["input_ce_s"]
        diff = diff.dropna()
        if len(diff) < 20:
            continue
        ratio = np.exp(diff)
        rows.append({"condition": "AAVE", "model": model,
                     "median_ratio": float(np.median(ratio)),
                     "frac_higher": float((ratio > 1).mean())})

    # Perturbations — paired by unique_id (perturbations carry the SAE unique_id)
    if len(perturbations):
        for (model, task, reasoning, pert), pg in perturbations.groupby(
            ["model", "task", "reasoning", "perturbation"]
        ):
            bs = sae[(sae["model"] == model) & (sae["task"] == task) & (sae["reasoning"] == reasoning)]
            merged = pg.merge(bs, on="unique_id", suffixes=("_p", "_s"))
            if len(merged) == 0:
                continue
            diff = merged["input_ce_p"] - merged["input_ce_s"]
            diff = diff.dropna()
            if len(diff) < 20:
                continue
            ratio = np.exp(diff)
            rows.append({"condition": pert, "model": model,
                         "median_ratio": float(np.median(ratio)),
                         "frac_higher": float((ratio > 1).mean())})

    per = pd.DataFrame(rows)
    if len(per) == 0:
        return pd.DataFrame()
    # Pool the per-(model, task, reasoning) cells: collapse to per-model medians
    agg = per.groupby(["condition", "model"]).agg(
        median_ratio=("median_ratio", "median"),
        frac_higher=("frac_higher", "mean"),
    ).reset_index()
    out = agg.groupby("condition").agg(
        median_ppl_ratio=("median_ratio", "median"),
        frac_higher=("frac_higher", "mean"),
        n_models=("model", "nunique"),
    ).reset_index()
    return out.sort_values("median_ppl_ratio", ascending=False).round(4)


def transform_perplexity_comparison(dialects, transforms):
    """Paired per-token perplexity ratio vs SAE for dialects, perturbations, and
    translations on a flat dataset (MultiVALUE / ParallelAAVE).

    Puts the dialect tax on the same axis as the §2 calibration transforms: each
    non-SAE dialect, each character perturbation, and each translation is paired
    to the SAE baseline by unique_id within a model (ratio = exp(CE_cond - CE_sae)),
    then aggregated across models. A ratio > 1 means the model is more surprised by
    that surface form than by SAE, even though all three preserve meaning.

    Args:
        dialects: logits._load_flat_dataset output (has `dialect`, `ce`, `unique_id`).
        transforms: logits._load_flat_perturbations output (SAE under each transform).

    Returns:
        DataFrame with columns: condition, type, median_ppl_ratio, frac_higher,
        n_models — one row per condition, sorted by median_ppl_ratio descending.
    """
    sae = (dialects[dialects["dialect"] == "sae"][["model", "unique_id", "ce"]]
           .rename(columns={"ce": "ce_sae"}))
    dia = (dialects[dialects["dialect"] != "sae"][["model", "unique_id", "dialect", "ce"]]
           .rename(columns={"dialect": "condition"}))
    cols = ["model", "unique_id", "perturbation", "ce"]
    tr = (transforms[cols].rename(columns={"perturbation": "condition"})
          if len(transforms) else pd.DataFrame(columns=["model", "unique_id", "condition", "ce"]))
    var = pd.concat([dia, tr], ignore_index=True)
    if len(var) == 0:
        return pd.DataFrame()

    merged = var.merge(sae, on=["model", "unique_id"], how="inner")
    # concat with an empty transforms frame can upcast `ce` to object dtype
    diff = pd.to_numeric(merged["ce"], errors="coerce") - pd.to_numeric(merged["ce_sae"], errors="coerce")
    merged = merged.assign(ratio=np.exp(diff)).dropna(subset=["ratio"])
    rows = []
    for (cond, model), g in merged.groupby(["condition", "model"]):
        if len(g) < 20:
            continue
        rows.append({"condition": cond, "model": model,
                     "median_ratio": float(np.median(g["ratio"])),
                     "frac_higher": float((g["ratio"] > 1).mean())})
    per = pd.DataFrame(rows)
    if len(per) == 0:
        return pd.DataFrame()
    agg = per.groupby("condition").agg(
        median_ppl_ratio=("median_ratio", "median"),
        frac_higher=("frac_higher", "mean"),
        n_models=("model", "nunique"),
    ).reset_index()
    agg["type"] = agg["condition"].map(_condition_type)
    return agg.sort_values("median_ppl_ratio", ascending=False).round(4)[
        ["condition", "type", "median_ppl_ratio", "frac_higher", "n_models"]
    ]


def _condition_to_embedding_label(cond):
    """Map a perplexity-table condition key to an embeddings.py label."""
    if cond in DIALECT_LABELS:
        return f"SAE → {DIALECT_LABELS[cond]}"
    if cond in PERTURBATION_LABELS:
        return PERTURBATION_LABELS[cond]
    if cond in TRANSLATION_LABELS:
        return TRANSLATION_LABELS[cond]
    return cond


def similarity_vs_perplexity_correlation(emb_dir, dataset, dialects, transforms, dim=768):
    """Verify that high dialect-SAE cosine similarity is not driven by surface
    predictability (LM input perplexity).

    For each transformation (dialect / character perturbation / translation),
    pairs the mean EmbeddingGemma cosine similarity from §2 to the paired
    per-token LM perplexity ratio from :func:`transform_perplexity_comparison`,
    then reports the Spearman rank correlation between similarity and log
    perplexity ratio. A non-negative correlation rules out the "easier text is
    more similar" confound: that confound would predict ρ < 0.

    Args:
        emb_dir: Root generate_embeddings directory (used by embeddings.py).
        dataset: "multivalue" or "parallelaave".
        dialects: logits._load_flat_dataset output for ``dataset``.
        transforms: logits._load_flat_perturbations output for ``dataset``.
        dim: Embedding dimension to use (default 768, the largest Matryoshka head).

    Returns:
        (merged DataFrame, spearman_rho, spearman_p). The DataFrame has columns
        label, type, mean_sim, median_ppl_ratio, sorted by ppl ratio descending.
    """
    cross = compute_cross_dialect_similarity(emb_dir, dataset)
    pert = compute_perturbation_similarity_sae(emb_dir, dataset)
    sim = pd.concat([cross, pert], ignore_index=True)
    sim = sim[sim["dim"] == dim][["label", "type", "mean_sim"]]

    ratios = transform_perplexity_comparison(dialects, transforms)
    if len(sim) == 0 or len(ratios) == 0:
        return pd.DataFrame(), float("nan"), float("nan")
    ratios = ratios.assign(label=ratios["condition"].map(_condition_to_embedding_label))

    merged = sim.merge(ratios[["label", "median_ppl_ratio"]], on="label")
    if len(merged) < 3:
        return merged, float("nan"), float("nan")
    rho, p = stats.spearmanr(merged["mean_sim"], np.log(merged["median_ppl_ratio"]))
    return (merged.sort_values("median_ppl_ratio", ascending=False).reset_index(drop=True),
            float(rho), float(p))


############
# PLOTTING #
############


########
# MAIN #
########

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default")
    parser.add_argument("--experiments-dir", default=None)
    parser.add_argument("--plots-dir", default="analysis/plots/perturbations")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    experiments_dir = args.experiments_dir
    if experiments_dir is None:
        cfg = dialecttax.utils.load_config(args.config)
        experiments_dir = cfg["directories"]["experiments"]

    redial = _load_redial(experiments_dir)
    perturbations = _load_redial_perturbations(experiments_dir)
    multivalue = _load_flat_dataset(experiments_dir, "multivalue")
    parallelaave = _load_flat_dataset(experiments_dir, "parallelaave")
    print(f"redial:        {len(redial):>7} rows  ({redial['model'].nunique()} models)")
    print(f"perturbations: {len(perturbations):>7} rows  ({perturbations['model'].nunique()} models, "
          f"{perturbations['perturbation'].nunique()} perturbations)")
    print(f"multivalue:    {len(multivalue):>7} rows  ({multivalue['model'].nunique() if len(multivalue) else 0} models)")
    print(f"parallelaave:  {len(parallelaave):>7} rows  ({parallelaave['model'].nunique() if len(parallelaave) else 0} models)")

    ###############
    # 1. ACCURACY #
    ###############

    acc = perturbation_accuracy(redial, perturbations)
    for reasoning in ["naive", "cot"]:
        print("\n" + "=" * 72)
        print(f" ACCURACY — {reasoning}")
        print("=" * 72)
        perturbation_accuracy_summary(acc, reasoning=reasoning)

    ################
    # 2. PERPLEXITY #
    ################

    ppl = perplexity_table(redial, perturbations)
    for reasoning in ["naive", "cot"]:
        print("\n" + "=" * 72)
        print(f" PERPLEXITY (ReDial: SAE vs AAVE vs SAE+perturbation) — {reasoning}")
        print("=" * 72)
        perplexity_summary(ppl, reasoning=reasoning)

    # §2 dialect perplexity on the parallel SAE-vs-dialect corpora (no perturbation
    # or translation log-probs exist for these datasets, only dialects).
    for name, flat in [("MultiVALUE", multivalue), ("ParallelAAVE", parallelaave)]:
        if len(flat) == 0:
            continue
        print("\n" + "=" * 72)
        print(f" §2  {name}: transformed-vs-SAE PERPLEXITY (same meaning, higher tax)")
        print("=" * 72)
        print(perplexity_by_dialect(flat))
        print("\n  Paired transformed/SAE perplexity ratio (aggregated across models):")
        print(summarize_paired_perplexity(paired_perplexity_ratio(flat)))

        transforms = _load_flat_perturbations(experiments_dir, name.lower())
        cmp = transform_perplexity_comparison(flat, transforms)
        print("\n  Dialects vs perturbations vs translations "
              "(paired per-token ppl ratio vs SAE, median across models):")
        if len(cmp):
            print(cmp.to_string(index=False))
        else:
            print("    (no transform logits yet — run "
                  "scripts/logits/generate_logits_transforms_{small,large}.sh)")

        emb_dir = os.path.join(experiments_dir, "generate_embeddings")
        if os.path.isdir(os.path.join(emb_dir, name.lower())):
            merged, rho, p = similarity_vs_perplexity_correlation(
                emb_dir, name.lower(), flat, transforms
            )
            if len(merged):
                print("\n  Semantic equivalence vs LM perplexity (§2 confound check):")
                print(merged.to_string(index=False))
                print(f"\n  Spearman(mean_sim, log ppl_ratio): "
                      f"ρ={rho:+.3f}  p={p:.3f}  n={len(merged)}  "
                      f"(ρ < 0 would indicate the surface-predictability confound)")

    ##############################
    # 3. LOG-PROB / ENTROPY Δ    #
    ##############################

    shifts = pd.concat([metric_shifts(redial, perturbations),
                        aave_shifts(redial)], ignore_index=True)
    summary = shifts_summary(shifts)
    for reasoning in ["naive", "cot"]:
        print("\n" + "=" * 72)
        print(f" Δ vs unperturbed SAE (mean across model × task) — {reasoning}")
        print("=" * 72)
        if reasoning in summary.index.get_level_values("reasoning"):
            print(summary.xs(reasoning, level="reasoning").to_string())

    #########################
    # 3b. BASE vs INSTRUCT  #
    #########################

    # Cited in paper §5.1 (base, gradient calibration) and §5.3 (instruct logits).
    variant_summary = shifts_summary_by_variant(shifts)
    for variant in ["base", "instruct"]:
        if variant not in variant_summary.index.get_level_values("variant"):
            continue
        for reasoning in ["naive", "cot"]:
            sub = variant_summary.xs(variant, level="variant")
            if reasoning not in sub.index.get_level_values("reasoning"):
                continue
            print("\n" + "=" * 72)
            print(f" Δ vs unperturbed SAE — {variant.upper()} × {reasoning}")
            print("=" * 72)
            print(sub.xs(reasoning, level="reasoning").to_string())

    acc_variant = accuracy_summary_by_variant(acc)
    for variant in ["base", "instruct"]:
        if len(acc_variant) and variant in acc_variant.index.get_level_values("variant"):
            for reasoning in ["naive", "cot"]:
                sub = acc_variant.xs(variant, level="variant")
                if reasoning not in sub.index.get_level_values("reasoning"):
                    continue
                print("\n" + "=" * 72)
                print(f" Δ accuracy from SAE (pp) — {variant.upper()} × {reasoning}")
                print("=" * 72)
                print(sub.xs(reasoning, level="reasoning").to_string())

    # Per-perturbation grand mean Δ input CE relative to the AAVE anchor
    print("\n" + "=" * 72)
    print(" RELATIVE STRENGTH: |Δ input CE| / |AAVE Δ input CE|")
    print("=" * 72)
    rows = []
    for reasoning, g in shifts.groupby("reasoning"):
        if "d_input_mean_log_prob" not in g.columns:
            continue
        per_pert = g.groupby("perturbation")["d_input_mean_log_prob"].mean().mul(-1)
        if "AAVE" not in per_pert.index:
            continue
        anchor = per_pert["AAVE"]
        for pert, val in per_pert.items():
            rows.append({"reasoning": reasoning, "perturbation": pert,
                         "delta_input_ce": float(val),
                         "vs_aave": float(val / anchor) if anchor else float("nan")})
    rel = pd.DataFrame(rows).round(3)
    if len(rel):
        print(rel.to_string(index=False))

    #############
    # 3. PLOTS  #
    #############

    if args.skip_plots:
        return
    print(f"\nSaving plots to {args.plots_dir}/ ...")
    for reasoning in ["naive", "cot"]:
        plot_perturbation_accuracy(acc, args.plots_dir, reasoning=reasoning)
    print("Done.")


if __name__ == "__main__":
    main()
