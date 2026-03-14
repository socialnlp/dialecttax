"""
Preliminary comparison of character-level vs canonical (logits) tokenization.

Loads metadata.jsonl from generate_characters and generate_logits experiments
and compares accuracy, answer agreement, and input metrics across matched
(model, task, reasoning, dialect) combinations.

Usage:
    python analysis/characters.py
    python analysis/characters.py --experiments-dir /data/gemini/ellang/dialecttax/experiments
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _save_fig(fig, out_dir, name):
    """Save figure as both PNG and PDF."""
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"))
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"))

FAMILY_ORDER = ["llama", "gemma", "qwen"]
FAMILY_COLORS = {"llama": "#4C72B0", "gemma": "#DD8452", "qwen": "#55A868"}
MODEL_SIZE_RE = {
    "llama": [("1b", 1.0), ("3b", 3.0), ("8b", 8.0), ("70b", 70.0)],
    "gemma": [("1b", 1.0), ("4b", 4.0), ("12b", 12.0), ("27b", 27.0)],
    "qwen":  [("1.7b", 1.7), ("4b", 4.0), ("8b", 8.0), ("9b", 9.0), ("27b", 27.0), ("32b", 32.0)],
}
# §4 analysis runs on the 9-model 3x3 grid (3 families x 3 sizes). The
# larger instruct models (gemma_27b, llama_70b, qwen_32b) are excluded
# so all per-section numbers and figures share the same model set.
INSTRUCT_9 = {
    "llama_1b_instruct", "llama_3b_instruct", "llama_8b_instruct",
    "gemma_1b_instruct", "gemma_4b_instruct", "gemma_12b_instruct",
    "qwen_1.7b_instruct", "qwen_4b_instruct", "qwen_8b_instruct",
}


def _parse_model(name):
    """Split a model dir name like 'llama_8b_instruct' into (family, size_b)."""
    family = name.split("_")[0]
    size = None
    for tok, val in MODEL_SIZE_RE.get(family, []):
        if f"_{tok}_" in f"_{name}_":
            size = val
            break
    return family, size


##########
# LOADING
##########

def _load_metadata(experiment_dir, experiment):
    """Walk {experiment_dir}/{experiment}/{model}/redial/{task}/{reasoning}/{dialect}/metadata.jsonl.

    Args:
        experiment_dir: Root experiments directory.
        experiment: Either "generate_characters" or "generate_logits".

    Returns:
        DataFrame with per-sample rows tagged by model/task/reasoning/dialect.
    """
    root = os.path.join(experiment_dir, experiment)
    pattern = os.path.join(root, "*", "redial", "*", "*", "*", "metadata.jsonl")
    rows = []
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        # {model}/redial/{task}/{reasoning}/{dialect}/metadata.jsonl
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
                r["experiment"] = experiment
                rows.append(r)
    return pd.DataFrame(rows)


def load_merged(experiment_dir):
    """Load characters and logits metadata, return merged per-sample DataFrame.

    Args:
        experiment_dir: Root experiments directory.

    Returns:
        DataFrame with one row per sample that exists in both experiments,
        suffixed _char and _can for character- and canonical-tokenization cols.
    """
    chars = _load_metadata(experiment_dir, "generate_characters")
    logits = _load_metadata(experiment_dir, "generate_logits")
    print(f"characters: {len(chars)} rows, "
          f"{chars['model'].nunique()} models, "
          f"{chars.groupby(['model', 'task', 'reasoning', 'dialect']).ngroups} combos")
    print(f"logits:     {len(logits)} rows, "
          f"{logits['model'].nunique()} models, "
          f"{logits.groupby(['model', 'task', 'reasoning', 'dialect']).ngroups} combos")

    keys = ["model", "task", "reasoning", "dialect", "unique_id"]
    merged = chars.merge(logits, on=keys, suffixes=("_char", "_can"))
    print(f"merged:     {len(merged)} matched samples, "
          f"{merged.groupby(['model', 'task', 'reasoning', 'dialect']).ngroups} combos")
    return merged


############
# ANALYSIS
############

def accuracy_table(merged):
    """Per-combo accuracy under each tokenization plus the gap.

    Returns:
        DataFrame indexed by (model, task, reasoning, dialect).
    """
    grp = merged.groupby(["model", "task", "reasoning", "dialect"])
    out = pd.DataFrame({
        "n": grp.size(),
        "acc_char": grp["correct_char"].mean() * 100,
        "acc_can": grp["correct_can"].mean() * 100,
    })
    out["delta"] = out["acc_char"] - out["acc_can"]
    out["agree_rate"] = grp.apply(
        lambda g: (g["predicted_answer_char"] == g["predicted_answer_can"]).mean() * 100,
        include_groups=False,
    )
    return out.round(2)


def expansion_table(merged):
    """Character vs canonical token counts per combo."""
    grp = merged.groupby(["model", "task", "reasoning", "dialect"])
    out = pd.DataFrame({
        "n": grp.size(),
        "mean_char_tokens": grp["n_char_tokens"].mean(),
        "mean_canonical_tokens": grp["n_canonical_tokens"].mean(),
        "mean_expansion": grp["char_expansion"].mean(),
    })
    return out.round(2)


def input_metric_table(merged):
    """Input log-prob / entropy under char vs canonical tokenization."""
    grp = merged.groupby(["model", "task", "reasoning", "dialect"])
    out = pd.DataFrame({
        "n": grp.size(),
        "logp_char": grp["input_mean_log_prob_char"].mean(),
        "logp_can": grp["input_mean_log_prob_can"].mean(),
        "ent_char": grp["input_mean_entropy_char"].mean(),
        "ent_can": grp["input_mean_entropy_can"].mean(),
    })
    out["logp_delta"] = out["logp_char"] - out["logp_can"]
    out["ent_delta"] = out["ent_char"] - out["ent_can"]
    return out.round(3)


def _pair_id(uid):
    """Strip '_original' / '_aave' dialect marker from a unique_id."""
    return uid.replace("_original-", "-").replace("_aave-", "-")


def _drop_broken_combos(merged, min_acc=1.0):
    """Drop (model, task, reasoning, dialect) combos whose char-accuracy is <min_acc%.

    Such combos are typically answer-extraction failures, not genuine drops.
    Drops the combo from BOTH dialects so paired comparisons stay balanced.
    """
    grp = merged.groupby(["model", "task", "reasoning", "dialect"])["correct_char"].mean() * 100
    bad_combos = grp[grp < min_acc].index
    if len(bad_combos) == 0:
        return merged, set()
    # Expand to whole (model, task, reasoning) triples — drop both dialects together
    bad_triples = {(m, t, r) for m, t, r, _ in bad_combos}
    keep = ~merged.set_index(["model", "task", "reasoning"]).index.isin(bad_triples)
    return merged[keep].copy(), bad_triples


def dialect_gap_table(merged):
    """Per (model, task, reasoning): SAE-AAVE accuracy gap under each tokenization.

    Positive gap = SAE favored; negative = AAVE favored.
    abs_gap_delta = |can_gap| - |char_gap|  (positive means char REDUCES unfairness).
    """
    pivot = (
        merged.groupby(["model", "task", "reasoning", "dialect"])
        [["correct_char", "correct_can"]].mean().mul(100)
        .unstack("dialect")
    )
    if ("correct_char", "aave") not in pivot.columns:
        return pd.DataFrame()
    out = pd.DataFrame({
        "acc_can_sae": pivot[("correct_can", "sae")],
        "acc_can_aave": pivot[("correct_can", "aave")],
        "acc_char_sae": pivot[("correct_char", "sae")],
        "acc_char_aave": pivot[("correct_char", "aave")],
    })
    out["can_gap"] = out["acc_can_sae"] - out["acc_can_aave"]
    out["char_gap"] = out["acc_char_sae"] - out["acc_char_aave"]
    out["abs_gap_delta"] = out["can_gap"].abs() - out["char_gap"].abs()
    return out.round(2).dropna()


def paired_dialect_table(merged):
    """Per (model, task, reasoning, tokenization): paired SAE/AAVE correctness.

    Pairs samples by sample index across dialects and reports how often the
    model is correct on SAE only vs AAVE only (the asymmetry of errors).
    """
    df = merged.copy()
    df["pair_id"] = df["unique_id"].map(_pair_id)
    rows = []
    for (model, task, reasoning), g in df.groupby(["model", "task", "reasoning"]):
        for tok, col in [("canonical", "correct_can"), ("char", "correct_char")]:
            wide = g.pivot_table(index="pair_id", columns="dialect", values=col, aggfunc="first")
            if not {"sae", "aave"}.issubset(wide.columns):
                continue
            wide = wide.dropna()
            if len(wide) == 0:
                continue
            sae_only = ((wide["sae"] == 1) & (wide["aave"] == 0)).mean() * 100
            aave_only = ((wide["aave"] == 1) & (wide["sae"] == 0)).mean() * 100
            both = ((wide["sae"] == 1) & (wide["aave"] == 1)).mean() * 100
            neither = ((wide["sae"] == 0) & (wide["aave"] == 0)).mean() * 100
            rows.append({
                "model": model, "task": task, "reasoning": reasoning, "tok": tok,
                "n_pairs": len(wide),
                "both": both, "sae_only": sae_only, "aave_only": aave_only, "neither": neither,
                "net_sae_advantage": sae_only - aave_only,
            })
    return pd.DataFrame(rows).round(2)


def perplexity_table(merged):
    """Per-token perplexity by dialect × tokenization.

    perplexity = exp(-input_mean_log_prob). Low = confident, high = surprised.
    Note: char- and canonical-tokenization ppl are NOT directly comparable
    because the token units differ — for that, use bits_per_byte_table.
    """
    df = merged.copy()
    df["ppl_char"] = np.exp(-df["input_mean_log_prob_char"])
    df["ppl_can"] = np.exp(-df["input_mean_log_prob_can"])
    grp = df.groupby(["model", "task", "reasoning", "dialect"])
    out = pd.DataFrame({
        "n": grp.size(),
        "ppl_can": grp["ppl_can"].mean(),
        "ppl_char": grp["ppl_char"].mean(),
    })
    pivot = out.unstack("dialect")
    pivot[("ppl_can_ratio", "aave_over_sae")] = (
        pivot[("ppl_can", "aave")] / pivot[("ppl_can", "sae")]
    )
    pivot[("ppl_char_ratio", "aave_over_sae")] = (
        pivot[("ppl_char", "aave")] / pivot[("ppl_char", "sae")]
    )
    return pivot.round(3)


def per_token_entropy_table(experiment_dir):
    """Tabular form of the entropy-violin data: per-token input entropy
    distribution summary per (model, dialect × tokenization).

    One row per model. Columns report p25 / median / p75 (nats) for SAE and
    AAVE under each tokenization, plus the median SAE−AAVE gap. More compact
    and directly comparable than the violin plot.
    """
    def _agg(experiment):
        d = _load_per_token_npz(experiment_dir, experiment, "input_entropy")
        out = {}
        for (m, _t, _r, dial), arr in d.items():
            out.setdefault((m, dial), []).append(arr)
        return {k: np.concatenate(v) for k, v in out.items()}

    can = _agg("generate_logits")
    ch = _agg("generate_characters")
    models = _sort_models(sorted({m for (m, _) in list(can) + list(ch)}))
    rows = []
    for m in models:
        r = {"model": m}
        for tok_label, agg in [("can", can), ("char", ch)]:
            for dial in ["sae", "aave"]:
                a = agg.get((m, dial))
                if a is None or a.size == 0:
                    r[f"{tok_label}_{dial}_p25"] = np.nan
                    r[f"{tok_label}_{dial}_median"] = np.nan
                    r[f"{tok_label}_{dial}_p75"] = np.nan
                    r[f"{tok_label}_{dial}_n"] = 0
                else:
                    q25, q50, q75 = np.percentile(a, [25, 50, 75])
                    r[f"{tok_label}_{dial}_p25"] = q25
                    r[f"{tok_label}_{dial}_median"] = q50
                    r[f"{tok_label}_{dial}_p75"] = q75
                    r[f"{tok_label}_{dial}_n"] = int(a.size)
            s = r.get(f"{tok_label}_sae_median")
            a = r.get(f"{tok_label}_aave_median")
            r[f"{tok_label}_gap_median"] = (
                None if (s is None or a is None or np.isnan(s) or np.isnan(a))
                else a - s
            )
        rows.append(r)
    df = pd.DataFrame(rows).set_index("model")
    # Reorder: SAE block, AAVE block, gap, per tokenization
    cols = []
    for tok in ["can", "char"]:
        for dial in ["sae", "aave"]:
            cols += [f"{tok}_{dial}_p25", f"{tok}_{dial}_median", f"{tok}_{dial}_p75"]
        cols += [f"{tok}_gap_median"]
    return df[cols].round(3)


def _load_logp_arrays(experiment_dir, experiment):
    """Load input_log_probs.npz for every combo as a dict keyed by path tuple.

    Returns:
        Dict (model, task, reasoning, dialect) -> list of 1-D float arrays
        (one per sample, length = seq_len - 1).
    """
    root = os.path.join(experiment_dir, experiment)
    pattern = os.path.join(root, "*", "redial", "*", "*", "*", "input_log_probs.npz")
    out = {}
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        model, _ds, task, reasoning, dialect, _ = parts
        with np.load(path) as z:
            arrays = [z[k] for k in z.files]
        out[(model, task, reasoning, dialect)] = arrays
    return out


def bits_per_byte_table(merged, experiment_dir):
    """Bits-per-byte by dialect × tokenization — the fair cross-tokenization metric.

    For each sample, total negative log-likelihood in bits is
        bits = -sum(log_probs) / ln(2)
    Dividing by raw text byte-length gives bits-per-byte, which is invariant
    to the choice of tokenizer. We approximate byte-length from n_char_tokens
    (the char-level tokenizer emits ~1 token per char, which is ~1 byte for
    ASCII-dominant prompts).

    Lower = model finds the text more predictable. A smaller aave-vs-sae BPB
    gap under char tokenization would be evidence that char tokenization
    equalizes per-character difficulty across dialects.
    """
    char_lps = _load_logp_arrays(experiment_dir, "generate_characters")
    can_lps = _load_logp_arrays(experiment_dir, "generate_logits")

    df = merged.copy()
    # For per-sample BPB we need per-sample totals. Rebuild by combo then align.
    rows = []
    for (model, task, reasoning, dialect), g in df.groupby(
        ["model", "task", "reasoning", "dialect"]
    ):
        key = (model, task, reasoning, dialect)
        if key not in char_lps or key not in can_lps:
            continue
        ch = char_lps[key]
        cn = can_lps[key]
        # Per-sample order matches metadata.jsonl order (both were saved together).
        n_samples = min(len(ch), len(cn), len(g))
        g = g.iloc[:n_samples]
        bytes_approx = g["n_char_tokens"].values  # ~ raw char count
        bits_char = np.array([-a.sum() / np.log(2) for a in ch[:n_samples]])
        bits_can = np.array([-a.sum() / np.log(2) for a in cn[:n_samples]])
        rows.append({
            "model": model, "task": task, "reasoning": reasoning, "dialect": dialect,
            "n": n_samples,
            "bpb_can": float(np.mean(bits_can / bytes_approx)),
            "bpb_char": float(np.mean(bits_char / bytes_approx)),
        })
    out = pd.DataFrame(rows).set_index(["model", "task", "reasoning", "dialect"])
    pivot = out.unstack("dialect")
    pivot[("bpb_can", "aave_minus_sae")] = (
        pivot[("bpb_can", "aave")] - pivot[("bpb_can", "sae")]
    )
    pivot[("bpb_char", "aave_minus_sae")] = (
        pivot[("bpb_char", "aave")] - pivot[("bpb_char", "sae")]
    )
    pivot[("gap_shrink", "can_minus_char")] = (
        pivot[("bpb_can", "aave_minus_sae")] - pivot[("bpb_char", "aave_minus_sae")]
    )
    return pivot.round(4)


def dialect_input_metrics(merged):
    """Per (model, task, reasoning, dialect, tokenization): input logp/entropy.

    Lets you see whether AAVE triggers lower log-prob / higher entropy than SAE
    and whether char tokenization narrows that gap.
    """
    df = merged.copy()
    long = []
    for tok, lp, ent, ntok in [
        ("canonical", "input_mean_log_prob_can", "input_mean_entropy_can", "n_canonical_tokens"),
        ("char", "input_mean_log_prob_char", "input_mean_entropy_char", "n_char_tokens"),
    ]:
        tmp = df[["model", "task", "reasoning", "dialect", lp, ent, ntok]].copy()
        tmp.columns = ["model", "task", "reasoning", "dialect", "logp", "ent", "ntok"]
        tmp["tok"] = tok
        long.append(tmp)
    long = pd.concat(long, ignore_index=True)
    grp = long.groupby(["model", "task", "reasoning", "tok", "dialect"])
    out = grp[["logp", "ent", "ntok"]].mean().unstack("dialect")
    # SAE - AAVE deltas: SAE more favored = higher logp, lower ent, fewer tokens
    out[("logp", "sae_minus_aave")] = out[("logp", "sae")] - out[("logp", "aave")]
    out[("ent", "aave_minus_sae")] = out[("ent", "aave")] - out[("ent", "sae")]
    out[("ntok", "aave_minus_sae")] = out[("ntok", "aave")] - out[("ntok", "sae")]
    return out.round(3)


######################
# DEPTH ANALYSES     #
######################

def accuracy_gap_tokenization_test(merged):
    """Test whether the SAE-AAVE accuracy gap differs between tokenizations.

    Computes per-model accuracy gaps under each tokenization, then tests
    whether the gap magnitude systematically changes using a paired test
    across models.
    """
    from scipy import stats
    df = merged.copy()
    df = df[~df["model"].isin({"qwen_32b_instruct", "llama_70b_instruct", "gemma_27b_instruct"})]
    df["pair_id"] = df["unique_id"].map(_pair_id)
    model_gaps = []
    for model in sorted(df["model"].unique()):
        mg = df[df["model"] == model]
        gaps_can, gaps_char = [], []
        for _, g in mg.groupby(["task", "reasoning"]):
            sae = g[g["dialect"] == "sae"].set_index("pair_id")
            aave = g[g["dialect"] == "aave"].set_index("pair_id")
            common = sae.index.intersection(aave.index)
            if len(common) < 5:
                continue
            gaps_can.append(sae.loc[common, "correct_can"].values.astype(int)
                            - aave.loc[common, "correct_can"].values.astype(int))
            gaps_char.append(sae.loc[common, "correct_char"].values.astype(int)
                             - aave.loc[common, "correct_char"].values.astype(int))
        if not gaps_can:
            continue
        can_mean = np.mean(np.concatenate(gaps_can)) * 100
        char_mean = np.mean(np.concatenate(gaps_char)) * 100
        widens = abs(char_mean) > abs(can_mean)
        model_gaps.append({"model": model, "can_gap": can_mean, "char_gap": char_mean,
                           "abs_can": abs(can_mean), "abs_char": abs(char_mean)})
        print(f"  {model:25s}  can={can_mean:+6.2f}  char={char_mean:+6.2f}  "
              f"{'WIDENS' if widens else 'NARROWS'}")

    if len(model_gaps) < 3:
        print("Too few models for paired test.")
        return
    abs_can = np.array([m["abs_can"] for m in model_gaps])
    abs_char = np.array([m["abs_char"] for m in model_gaps])
    n_widens = (abs_char > abs_can).sum()
    n_models = len(model_gaps)

    t_stat, t_p = stats.ttest_rel(abs_char, abs_can)
    wilcox_stat, wilcox_p = stats.wilcoxon(abs_char - abs_can)
    sign_p = stats.binomtest(n_widens, n_models, 0.5).pvalue

    print(f"\n  n_models={n_models}, widens={n_widens}, narrows={n_models - n_widens}")
    print(f"  paired t-test on |gap|: t={t_stat:.3f}, p={t_p:.3e}")
    print(f"  Wilcoxon on |gap|: stat={wilcox_stat:.0f}, p={wilcox_p:.3e}")
    print(f"  sign test: p={sign_p:.3e}")


def input_entropy_gap_persistence(merged):
    """Test whether the input entropy gap direction persists across tokenizations.

    For each SAE-AAVE pair, computes the input entropy gap (SAE - AAVE) under
    canonical and character tokenization, then tests whether the two gaps agree
    in sign and are correlated.
    """
    from scipy import stats
    df = merged.copy()
    df["pair_id"] = df["unique_id"].map(_pair_id)
    gaps_can, gaps_char = [], []
    for _, g in df.groupby(["model", "task", "reasoning"]):
        sae = g[g["dialect"] == "sae"].set_index("pair_id")
        aave = g[g["dialect"] == "aave"].set_index("pair_id")
        common = sae.index.intersection(aave.index)
        if len(common) < 5:
            continue
        s_can = sae.loc[common, "input_mean_entropy_can"]
        a_can = aave.loc[common, "input_mean_entropy_can"]
        s_char = sae.loc[common, "input_mean_entropy_char"]
        a_char = aave.loc[common, "input_mean_entropy_char"]
        valid = s_can.notna() & a_can.notna() & s_char.notna() & a_char.notna()
        if valid.sum() < 5:
            continue
        gaps_can.append((s_can[valid] - a_can[valid]).values)
        gaps_char.append((s_char[valid] - a_char[valid]).values)
    if not gaps_can:
        print("Too few paired samples for input entropy gap persistence test.")
        return
    gap_can = np.concatenate(gaps_can)
    gap_char = np.concatenate(gaps_char)

    for label, gap in [("canonical", gap_can), ("character", gap_char)]:
        n_neg = (gap < 0).sum()
        n_nonzero = (gap != 0).sum()
        binom_p = stats.binomtest(n_neg, n_nonzero, 0.5).pvalue
        wilcox_stat, wilcox_p = stats.wilcoxon(gap, alternative="less")
        print(f"  {label}: mean={np.mean(gap):.4f}, median={np.median(gap):.4f}, "
              f"AAVE-higher={n_neg}/{n_nonzero} ({100 * n_neg / n_nonzero:.1f}%), "
              f"sign-test p={binom_p:.2e}, Wilcoxon p={wilcox_p:.2e}")

    valid = (gap_can != 0) & (gap_char != 0)
    agree = ((gap_can < 0) & (gap_char < 0)) | ((gap_can > 0) & (gap_char > 0))
    n_agree = agree[valid].sum()
    n_valid = valid.sum()
    agree_p = stats.binomtest(n_agree, n_valid, 0.5).pvalue
    r, r_p = stats.pearsonr(gap_can, gap_char)
    print(f"  sign agreement: {n_agree}/{n_valid} ({100 * n_agree / n_valid:.1f}%), "
          f"binomial p={agree_p:.2e}")
    print(f"  gap correlation: Pearson r={r:.3f}, p={r_p:.2e}")


def _load_per_sample_npz(experiment_dir, experiment, name):
    """Like _load_per_token_npz but keeps arrays per-sample (list of 1-D arrays).

    Returns:
        Dict (model, task, reasoning, dialect) -> list[np.ndarray]
    """
    root = os.path.join(experiment_dir, experiment)
    pattern = os.path.join(root, "*", "redial", "*", "*", "*", f"{name}.npz")
    out = {}
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        model, _ds, task, reasoning, dialect, _ = parts
        with np.load(path) as z:
            keys = sorted(z.files, key=lambda k: int(k.replace("arr_", "")))
            out[(model, task, reasoning, dialect)] = [z[k] for k in keys]
    return out


def output_entropy_trajectory(experiment_dir, out_dir):
    """Per-token output entropy gap (SAE − AAVE) across generation steps.

    Uses gen_entropy.npz from canonical naive runs (the only configuration
    that saves per-token output entropy). Shows whether the dialect gap
    emerges at the first token or builds over the generation.

    Plots: (1) aggregate across all models, (2) per-model family thin lines
    overlaid on the aggregate to show consistency.
    """
    data = _load_per_sample_npz(experiment_dir, "generate_logits", "gen_entropy")
    naive_keys = [(m, t, r, d) for (m, t, r, d) in data if r == "naive"]

    max_steps = 30

    # Per-model gap trajectories
    models = _sort_models(sorted({m for (m, _, _, _) in naive_keys}))
    model_trajectories = {}
    for model in models:
        sae_by_step = [[] for _ in range(max_steps)]
        aave_by_step = [[] for _ in range(max_steps)]
        for (m, t, r, d) in naive_keys:
            if m != model:
                continue
            target = sae_by_step if d == "sae" else aave_by_step
            for arr in data[(m, t, r, d)]:
                for step_i, val in enumerate(arr[:max_steps]):
                    target[step_i].append(float(val))
        steps, gaps = [], []
        for i in range(max_steps):
            if len(sae_by_step[i]) < 10 or len(aave_by_step[i]) < 10:
                break
            gaps.append(np.mean(sae_by_step[i]) - np.mean(aave_by_step[i]))
            steps.append(i + 1)
        if steps:
            model_trajectories[model] = (np.array(steps), np.array(gaps))

    # Aggregate across all models
    all_sae = [[] for _ in range(max_steps)]
    all_aave = [[] for _ in range(max_steps)]
    for (m, t, r, d) in naive_keys:
        target = all_sae if d == "sae" else all_aave
        for arr in data[(m, t, r, d)]:
            for step_i, val in enumerate(arr[:max_steps]):
                target[step_i].append(float(val))

    agg_steps, agg_gaps, agg_ses = [], [], []
    for i in range(max_steps):
        if len(all_sae[i]) < 10 or len(all_aave[i]) < 10:
            break
        sae_arr = np.array(all_sae[i])
        aave_arr = np.array(all_aave[i])
        gap = sae_arr.mean() - aave_arr.mean()
        se = np.sqrt(sae_arr.var() / len(sae_arr) + aave_arr.var() / len(aave_arr))
        agg_steps.append(i + 1)
        agg_gaps.append(gap)
        agg_ses.append(se)

    if not agg_steps:
        return
    agg_steps = np.array(agg_steps)
    agg_gaps = np.array(agg_gaps)
    agg_ses = np.array(agg_ses)

    fig, ax = plt.subplots(figsize=(7, 4))
    for model, (s, g) in model_trajectories.items():
        family, _ = _parse_model(model)
        color = FAMILY_COLORS.get(family, "gray")
        ax.plot(s, g, "-", color=color, alpha=0.5, lw=1.0)
    ax.fill_between(agg_steps, agg_gaps - 1.96 * agg_ses, agg_gaps + 1.96 * agg_ses,
                     alpha=0.2, color="black")
    ax.plot(agg_steps, agg_gaps, "-o", color="black", markersize=3, lw=1.2,
            label="All models (mean ± 95% CI)")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("Generation step", fontsize=12)
    ax.set_ylabel("SAE − AAVE entropy gap (nats)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.lines import Line2D
    family_handles = [Line2D([0], [0], color=FAMILY_COLORS[f], lw=1.5,
                              label=f.capitalize()) for f in FAMILY_ORDER
                       if f in {_parse_model(m)[0] for m in model_trajectories}]
    family_handles.append(Line2D([0], [0], color="black", lw=2.0, label="Aggregate"))
    ax.legend(handles=family_handles, fontsize=10, loc="upper right")

    _save_fig(fig, out_dir, "output_entropy_trajectory")
    plt.close(fig)


def hidden_separability_comparison(experiment_dir, allowed_triples=None):
    """Hidden-state dialect separability under both tokenizations.

    Returns:
        DataFrame with separability results.
    """
    df = hidden_dialect_separability(experiment_dir, allowed_triples=allowed_triples)
    if len(df) == 0 or len(df["tok"].unique()) < 2:
        print("Hidden separability comparison requires both tokenizations.")
        return df

    agg = df.groupby(["model", "tok"])["acc"].agg(["mean", "std"]).reset_index()
    models = _sort_models(agg["model"].unique().tolist())

    print("\nHidden-state dialect separability (canonical vs character):")
    pivot = agg.pivot_table(index="model", columns="tok", values="mean")
    pivot = pivot.reindex(models)
    pivot["delta"] = pivot.get("character", np.nan) - pivot.get("canonical", np.nan)
    print(pivot.round(2).to_string())
    return df


#########
# PLOTS #
#########

def _model_accuracy_frame(merged):
    """Per (model, dialect) accuracy under char and canonical tokenization."""
    grp = merged.groupby(["model", "dialect"])
    acc = pd.DataFrame({
        "n": grp.size(),
        "acc_char": grp["correct_char"].mean() * 100,
        "acc_can": grp["correct_can"].mean() * 100,
    }).reset_index()
    fam_size = acc["model"].map(lambda m: _parse_model(m))
    acc["family"] = [f for f, _ in fam_size]
    acc["size_b"] = [s for _, s in fam_size]
    return acc


def _sort_models(models):
    """Sort model names by (family, size_b) using FAMILY_ORDER."""
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    def key(m):
        f, s = _parse_model(m)
        return (rank.get(f, 99), s if s is not None else 99)
    return sorted(models, key=key)


def _pretty_model(name):
    """Render a directory-name like 'llama_3b_instruct' as 'Llama 3B'."""
    family, size = _parse_model(name)
    fam_label = {"llama": "Llama", "gemma": "Gemma", "qwen": "Qwen"}.get(family, family.capitalize())
    if size is None:
        return fam_label
    size_str = f"{size:g}B"
    return f"{fam_label} {size_str}"


############
# NPZ I/O  #
############

def _load_per_token_npz(experiment_dir, experiment, name):
    """For entropy/logp: (model, task, reasoning, dialect) -> 1-D concat of per-token values."""
    root = os.path.join(experiment_dir, experiment)
    pattern = os.path.join(root, "*", "redial", "*", "*", "*", f"{name}.npz")
    out = {}
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        model, _ds, task, reasoning, dialect, _ = parts
        with np.load(path) as z:
            arrs = [z[k] for k in z.files]
        if arrs:
            out[(model, task, reasoning, dialect)] = np.concatenate([a.ravel() for a in arrs])
    return out


def _load_hidden_by_combo(experiment_dir, experiment):
    """Return (model, task, reasoning, dialect) -> (n_samples, hidden_dim) float array.

    hidden.npz is keyed by the sample index (as a string) at the answer-extraction
    step. Skipped samples (no answer step found) are absent from the archive.
    """
    root = os.path.join(experiment_dir, experiment)
    pattern = os.path.join(root, "*", "redial", "*", "*", "*", "hidden.npz")
    out = {}
    for path in sorted(glob.glob(pattern)):
        parts = os.path.relpath(path, root).split(os.sep)
        model, _ds, task, reasoning, dialect, _ = parts
        with np.load(path) as z:
            if len(z.files) == 0:
                continue
            keys = sorted(z.files, key=lambda k: int(k))
            arrs = [z[k] for k in keys if z[k].ndim == 1]
        if arrs:
            out[(model, task, reasoning, dialect)] = np.stack(arrs)
    return out


##################
# DIALECT GAPS   #
##################


def hidden_dialect_separability(experiment_dir, allowed_triples=None):
    """5-fold CV accuracy of logistic regression predicting dialect from the
    answer-step hidden state, per (model, task, reasoning, tokenization).

    50% = indistinguishable; 100% = perfectly linearly separable. Shrinking
    separability under character tokenization would mean dialect identity is
    less linearly encoded in the representation used to emit the final answer.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    rows = []
    for experiment, tok_label in [("generate_logits", "canonical"),
                                   ("generate_characters", "character")]:
        hiddens = _load_hidden_by_combo(experiment_dir, experiment)
        triples = {(m, t, r) for (m, t, r, _) in hiddens}
        for (m, t, r) in triples:
            if allowed_triples is not None and (m, t, r) not in allowed_triples:
                continue
            sae = hiddens.get((m, t, r, "sae"))
            aave = hiddens.get((m, t, r, "aave"))
            if sae is None or aave is None or len(sae) < 10 or len(aave) < 10:
                continue
            if sae.shape[1] != aave.shape[1]:
                continue
            X = np.concatenate([sae, aave]).astype(np.float32)
            y = np.concatenate([np.zeros(len(sae)), np.ones(len(aave))])
            try:
                clf = LogisticRegression(max_iter=400, C=1.0, solver="liblinear")
                acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
            except Exception:
                continue
            rows.append({"model": m, "task": t, "reasoning": r,
                         "tok": tok_label, "acc": acc * 100, "n": len(X)})
    return pd.DataFrame(rows)


##################
# DENSE PLOTS    #
##################


def plot_effect_size_summary(merged, experiment_dir, out_dir, allowed_triples=None):
    """Per-model SAE−AAVE effect size across metrics, split by reasoning.

    Color = tokenization (canonical vs character).
    Marker = reasoning (naive=circle, cot=triangle).
    Grey lines connect the four points for each model to make the per-model
    shift legible at a glance.
    """
    rows = []
    for tok_suffix, tok_label in [("can", "canonical"), ("char", "character")]:
        cols = [(f"correct_{tok_suffix}", "Accuracy gap (pts)", 100.0),
                (f"input_mean_log_prob_{tok_suffix}", "Input logp gap (nats)", 1.0),
                (f"input_mean_entropy_{tok_suffix}", "Input entropy gap (nats)", 1.0)]
        for col, metric, scale in cols:
            if col not in merged.columns:
                continue
            g = (merged.groupby(["model", "reasoning", "dialect"])[col].mean()
                        .unstack("dialect"))
            if not {"sae", "aave"}.issubset(g.columns):
                continue
            gap = (g["sae"] - g["aave"]) * scale
            for (m, r), v in gap.items():
                rows.append({"model": m, "reasoning": r, "tok": tok_label,
                             "metric": metric, "gap": v})
    # Hidden separability: canonical pipeline only. Include per-reasoning if both
    # canonical + character exist; otherwise drop this column entirely (as before).
    sep = hidden_dialect_separability(experiment_dir, allowed_triples=allowed_triples)
    if len(sep) and set(sep["tok"].unique()) >= {"canonical", "character"}:
        agg = sep.groupby(["model", "reasoning", "tok"])["acc"].mean().reset_index()
        for _, r in agg.iterrows():
            rows.append({"model": r["model"], "reasoning": r["reasoning"],
                         "tok": r["tok"], "metric": "Hidden sep. (acc − 50%)",
                         "gap": r["acc"] - 50})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return
    metrics = ["Accuracy gap (pts)", "Input logp gap (nats)",
               "Input entropy gap (nats)", "Hidden sep. (acc − 50%)"]
    metrics = [m for m in metrics if m in df["metric"].unique()]
    models = _sort_models(df["model"].unique().tolist())

    TOK_COLOR = {"canonical": "#4C72B0", "character": "#DD8452"}
    REASON_MARKER = {"naive": "o", "cot": "^"}
    reasonings_present = [r for r in ["cot", "naive"] if r in df["reasoning"].unique()]

    # Rows are (model, reasoning) pairs: within each model, list CoT above Naive.
    rows_mr = [(m, r) for m in models for r in reasonings_present]
    row_labels = [f"{_pretty_model(m)} ({'CoT' if r == 'cot' else 'Naive'})"
                  for (m, r) in rows_mr]

    # Accuracy panel gets the largest x-range (~−35..+10 pts), so give it
    # proportionally more width; overall figure is widened for room.
    width_ratios = [2.0 if m == "Accuracy gap (pts)" else 1.2 for m in metrics]
    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(4.2 * sum(width_ratios), 7.0),
                              sharey=True,
                              gridspec_kw={"width_ratios": width_ratios})
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        y = np.arange(len(rows_mr))
        for idx, (m, r) in enumerate(rows_mr):
            xs = sub[(sub["model"] == m) & (sub["reasoning"] == r)]["gap"].dropna().values
            if len(xs) >= 2:
                ax.plot([xs.min(), xs.max()], [idx, idx], "-",
                        color="lightgray", lw=1.3, zorder=1)
        for tok in ["canonical", "character"]:
            for idx, (m, r) in enumerate(rows_mr):
                match = sub[(sub["model"] == m) & (sub["reasoning"] == r) & (sub["tok"] == tok)]
                if len(match) == 0:
                    continue
                ax.scatter(match["gap"].iloc[0], idx,
                           color=TOK_COLOR[tok], marker=REASON_MARKER[r],
                           zorder=3, s=220, alpha=1.0,
                           edgecolor="black", linewidth=1.1)
        ax.axvline(0, color="k", lw=0.7, ls=":")
        ax.set_xlabel(metric, fontsize=18)
        ax.tick_params(axis="x", labelsize=16)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(np.arange(len(rows_mr)))
    axes[0].set_yticklabels(row_labels, fontsize=17)
    axes[0].invert_yaxis()

    from matplotlib.lines import Line2D
    tok_handles = [Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=TOK_COLOR[t], markeredgecolor="black",
                          markersize=18, markeredgewidth=1.1, label=t.capitalize())
                   for t in ["canonical", "character"]]
    reason_handles = [Line2D([0], [0], marker=REASON_MARKER[r], linestyle="",
                              markerfacecolor="white", markeredgecolor="black",
                              markersize=18, markeredgewidth=1.1,
                              label="CoT" if r == "cot" else r.capitalize())
                      for r in reasonings_present]
    leg1 = axes[0].legend(handles=tok_handles, title="Tokenizer", fontsize=14,
                           title_fontsize=16, loc="upper left",
                           bbox_to_anchor=(0.01, 0.99))
    axes[0].add_artist(leg1)
    axes[0].legend(handles=reason_handles, title="Reasoning", fontsize=14,
                    title_fontsize=16, loc="upper left",
                    bbox_to_anchor=(0.28, 0.99))

    _save_fig(fig, out_dir, "character_effect_size_summary")
    plt.close(fig)


def plot_output_effect_size_summary(merged, out_dir):
    """Per-model SAE−AAVE effect size for GENERATED (output) logprob & entropy.

    Same visual encoding as plot_effect_size_summary:
    color = tokenization, marker = reasoning, grey lines = per-model shift.

    Column mapping:
        - character (all): answer_log_prob, answer_entropy_char
        - canonical (all): answer_entropy_can
    """
    rows = []

    def _add_gaps(col, sub_df, tok_label, metric, scale=1.0):
        if col not in sub_df.columns:
            return
        g = (sub_df.groupby(["model", "reasoning", "dialect"])[col].mean()
                   .unstack("dialect"))
        if not {"sae", "aave"}.issubset(g.columns):
            return
        gap = (g["sae"] - g["aave"]) * scale
        for (m, r), v in gap.items():
            rows.append({"model": m, "reasoning": r, "tok": tok_label,
                         "metric": metric, "gap": v})

    # Accuracy: both tokenizations (suffixed in merged)
    for tok_suffix, tok_label in [("can", "canonical"), ("char", "character")]:
        _add_gaps(f"correct_{tok_suffix}", merged, tok_label, "Accuracy gap (pts)", scale=100.0)

    # Input entropy: both tokenizations (suffixed in merged)
    for tok_suffix, tok_label in [("can", "canonical"), ("char", "character")]:
        _add_gaps(f"input_mean_entropy_{tok_suffix}", merged, tok_label, "Input entropy gap (nats)")

    # Output entropy
    _add_gaps("answer_entropy_char", merged, "character", "Output entropy gap (nats)")
    _add_gaps("answer_entropy_can", merged, "canonical", "Output entropy gap (nats)")

    df = pd.DataFrame(rows)
    df = df[~df["model"].isin({"qwen_32b_instruct", "llama_70b_instruct", "gemma_27b_instruct"})]
    if len(df) == 0:
        return
    metrics = ["Accuracy gap (pts)", "Input entropy gap (nats)", "Output entropy gap (nats)"]
    metrics = [m for m in metrics if m in df["metric"].unique()]
    models = _sort_models(df["model"].unique().tolist())

    TOK_COLOR = {"canonical": "#4C72B0", "character": "#DD8452"}
    REASON_MARKER = {"naive": "o", "cot": "^"}
    reasonings_present = [r for r in ["cot", "naive"] if r in df["reasoning"].unique()]

    rows_mr = [(m, r) for m in models for r in reasonings_present]
    row_labels = [f"{_pretty_model(m)} ({'CoT' if r == 'cot' else 'Naive'})"
                  for (m, r) in rows_mr]

    width_ratios = [2.0 if m == "Accuracy gap (pts)" else 1.2 for m in metrics]
    fig, axes = plt.subplots(1, len(metrics),
                              figsize=(4.2 * sum(width_ratios), 7.0),
                              sharey=True,
                              gridspec_kw={"width_ratios": width_ratios})
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        sub = df[df["metric"] == metric]
        for idx, (m, r) in enumerate(rows_mr):
            xs = sub[(sub["model"] == m) & (sub["reasoning"] == r)]["gap"].dropna().values
            if len(xs) >= 2:
                ax.plot([xs.min(), xs.max()], [idx, idx], "-",
                        color="lightgray", lw=1.3, zorder=1)
        for tok in ["canonical", "character"]:
            for idx, (m, r) in enumerate(rows_mr):
                match = sub[(sub["model"] == m) & (sub["reasoning"] == r) & (sub["tok"] == tok)]
                if len(match) == 0:
                    continue
                ax.scatter(match["gap"].iloc[0], idx,
                           color=TOK_COLOR[tok], marker=REASON_MARKER[r],
                           zorder=3, s=220, alpha=1.0,
                           edgecolor="black", linewidth=1.1)
        ax.axvline(0, color="k", lw=0.7, ls=":")
        ax.set_xlabel(metric, fontsize=18)
        ax.tick_params(axis="x", labelsize=16)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(np.arange(len(rows_mr)))
    axes[0].set_yticklabels(row_labels, fontsize=17)
    axes[0].invert_yaxis()

    from matplotlib.lines import Line2D
    tok_handles = [Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=TOK_COLOR[t], markeredgecolor="black",
                          markersize=18, markeredgewidth=1.1, label=t.capitalize())
                   for t in ["canonical", "character"]]
    reason_handles = [Line2D([0], [0], marker=REASON_MARKER[r], linestyle="",
                              markerfacecolor="white", markeredgecolor="black",
                              markersize=18, markeredgewidth=1.1,
                              label="CoT" if r == "cot" else r.capitalize())
                      for r in reasonings_present]
    leg1 = axes[0].legend(handles=tok_handles, title="Tokenizer", fontsize=14,
                           title_fontsize=16, loc="upper left",
                           bbox_to_anchor=(0.01, 1.00))
    axes[0].add_artist(leg1)
    axes[0].legend(handles=reason_handles, title="Reasoning", fontsize=14,
                    title_fontsize=16, loc="upper left",
                    bbox_to_anchor=(0.28, 1.00))

    os.makedirs(out_dir, exist_ok=True)
    _save_fig(fig, out_dir, "character_accuracy_entropy_gaps")
    plt.close(fig)


def plot_bits_per_byte_by_model(merged, experiment_dir, out_dir):
    """Per (model, reasoning), AAVE − SAE bits-per-byte gap under each tokenization.

    BPB shares units across tokenizations (bits ÷ raw text byte length), so
    canonical and character points live on a single axis. The unit is
    tokenizer-agnostic but the value is not — each forward pass tokenizes
    differently, so canonical-BPB and character-BPB are distinct quantities.
    The gap between them is the object of interest: it localizes how much of
    the dialect-difficulty signal is tokenization-induced.
    """
    char_lps = _load_logp_arrays(experiment_dir, "generate_characters")
    can_lps = _load_logp_arrays(experiment_dir, "generate_logits")

    rows = []
    for (model, task, reasoning, dialect), g in merged.groupby(
        ["model", "task", "reasoning", "dialect"]
    ):
        key = (model, task, reasoning, dialect)
        if key not in char_lps or key not in can_lps:
            continue
        ch = char_lps[key]
        cn = can_lps[key]
        n = min(len(ch), len(cn), len(g))
        if n == 0:
            continue
        g = g.iloc[:n]
        bytes_approx = g["n_char_tokens"].values
        bpb_char = np.array([-a.sum() / np.log(2) for a in ch[:n]]) / bytes_approx
        bpb_can = np.array([-a.sum() / np.log(2) for a in cn[:n]]) / bytes_approx
        rows.append({"model": model, "reasoning": reasoning, "dialect": dialect,
                     "bpb_can": float(np.mean(bpb_can)),
                     "bpb_char": float(np.mean(bpb_char))})
    if not rows:
        return
    df = pd.DataFrame(rows)
    # Mean across tasks per (model, reasoning, dialect)
    agg = df.groupby(["model", "reasoning", "dialect"])[["bpb_can", "bpb_char"]].mean()
    piv = agg.unstack("dialect")
    if ("bpb_can", "aave") not in piv.columns or ("bpb_char", "aave") not in piv.columns:
        return
    gap_can = piv[("bpb_can", "aave")] - piv[("bpb_can", "sae")]
    gap_char = piv[("bpb_char", "aave")] - piv[("bpb_char", "sae")]
    gap = pd.DataFrame({"canonical": gap_can, "character": gap_char}).reset_index()

    models = _sort_models(gap["model"].unique().tolist())
    reasonings_present = [r for r in ["cot", "naive"] if r in gap["reasoning"].unique()]
    rows_mr = [(m, r) for m in models for r in reasonings_present]
    row_labels = [f"{_pretty_model(m)} ({'CoT' if r == 'cot' else 'Naive'})"
                  for (m, r) in rows_mr]

    TOK_COLOR = {"canonical": "#4C72B0", "character": "#DD8452"}
    REASON_MARKER = {"naive": "o", "cot": "^"}

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    gap_idx = gap.set_index(["model", "reasoning"])
    y = np.arange(len(rows_mr))
    for idx, (m, r) in enumerate(rows_mr):
        if (m, r) not in gap_idx.index:
            continue
        row = gap_idx.loc[(m, r)]
        xs = [row["canonical"], row["character"]]
        xs = [x for x in xs if not np.isnan(x)]
        if len(xs) >= 2:
            ax.plot([min(xs), max(xs)], [idx, idx], "-",
                    color="lightgray", lw=1.3, zorder=1)
    for tok in ["canonical", "character"]:
        for idx, (m, r) in enumerate(rows_mr):
            if (m, r) not in gap_idx.index:
                continue
            v = gap_idx.loc[(m, r), tok]
            if np.isnan(v):
                continue
            ax.scatter(v, idx, color=TOK_COLOR[tok], marker=REASON_MARKER[r],
                       zorder=3, s=95, alpha=1.0,
                       edgecolor="black", linewidth=0.9)
    ax.axvline(0, color="k", lw=0.7, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("AAVE − SAE bits / byte")
    ax.grid(True, axis="x", alpha=0.3)

    from matplotlib.lines import Line2D
    tok_handles = [Line2D([0], [0], marker="o", linestyle="",
                          markerfacecolor=TOK_COLOR[t], markeredgecolor="black",
                          markersize=9, markeredgewidth=0.9, label=t.capitalize())
                   for t in ["canonical", "character"]]
    reason_handles = [Line2D([0], [0], marker=REASON_MARKER[r], linestyle="",
                              markerfacecolor="white", markeredgecolor="black",
                              markersize=9, markeredgewidth=0.9,
                              label="CoT" if r == "cot" else r.capitalize())
                      for r in reasonings_present]
    leg1 = ax.legend(handles=tok_handles, title="Tokenizer", fontsize=8,
                     title_fontsize=8, loc="upper left",
                     bbox_to_anchor=(0.01, 0.99))
    ax.add_artist(leg1)
    ax.legend(handles=reason_handles, title="Reasoning", fontsize=8,
              title_fontsize=8, loc="upper left",
              bbox_to_anchor=(0.27, 0.99))

    fig.suptitle("AAVE − SAE bits-per-byte gap by model (shared bits-per-byte axis)", fontsize=11)
    _save_fig(fig, out_dir, "character_bpb_by_model")
    plt.close(fig)


########
# MAIN #
########

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default")
    parser.add_argument("--experiments-dir", default=None)
    parser.add_argument("--plots-dir", default="analysis/plots/characters")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    if args.experiments_dir is None:
        cfg = dialecttax.utils.load_config(args.config)
        args.experiments_dir = cfg["directories"]["experiments"]

    merged = load_merged(args.experiments_dir)
    if len(merged) == 0:
        print("No matched samples.")
        return

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    clean, dropped = _drop_broken_combos(merged, min_acc=1.0)
    if dropped:
        print(f"\nDropping {len(dropped)} (model, task, reasoning) triples with char-acc <1% "
              f"(likely extraction failure): {sorted(dropped)}")

    # §4 uses the 9-model 3x3 grid (3 families x 3 sizes). Drop the larger
    # instruct models so every analysis and figure below shares the same set.
    excluded = sorted(set(clean["model"].unique()) - INSTRUCT_9)
    if excluded:
        print(f"\nDropping {len(excluded)} models outside the §4 9-model grid: {excluded}")
        clean = clean[clean["model"].isin(INSTRUCT_9)].copy()

    print("\n" + "=" * 72)
    print(" DIALECT GAPS (SAE − AAVE %, matched samples)")
    print(" abs_gap_delta > 0 = char tokenization REDUCES unfairness")
    print("=" * 72)
    print(dialect_gap_table(clean))

    print("\n" + "=" * 72)
    print(" PAIRED SAMPLE OUTCOMES (% of paired SAE/AAVE items)")
    print(" net_sae_advantage = sae_only − aave_only")
    print("=" * 72)
    print(paired_dialect_table(clean))

    print("\n" + "=" * 72)
    print(" INPUT METRICS BY DIALECT × TOKENIZATION")
    print("=" * 72)
    print(dialect_input_metrics(clean))

    print("\n" + "=" * 72)
    print(" PERPLEXITY BY DIALECT × TOKENIZATION")
    print(" (aave/sae ratio > 1 = model more surprised by AAVE)")
    print("=" * 72)
    print(perplexity_table(clean))

    print("\n" + "=" * 72)
    print(" PER-TOKEN INPUT ENTROPY DISTRIBUTION (replaces violin plot)")
    print(" p25/median/p75 in nats; *_gap_median = AAVE − SAE median (nats)")
    print("=" * 72)
    print(per_token_entropy_table(args.experiments_dir))

    print("\n" + "=" * 72)
    print(" BITS-PER-BYTE BY DIALECT × TOKENIZATION (fair cross-tok metric)")
    print(" gap_shrink > 0 = char tokenization REDUCES AAVE-SAE difficulty gap")
    print("=" * 72)
    print(bits_per_byte_table(clean, args.experiments_dir))

    print("\n" + "=" * 72)
    print(" ACCURACY GAP TOKENIZATION TEST (canonical vs character)")
    print("=" * 72)
    accuracy_gap_tokenization_test(clean)

    print("\n" + "=" * 72)
    print(" INPUT ENTROPY GAP PERSISTENCE (canonical vs character)")
    print("=" * 72)
    input_entropy_gap_persistence(clean)

    if not args.skip_plots:
        allowed = {(m, t, r) for (m, t, r) in
                   clean[["model", "task", "reasoning"]].drop_duplicates().values}
        plot_effect_size_summary(clean, args.experiments_dir, args.plots_dir,
                                 allowed_triples=allowed)
        plot_output_effect_size_summary(clean, args.plots_dir)
        plot_bits_per_byte_by_model(clean, args.experiments_dir, args.plots_dir)
        output_entropy_trajectory(args.experiments_dir, args.plots_dir)
        hidden_separability_comparison(args.experiments_dir,
                                        allowed_triples=allowed)
        print(f"\nSaved plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
