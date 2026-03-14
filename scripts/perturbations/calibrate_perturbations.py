"""
Calibrate perturbation strength p to match dialect perplexity.

For each surface-form perturbation with a continuous strength (swap, drop, insert),
find the p such that perplexity(perturbation_p(SAE)) == perplexity(AAVE), under a
reference LM. Perplexity is exp of the mean per-sample cross-entropy (nats), where
per-sample CE = -mean(next-token log-prob) -- matching generate_logits and
analysis/perturbations.perplexity_by_dialect.

Perturbations are applied to the SAE side of a parallel dataset; the AAVE side of the
same dataset is the target. capitalize (discrete modes) and translate (a language) have
no continuous p and are not calibrated.

Usage:
    python scripts/perturbations/calibrate_perturbations.py
    python scripts/perturbations/calibrate_perturbations.py --datasets parallelaave
    python scripts/perturbations/calibrate_perturbations.py --model-id Qwen/Qwen3-4B-Base
"""

import argparse
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import dialecttax

log = logging.getLogger(__name__)


#############
# CONSTANTS #
#############

DEFAULT_MODEL_ID = "Qwen/Qwen3-8B-Base"  # non-gated; runs without an HF token
DEFAULT_DATASETS = ["parallelaave", "multivalue"]
PERTURBATIONS = {
    "swap": dialecttax.perturbations.swap,
    "drop": dialecttax.perturbations.drop,
    "insert": dialecttax.perturbations.insert,
}

# Largest batch that fits; persisted across calls so we don't re-probe OOM every eval.
_EFFECTIVE_BATCH: int | None = None


###########
# HELPERS #
###########

def _load_pair(dataset: str, datasets_dir: str) -> tuple[list[str], list[str]]:
    """Load the SAE and AAVE text lists for a flat parallel dataset.

    Args:
        dataset: Dataset key in dialecttax.data.DATASET_MODULES (flat, has sae/aave).
        datasets_dir: Root datasets directory from the project config.

    Returns:
        (sae_texts, aave_texts), line-aligned.
    """
    mod = dialecttax.data.DATASET_MODULES[dataset]

    def load(dialect: str) -> list[str]:
        path_file = os.path.join(mod.DIRECTORY_NAME, mod.FILE_NAME_FORMAT.format(dialect=dialect))
        return mod.load_dataset(datasets_dir, path_file, return_id=False)

    return load("sae"), load("aave")


@torch.no_grad()
def _perplexity(texts: list[str], model, tok, device: str, batch_size: int, max_len: int) -> float:
    """Perplexity = exp(mean over samples of per-sample mean next-token NLL).

    Args:
        texts: Input strings.
        model: Causal LM.
        tok: Tokenizer.
        device: Torch device.
        batch_size: Batch size.
        max_len: Max tokens per sample (truncation).

    Returns:
        Corpus perplexity under the model.
    """
    global _EFFECTIVE_BATCH
    if _EFFECTIVE_BATCH is None:
        _EFFECTIVE_BATCH = batch_size
    ces: list[float] = []
    i = 0
    while i < len(texts):
        batch = [t if t.strip() else " " for t in texts[i:i + _EFFECTIVE_BATCH]]
        try:
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
            ids, attn = enc.input_ids.to(device), enc.attention_mask.to(device)
            logits = model(input_ids=ids, attention_mask=attn).logits
            shift_logits, shift_labels = logits[:, :-1], ids[:, 1:]
            vocab = shift_logits.size(-1)
            # Fused cross-entropy: per-token NLL without materializing a full-vocab fp32 softmax.
            ce_tok = F.cross_entropy(
                shift_logits.reshape(-1, vocab), shift_labels.reshape(-1), reduction="none"
            ).reshape(shift_labels.shape)
            mask = attn[:, 1:].to(ce_tok.dtype)
            n = mask.sum(1).clamp(min=1)
            ce = (ce_tok * mask).sum(1) / n  # per-sample CE (nats)
            ces.extend(ce.float().cpu().tolist())
            del logits, shift_logits, ce_tok
            i += _EFFECTIVE_BATCH
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if _EFFECTIVE_BATCH == 1:
                raise
            _EFFECTIVE_BATCH = max(1, _EFFECTIVE_BATCH // 2)  # shrink once; persists across calls
            log.info(f"  OOM -> batch reduced to {_EFFECTIVE_BATCH} (persists)")
    return float(np.exp(np.mean(ces)))


def _perturbed_ppl(sae: list[str], fn: "callable", p: float, seed: int, score: "callable") -> float:
    """Perplexity of the SAE texts perturbed at strength p (seeded like generate_perturbations)."""
    random.seed(seed)
    np.random.seed(seed)
    return score(fn(sae, p=p))


def _calibrate(sae: list[str], target: float, fn: "callable", seed: int, score: "callable",
               tol: float, iters: int) -> tuple[float | None, float]:
    """Bisect p in [0, 1] so perplexity(perturb_p(sae)) matches target.

    Perplexity is monotonic in p (same seed each step), so bisection converges.

    Args:
        sae: SAE texts to perturb.
        target: Target perplexity (the AAVE perplexity).
        fn: Perturbation function taking (texts, p=...).
        seed: Perturbation seed.
        score: Callable mapping texts -> perplexity.
        tol: Relative tolerance on perplexity for early stop.
        iters: Max bisection iterations.

    Returns:
        (p, perplexity_at_p); p is None if the target is unreachable for p <= 1.
    """
    lo, hi = 0.0, 0.5
    f_hi = _perturbed_ppl(sae, fn, hi, seed, score)
    tries = 0
    while f_hi < target and hi < 1.0 and tries < 4:
        hi = min(1.0, hi * 2)
        f_hi = _perturbed_ppl(sae, fn, hi, seed, score)
        tries += 1
    if f_hi < target:
        return None, f_hi
    best = (hi, f_hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        fm = _perturbed_ppl(sae, fn, mid, seed, score)
        best = (mid, fm)
        if abs(fm - target) / target < tol:
            break
        if fm < target:
            lo = mid
        else:
            hi = mid
    return best


########
# MAIN #
########

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate perturbation p to match AAVE perplexity.")
    parser.add_argument("--config", default="default", help="Config file name (without .yaml)")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Reference LM (HuggingFace id)")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, help="Flat parallel datasets")
    parser.add_argument("--device", default="cuda", help="Torch device")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for perplexity")
    parser.add_argument("--max-len", type=int, default=1024, help="Max tokens per sample")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per dataset (speed; default all)")
    parser.add_argument("--seed", type=int, default=42, help="Perturbation seed")
    parser.add_argument("--tol", type=float, default=0.01, help="Relative perplexity tolerance")
    parser.add_argument("--iters", type=int, default=14, help="Max bisection iterations")
    parser.add_argument("--output", default=None, help="Optional path to write results JSON")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    config = dialecttax.utils.load_config(args.config)
    datasets_dir = config["directories"]["datasets"]

    log.info(f"Loading {args.model_id} ...")
    tok = AutoTokenizer.from_pretrained(args.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=torch.bfloat16).to(args.device).eval()

    def score(texts: list[str]) -> float:
        return _perplexity(texts, model, tok, args.device, args.batch_size, args.max_len)

    results: dict = {"model_id": args.model_id, "seed": args.seed, "datasets": {}}
    for dataset in args.datasets:
        sae, aave = _load_pair(dataset, datasets_dir)
        if args.max_samples:
            sae, aave = sae[:args.max_samples], aave[:args.max_samples]
        ppl_sae, ppl_aave = score(sae), score(aave)
        log.info(f"\n=== {dataset} (n={len(sae)}) ===")
        log.info(f"  ppl(sae)={ppl_sae:.3f}  ppl(aave)={ppl_aave:.3f}  ratio={ppl_aave / ppl_sae:.3f}")
        entry = {"n": len(sae), "ppl_sae": ppl_sae, "ppl_aave": ppl_aave, "perturbations": {}}
        for name, fn in PERTURBATIONS.items():
            p, achieved = _calibrate(sae, ppl_aave, fn, args.seed, score, args.tol, args.iters)
            entry["perturbations"][name] = {"p": p, "ppl_at_p": achieved}
            pstr = f"{p:.4f}" if p is not None else "UNREACHABLE(p<=1.0)"
            log.info(f"  {name:7} -> p={pstr}  (ppl={achieved:.3f} vs target {ppl_aave:.3f})")
        results["datasets"][dataset] = entry

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"\nSaved results: {args.output}")


if __name__ == "__main__":
    main()
