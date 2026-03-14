"""
Generate the MultiValue dialect corpus: CoQA stories transformed into English dialects.

Recreates ``{datasets}/multivalue/coqa_{dialect}.txt`` for dialect in
{sae, aave, appalachian, chicano, indian, singapore}. ``coqa_sae.txt`` holds the
original CoQA stories (one per line); each dialect file is the line-aligned
Multi-VALUE transform of those same stories.

Multi-VALUE (https://github.com/SALT-NLP/multi-value) has heavy, version-sensitive
dependencies that conflict with the dialecttax env, so run this in a THROWAWAY
virtualenv. Do NOT install it into the dialecttax conda env. The PyPI package is
`value-nlp` (it provides the `multivalue` import), and spaCy needs Python <= 3.12
(the dialecttax env is 3.14). These exact pins are required and were painful to find:
  - torch < 2.6 (>= 2.6 flips torch.load weights_only=True, breaking stanza models)
  - transformers < 4.50 (>= 4.50 refuses torch.load unless torch >= 2.6 — CVE gate)
  - spaCy model installed from the release wheel (`spacy download` builds a bad URL)
If disk is tight, point pip's temp/cache at a roomy partition (TMPDIR, PIP_CACHE_DIR)
and use CPU-only torch (skips ~1.5 GB of unused CUDA libraries).

    python3.12 -m venv /tmp/mv-venv
    source /tmp/mv-venv/bin/activate
    pip install value-nlp datasets pyyaml
    pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
    pip install "transformers==4.46.3"
    pip install "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl"
    python scripts/multivalue/generate_multivalue.py --list-dialects   # sanity-check class names
    python scripts/multivalue/generate_multivalue.py --workers 40      # ~15 min (sequential ~9 h)
    deactivate

This script is deliberately self-contained (no ``dialecttax`` import) so the venv
stays minimal; it reads the datasets directory straight from configs/{config}.yaml.

Note: Multi-VALUE dialect rules fire stochastically, so output is not byte-identical
to the lost originals even with a fixed seed. The corpus it produces is internally
consistent (all dialects derived from the same SAE lines, line-aligned).

Usage:
    python scripts/multivalue/generate_multivalue.py
    python scripts/multivalue/generate_multivalue.py --num-samples 429 --split validation
    python scripts/multivalue/generate_multivalue.py --dialects aave indian
    python scripts/multivalue/generate_multivalue.py --list-dialects   # print Multi-VALUE class names
"""

import argparse
import logging
import multiprocessing as mp
import os
import random
import re
import socket

# Cap intra-process math threads so many parallel workers don't oversubscribe cores.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import yaml

log = logging.getLogger(__name__)


##########
# CONFIG #
##########

# Mirror of dialecttax.data.multivalue (kept local so the isolated venv needs no dialecttax/torch).
DIRECTORY_NAME = "multivalue"
FILE_NAME_FORMAT = "coqa_{dialect}.txt"
SAE_DIALECT = "sae"
DIALECTS = ["sae", "aave", "appalachian", "chicano", "indian", "singapore"]

# Dialect key -> Multi-VALUE `Dialects` class name. If a name is wrong for your
# multivalue version, run with --list-dialects to see the available classes and fix here.
DIALECT_TO_MULTIVALUE_CLASS = {
    "aave": "AfricanAmericanVernacular",
    "appalachian": "AppalachianDialect",
    "chicano": "ChicanoDialect",
    "indian": "IndianDialect",
    "singapore": "ColloquialSingaporeDialect",
}

DEFAULT_NUM_SAMPLES = 429
DEFAULT_SPLIT = "validation"
DEFAULT_SEED = 42

# Lazily-loaded spaCy pipeline for sentence splitting (only touched on the transform path).
_SPACY_NLP = None
# Per-worker Multi-VALUE transformer, populated by the pool initializer.
_WORKER_TRANSFORM = None


###########
# HELPERS #
###########

def _load_datasets_dir(config_name: str) -> str:
    """Resolve the datasets directory from configs/{config_name}.yaml.

    Minimal reimplementation of dialecttax.utils.load_config so this script needs no
    dialecttax import (keeps the Multi-VALUE venv free of the torch stack).

    Args:
        config_name: Config file stem under configs/.

    Returns:
        Absolute path to the datasets directory.
    """
    path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"../../configs/{config_name}.yaml"))
    with open(path) as f:
        raw = f.read()
    raw = raw.replace("{hostname}", socket.gethostname().split(".")[0])
    return yaml.safe_load(raw)["directories"]["datasets"]


def _flatten(text: str) -> str:
    """Collapse whitespace/newlines so a story occupies a single line."""
    return re.sub(r"\s+", " ", text).strip()


def _load_stories(split: str, num_samples: int) -> list[str]:
    """Load CoQA stories, dedupe (order-preserving), flatten, and take the first num_samples.

    Args:
        split: CoQA split name.
        num_samples: Number of stories to keep.

    Returns:
        List of flattened, unique story strings.
    """
    from datasets import load_dataset

    coqa = load_dataset("stanfordnlp/coqa", split=split)
    seen: set[str] = set()
    stories: list[str] = []
    for story in coqa["story"]:
        flat = _flatten(story)
        if flat and flat not in seen:
            seen.add(flat)
            stories.append(flat)
        if len(stories) >= num_samples:
            break
    if len(stories) < num_samples:
        log.warning(f"Only {len(stories)} unique stories in split={split!r} (requested {num_samples})")
    return stories


def _build_transformer(dialect: str) -> "callable":
    """Return the Multi-VALUE SAE->dialect transform callable for a dialect.

    Args:
        dialect: Dialect key present in DIALECT_TO_MULTIVALUE_CLASS.

    Returns:
        A callable mapping an SAE string to the dialect string.
    """
    from multivalue import Dialects

    class_name = DIALECT_TO_MULTIVALUE_CLASS[dialect]
    if not hasattr(Dialects, class_name):
        available = "\n  ".join(sorted(c for c in dir(Dialects) if c[0].isupper()))
        raise SystemExit(f"Multi-VALUE has no class {class_name!r}. Available classes:\n  {available}")
    obj = getattr(Dialects, class_name)()
    fn = getattr(obj, "transform", None) or getattr(obj, "convert_sae_to_dialect", None)
    if fn is None:
        raise SystemExit(f"{class_name} exposes no transform()/convert_sae_to_dialect(); check the Multi-VALUE API.")
    return fn


def _iter_sentences(text: str) -> list[str]:
    """Split text into sentences with spaCy (whole text if spaCy is unavailable)."""
    global _SPACY_NLP
    try:
        if _SPACY_NLP is None:
            import spacy

            _SPACY_NLP = spacy.load("en_core_web_sm")
        return [s.text.strip() for s in _SPACY_NLP(text).sents if s.text.strip()]
    except Exception:
        return [text]


def _transform_story(transform_fn: "callable", story: str) -> str:
    """Transform one story sentence-by-sentence, keeping SAE text on any failure.

    Multi-VALUE asserts spaCy and Stanza agree on sentence count, which fails on
    multi-sentence input, so each story is split into sentences first. A failed or
    empty sentence transform falls back to the original sentence.

    Args:
        transform_fn: SAE->dialect callable.
        story: One SAE story line.

    Returns:
        The dialect story line (never empty; falls back to the SAE story).
    """
    parts = []
    for sentence in _iter_sentences(story):
        try:
            transformed = _flatten(transform_fn(sentence))
        except Exception:  # noqa: BLE001 - keep alignment; a mismatched sentence stays SAE
            transformed = ""
        parts.append(transformed or sentence)
    return _flatten(" ".join(parts)) or story


def _init_worker(dialect: str, seed: int) -> None:
    """Pool initializer: seed RNG and build this worker's dialect transformer once."""
    global _WORKER_TRANSFORM
    random.seed(seed)
    _WORKER_TRANSFORM = _build_transformer(dialect)


def _worker_transform_story(story: str) -> str:
    """Pool task: transform one story with the worker's transformer."""
    return _transform_story(_WORKER_TRANSFORM, story)


def _transform(dialect: str, stories: list[str], seed: int, workers: int) -> list[str]:
    """Transform all stories to a dialect, optionally across a process pool.

    Args:
        dialect: Dialect key.
        stories: SAE story lines.
        seed: Seed for Multi-VALUE stochastic rules.
        workers: Parallel worker processes (<=1 runs in-process, no pool).

    Returns:
        Dialect story lines, one per input story (line-aligned with `stories`).
    """
    if workers <= 1:
        _init_worker(dialect, seed)
        return [_worker_transform_story(s) for s in stories]
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(dialect, seed)) as pool:
        return pool.map(_worker_transform_story, stories, chunksize=1)


def _write(path: str, lines: list[str]) -> None:
    """Write one line per sample to path."""
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


########
# MAIN #
########

def parse_args():
    parser = argparse.ArgumentParser(description="Generate the MultiValue CoQA dialect corpus.")
    parser.add_argument("--config", default="default", help="Config file name (without .yaml)")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="CoQA split to draw stories from")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Number of stories")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed for Multi-VALUE stochastic rules")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                        help="Parallel worker processes for transforms (sentence transforms are CPU-bound)")
    parser.add_argument("--dialects", nargs="+", choices=[d for d in DIALECTS if d != SAE_DIALECT],
                        help="Subset of dialects to generate (default: all)")
    parser.add_argument("--list-dialects", action="store_true", help="Print Multi-VALUE class names and exit")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if args.list_dialects:
        from multivalue import Dialects

        print("\n".join(sorted(c for c in dir(Dialects) if c[0].isupper())))
        return

    out_dir = os.path.join(_load_datasets_dir(args.config), DIRECTORY_NAME)
    os.makedirs(out_dir, exist_ok=True)

    stories = _load_stories(args.split, args.num_samples)
    log.info(f"Loaded {len(stories)} CoQA stories (split={args.split})")

    sae_path = os.path.join(out_dir, FILE_NAME_FORMAT.format(dialect=SAE_DIALECT))
    _write(sae_path, stories)
    log.info(f"Wrote {len(stories)} lines -> {sae_path}")

    dialects = args.dialects or [d for d in DIALECTS if d != SAE_DIALECT]
    for dialect in dialects:
        log.info(f"Transforming -> {dialect} ({args.workers} worker(s))")
        lines = _transform(dialect, stories, args.seed, args.workers)
        path = os.path.join(out_dir, FILE_NAME_FORMAT.format(dialect=dialect))
        _write(path, lines)
        log.info(f"Wrote {len(lines)} lines -> {path}")

    log.info("Done!")


if __name__ == "__main__":
    main()
