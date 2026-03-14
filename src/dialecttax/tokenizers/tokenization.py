from collections import Counter
from collections.abc import Callable
from functools import reduce
from operator import add

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

TOKENIZER_NAMES = ["bpe", "unigram", "wordpiece"]
TOKENIZER_NAME_MAP = {
    # BPE
    "bpe": "gpt-5",
    "gpt2": "openai-community/gpt2",
    "gemma": "google/gemma-3-27b-it",
    "llama": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen": "Qwen/Qwen3-32B",
    # Unigram
    "unigram": "t5-small",
    # WordPiece
    "wordpiece": "bert-base-uncased",
}

TOKENIZER_NAME_TO_TYPE = {
    # BPE (tiktoken)
    "bpe": "gpt",
    # BPE (HuggingFace)
    "gpt2": "bpe",
    "gemma": "bpe",
    "llama": "bpe",
    "qwen": "bpe",
    # Unigram
    "unigram": "unigram",
    # WordPiece
    "wordpiece": "wordpiece",
}

RESULT_COLUMNS = [
    "RID", "tokens", "encoded",
    "n_tokens", "n_types", "n_words", "n_chars",
    "fertility", "p_in_vocab", "avg_tokens_per_word", "avg_types_per_word",
]


def get_tokenizer(tokenizer_name: str):
    """Load a pretrained HuggingFace tokenizer by short name.

    Args:
        tokenizer_name: Key into TOKENIZER_NAME_MAP.

    Returns:
        HuggingFace tokenizer instance.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME_MAP[tokenizer_name])
    return tokenizer


##############
# EXTRACTION #
##############

def extract_tokens(ds: list[dict], row_text: str, tokenizer_name: str) -> dict[str, int]:
    """Extract token counts from all samples using a tokenizer.

    Args:
        ds: List of sample dicts.
        row_text: Key for the text field in each sample.
        tokenizer_name: Key into TOKENIZER_NAME_MAP.

    Returns:
        Dict mapping each decoded token string to its count, sorted by count descending.
    """
    import tiktoken

    is_tiktoken = TOKENIZER_NAME_TO_TYPE[tokenizer_name] == "gpt"
    if is_tiktoken:
        tokenizer = tiktoken.encoding_for_model(TOKENIZER_NAME_MAP[tokenizer_name])
    else:
        tokenizer = get_tokenizer(tokenizer_name)

    counts = Counter()
    for sample in ds:
        text = sample[row_text]
        if text is not None:
            if is_tiktoken:
                token_ids = tokenizer.encode(text)
                tokens = [tokenizer.decode([tid]) for tid in token_ids]
            else:
                raw_tokens = tokenizer.tokenize(text)
                tokens = [tokenizer.convert_tokens_to_string([t]) for t in raw_tokens]
            counts.update(tokens)
    return dict(counts.most_common())


##############
# VOCABULARY #
##############

def count_vocab(
    counts: dict[str, int],
    items: list[str],
) -> dict[str, int]:
    """Merge item frequencies into an existing count dict.

    Args:
        counts: Running vocabulary counts.
        items: New items to count and merge.

    Returns:
        Updated count dict.
    """
    return dict(reduce(add, map(Counter, [counts, Counter(items)])))


def get_counts(
    corpus: list[dict],
    run_get_items: Callable[[str | None], list[str]],
    row_text: str = "text",
) -> dict[str, int]:
    """Count item frequencies across all rows in a corpus.

    Args:
        corpus: List of row dicts.
        run_get_items: Extracts items from a text string.
        row_text: Key for the text field in each row dict.

    Returns:
        Dict mapping items to their total counts.
    """
    counts = {}
    for i, row in tqdm(enumerate(corpus)):
        counts = count_vocab(counts, run_get_items(row[row_text]))
    return counts


def get_words() -> Callable[[str | None], list[str]]:
    """Create a function that extracts words with punctuation removed.

    Returns:
        Function mapping text to whitespace-delimited word list.
    """
    def run_get_words(text):
        if text is None:
            return []
        row_no_punctuation = text.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        words_no_punctuation = row_no_punctuation.split()
        return words_no_punctuation
    return run_get_words


def get_tokens(tokenizer_model) -> Callable[[str | None], list[str]]:
    """Create a function that tokenizes text with a given tokenizer.

    Args:
        tokenizer_model: HuggingFace tokenizer instance.

    Returns:
        Function mapping text to token list.
    """
    def run_get_tokens(text):
        if text is None:
            return []
        return tokenizer_model.tokenize(text)
    return run_get_tokens


def get_row_counts_with_trait(
    corpus: list[dict],
    row_names: list[str],
    run_get_items: Callable[[str | None], list[str]],
    row_text: str = "text",
) -> list[dict]:
    """Count items per row, attaching demographic trait columns.

    Args:
        corpus: List of row dicts.
        row_names: Trait column names to include per row.
        run_get_items: Extracts items from text.
        row_text: Key for the text field in each row dict.

    Returns:
        List of dicts with trait values and item Counter per row.
    """
    def get_bin(x, bins):
        for i, edge in enumerate(bins):
            if x <= edge:
                return i - 1
        return i - 1

    result = []
    for row in tqdm(corpus):
        if COL_AGE_RANGE in row_names:
            row[COL_AGE_RANGE] = AGE_LABELS[get_bin(row[COL_AGE], AGE_BINS)]
        if COL_GENDER in row_names:
            row[COL_GENDER] = GENDER_MAPPING[row[COL_GENDER]]
        if COL_INCOME in row_names:
            income = JOB2INCOME.get(row[COL_JOB])
            row[COL_INCOME] = "indUnk" if income is None else income

        curr = {n: row[n] for n in row_names}
        curr["count"] = Counter(run_get_items(row[row_text]))
        result.append(curr)
    return result
