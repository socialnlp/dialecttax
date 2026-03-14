import re
import string
from tqdm import tqdm

import numpy as np

from dialecttax.tokenizers.tokenization import RESULT_COLUMNS, get_tokenizer
from dialecttax.utils import divide_nan


#############
# WORDPIECE #
#############

def wordpiece_results(
    corpus: list[dict],
    row_names: list[str],
    tokenizer_name: str = "wordpiece",
    row_text: str = "text",
) -> list[dict]:
    """Compute tokenization metrics for each row in a corpus.

    Args:
        corpus: Row dicts, each containing at least a text field.
        row_names: Column names to carry through from each row.
        tokenizer_name: Key into TOKENIZER_NAME_MAP.
        row_text: Key for the text field in each row dict.

    Returns:
        List of dicts with token counts, fertility, and vocab coverage.
    """
    tokenizer = get_tokenizer(tokenizer_name)

    vocab = set(tokenizer.vocab.keys())
    vocab.add("urllink")
    results = []
    for i, row in tqdm(enumerate(corpus)):
        if row[row_text] is None:
            result = {col: None for col in RESULT_COLUMNS}
            result["RID"] = i
            for n in row_names:
                result[n] = row[n]
            results.append(result)
            continue

        result = {}

        result["n_chars"] = len(row[row_text])

        tokens = tokenizer.tokenize(row[row_text])
        token_ids = tokenizer.encode(row[row_text])
        result["n_tokens"] = len(tokens)
        result["n_types"] = len(set(tokens))

        result["tokens"] = tokens
        result["encoded"] = token_ids
        result["n_words"] = len(row[row_text].split())
        result["fertility"] = divide_nan(result["n_tokens"], result["n_words"])

        normalizer = tokenizer.backend_tokenizer.normalizer
        row_normalized = normalizer.normalize_str(row[row_text]) if normalizer else row[row_text]
        row_no_punctuation = row_normalized.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        words_no_punctuation = row_no_punctuation.split()
        n_total_words = len(words_no_punctuation)
        n_words = len(set(words_no_punctuation))

        words = words_no_punctuation
        words_in_vocab = vocab.intersection(words)
        result["p_in_vocab"] = divide_nan(len(words_in_vocab), len(words))

        tokens_no_punctuation = tokenizer.tokenize(row_no_punctuation)
        result["avg_tokens_per_word"] = divide_nan(len(tokens_no_punctuation), n_total_words)
        types_no_punctuation = set(tokens_no_punctuation)
        result["avg_types_per_word"] = divide_nan(len(types_no_punctuation), n_words)

        result["RID"] = i
        for n in row_names:
            result[n] = row[n]
        results.append(result)
    return results


def wordpiece_tokens(
    texts: str | list[str],
    tokenizer_name: str = "wordpiece",
) -> list[list[str]]:
    """Tokenize one or more texts with a WordPiece-family tokenizer.

    Args:
        texts: Single string or list of strings to tokenize.
        tokenizer_name: Key into TOKENIZER_NAME_MAP.

    Returns:
        List of token lists, one per input text.
    """
    if isinstance(texts, str):
        texts = [texts]

    tokenizer = get_tokenizer(tokenizer_name)
    tokenizations = []
    for text in texts:
        if text is np.nan:
            tokenizations.append([])
        else:
            tokens = tokenizer.tokenize(text)
            tokenizations.append(tokens)
    return tokenizations


def wordpiece_subtokenize(tokens: list[str]) -> list[list[str]]:
    """Group WordPiece tokens into word-level clusters via ## prefix.

    Args:
        tokens: Flat list of WordPiece tokens.

    Returns:
        List of token groups, one per word boundary.
    """
    result = []
    curr_index = 0
    next_index = 1
    n_tokens = len(tokens)
    while next_index <= n_tokens:
        if next_index == n_tokens or not tokens[next_index].startswith("##"):
            result.append(tokens[curr_index:next_index])
            curr_index = next_index
        next_index += 1
    return result
