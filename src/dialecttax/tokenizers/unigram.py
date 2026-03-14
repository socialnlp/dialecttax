import re
import string
from tqdm import tqdm

import numpy as np

from dialecttax.tokenizers.tokenization import RESULT_COLUMNS, get_tokenizer
from dialecttax.utils import divide_nan


###########
# UNIGRAM #
###########

def unigram_punctuation(word: str) -> set[str]:
    """Strip punctuation and prepend ▁ for Unigram vocab lookup.

    Args:
        word: Raw whitespace-delimited word.

    Returns:
        Unigram-formatted word fragments, split on apostrophes.
    """
    punctuations = string.punctuation.replace("'", '')
    word = word.translate(str.maketrans(punctuations, ' '*len(punctuations))).replace(' ', '')
    word = '▁' + word
    # apostrophes treated differently
    # a word like can't -> ▁can, t
    if "'" in word:
        return set(word.split("'"))
    return set([word])


def unigram_results(
    corpus: list[dict],
    row_names: list[str],
    tokenizer_name: str = "unigram",
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
    vocab.add("▁urlLink")
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
        # row_pre_tokenized = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(row_normalized)
        row_no_punctuation = row_normalized.translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        words_no_punctuation = row_no_punctuation.split()
        n_total_words = len(words_no_punctuation)
        n_words = len(set(words_no_punctuation))

        words = row_normalized.split()
        words_ = set()
        for w in words:
            words_ = words_.union(unigram_punctuation(w))
        words_in_vocab = vocab.intersection(words_)
        result["p_in_vocab"] = divide_nan(len(words_in_vocab), len(words_))

        tokens_no_punctuation = tokenizer.tokenize(row_no_punctuation)
        result["avg_tokens_per_word"] = divide_nan(len(tokens_no_punctuation), n_total_words)
        types_no_punctuation = set(tokens_no_punctuation)
        result["avg_types_per_word"] = divide_nan(len(types_no_punctuation), n_words)

        result["RID"] = i
        for n in row_names:
            result[n] = row[n]
        results.append(result)
    return results


def unigram_tokens(
    texts: str | list[str],
    tokenizer_name: str = "unigram",
) -> list[list[str]]:
    """Tokenize one or more texts with a Unigram-family tokenizer.

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


def unigram_subtokenize(tokens: list[str]) -> list[list[str]]:
    """Group Unigram tokens into word-level clusters via the ▁ prefix.

    Args:
        tokens: Flat list of Unigram tokens.

    Returns:
        List of token groups, one per word boundary.
    """
    result = []
    curr_index = 0
    next_index = 1
    n_tokens = len(tokens)
    while next_index <= n_tokens:
        if next_index == n_tokens or tokens[next_index].startswith('▁'):
            result.append(tokens[curr_index:next_index])
            curr_index = next_index
        next_index += 1
    return result
