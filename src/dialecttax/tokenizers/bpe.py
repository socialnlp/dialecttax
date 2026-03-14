import re
import string
from tqdm import tqdm

import numpy as np
import tiktoken

from dialecttax.tokenizers.tokenization import RESULT_COLUMNS, TOKENIZER_NAME_MAP, get_tokenizer
from dialecttax.utils import divide_nan


#######
# BPE #
#######

PREFIX_DISPATCH = {
    "gpt2": "Ġ",
    "gemma": "▁",
    "llama": "Ġ",
    "qwen": "Ġ",
}


def bpe_punctuation(word: str, tokenizer_name: str = "gpt2") -> set[str]:
    """Strip punctuation and prepend word-boundary prefix for vocab lookup.

    Args:
        word: Raw whitespace-delimited word.
        tokenizer_name: Key into PREFIX_DISPATCH for the boundary prefix.

    Returns:
        Formatted word fragments, split on apostrophes.
    """
    prefix = PREFIX_DISPATCH.get(tokenizer_name, "Ġ")
    punctuations = string.punctuation.replace("'", '')
    word = word.translate(str.maketrans(punctuations, ' '*len(punctuations))).replace(' ', '')
    word = prefix + word
    # apostrophes treated differently
    # a word like can't -> {prefix}can, 't
    if "'" in word:
        results = []
        i_apostrophes = list(re.finditer(r"(')", word))
        i, j = 0, 0
        while i < len(i_apostrophes):
            results.append(word[j:i_apostrophes[i].start(0)])
            j = i_apostrophes[i].start(0)
            i += 1
        results.append(word[i_apostrophes[-1].start(0):])
        return set(results)
    return set([word])


def bpe_results(corpus: list[dict], row_names: list[str], tokenizer_name: str = "bpe", row_text: str = "text") -> list[dict]:
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
    vocab.add("ĠurlLink")
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
        result["tokens"] = tokens
        result["encoded"] = token_ids

        result["n_tokens"] = len(tokens)
        result["n_types"] = len(set(tokens))

        result["n_words"] = len(row[row_text].split())
        result["fertility"] = divide_nan(result["n_tokens"], result["n_words"])

        row_no_punctuation = row[row_text].translate(str.maketrans(string.punctuation, ' '*len(string.punctuation)))
        words_no_punctuation = row_no_punctuation.split()
        n_total_words = len(words_no_punctuation)
        n_words = len(set(words_no_punctuation))

        words = row[row_text].split()
        words_ = set()
        for w in words:
            words_ = words_.union(bpe_punctuation(w, tokenizer_name=tokenizer_name))
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


def bpe_tokens(texts: str | list[str], tokenizer_name: str = "bpe") -> list[list[str]]:
    """Tokenize one or more texts with a BPE-family tokenizer.

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


def bpe_subtokenize(tokens: list[str], tokenizer_name: str = "bpe") -> list[list[str]]:
    """Group BPE tokens into word-level clusters via word-boundary prefix.

    Args:
        tokens: Flat list of BPE tokens.
        tokenizer_name: Tokenizer name, used to select the prefix ("▁" for gemma, "Ġ" otherwise).

    Returns:
        List of token groups, one per word boundary.
    """
    prefix = PREFIX_DISPATCH.get(tokenizer_name, "Ġ")
    result = []
    curr_index = 0
    next_index = 1
    n_tokens = len(tokens)
    while next_index <= n_tokens:
        if next_index == n_tokens or tokens[next_index].startswith(prefix):
            result.append(tokens[curr_index:next_index])
            curr_index = next_index
        next_index += 1
    return result


#######
# GPT #
#######

def gpt_punctuation(word: str) -> set[str]:
    """Strip punctuation and prepend space for tiktoken vocab lookup.

    Args:
        word: Raw whitespace-delimited word.

    Returns:
        Formatted word fragments for vocab intersection.
    """
    punctuations = string.punctuation.replace("'", "")
    word = word.translate(str.maketrans(punctuations, " " * len(punctuations))).replace(" ", "")
    word = " " + word
    # apostrophes treated differently
    # a word like can't -> " can", "'t"
    if "'" in word:
        results = []
        i_apostrophes = list(re.finditer(r"(')", word))
        i, j = 0, 0
        while i < len(i_apostrophes):
            results.append(word[j:i_apostrophes[i].start(0)])
            j = i_apostrophes[i].start(0)
            i += 1
        results.append(word[i_apostrophes[-1].start(0):])
        return set(results)
    return {word}


def gpt_results(corpus: list[dict], row_names: list[str], tokenizer_name: str = "gpt-5", row_text: str = "text") -> list[dict]:
    """Compute tokenization metrics for each row using tiktoken.

    Args:
        corpus: Row dicts, each containing at least a text field.
        row_names: Column names to carry through from each row.
        tokenizer_name: OpenAI model name for tiktoken encoding.
        row_text: Key for the text field in each row dict.

    Returns:
        List of dicts with token counts, fertility, and vocab coverage.
    """
    model = TOKENIZER_NAME_MAP[tokenizer_name]
    tokenizer = tiktoken.encoding_for_model(model)
    vocab = {b.decode("utf-8", errors="replace") for b in tokenizer._mergeable_ranks}
    vocab.add(" urlLink")

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

        token_ids = tokenizer.encode(row[row_text])
        tokens = [tokenizer.decode([tid]) for tid in token_ids]
        result["tokens"] = tokens
        result["encoded"] = token_ids

        result["n_tokens"] = len(tokens)
        result["n_types"] = len(set(tokens))

        result["n_words"] = len(row[row_text].split())
        result["fertility"] = divide_nan(result["n_tokens"], result["n_words"])

        row_no_punctuation = row[row_text].translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
        words_no_punctuation = row_no_punctuation.split()
        n_total_words = len(words_no_punctuation)
        n_words = len(set(words_no_punctuation))

        words = row[row_text].split()
        words_ = set()
        for w in words:
            words_ = words_.union(gpt_punctuation(w))
        words_in_vocab = vocab.intersection(words_)
        result["p_in_vocab"] = divide_nan(len(words_in_vocab), len(words_))

        ids_no_punct = tokenizer.encode(row_no_punctuation)
        tokens_no_punct = [tokenizer.decode([tid]) for tid in ids_no_punct]
        result["avg_tokens_per_word"] = divide_nan(len(tokens_no_punct), n_total_words)
        types_no_punctuation = set(tokens_no_punct)
        result["avg_types_per_word"] = divide_nan(len(types_no_punctuation), n_words)

        result["RID"] = i
        for n in row_names:
            result[n] = row[n]
        results.append(result)
    return results


def gpt_tokens(texts: str | list[str], tokenizer_name: str = "gpt-5") -> list[list[str]]:
    """Tokenize one or more texts using tiktoken.

    Args:
        texts: Single string or list of strings to tokenize.
        tokenizer_name: OpenAI model name for tiktoken encoding.

    Returns:
        List of token lists, one per input text.
    """
    if isinstance(texts, str):
        texts = [texts]

    model = TOKENIZER_NAME_MAP[tokenizer_name]
    tokenizer = tiktoken.encoding_for_model(model)
    tokenizations = []
    for text in texts:
        if text is np.nan:
            tokenizations.append([])
        else:
            token_ids = tokenizer.encode(text)
            tokenizations.append([tokenizer.decode([tid]) for tid in token_ids])
    return tokenizations


def gpt_subtokenize(tokens: list[str]) -> list[list[str]]:
    """Group tiktoken tokens into word-level clusters via leading space.

    Args:
        tokens: Flat list of decoded tiktoken token strings.

    Returns:
        List of token groups, one per word boundary.
    """
    result = []
    curr_index = 0
    next_index = 1
    n_tokens = len(tokens)
    while next_index <= n_tokens:
        if next_index == n_tokens or tokens[next_index].startswith(" "):
            result.append(tokens[curr_index:next_index])
            curr_index = next_index
        next_index += 1
    return result
