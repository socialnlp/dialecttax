from . import bpe, unigram, wordpiece
from .tokenization import TOKENIZER_NAME_TO_TYPE


############
# DISPATCH #
############

TOKENS_DISPATCH = {
    "gpt": bpe.gpt_tokens,
    "bpe": bpe.bpe_tokens,
    "unigram": unigram.unigram_tokens,
    "wordpiece": wordpiece.wordpiece_tokens,
}

SUBTOKENIZE_DISPATCH = {
    "gpt": bpe.gpt_subtokenize,
    "bpe": bpe.bpe_subtokenize,
    "unigram": unigram.unigram_subtokenize,
    "wordpiece": wordpiece.wordpiece_subtokenize,
}

RESULTS_DISPATCH = {
    "gpt": bpe.gpt_results,
    "bpe": bpe.bpe_results,
    "unigram": unigram.unigram_results,
    "wordpiece": wordpiece.wordpiece_results,
}


def get_tokenizer_results(tokenizer_type: str):
    """Return the results function for a tokenizer type.

    Args:
        tokenizer_type: One of "bpe", "unigram", "wordpiece".

    Returns:
        Callable that computes a list of result dicts.
    """
    if tokenizer_type not in RESULTS_DISPATCH:
        raise ValueError(f"Tokenizer type `{tokenizer_type}` is invalid.")
    return RESULTS_DISPATCH[tokenizer_type]


def tokenize(tokenizer_name: str, texts: str | list[str]) -> list[list[str]]:
    """Tokenize texts using the appropriate tokenizer module.

    Args:
        tokenizer_name: Any key in TOKENIZER_NAME_TO_TYPE (e.g. "bpe", "gpt2", "llama", "unigram", "wordpiece").
        texts: Single string or list of strings to tokenize.

    Returns:
        List of token lists, one per input text.
    """
    tokenizer_type = TOKENIZER_NAME_TO_TYPE[tokenizer_name]
    return TOKENS_DISPATCH[tokenizer_type](texts, tokenizer_name=tokenizer_name)


def subtokenize(tokenizer_name: str, text: str) -> list[list[str]]:
    """Tokenize then group tokens into word-level clusters.

    Args:
        tokenizer_name: Any key in TOKENIZER_NAME_TO_TYPE (e.g. "bpe", "gpt2", "llama", "unigram", "wordpiece").
        text: Single string to tokenize and subtokenize.

    Returns:
        List of token groups, one per word boundary.
    """
    tokens = tokenize(tokenizer_name, text)[0]
    tokenizer_type = TOKENIZER_NAME_TO_TYPE[tokenizer_name]
    return SUBTOKENIZE_DISPATCH[tokenizer_type](tokens)
