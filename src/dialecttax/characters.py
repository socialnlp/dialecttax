"""Forced character-level tokenization for language models.

Splits each subword token into its constituent characters while preserving
the tokenizer's internal representation (e.g., the Ġ space prefix in BPE).
Special tokens are kept as canonical single-token IDs.

Follows Zheng et al. (2025), "Broken Tokens? Your Language Model can
Secretly Handle Non-Canonical Tokenizations."
"""

import logging

import torch

log = logging.getLogger(__name__)


def _get_special_token_ids(tokenizer):
    """Return the set of all special/added token IDs.

    Args:
        tokenizer: HuggingFace tokenizer.

    Returns:
        Set of integer token IDs that are special tokens.
    """
    return set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())


def text_to_char_ids(text, tokenizer):
    """Convert text to character-level token IDs, preserving special tokens.

    For each canonical token, splits its token string into individual characters
    and maps each back to a token ID via the vocabulary. This preserves internal
    markers (e.g., Ġ space prefix in BPE) as separate tokens.

    Args:
        text: Input string.
        tokenizer: HuggingFace tokenizer.

    Returns:
        List of token IDs.
    """
    canonical_ids = tokenizer.encode(text, add_special_tokens=False)
    special_ids = _get_special_token_ids(tokenizer)
    vocab = tokenizer.get_vocab()

    result = []
    for tid in canonical_ids:
        if tid in special_ids:
            result.append(tid)
        else:
            # Get the token string (e.g., "Ġworld") and split into characters
            token_str = tokenizer.convert_ids_to_tokens(tid)
            for char in token_str:
                if char in vocab:
                    result.append(vocab[char])
                else:
                    # Fallback: encode the character independently
                    result.extend(tokenizer.encode(char, add_special_tokens=False))

    return result


def char_tokenize(text, tokenizer, add_bos=True):
    """Produce character-level input_ids tensor, preserving special tokens.

    Args:
        text: Input string.
        tokenizer: HuggingFace tokenizer.
        add_bos: If True, prepend the BOS token (if not already present).

    Returns:
        input_ids tensor of shape (1, seq_len).
    """
    ids = text_to_char_ids(text, tokenizer)
    if add_bos and tokenizer.bos_token_id is not None and (not ids or ids[0] != tokenizer.bos_token_id):
        ids = [tokenizer.bos_token_id] + ids
    return torch.tensor([ids], dtype=torch.long)
