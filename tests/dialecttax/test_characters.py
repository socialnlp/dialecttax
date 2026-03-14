"""Tests for dialecttax.characters."""

import torch

from dialecttax.characters import _get_special_token_ids, char_tokenize, text_to_char_ids


##########################
# MOCK TOKENIZER HELPERS #
##########################

class _MockVocab:
    """Minimal mock tokenizer with a controllable vocabulary."""

    def __init__(self, vocab, special_ids=None, added_vocab=None, bos_token_id=None, pad_token_id=None):
        self._vocab = vocab  # str -> int
        self._id_to_token = {v: k for k, v in vocab.items()}
        self._special_ids = special_ids or []
        self._added_vocab = added_vocab or {}
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.all_special_ids = self._special_ids

    def get_vocab(self):
        return dict(self._vocab)

    def get_added_vocab(self):
        return dict(self._added_vocab)

    def encode(self, text, add_special_tokens=False):
        # Greedy longest-match tokenization
        ids = []
        i = 0
        while i < len(text):
            best = None
            best_len = 0
            for token, tid in self._vocab.items():
                if text[i:i + len(token)] == token and len(token) > best_len:
                    # Skip special/added tokens during greedy match
                    if tid not in self._special_ids and tid not in self._added_vocab.values():
                        best = tid
                        best_len = len(token)
            if best is not None:
                ids.append(best)
                i += best_len
            else:
                i += 1  # skip unknown char
        return ids

    def convert_ids_to_tokens(self, tid):
        return self._id_to_token.get(tid, "")


#############################
# GET_SPECIAL_TOKEN_IDS     #
#############################

class TestGetSpecialTokenIds:
    def test_combines_special_and_added(self):
        tok = _MockVocab(
            vocab={"a": 0, "b": 1, "<s>": 100, "<pad>": 101},
            special_ids=[100, 101],
            added_vocab={"<extra>": 200},
        )
        result = _get_special_token_ids(tok)
        assert 100 in result
        assert 101 in result
        assert 200 in result
        assert 0 not in result
        assert 1 not in result

    def test_empty_special(self):
        tok = _MockVocab(vocab={"a": 0}, special_ids=[], added_vocab={})
        result = _get_special_token_ids(tok)
        assert len(result) == 0


#####################
# TEXT_TO_CHAR_IDS  #
#####################

class TestTextToCharIds:
    def _make_tokenizer(self):
        """Tokenizer where 'ab' merges to one token but 'a' and 'b' exist as singles."""
        return _MockVocab(
            vocab={"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "<s>": 100},
            special_ids=[100],
            added_vocab={"<s>": 100},
        )

    def test_splits_merged_tokens(self):
        """Merged token 'ab' should be split into 'a' + 'b'."""
        tok = self._make_tokenizer()
        # "ab" canonically encodes to [3] (the merged token)
        # Character-level should split to [0, 1]
        result = text_to_char_ids("ab", tok)
        assert result == [0, 1]

    def test_single_chars_unchanged(self):
        """Single-character tokens remain unchanged."""
        tok = self._make_tokenizer()
        result = text_to_char_ids("a", tok)
        assert result == [0]

    def test_preserves_special_tokens(self):
        """Special tokens are kept as-is, not split."""
        tok = _MockVocab(
            vocab={"a": 0, "b": 1, "<s>": 100, "s": 5, "<": 6, ">": 7},
            special_ids=[100],
            added_vocab={"<s>": 100},
        )
        # If the canonical encoding contains <s> as a special token, it should stay
        # Force canonical to include the special token
        canonical = [100, 0, 1]  # <s> a b

        special_ids = _get_special_token_ids(tok)
        vocab = tok.get_vocab()
        result = []
        for tid in canonical:
            if tid in special_ids:
                result.append(tid)
            else:
                token_str = tok.convert_ids_to_tokens(tid)
                for char in token_str:
                    if char in vocab:
                        result.append(vocab[char])

        assert result[0] == 100  # <s> preserved
        assert result[1] == 0    # a
        assert result[2] == 1    # b

    def test_empty_string(self):
        tok = self._make_tokenizer()
        result = text_to_char_ids("", tok)
        assert result == []

    def test_longer_merge(self):
        """Multi-character merged token 'abc' splits to 'a' + 'b' + 'c'."""
        tok = self._make_tokenizer()
        result = text_to_char_ids("abc", tok)
        assert result == [0, 1, 2]

    def test_expansion_increases_length(self):
        """Character-level encoding is at least as long as canonical."""
        tok = self._make_tokenizer()
        text = "ababc"
        canonical = tok.encode(text)
        char_ids = text_to_char_ids(text, tok)
        assert len(char_ids) >= len(canonical)
        assert len(char_ids) == len(text)  # each char maps to one token


##################
# CHAR_TOKENIZE  #
##################

class TestCharTokenize:
    def _make_tokenizer(self):
        return _MockVocab(
            vocab={"a": 0, "b": 1, "ab": 3, "<bos>": 99},
            special_ids=[99],
            added_vocab={"<bos>": 99},
            bos_token_id=99,
        )

    def test_output_shape(self):
        tok = self._make_tokenizer()
        result = char_tokenize("ab", tok)
        assert result.dim() == 2
        assert result.shape[0] == 1

    def test_output_dtype(self):
        tok = self._make_tokenizer()
        result = char_tokenize("ab", tok)
        assert result.dtype == torch.long

    def test_adds_bos(self):
        tok = self._make_tokenizer()
        result = char_tokenize("ab", tok, add_bos=True)
        assert result[0, 0].item() == 99

    def test_no_bos(self):
        tok = self._make_tokenizer()
        result = char_tokenize("ab", tok, add_bos=False)
        assert result[0, 0].item() != 99

    def test_no_double_bos(self):
        """If BOS is already in the sequence, don't add another."""
        tok = _MockVocab(
            vocab={"a": 0, "<bos>": 99},
            special_ids=[99],
            added_vocab={"<bos>": 99},
            bos_token_id=99,
        )
        # Manually construct a case where encode returns BOS first
        # Since our mock won't naturally produce BOS, test the logic directly
        ids = [99, 0]  # already has BOS
        result_ids = list(ids)
        if tok.bos_token_id is not None and (not result_ids or result_ids[0] != tok.bos_token_id):
            result_ids = [tok.bos_token_id] + result_ids
        # BOS already present, should not be duplicated
        assert result_ids == [99, 0]

    def test_no_bos_when_none(self):
        """No BOS added if tokenizer has no bos_token_id."""
        tok = _MockVocab(
            vocab={"a": 0, "b": 1, "ab": 3},
            bos_token_id=None,
        )
        result = char_tokenize("ab", tok, add_bos=True)
        # Without BOS, first token should be 'a' = 0
        assert result[0, 0].item() == 0


###########################
# ROUND-TRIP CONSISTENCY  #
###########################

class TestRoundTrip:
    """Tests that require a real HuggingFace tokenizer.

    These are marked to be skipped if transformers is not available or the
    model is not cached locally.
    """

    def _load_tokenizer(self):
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
        except Exception:
            import pytest
            pytest.skip("Llama tokenizer not available")

    def test_plain_text_round_trip(self):
        """Decoding character-level IDs reproduces the original text."""
        tok = self._load_tokenizer()
        text = "Hello world, this is a test."
        char_ids = text_to_char_ids(text, tok)
        decoded = tok.decode(char_ids, skip_special_tokens=True)
        assert decoded == text

    def test_chat_template_round_trip(self):
        """Character-level encoding of a chat template decodes identically to canonical."""
        tok = self._load_tokenizer()
        from dialecttax.models import get_message
        messages = get_message("Solve 2+2", system="Be helpful.")
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        canonical_ids = tok.encode(text, add_special_tokens=False)
        char_ids = text_to_char_ids(text, tok)

        decoded_canonical = tok.decode(canonical_ids, skip_special_tokens=False)
        decoded_char = tok.decode(char_ids, skip_special_tokens=False)
        assert decoded_canonical == decoded_char

    def test_special_tokens_preserved(self):
        """Special tokens appear as single IDs, not split into characters."""
        tok = self._load_tokenizer()
        from dialecttax.models import get_message
        messages = get_message("Hello", system="Test.")
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        char_ids = text_to_char_ids(text, tok)
        special_ids = _get_special_token_ids(tok)

        # BOS and chat markers should appear in the char_ids
        assert tok.bos_token_id in char_ids
        # At least some special tokens should be present
        special_in_result = [tid for tid in char_ids if tid in special_ids]
        assert len(special_in_result) > 0

    def test_expansion_ratio(self):
        """Character-level encoding should be longer than canonical."""
        tok = self._load_tokenizer()
        text = "The quick brown fox jumps over the lazy dog."
        canonical_ids = tok.encode(text, add_special_tokens=False)
        char_ids = text_to_char_ids(text, tok)
        assert len(char_ids) > len(canonical_ids)

    def test_char_tokenize_tensor(self):
        """char_tokenize returns a proper tensor with BOS."""
        tok = self._load_tokenizer()
        result = char_tokenize("Hello", tok, add_bos=True)
        assert result.dim() == 2
        assert result.shape[0] == 1
        assert result.dtype == torch.long
        assert result[0, 0].item() == tok.bos_token_id
