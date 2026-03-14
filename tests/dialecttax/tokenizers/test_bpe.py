"""Tests for dialecttax.tokenizers.bpe."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import tiktoken

from dialecttax.tokenizers.bpe import (
    bpe_punctuation,
    bpe_subtokenize,
    bpe_tokens,
    gpt_punctuation,
    gpt_results,
    gpt_subtokenize,
    gpt_tokens,
)


###################
# BPE PUNCTUATION #
###################

class TestBPEPunctuation:
    def test_simple_word(self):
        assert bpe_punctuation("hello") == {"Ġhello"}

    def test_word_with_punctuation(self):
        result = bpe_punctuation("hello!")
        assert "Ġhello" in result

    def test_apostrophe_split(self):
        result = bpe_punctuation("can't")
        assert "Ġcan" in result
        assert "'t" in result

    def test_multiple_apostrophes(self):
        result = bpe_punctuation("it's'ok")
        assert len(result) == 3

    def test_only_punctuation(self):
        result = bpe_punctuation("...")
        assert "Ġ" in result


####################
# BPE SUBTOKENIZE #
####################

class TestBPESubtokenize:
    def test_single_word(self):
        tokens = ["Ġhello"]
        assert bpe_subtokenize(tokens) == [["Ġhello"]]

    def test_two_words(self):
        tokens = ["Ġhello", "Ġworld"]
        assert bpe_subtokenize(tokens) == [["Ġhello"], ["Ġworld"]]

    def test_subword_tokens(self):
        tokens = ["Ġun", "believ", "able"]
        assert bpe_subtokenize(tokens) == [["Ġun", "believ", "able"]]

    def test_mixed(self):
        tokens = ["Ġhello", "Ġun", "believ", "able", "Ġworld"]
        assert bpe_subtokenize(tokens) == [
            ["Ġhello"],
            ["Ġun", "believ", "able"],
            ["Ġworld"],
        ]

    def test_empty(self):
        assert bpe_subtokenize([]) == []


##############
# BPE TOKENS #
##############

class TestBPETokens:
    def _mock_tokenizer(self):
        mock_bpe = MagicMock()
        mock_bpe.tokenize.side_effect = lambda t: list(t.split())
        return mock_bpe

    @patch("dialecttax.tokenizers.bpe.get_tokenizer", create=True)
    def test_single_string(self, mock_get):
        mock_get.return_value = self._mock_tokenizer()
        result = bpe_tokens("hello world")
        assert result == [["hello", "world"]]

    @patch("dialecttax.tokenizers.bpe.get_tokenizer", create=True)
    def test_list_of_strings(self, mock_get):
        mock_get.return_value = self._mock_tokenizer()
        result = bpe_tokens(["hello world", "foo bar"])
        assert len(result) == 2
        assert result[0] == ["hello", "world"]
        assert result[1] == ["foo", "bar"]

    @patch("dialecttax.tokenizers.bpe.get_tokenizer", create=True)
    def test_nan_input(self, mock_get):
        mock_get.return_value = self._mock_tokenizer()
        result = bpe_tokens([np.nan, "hello"])
        assert result[0] == []
        assert result[1] == ["hello"]

    @patch("dialecttax.tokenizers.bpe.get_tokenizer", create=True)
    def test_tokenizer_name_forwarded(self, mock_get):
        mock_get.return_value = self._mock_tokenizer()
        bpe_tokens("text", tokenizer_name="custom")
        mock_get.assert_called_with("custom")


###################
# GPT PUNCTUATION #
###################

class TestGPTPunctuation:
    def test_simple_word(self):
        assert gpt_punctuation("hello") == {" hello"}

    def test_word_with_punctuation(self):
        result = gpt_punctuation("hello!")
        assert " hello" in result

    def test_apostrophe_split(self):
        result = gpt_punctuation("can't")
        assert " can" in result
        assert "'t" in result

    def test_multiple_apostrophes(self):
        result = gpt_punctuation("it's'ok")
        assert len(result) == 3

    def test_only_punctuation(self):
        result = gpt_punctuation("...")
        assert " " in result

    def test_mirrors_bpe_punctuation(self):
        """gpt_punctuation uses space prefix where bpe_punctuation uses Ġ."""
        words = ["hello", "can't", "test!", "foo.bar"]
        for word in words:
            bpe_result = bpe_punctuation(word)
            gpt_result = gpt_punctuation(word)
            bpe_normalized = {s.replace("Ġ", " ") for s in bpe_result}
            assert bpe_normalized == gpt_result


####################
# GPT SUBTOKENIZE #
####################

class TestGPTSubtokenize:
    def test_single_word(self):
        tokens = [" hello"]
        assert gpt_subtokenize(tokens) == [[" hello"]]

    def test_two_words(self):
        tokens = [" hello", " world"]
        assert gpt_subtokenize(tokens) == [[" hello"], [" world"]]

    def test_subword_tokens(self):
        tokens = [" un", "believ", "able"]
        assert gpt_subtokenize(tokens) == [[" un", "believ", "able"]]

    def test_mixed(self):
        tokens = [" hello", " un", "believ", "able", " world"]
        assert gpt_subtokenize(tokens) == [
            [" hello"],
            [" un", "believ", "able"],
            [" world"],
        ]

    def test_empty(self):
        assert gpt_subtokenize([]) == []

    def test_first_token_no_space(self):
        """First token often lacks a leading space (start of text)."""
        tokens = ["Hello", " world"]
        assert gpt_subtokenize(tokens) == [["Hello"], [" world"]]


##############
# GPT TOKENS #
##############

class TestGPTTokens:
    def test_single_string(self):
        result = gpt_tokens("hello")
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert "".join(result[0]) == "hello"

    def test_list_of_strings(self):
        result = gpt_tokens(["hello", "world"])
        assert len(result) == 2
        assert "".join(result[0]) == "hello"
        assert "".join(result[1]) == "world"

    def test_nan_input(self):
        result = gpt_tokens([np.nan, "hello"])
        assert result[0] == []
        assert len(result[1]) >= 1

    def test_roundtrip(self):
        text = "The quick brown fox jumps over the lazy dog."
        result = gpt_tokens(text)
        assert "".join(result[0]) == text

    def test_tokens_are_strings(self):
        result = gpt_tokens("hello world")
        for token in result[0]:
            assert isinstance(token, str)

    def test_model_parameter(self):
        result_4o = gpt_tokens("hello", model="gpt-4o")
        result_4 = gpt_tokens("hello", model="gpt-4")
        # Both should roundtrip correctly regardless of encoding
        assert "".join(result_4o[0]) == "hello"
        assert "".join(result_4[0]) == "hello"

    def test_empty_string(self):
        result = gpt_tokens("")
        assert result == [[]]

    def test_unicode(self):
        text = "cafe\u0301"
        result = gpt_tokens(text)
        assert "".join(result[0]) == text

    def test_multiword_token_count(self):
        """Longer text should produce multiple tokens."""
        text = "This is a somewhat longer sentence with several words."
        result = gpt_tokens(text)
        assert len(result[0]) > 1


################
# GPT RESULTS #
################

class TestGPTResults:
    @patch("dialecttax.tokenizers.bpe.divide_nan", create=True, side_effect=lambda a, b: a / b if b else float("nan"))
    @patch("dialecttax.tokenizers.bpe.RESULT_COLUMNS", create=True, new=[
        "n_chars", "tokens", "n_tokens", "n_types", "n_words",
        "fertility", "p_in_vocab", "avg_tokens_per_word",
        "avg_types_per_word", "RID",
    ])
    def test_basic_corpus(self, mock_divide):
        corpus = [{"text": "hello world", "id": 1}]
        df = gpt_results(corpus, ["id"], model="gpt-4o")
        assert len(df) == 1
        assert df["n_chars"].iloc[0] == len("hello world")
        assert df["n_words"].iloc[0] == 2
        assert df["n_tokens"].iloc[0] >= 1
        assert df["id"].iloc[0] == 1

    @patch("dialecttax.tokenizers.bpe.divide_nan", create=True, side_effect=lambda a, b: a / b if b else float("nan"))
    @patch("dialecttax.tokenizers.bpe.RESULT_COLUMNS", create=True, new=[
        "n_chars", "tokens", "n_tokens", "n_types", "n_words",
        "fertility", "p_in_vocab", "avg_tokens_per_word",
        "avg_types_per_word", "RID",
    ])
    def test_none_text_row(self, mock_divide):
        corpus = [{"text": None, "id": 1}]
        df = gpt_results(corpus, ["id"], model="gpt-4o")
        assert len(df) == 1
        assert np.isnan(df["n_chars"].iloc[0])

    @patch("dialecttax.tokenizers.bpe.divide_nan", create=True, side_effect=lambda a, b: a / b if b else float("nan"))
    @patch("dialecttax.tokenizers.bpe.RESULT_COLUMNS", create=True, new=[
        "n_chars", "tokens", "n_tokens", "n_types", "n_words",
        "fertility", "p_in_vocab", "avg_tokens_per_word",
        "avg_types_per_word", "RID",
    ])
    def test_multiple_rows(self, mock_divide):
        corpus = [
            {"text": "hello", "id": 1},
            {"text": "world", "id": 2},
        ]
        df = gpt_results(corpus, ["id"], model="gpt-4o")
        assert len(df) == 2
        assert df["RID"].tolist() == [0, 1]

    @patch("dialecttax.tokenizers.bpe.divide_nan", create=True, side_effect=lambda a, b: a / b if b else float("nan"))
    @patch("dialecttax.tokenizers.bpe.RESULT_COLUMNS", create=True, new=[
        "n_chars", "tokens", "n_tokens", "n_types", "n_words",
        "fertility", "p_in_vocab", "avg_tokens_per_word",
        "avg_types_per_word", "RID",
    ])
    def test_fertility(self, mock_divide):
        corpus = [{"text": "hello world", "id": 1}]
        df = gpt_results(corpus, ["id"], model="gpt-4o")
        n_tokens = df["n_tokens"].iloc[0]
        n_words = df["n_words"].iloc[0]
        expected_fertility = n_tokens / n_words
        assert df["fertility"].iloc[0] == pytest.approx(expected_fertility)
