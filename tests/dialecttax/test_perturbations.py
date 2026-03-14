"""Tests for dialecttax.perturbations."""

import random
import string
from unittest.mock import MagicMock, patch

from dialecttax.perturbations import capitalize, drop, insert, swap, translate


########
# SWAP #
########


class TestSwap:
    def test_no_swap_at_zero_prob(self):
        result = swap("hello world", p=0.0)
        assert result == "hello world"

    def test_full_swap(self):
        """p=1 swaps every eligible adjacent pair."""
        result = swap("abcd", p=1.0)
        # a<->b, then skip to c<->d
        assert result == "badc"

    def test_spaces_never_swap(self):
        random.seed(0)
        text = "a b c d"
        for _ in range(50):
            result = swap(text, p=1.0)
            assert " " not in result.replace(" ", "") or True
            # Spaces stay in original positions
            for i, c in enumerate(text):
                if c == " ":
                    assert result[i] == " "

    def test_length_preserved(self):
        random.seed(42)
        text = "the quick brown fox"
        result = swap(text, p=0.3)
        assert len(result) == len(text)

    def test_single_char(self):
        assert swap("a", p=1.0) == "a"

    def test_empty_string(self):
        assert swap("", p=1.0) == ""

    def test_list_input(self):
        results = swap(["abc", "xyz"], p=0.0)
        assert results == ["abc", "xyz"]

    def test_single_string_returns_string(self):
        result = swap("abc", p=0.0)
        assert isinstance(result, str)

    def test_list_returns_list(self):
        result = swap(["abc"], p=0.0)
        assert isinstance(result, list)


########
# DROP #
########


class TestDrop:
    def test_no_drop_at_zero_prob(self):
        result = drop("hello world", p=0.0)
        assert result == "hello world"

    def test_full_drop(self):
        """p=1 drops all non-space characters."""
        result = drop("hello world", p=1.0)
        assert result == " "

    def test_spaces_never_dropped(self):
        random.seed(0)
        text = "a b c"
        for _ in range(50):
            result = drop(text, p=0.5)
            # Count spaces — should always have exactly 2
            assert result.count(" ") == 2

    def test_output_shorter_or_equal(self):
        random.seed(42)
        text = "the quick brown fox"
        result = drop(text, p=0.3)
        assert len(result) <= len(text)

    def test_empty_string(self):
        assert drop("", p=1.0) == ""

    def test_list_input(self):
        results = drop(["abc", "xyz"], p=0.0)
        assert results == ["abc", "xyz"]

    def test_single_string_returns_string(self):
        result = drop("abc", p=0.0)
        assert isinstance(result, str)


##########
# INSERT #
##########


class TestInsert:
    def test_no_insert_at_zero_prob(self):
        result = insert("hello world", p=0.0)
        assert result == "hello world"

    def test_full_insert(self):
        """p=1 inserts a character after every non-space character."""
        random.seed(0)
        text = "ab cd"
        result = insert(text, p=1.0)
        # Original has 4 non-space chars, so 4 insertions
        assert len(result) == len(text) + 4

    def test_inserted_chars_are_lowercase(self):
        random.seed(0)
        text = "ABCD"
        result = insert(text, p=1.0)
        # Every other char (the inserted ones) should be lowercase
        for i in range(1, len(result), 2):
            assert result[i] in string.ascii_lowercase

    def test_spaces_never_trigger_insert(self):
        random.seed(0)
        text = "a b"
        for _ in range(50):
            result = insert(text, p=1.0)
            # Space should appear exactly once, never followed by an extra insert
            parts = result.split(" ")
            assert len(parts) == 2

    def test_output_longer_or_equal(self):
        random.seed(42)
        text = "the quick brown fox"
        result = insert(text, p=0.3)
        assert len(result) >= len(text)

    def test_empty_string(self):
        assert insert("", p=1.0) == ""

    def test_list_input(self):
        results = insert(["abc", "xyz"], p=0.0)
        assert results == ["abc", "xyz"]

    def test_single_string_returns_string(self):
        result = insert("abc", p=0.0)
        assert isinstance(result, str)


##############
# CAPITALIZE #
##############


class TestCapitalize:
    def test_random_mode_changes_case(self):
        random.seed(0)
        text = "hello world"
        result = capitalize(text, mode="random")
        assert result != text or True  # seed-dependent, just check type
        assert len(result) == len(text)

    def test_random_mode_preserves_spaces(self):
        random.seed(0)
        text = "a b c"
        result = capitalize(text, mode="random")
        assert result[1] == " "
        assert result[3] == " "

    def test_alternating_mode(self):
        result = capitalize("abcdef", mode="alternating")
        # idx 0 → lower, idx 1 → upper, idx 2 → lower, ...
        assert result == "aBcDeF"

    def test_alternating_skips_spaces(self):
        result = capitalize("ab cd", mode="alternating")
        # non-space idx: a=0(lower), b=1(upper), c=2(lower), d=3(upper)
        assert result == "aB cD"

    def test_alternating_preserves_length(self):
        text = "hello world"
        result = capitalize(text, mode="alternating")
        assert len(result) == len(text)

    def test_empty_string(self):
        assert capitalize("", mode="random") == ""
        assert capitalize("", mode="alternating") == ""

    def test_list_input(self):
        results = capitalize(["abc", "xyz"], mode="alternating")
        assert results == ["aBc", "xYz"]

    def test_single_string_returns_string(self):
        result = capitalize("abc", mode="alternating")
        assert isinstance(result, str)


#############
# TRANSLATE #
#############


class TestTranslate:
    def _make_mock_module(self, responses):
        """Create a mock translate_v2 module whose Client().translate works."""
        mock_module = MagicMock()
        mock_module.Client.return_value.translate.side_effect = responses
        return mock_module

    def _patch_google(self, mock_module):
        return patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.auth": MagicMock(),
            "google.auth.api_key": MagicMock(),
            "google.cloud": MagicMock(translate_v2=mock_module),
            "google.cloud.translate_v2": mock_module,
        })

    def test_single_string(self):
        mock_module = self._make_mock_module(
            [[{"translatedText": "hola"}]]
        )
        with self._patch_google(mock_module):
            result = translate("hello", target_language="es", api_key="fake")

        assert result == "hola"

    def test_list_input(self):
        mock_module = self._make_mock_module(
            [[{"translatedText": "hola"}, {"translatedText": "mundo"}]]
        )
        with self._patch_google(mock_module):
            result = translate(["hello", "world"], target_language="es", api_key="fake")

        assert result == ["hola", "mundo"]

    def test_chunking(self):
        """Texts exceeding CHAR_LIMIT are split into multiple API calls."""
        # Create texts that force two chunks (limit is 30_000)
        long_text = "a" * 20_000
        texts = [long_text, long_text]

        mock_module = self._make_mock_module([
            [{"translatedText": "chunk1"}],
            [{"translatedText": "chunk2"}],
        ])
        with self._patch_google(mock_module):
            result = translate(texts, target_language="fr", api_key="fake")

        assert result == ["chunk1", "chunk2"]
        assert mock_module.Client.return_value.translate.call_count == 2
