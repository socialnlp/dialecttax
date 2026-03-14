"""Tests for scripts/ReDial/preprocess_redial.py.

Verifies that the preprocessing pipeline:
- Keeps only ReDial math questions that also appear in GSM8K.
- Produces matching answers/solutions for SAE–AAVE pairs.
- Assigns correct unique_id and original_id values.
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest

# The script lives outside the installed package, so add it to sys.path.
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "ReDial")
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

import preprocess_redial  # noqa: E402


############
# FIXTURES #
############

# Fake GSM8K test set (5 questions).
GSM8K_ROWS = [
    {"question": "How many apples does Ann have?",   "answer": "She buys 3 more.\n#### 8"},
    {"question": "What is the total cost?",          "answer": "Each costs $5.\n#### 15"},
    {"question": "How many pages did Tom read?",     "answer": "He read 20 each day.\n#### 60"},
    {"question": "How far did the car travel?",      "answer": "At 60 mph for 2 hrs.\n#### 120"},
    {"question": "How many cookies are left?",       "answer": "She ate 4.\n#### 6"},
]


def _make_gsm8k_dataset():
    """Return a dict-like object mimicking a HuggingFace Dataset for GSM8K."""
    ds = MagicMock()
    ds.__getitem__ = lambda self, key: (
        [r[key] for r in GSM8K_ROWS] if isinstance(key, str)
        else GSM8K_ROWS[key]
    )
    ds.__len__ = lambda self: len(GSM8K_ROWS)
    return ds


def _wrap_redial_question(question: str) -> str:
    """Wrap a core question into the ReDial math prompt format."""
    return f"Solve the following math problem.\nQuestion: {question}\nAnswer:"


def _wrap_redial_question_aave(question: str) -> str:
    """Wrap a core question into an AAVE-dialect ReDial math prompt."""
    return f"Solve dis math problem yo.\nQuestion: {question}\nAnswer:"


# ReDial "math_vanilla_original" split.
# Indices 0, 2, 4 match GSM8K questions 0, 2, 4.
# Indices 1, 3 do NOT appear in GSM8K (should be filtered out).
REDIAL_ORIGINAL_ROWS = [
    {"question": _wrap_redial_question("How many apples does Ann have?"),   "answer": "8"},      # idx 0 -> gsm8k 0
    {"question": _wrap_redial_question("Unrelated question A?"),            "answer": "99"},     # idx 1 -> NOT in gsm8k
    {"question": _wrap_redial_question("How many pages did Tom read?"),     "answer": "60"},     # idx 2 -> gsm8k 2
    {"question": _wrap_redial_question("Unrelated question B?"),            "answer": "77"},     # idx 3 -> NOT in gsm8k
    {"question": _wrap_redial_question("How many cookies are left?"),       "answer": "6"},      # idx 4 -> gsm8k 4
]

# ReDial "math_vanilla_aave" split — same indices as original, AAVE wording.
REDIAL_AAVE_ROWS = [
    {"question": _wrap_redial_question_aave("How many apples Ann got?"),       "answer": "8"},
    {"question": _wrap_redial_question_aave("Unrelated question A yo?"),       "answer": "99"},
    {"question": _wrap_redial_question_aave("How many pages Tom read?"),       "answer": "60"},
    {"question": _wrap_redial_question_aave("Unrelated question B yo?"),       "answer": "77"},
    {"question": _wrap_redial_question_aave("How many cookies be left?"),      "answer": "6"},
]


def _make_redial_split(rows):
    """Turn a list of dicts into something iterable with __getitem__."""
    ds = MagicMock()
    ds.__iter__ = lambda self: iter(rows)
    ds.__getitem__ = lambda self, idx: rows[idx]
    ds.__len__ = lambda self: len(rows)
    return ds


@pytest.fixture()
def preprocessed():
    """Run the full SAE + AAVE preprocessing with mocked datasets.

    Returns (sae_split, aave_split, redial_original_mapping).
    """
    fake_gsm8k = _make_gsm8k_dataset()
    split_original = _make_redial_split(REDIAL_ORIGINAL_ROWS)
    split_aave = _make_redial_split(REDIAL_AAVE_ROWS)

    with patch.object(preprocess_redial, "load_dataset", return_value=fake_gsm8k):
        sae_split, mapping = preprocess_redial.preprocess_math(
            split_original, "math_vanilla_original",
        )

    aave_split = preprocess_redial.preprocess_math_aave(
        split_aave, "math_vanilla_aave", sae_split, mapping,
    )

    return sae_split, aave_split, mapping


####################
# GSM8K FILTERING #
####################

class TestGSM8KFiltering:
    def test_only_gsm8k_questions_kept(self, preprocessed):
        """Non-GSM8K rows (indices 1, 3) must be dropped."""
        sae_split, _, _ = preprocessed
        assert len(sae_split) == 3  # out of 5 original rows

    def test_problems_are_gsm8k_questions(self, preprocessed):
        """Every retained problem must be a literal GSM8K question."""
        sae_split, _, _ = preprocessed
        gsm8k_questions = {r["question"] for r in GSM8K_ROWS}
        for row in sae_split:
            assert row["problem"] in gsm8k_questions

    def test_non_gsm8k_questions_absent(self, preprocessed):
        """Questions not in GSM8K must not appear."""
        sae_split, _, _ = preprocessed
        problems = {row["problem"] for row in sae_split}
        assert "Unrelated question A?" not in problems
        assert "Unrelated question B?" not in problems


##################
# SAE-AAVE PARITY #
##################

class TestSAEAAVEParity:
    def test_same_length(self, preprocessed):
        """SAE and AAVE splits must have the same number of rows."""
        sae_split, aave_split, _ = preprocessed
        assert len(sae_split) == len(aave_split)

    def test_same_answers(self, preprocessed):
        """Corresponding SAE and AAVE rows must have identical answers."""
        sae_split, aave_split, _ = preprocessed
        for sae_row, aave_row in zip(sae_split, aave_split):
            assert sae_row["answer"] == aave_row["answer"]

    def test_same_solutions(self, preprocessed):
        """AAVE rows get their solution from the SAE row, so they must match."""
        sae_split, aave_split, _ = preprocessed
        for sae_row, aave_row in zip(sae_split, aave_split):
            assert sae_row["solution"] == aave_row["solution"]

    def test_same_original_ids(self, preprocessed):
        """Both dialects point back to the same GSM8K example."""
        sae_split, aave_split, _ = preprocessed
        for sae_row, aave_row in zip(sae_split, aave_split):
            assert sae_row["original_id"] == aave_row["original_id"]

    def test_different_problems(self, preprocessed):
        """The problem text should differ (different dialect wording)."""
        sae_split, aave_split, _ = preprocessed
        for sae_row, aave_row in zip(sae_split, aave_split):
            assert sae_row["problem"] != aave_row["problem"]

    def test_different_unique_ids(self, preprocessed):
        """Unique IDs must differ because the split names differ."""
        sae_split, aave_split, _ = preprocessed
        for sae_row, aave_row in zip(sae_split, aave_split):
            assert sae_row["unique_id"] != aave_row["unique_id"]


######
# IDS #
######

class TestIDs:
    def test_sae_unique_id_format(self, preprocessed):
        """SAE unique_id must be 'redial-{split_name}-{redial_index}'."""
        sae_split, _, _ = preprocessed
        expected_redial_indices = [0, 2, 4]  # the rows that match GSM8K
        for row, redial_idx in zip(sae_split, expected_redial_indices):
            assert row["unique_id"] == f"redial-math_vanilla_original-{redial_idx}"

    def test_aave_unique_id_format(self, preprocessed):
        """AAVE unique_id must be 'redial-{split_name}-{redial_index}'."""
        _, aave_split, _ = preprocessed
        expected_redial_indices = [0, 2, 4]
        for row, redial_idx in zip(aave_split, expected_redial_indices):
            assert row["unique_id"] == f"redial-math_vanilla_aave-{redial_idx}"

    def test_original_id_format(self, preprocessed):
        """original_id must be 'gsm8k-test-{gsm8k_index}'."""
        sae_split, _, _ = preprocessed
        expected_gsm8k_indices = [0, 2, 4]  # GSM8K positions of the matched questions
        for row, gsm8k_idx in zip(sae_split, expected_gsm8k_indices):
            assert row["original_id"] == f"gsm8k-test-{gsm8k_idx}"

    def test_unique_ids_are_unique(self, preprocessed):
        sae_split, aave_split, _ = preprocessed
        all_ids = [r["unique_id"] for r in sae_split] + [r["unique_id"] for r in aave_split]
        assert len(all_ids) == len(set(all_ids))


###########
# MAPPING #
###########

class TestMapping:
    def test_mapping_keys_are_retained_redial_indices(self, preprocessed):
        _, _, mapping = preprocessed
        assert set(mapping.keys()) == {0, 2, 4}

    def test_mapping_values_are_gsm8k_indices(self, preprocessed):
        _, _, mapping = preprocessed
        assert mapping[0] == 0
        assert mapping[2] == 2
        assert mapping[4] == 4

    def test_mapping_length_matches_sae_split(self, preprocessed):
        sae_split, _, mapping = preprocessed
        assert len(mapping) == len(sae_split)


################
# FIELD VALUES #
################

class TestFieldValues:
    REQUIRED_KEYS = {"problem", "solution", "answer", "task", "unique_id", "original_id", "meta"}

    def test_sae_rows_have_all_keys(self, preprocessed):
        sae_split, _, _ = preprocessed
        for row in sae_split:
            assert set(row.keys()) == self.REQUIRED_KEYS

    def test_aave_rows_have_all_keys(self, preprocessed):
        _, aave_split, _ = preprocessed
        for row in aave_split:
            assert set(row.keys()) == self.REQUIRED_KEYS

    def test_task_is_math(self, preprocessed):
        sae_split, aave_split, _ = preprocessed
        for row in sae_split + aave_split:
            assert row["task"] == "math"

    def test_meta_is_none(self, preprocessed):
        sae_split, aave_split, _ = preprocessed
        for row in sae_split + aave_split:
            assert row["meta"] is None

    def test_answers_are_ints(self, preprocessed):
        sae_split, aave_split, _ = preprocessed
        for row in sae_split + aave_split:
            assert isinstance(row["answer"], int)

    def test_solutions_come_from_gsm8k(self, preprocessed):
        """Each solution must be the text before '#### ...' in the GSM8K answer."""
        sae_split, _, _ = preprocessed
        gsm8k_solutions = {
            r["question"]: r["answer"].split("\n#### ")[0] for r in GSM8K_ROWS
        }
        for row in sae_split:
            assert row["solution"] == gsm8k_solutions[row["problem"]]

    def test_answers_match_gsm8k(self, preprocessed):
        """Each answer must equal the integer after '####' in GSM8K."""
        sae_split, _, _ = preprocessed
        gsm8k_answers = {
            r["question"]: int(r["answer"].split("\n#### ")[1]) for r in GSM8K_ROWS
        }
        for row in sae_split:
            assert row["answer"] == gsm8k_answers[row["problem"]]


##########################
# _PARSE_HUMANEVAL_TESTS #
##########################

class TestParseHumanevalTests:
    """Tests for _parse_humaneval_tests()."""

    def test_simple_asserts(self):
        """Simple single-line asserts are returned as individual tests."""
        raw = (
            "\n\nMETADATA = {}\n\n\n"
            "def check(candidate):\n"
            "    assert candidate([1, 2], 0.3) == True\n"
            "    assert candidate([1, 2], 0.05) == False\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "has_close_elements(")
        assert imports == []
        assert len(tests) == 2
        assert tests[0] == "assert python_function([1, 2], 0.3) == True"
        assert tests[1] == "assert python_function([1, 2], 0.05) == False"

    def test_candidate_replaced(self):
        """Both 'candidate(' and entry_point are replaced with python_function."""
        raw = (
            "def check(candidate):\n"
            "    assert candidate(1) == 2\n"
        )
        _, tests = preprocess_redial._parse_humaneval_tests(raw, "my_func(")
        assert "python_function(" in tests[0]
        assert "candidate(" not in tests[0]
        assert "my_func(" not in tests[0]

    def test_imports_extracted(self):
        """Import statements inside check() go to test_imports."""
        raw = (
            "def check(candidate):\n"
            "    import math\n"
            "    from random import randint\n"
            "    assert candidate(1) == 1\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert imports == ["import math", "from random import randint"]
        assert len(tests) == 1

    def test_multiline_assert_condensed(self):
        """Multi-line bracket asserts are condensed to a single line."""
        raw = (
            "def check(candidate):\n"
            "    assert candidate('abc') == [\n"
            "        'a', 'b', 'c'\n"
            "    ]\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert len(tests) == 1
        assert "\n" not in tests[0]
        assert "'a', 'b', 'c'" in tests[0]

    def test_for_loop_kept_as_block(self):
        """A for loop is kept as a single multi-line test entry."""
        raw = (
            "def check(candidate):\n"
            "    for x in range(2, 8):\n"
            "        assert candidate(x, x + 1) == str(x)\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert len(tests) == 1
        assert "\n" in tests[0]
        assert tests[0].startswith("for x in range(2, 8):")

    def test_setup_bundled_with_for_loop(self):
        """Setup variables before a for-loop are bundled into one test."""
        raw = (
            "def check(candidate):\n"
            "    import string\n"
            "    letters = string.ascii_lowercase\n"
            "    for _ in range(10):\n"
            "        assert candidate(letters) == letters\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert imports == ["import string"]
        assert len(tests) == 1
        assert "letters = string.ascii_lowercase" in tests[0]
        assert "for _ in range(10):" in tests[0]

    def test_setup_bundled_with_assert(self):
        """Setup variables before an assert are bundled together."""
        raw = (
            "def check(candidate):\n"
            "    lst = list(range(10))\n"
            "    total = sum(lst)\n"
            "    assert candidate(lst) == total\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert len(tests) == 1
        assert "lst = list(range(10))" in tests[0]
        assert "total = sum(lst)" in tests[0]
        assert "assert python_function(lst) == total" in tests[0]

    def test_comments_skipped(self):
        """Comments are not included in tests or imports."""
        raw = (
            "def check(candidate):\n"
            "    # Check some cases\n"
            "    assert candidate(1) == 1\n"
            "    # Edge cases\n"
            "    assert candidate(0) == 0\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert len(tests) == 2
        for t in tests:
            assert not t.startswith("#")

    def test_bare_print_skipped(self):
        """A bare 'print' statement is skipped."""
        raw = (
            "def check(candidate):\n"
            "    print\n"
            "    assert candidate(1) == 1\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert len(tests) == 1
        assert "print" not in tests[0]

    def test_no_check_function(self):
        """Returns empty lists if def check(candidate) is not found."""
        raw = "METADATA = {}\n\ndef other():\n    pass\n"
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert imports == []
        assert tests == []

    def test_nested_for_loop(self):
        """Nested for loops are kept as a single block."""
        raw = (
            "def check(candidate):\n"
            "    import copy\n"
            "    rng = 42\n"
            "    for _ in range(5):\n"
            "        coeffs = []\n"
            "        for j in range(3):\n"
            "            coeffs.append(j)\n"
            "        assert candidate(copy.deepcopy(coeffs)) == coeffs\n"
        )
        imports, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert imports == ["import copy"]
        assert len(tests) == 1
        assert "rng = 42" in tests[0]
        assert "for _ in range(5):" in tests[0]
        assert "for j in range(3):" in tests[0]

    def test_assert_true_kept(self):
        """Bare 'assert True' is kept as a test."""
        raw = (
            "def check(candidate):\n"
            "    assert candidate(1) == 1\n"
            "    assert True\n"
        )
        _, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert len(tests) == 2
        assert tests[1] == "assert True"

    def test_brackets_inside_strings_ignored(self):
        """Brackets inside string literals don't affect statement splitting."""
        raw = (
            "def check(candidate):\n"
            '    assert candidate("(")\n'
            '    assert candidate(")")\n'
            '    assert not candidate("((")\n'
        )
        _, tests = preprocess_redial._parse_humaneval_tests(raw, "f(")
        assert len(tests) == 3
        assert 'python_function("(")' in tests[0]
        assert 'python_function(")")' in tests[1]
        assert 'python_function("((")' in tests[2]


####################
# EXTRACT_QUESTION #
####################

class TestExtractQuestion:
    def test_strips_preamble_and_answer_tag(self):
        text = "Some instructions here.\nQuestion: What is 2+2?\nAnswer:"
        assert preprocess_redial.extract_question(text, "math") == "What is 2+2?"

    def test_preserves_multiline_question(self):
        text = "Instructions.\nQuestion: A store has 5 items.\nEach costs $3.\nAnswer:"
        result = preprocess_redial.extract_question(text, "math")
        assert result == "A store has 5 items.\nEach costs $3."

    def test_minimal_wrapping(self):
        text = "Question: Q?\nAnswer:"
        assert preprocess_redial.extract_question(text, "math") == "Q?"


##############
# EDGE CASES #
##############

class TestEdgeCases:
    def test_answer_with_commas_and_underscores(self):
        """Answers like '1,000' or '1_000' must be parsed to int correctly."""
        gsm8k_rows = [
            {"question": "Big number?", "answer": "Steps here.\n#### 1,000"},
        ]
        fake_gsm8k = MagicMock()
        fake_gsm8k.__getitem__ = lambda self, key: (
            [r[key] for r in gsm8k_rows] if isinstance(key, str)
            else gsm8k_rows[key]
        )
        fake_gsm8k.__len__ = lambda self: len(gsm8k_rows)

        redial_rows = [
            {"question": "Preamble.\nQuestion: Big number?\nAnswer:", "answer": "1_000"},
        ]
        split = _make_redial_split(redial_rows)

        with patch.object(preprocess_redial, "load_dataset", return_value=fake_gsm8k):
            result, mapping = preprocess_redial.preprocess_math(split, "test_split")

        assert len(result) == 1
        assert result[0]["answer"] == 1000

    def test_empty_split_produces_empty_output(self):
        """A split with zero rows matching GSM8K yields an empty list."""
        fake_gsm8k = _make_gsm8k_dataset()
        redial_rows = [
            {"question": "Preamble.\nQuestion: Not in GSM8K?\nAnswer:", "answer": "0"},
        ]
        split = _make_redial_split(redial_rows)

        with patch.object(preprocess_redial, "load_dataset", return_value=fake_gsm8k):
            result, mapping = preprocess_redial.preprocess_math(split, "empty_split")

        assert result == []
        assert mapping == {}

    def test_aave_empty_when_original_empty(self):
        """If no original rows survived, the AAVE split is empty too."""
        aave_rows = [
            {"question": "Preamble.\nQuestion: Not here?\nAnswer:", "answer": "0"},
        ]
        split = _make_redial_split(aave_rows)
        result = preprocess_redial.preprocess_math_aave(split, "aave_empty", [], {})
        assert result == []
