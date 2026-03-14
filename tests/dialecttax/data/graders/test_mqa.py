"""Tests for dialecttax.data.graders.mqa."""

import pytest

from dialecttax.data.graders import mqa


##################
# EXTRACT_ANSWER #
##################


class TestExtractAnswer:
    """Tests for extract_answer (strict #### marker)."""
    def test_basic(self):
        """Extracts text after #### marker."""
        assert mqa.extract_answer("The answer is #### B") == "B"

    def test_lowercase(self):
        """Preserves case of extracted answer."""
        assert mqa.extract_answer("#### c") == "c"

    def test_with_trailing_text(self):
        """Extracts only the first token after ####."""
        assert mqa.extract_answer("#### B is correct") == "B"

    def test_no_marker(self):
        """Returns None when no #### found."""
        assert mqa.extract_answer("The answer is B") is None

    def test_none_input(self):
        """Returns None for None input."""
        assert mqa.extract_answer(None) is None

    def test_no_space_after_marker(self):
        """Handles ####B without space."""
        assert mqa.extract_answer("####B") == "B"

    def test_multiple_markers(self):
        """Extracts first token from first #### marker."""
        assert mqa.extract_answer("#### A then #### B") == "A"

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace from extracted answer."""
        assert mqa.extract_answer("####   B  ") == "B"


########################
# EXTRACT_ANSWER_FUZZY #
########################


class TestExtractAnswerFuzzy:
    """Tests for extract_answer_fuzzy (last standalone A-D letter)."""

    def test_single_letter(self):
        """Extracts a single standalone letter."""
        assert mqa.extract_answer_fuzzy("The answer is B") == "B"

    def test_last_letter(self):
        """Returns the last standalone A-D letter."""
        assert mqa.extract_answer_fuzzy("I think A but actually B") == "B"

    def test_no_letter(self):
        """Returns None when no A-D letter found."""
        assert mqa.extract_answer_fuzzy("The answer is 42") is None

    def test_none_input(self):
        """Returns None for None input."""
        assert mqa.extract_answer_fuzzy(None) is None

    def test_empty_string(self):
        """Returns None for empty string."""
        assert mqa.extract_answer_fuzzy("") is None

    def test_lowercase_ignored(self):
        """Only matches uppercase A-D."""
        assert mqa.extract_answer_fuzzy("the answer is b") is None

    def test_letter_in_word_ignored(self):
        """Letters inside words are not matched."""
        assert mqa.extract_answer_fuzzy("Banana is delicious") is None

    def test_number_then_letter(self):
        """Matches letter after a number."""
        assert mqa.extract_answer_fuzzy("12 D") == "D"


####################
# NORMALIZE_ANSWER #
####################


class TestNormalizeAnswer:
    def test_uppercase(self):
        """Converts to upper case."""
        assert mqa.normalize_answer("b") == "B"

    def test_strips_whitespace(self):
        """Strips leading/trailing whitespace."""
        assert mqa.normalize_answer("  B  ") == "B"

    def test_removes_commas(self):
        """Removes commas."""
        assert mqa.normalize_answer("1,000") == "1000"

    def test_removes_periods(self):
        """Removes periods."""
        assert mqa.normalize_answer("B.") == "B"

    def test_numeric_input(self):
        """Handles non-string input."""
        assert mqa.normalize_answer(4) == "4"


#########
# GRADE #
#########


class TestGrade:
    def test_correct(self):
        """Matching answers return True."""
        assert mqa.grade("B", "B") is True

    def test_case_insensitive(self):
        """Comparison is case-insensitive."""
        assert mqa.grade("b", "B") is True

    def test_incorrect(self):
        """Non-matching answers return False."""
        assert mqa.grade("A", "B") is False

    def test_none_predicted(self):
        """None predicted returns False."""
        assert mqa.grade(None, "B") is False

    def test_with_punctuation(self):
        """Punctuation is stripped before comparison."""
        assert mqa.grade("B.", "B") is True


##############
# IS_REFUSAL #
##############


class TestIsRefusal:
    def test_refusal(self):
        """Detects 'I cannot' pattern."""
        assert mqa.is_refusal("I cannot answer this question.", None) is True

    def test_not_refusal_with_answer(self):
        """Not a refusal if answer was extracted."""
        assert mqa.is_refusal("I cannot answer... #### B", "B") is False

    def test_not_refusal_normal(self):
        """Normal completion is not a refusal."""
        assert mqa.is_refusal("The answer is #### B", None) is False

    def test_none_completion(self):
        """None completion is not a refusal."""
        assert mqa.is_refusal(None, None) is False


#####################
# GRADE_COMPLETIONS #
#####################


class TestGradeCompletions:
    def test_single_correct(self):
        """Grades a single correct completion."""
        results = mqa.grade_completions("#### B", "B")
        assert len(results) == 1
        assert results[0]["correct"] is True
        assert results[0]["extracted"] == "B"
        assert results[0]["gold"] == "B"

    def test_single_incorrect(self):
        """Grades a single incorrect completion."""
        results = mqa.grade_completions("#### A", "B")
        assert len(results) == 1
        assert results[0]["correct"] is False

    def test_list_input(self):
        """Grades a list of completions."""
        results = mqa.grade_completions(
            ["#### A", "#### B", "#### C"],
            ["A", "B", "A"],
        )
        assert len(results) == 3
        assert results[0]["correct"] is True
        assert results[1]["correct"] is True
        assert results[2]["correct"] is False

    def test_no_answer(self):
        """Handles completions without #### marker."""
        results = mqa.grade_completions("The answer is 42", "B")
        assert results[0]["extracted"] is None
        assert results[0]["correct"] is False

    def test_refusal_detected(self):
        """Detects refusal in results."""
        results = mqa.grade_completions("I cannot answer this.", "B")
        assert results[0]["refusal"] is True
        assert results[0]["correct"] is False

    def test_result_keys(self):
        """Results contain all expected keys."""
        results = mqa.grade_completions("#### B", "B")
        assert set(results[0].keys()) == {"completion", "extracted", "refusal", "gold", "correct"}

    def test_fuzzy_correct(self):
        """Fuzzy fallback finds last standalone letter when #### is missing."""
        results = mqa.grade_completions("I think the answer is B", "B")
        assert results[0]["correct"] is False
        assert results[0]["calculated"] == "B"
        assert results[0]["fuzzy"] is True

    def test_fuzzy_incorrect(self):
        """Fuzzy fallback with wrong letter."""
        results = mqa.grade_completions("I think A", "B")
        assert results[0]["correct"] is False
        assert results[0]["calculated"] == "A"
        assert results[0]["fuzzy"] is False

    def test_no_fuzzy_when_extracted(self):
        """No calculated/fuzzy keys when #### extraction succeeds."""
        results = mqa.grade_completions("#### B", "B")
        assert "calculated" not in results[0]
        assert "fuzzy" not in results[0]

    def test_no_fuzzy_when_no_letter(self):
        """No calculated/fuzzy keys when no letter found at all."""
        results = mqa.grade_completions("The answer is 42", "B")
        assert "calculated" not in results[0]
        assert "fuzzy" not in results[0]

    def test_fuzzy_number_then_letter(self):
        """Fuzzy fallback picks letter from '12 D' pattern."""
        results = mqa.grade_completions("12 D", "D")
        assert results[0]["correct"] is False
        assert results[0]["calculated"] == "D"
        assert results[0]["fuzzy"] is True
