"""Tests for dialecttax.data.graders.planning."""

import pytest

from dialecttax.data.graders.planning import (
    extract_answer,
    extract_answer_fuzzy,
    grade,
    grade_completions,
    normalize_answer,
    time_str_to_int,
)


####################
# FUZZY EXTRACTION #
####################

class TestFuzzyExtractAnswer:
    """Tests for extract_answer_fuzzy."""

    # --- basic cases ---

    def test_single_unit(self):
        assert extract_answer_fuzzy("The answer is 120 seconds.") == 120

    def test_multi_part(self):
        # 3 minutes 20 seconds = 200s
        assert extract_answer_fuzzy("It takes 3 minutes 20 seconds.") == 200

    def test_abbreviated_units(self):
        # 3 mins, 20 secs = 200s
        assert extract_answer_fuzzy("about 3 mins, 20 secs") == 200

    def test_and_connector(self):
        # 2 hours and 30 minutes = 9000s
        assert extract_answer_fuzzy("roughly 2 hours and 30 minutes") == 9000

    # --- malformed numbers ---

    def test_two_numbers_before_unit(self):
        # "60 120 seconds" — space-separated digits concatenate to 60120
        assert extract_answer_fuzzy("60 120 seconds") == 60120

    def test_stray_number_then_time(self):
        assert extract_answer_fuzzy("step 5: wait 45 seconds") == 45

    # --- takes last group ---

    def test_takes_last_group(self):
        text = "First we wait 10 seconds. Then 5 minutes and 30 seconds."
        # Last group: "5 minutes and 30 seconds" = 330s
        assert extract_answer_fuzzy(text) == 330

    def test_takes_last_group_separated_by_prose(self):
        text = "Phase 1: 2 hours. Phase 2: 45 minutes."
        # Two separate groups; last is "45 minutes" = 2700s
        assert extract_answer_fuzzy(text) == 2700

    # --- larger units ---

    def test_days(self):
        assert extract_answer_fuzzy("about 3 days") == 3 * 86400

    def test_hours_and_minutes(self):
        # 1 hour 15 minutes = 4500s
        assert extract_answer_fuzzy("1 hour, 15 minutes") == 4500

    # --- short units ---

    def test_short_unit_h(self):
        assert extract_answer_fuzzy("2h") == 7200

    def test_short_unit_m(self):
        assert extract_answer_fuzzy("30m") == 1800

    def test_short_unit_s(self):
        assert extract_answer_fuzzy("90s") == 90

    # --- edge cases ---

    def test_none_input(self):
        assert extract_answer_fuzzy(None) is None

    def test_no_time_expression(self):
        assert extract_answer_fuzzy("no time here") is None

    def test_number_without_unit(self):
        assert extract_answer_fuzzy("the answer is 42") is None

    def test_comma_formatted_number(self):
        assert extract_answer_fuzzy("1,200 seconds") == 1200

    def test_decimal_number(self):
        # 1.5 hours = 5400s
        assert extract_answer_fuzzy("1.5 hours") == 5400


###########
# EXTRACT #
###########

class TestExtractAnswer:
    def test_basic(self):
        assert extract_answer("#### 120 seconds") == "120"

    def test_with_commas(self):
        assert extract_answer("#### 1,200 seconds") == "1,200"

    def test_plural(self):
        assert extract_answer("#### 1 second") == "1"

    def test_none_input(self):
        assert extract_answer(None) is None

    def test_no_match(self):
        assert extract_answer("no pattern here") is None


#############
# NORMALIZE #
#############

class TestNormalizeAnswer:
    def test_strips_whitespace(self):
        assert normalize_answer("  120  ") == "120"

    def test_removes_commas(self):
        assert normalize_answer("1,200") == "1200"

    def test_removes_underscores(self):
        assert normalize_answer("1_200") == "1200"

    def test_lowercases(self):
        assert normalize_answer("ABC") == "abc"

    def test_non_string(self):
        assert normalize_answer(120) == "120"


###############
# TIME TO INT #
###############

class TestTimeStrToInt:
    def test_seconds(self):
        assert time_str_to_int("120 seconds") == 120

    def test_minutes(self):
        assert time_str_to_int("5 minutes") == 300

    def test_hours(self):
        assert time_str_to_int("2 hours") == 7200

    def test_mixed(self):
        assert time_str_to_int("1 hour 30 minutes") == 5400

    def test_no_match_returns_string(self):
        assert time_str_to_int("none") == "none"


#########
# GRADE #
#########

class TestGrade:
    def test_within_range(self):
        assert grade(150, [100, 200]) is True

    def test_at_lower_bound(self):
        assert grade(100, [100, 200]) is True

    def test_at_upper_bound(self):
        assert grade(200, [100, 200]) is True

    def test_below_range(self):
        assert grade(50, [100, 200]) is False

    def test_above_range(self):
        assert grade(250, [100, 200]) is False

    def test_none_predicted(self):
        assert grade(None, [100, 200]) is False


#####################
# GRADE COMPLETIONS #
#####################

class TestGradeCompletions:
    def test_exact_match(self):
        results = grade_completions(
            "#### 120 seconds",
            "[100, 200]",
        )
        assert len(results) == 1
        assert results[0]["extracted"] == "120"
        assert results[0]["correct"] is True

    def test_out_of_range(self):
        results = grade_completions(
            "#### 500 seconds",
            "[100, 200]",
        )
        assert results[0]["correct"] is False

    def test_fuzzy_multi_part(self):
        """Non-integer extraction falls back to fuzzy on raw completion."""
        # "about 3 minutes 20 seconds" → extract_answer gets "about 3 minutes 20"
        # int("about 3 minutes 20") fails → extract_answer_fuzzy finds
        # "3 minutes 20 seconds" = 200s in raw text
        completion = "#### about 3 minutes 20 seconds"
        results = grade_completions(completion, "[180, 220]")
        assert results[0]["correct"] is False
        assert results[0]["fuzzy"] is True
        assert results[0]["calculated"] == 200

    def test_fuzzy_no_hash_marker(self):
        """When #### marker is absent, extract_answer returns None,
        but fuzzy still finds the time expression."""
        completion = "The total time is 5 minutes and 30 seconds."
        results = grade_completions(completion, "[300, 350]")
        assert results[0]["extracted"] is None
        assert results[0]["correct"] is False
        assert results[0]["fuzzy"] is True
        assert results[0]["calculated"] == 330

    def test_fuzzy_malformed_number(self):
        """'60 120 seconds' — space-separated digits read as 60120."""
        completion = "#### 60 120 seconds"
        results = grade_completions(completion, "[60000, 61000]")
        assert results[0]["correct"] is False
        assert results[0]["calculated"] == 60120
        assert results[0]["fuzzy"] is True

    def test_no_time_at_all(self):
        """Completely unparseable completion — no fuzzy keys."""
        completion = "I don't know the answer"
        results = grade_completions(completion, "[100, 200]")
        assert results[0]["correct"] is False
        assert "fuzzy" not in results[0]
        assert "calculated" not in results[0]

    def test_multiple_completions(self):
        completions = ["#### 120 seconds", "#### 500 seconds"]
        golds = ["[100, 200]", "[100, 200]"]
        results = grade_completions(completions, golds)
        assert len(results) == 2
        assert results[0]["correct"] is True
        assert results[1]["correct"] is False
