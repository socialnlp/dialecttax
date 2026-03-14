"""Tests for scripts/ReDial/preprocess_redial_qa.py."""

import json
import os
import random
import sys
import tempfile

import pytest

# The script lives outside the installed package, so add it to sys.path.
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts", "ReDial")
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

import preprocess_redial_qa  # noqa: E402


######################
# _PARSE_DISTRACTORS #
######################

class TestParseDistractors:
    def test_valid_json_array(self):
        assert preprocess_redial_qa._parse_distractors("[10, 20, 30]", "math") == [10, 20, 30]

    def test_valid_with_whitespace(self):
        assert preprocess_redial_qa._parse_distractors("  [1, 2, 3]  ", "math") == [1, 2, 3]

    def test_markdown_fenced(self):
        text = "```json\n[100, 200, 300]\n```"
        assert preprocess_redial_qa._parse_distractors(text, "math") == [100, 200, 300]

    def test_markdown_fenced_no_language(self):
        text = "```\n[5, 10, 15]\n```"
        assert preprocess_redial_qa._parse_distractors(text, "math") == [5, 10, 15]

    def test_none_completion(self):
        assert preprocess_redial_qa._parse_distractors(None, "math") is None

    def test_invalid_json(self):
        assert preprocess_redial_qa._parse_distractors("not json", "math") is None

    def test_wrong_length(self):
        assert preprocess_redial_qa._parse_distractors("[1, 2]", "math") is None
        assert preprocess_redial_qa._parse_distractors("[1, 2, 3, 4]", "math") is None

    def test_not_array(self):
        assert preprocess_redial_qa._parse_distractors('{"a": 1}', "math") is None

    def test_string_distractors(self):
        text = '["code1", "code2", "code3"]'
        result = preprocess_redial_qa._parse_distractors(text, "algorithm")
        assert result == ["code1", "code2", "code3"]


#####################
# _NORMALIZE_INDENT #
#####################

class TestNormalizeIndent:
    def test_no_change_already_at_zero(self):
        code = "for x in y:\nprint(x)\nreturn x"
        assert preprocess_redial_qa._normalize_indent(code) == code

    def test_single_line(self):
        assert preprocess_redial_qa._normalize_indent("return x + 1") == "return x + 1"

    def test_strips_common_non_first_indent(self):
        # First line at 0, rest at 8+ (typical bad extraction)
        code = "for x in y:\n        print(x)\n        return x"
        expected = "for x in y:\nprint(x)\nreturn x"
        assert preprocess_redial_qa._normalize_indent(code) == expected

    def test_mixed_indentation(self):
        # First line at 0, inner lines at 8/12, return at 4
        code = "for x in y:\n        if x:\n            print(x)\n    return None"
        expected = "for x in y:\n    if x:\n        print(x)\nreturn None"
        assert preprocess_redial_qa._normalize_indent(code) == expected

    def test_empty_lines_preserved(self):
        code = "for x in y:\n        print(x)\n\n    return x"
        expected = "for x in y:\n    print(x)\n\nreturn x"
        assert preprocess_redial_qa._normalize_indent(code) == expected

    def test_all_at_zero(self):
        code = "x = 1\ny = 2\nreturn x + y"
        assert preprocess_redial_qa._normalize_indent(code) == code

    def test_empty_string(self):
        assert preprocess_redial_qa._normalize_indent("") == ""


################
# _MAKE_CHOICES #
################

class TestMakeChoices:
    def test_math_has_four_choices(self):
        rng = random.Random(0)
        choices, letter = preprocess_redial_qa._make_choices(100, [50, 150, 200], "math", rng)
        assert len(choices) == 4
        assert set(choices.keys()) == {"A", "B", "C", "D"}
        assert letter in ("A", "B", "C", "D")

    def test_math_correct_answer_present(self):
        rng = random.Random(0)
        choices, letter = preprocess_redial_qa._make_choices(42, [10, 20, 30], "math", rng)
        assert choices[letter] == "42"

    def test_math_all_values_present(self):
        rng = random.Random(0)
        choices, letter = preprocess_redial_qa._make_choices(100, [50, 150, 200], "math", rng)
        values = set(choices.values())
        assert values == {"100", "50", "150", "200"}

    def test_planning_wraps_pairs(self):
        rng = random.Random(0)
        correct = [4800.0, 4800.0]
        choices, letter = preprocess_redial_qa._make_choices(correct, [3600, 5400, 7200], "planning", rng)
        assert choices[letter] == [4800.0, 4800.0]
        # All distractors should be [val, val] pairs
        for k, v in choices.items():
            assert isinstance(v, list) and len(v) == 2

    def test_planning_different_pair_preserved(self):
        rng = random.Random(0)
        correct = [3600.0, 5400.0]
        choices, letter = preprocess_redial_qa._make_choices(correct, [1000, 2000, 3000], "planning", rng)
        assert choices[letter] == [3600.0, 5400.0]

    def test_algorithm_string_choices(self):
        rng = random.Random(0)
        correct = "return x + 1"
        distractors = ["return x", "return x - 1", "return x * 2"]
        choices, letter = preprocess_redial_qa._make_choices(correct, distractors, "algorithm", rng)
        assert choices[letter] == "return x + 1"
        assert set(choices.values()) == {"return x + 1", "return x", "return x - 1", "return x * 2"}

    def test_algorithm_normalizes_indent(self):
        rng = random.Random(0)
        correct = "for x in y:\n        print(x)\n    return x"
        distractors = [
            "for x in y:\n    print(x + 1)\n    return x",
            "for x in y:\n    print(x)\n    return x + 1",
            "for x in y:\n    pass\n    return x",
        ]
        choices, letter = preprocess_redial_qa._make_choices(correct, distractors, "algorithm", rng)
        # Correct answer should be normalized to 4-space indent
        assert choices[letter] == "for x in y:\n    print(x)\nreturn x"

    def test_shuffle_is_deterministic(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        c1, l1 = preprocess_redial_qa._make_choices(100, [50, 150, 200], "math", rng1)
        c2, l2 = preprocess_redial_qa._make_choices(100, [50, 150, 200], "math", rng2)
        assert c1 == c2
        assert l1 == l2

    def test_duplicate_distractor_returns_none(self):
        rng = random.Random(0)
        choices, letter = preprocess_redial_qa._make_choices(100, [50, 50, 200], "math", rng)
        assert choices is None
        assert letter is None

    def test_distractor_equals_correct_returns_none(self):
        rng = random.Random(0)
        choices, letter = preprocess_redial_qa._make_choices(100, [100, 150, 200], "math", rng)
        assert choices is None
        assert letter is None

    def test_planning_duplicate_returns_none(self):
        rng = random.Random(0)
        choices, letter = preprocess_redial_qa._make_choices([4800.0, 4800.0], [3600, 3600, 7200], "planning", rng)
        assert choices is None
        assert letter is None

    def test_unsupported_task(self):
        rng = random.Random(0)
        with pytest.raises(ValueError):
            preprocess_redial_qa._make_choices(1, [2, 3, 4], "unknown", rng)


############################
# _BUILD_DISTRACTOR_MESSAGES #
############################

class TestBuildDistractorMessages:
    def test_math_messages(self):
        samples = [{"problem": "What is 2+2?", "answer": 4, "unique_id": "test-0"}]
        messages = preprocess_redial_qa._build_distractor_messages(samples, "math")
        assert len(messages) == 1
        assert len(messages[0]) == 2  # system + user
        assert messages[0][0]["role"] == "system"
        assert "4" in messages[0][1]["content"]

    def test_planning_uses_first_value(self):
        samples = [{"problem": "Plan a trip", "answer": [4800.0, 4800.0], "unique_id": "test-0"}]
        messages = preprocess_redial_qa._build_distractor_messages(samples, "planning")
        assert "4800" in messages[0][1]["content"]

    def test_algorithm_messages(self):
        samples = [{"problem": "Sort a list", "context": "def f():", "answer": "return sorted(x)", "unique_id": "test-0"}]
        messages = preprocess_redial_qa._build_distractor_messages(samples, "algorithm")
        assert "Sort a list" in messages[0][1]["content"]
        assert "return sorted(x)" in messages[0][1]["content"]

    def test_unsupported_task(self):
        with pytest.raises(ValueError):
            preprocess_redial_qa._build_distractor_messages([{"problem": "x"}], "logic")


#################
# CONVERT_LOGIC #
#################

class TestConvertLogic:
    def test_passthrough(self):
        samples = [
            {
                "premises": "All cats are animals.",
                "conclusion": "Some animals are cats.",
                "choices": {"A": "True", "B": "False", "C": "Uncertain"},
                "answer": "A",
                "task": "logic",
                "unique_id": "test-0",
            },
        ]
        result = preprocess_redial_qa.convert_logic(samples)
        assert len(result) == 1
        assert result[0]["answer"] == "A"
        assert result[0]["choices"] == {"A": "True", "B": "False", "C": "Uncertain"}

    def test_choices_are_copies(self):
        original_choices = {"A": "True", "B": "False", "C": "Uncertain"}
        samples = [{"choices": original_choices, "answer": "A", "unique_id": "test-0"}]
        result = preprocess_redial_qa.convert_logic(samples)
        # Modifying output should not affect input
        result[0]["choices"]["D"] = "Maybe"
        assert "D" not in original_choices


##########################
# _LOAD_EXISTING_RESPONSES #
##########################

class TestLoadExistingResponses:
    def test_no_files(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        responses, done = preprocess_redial_qa._load_existing_responses(path, 5)
        assert len(responses) == 5
        assert all(r is None for r in responses)
        assert done == set()

    def test_none_path(self):
        responses, done = preprocess_redial_qa._load_existing_responses(None, 3)
        assert len(responses) == 3
        assert done == set()

    def test_complete_file(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        with open(path, "w") as f:
            for i in range(3):
                f.write(json.dumps({"_idx": i, "choices": [{"message": {"content": f"[{i}, {i+1}, {i+2}]"}}]}) + "\n")
        responses, done = preprocess_redial_qa._load_existing_responses(path, 3)
        assert done == {0, 1, 2}
        assert all(r is not None for r in responses)

    def test_partial_file(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"_idx": 0, "data": "first"}) + "\n")
            f.write(json.dumps({"_idx": 2, "data": "third"}) + "\n")
        responses, done = preprocess_redial_qa._load_existing_responses(path, 4)
        assert done == {0, 2}
        assert responses[0] is not None
        assert responses[1] is None
        assert responses[2] is not None
        assert responses[3] is None

    def test_tmp_crash_recovery(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        tmp_file = path + ".tmp"
        idx_file = path + ".idx"
        # Write .tmp with 2 responses and .idx mapping them to indices [1, 3]
        with open(tmp_file, "w") as f:
            f.write(json.dumps({"data": "second"}) + "\n")
            f.write(json.dumps({"data": "fourth"}) + "\n")
        with open(idx_file, "w") as f:
            json.dump([1, 3], f)
        responses, done = preprocess_redial_qa._load_existing_responses(path, 5)
        assert done == {1, 3}
        assert responses[1] is not None
        assert responses[3] is not None
        assert responses[0] is None

    def test_merge_complete_and_tmp(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        # Complete file has index 0
        with open(path, "w") as f:
            f.write(json.dumps({"_idx": 0, "data": "first"}) + "\n")
        # .tmp has index 1 (from a crashed resume)
        with open(path + ".tmp", "w") as f:
            f.write(json.dumps({"data": "second"}) + "\n")
        with open(path + ".idx", "w") as f:
            json.dump([1], f)
        responses, done = preprocess_redial_qa._load_existing_responses(path, 3)
        assert done == {0, 1}

    def test_tmp_does_not_overwrite_complete(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        # Complete file has index 0 with value "from_complete"
        with open(path, "w") as f:
            f.write(json.dumps({"_idx": 0, "data": "from_complete"}) + "\n")
        # .tmp also maps to index 0 with value "from_tmp"
        with open(path + ".tmp", "w") as f:
            f.write(json.dumps({"data": "from_tmp"}) + "\n")
        with open(path + ".idx", "w") as f:
            json.dump([0], f)
        responses, done = preprocess_redial_qa._load_existing_responses(path, 1)
        assert responses[0]["data"] == "from_complete"

    def test_out_of_range_idx_ignored(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps({"_idx": 10, "data": "oob"}) + "\n")
        responses, done = preprocess_redial_qa._load_existing_responses(path, 3)
        assert done == set()


#####################
# _SAVE_RESPONSES   #
#####################

class TestSaveResponses:
    def test_saves_with_idx(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        responses = [{"data": "a"}, {"data": "b"}, {"data": "c"}]
        preprocess_redial_qa._save_responses(path, responses)
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 3
        assert [l["_idx"] for l in lines] == [0, 1, 2]

    def test_skips_none(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        responses = [{"data": "a"}, None, {"data": "c"}]
        preprocess_redial_qa._save_responses(path, responses)
        with open(path) as f:
            lines = [json.loads(l) for l in f]
        assert len(lines) == 2
        assert lines[0]["_idx"] == 0
        assert lines[1]["_idx"] == 2

    def test_cleans_up_tmp(self, tmp_path):
        path = str(tmp_path / "responses.jsonl")
        for suffix in (".tmp", ".idx"):
            with open(path + suffix, "w") as f:
                f.write("leftover")
        preprocess_redial_qa._save_responses(path, [{"data": "x"}])
        assert not os.path.exists(path + ".tmp")
        assert not os.path.exists(path + ".idx")

    def test_none_path_noop(self):
        preprocess_redial_qa._save_responses(None, [{"data": "x"}])


#####################
# _NORMALIZE_ANSWER #
#####################

class TestNormalizeAnswer:
    def test_math(self):
        assert preprocess_redial_qa._normalize_answer(42, "math") == "42"
        assert preprocess_redial_qa._normalize_answer(42.0, "math") == "42"

    def test_planning_list(self):
        assert preprocess_redial_qa._normalize_answer([4800.0, 4800.0], "planning") == [4800.0, 4800.0]

    def test_planning_scalar(self):
        assert preprocess_redial_qa._normalize_answer(3600, "planning") == [3600.0, 3600.0]

    def test_algorithm(self):
        assert preprocess_redial_qa._normalize_answer("return x", "algorithm") == "return x"

    def test_logic(self):
        assert preprocess_redial_qa._normalize_answer("A", "logic") == "A"


######################
# CHECK_QA_ALIGNMENT #
######################

class TestCheckQAAlignment:
    def _write_jsonl(self, path, samples):
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

    def test_math_aligned(self, tmp_path):
        orig = [{"unique_id": "m0", "answer": 42}, {"unique_id": "m1", "answer": 100}]
        qa = [
            {"unique_id": "m0", "choices": {"A": "10", "B": "42", "C": "20", "D": "30"}, "answer": "B"},
            {"unique_id": "m1", "choices": {"A": "100", "B": "50", "C": "150", "D": "200"}, "answer": "A"},
        ]
        p_orig = str(tmp_path / "orig.jsonl")
        p_qa = str(tmp_path / "qa.jsonl")
        self._write_jsonl(p_orig, orig)
        self._write_jsonl(p_qa, qa)
        assert preprocess_redial_qa.check_qa_alignment(p_orig, p_qa, "math") == 0

    def test_math_misaligned(self, tmp_path):
        orig = [{"unique_id": "m0", "answer": 42}]
        qa = [{"unique_id": "m0", "choices": {"A": "10", "B": "99", "C": "20", "D": "30"}, "answer": "B"}]
        p_orig = str(tmp_path / "orig.jsonl")
        p_qa = str(tmp_path / "qa.jsonl")
        self._write_jsonl(p_orig, orig)
        self._write_jsonl(p_qa, qa)
        assert preprocess_redial_qa.check_qa_alignment(p_orig, p_qa, "math") == 1

    def test_logic_aligned(self, tmp_path):
        orig = [{"unique_id": "l0", "answer": "A"}]
        qa = [{"unique_id": "l0", "choices": {"A": "True", "B": "False"}, "answer": "A"}]
        p_orig = str(tmp_path / "orig.jsonl")
        p_qa = str(tmp_path / "qa.jsonl")
        self._write_jsonl(p_orig, orig)
        self._write_jsonl(p_qa, qa)
        assert preprocess_redial_qa.check_qa_alignment(p_orig, p_qa, "logic") == 0

    def test_logic_misaligned(self, tmp_path):
        orig = [{"unique_id": "l0", "answer": "A"}]
        qa = [{"unique_id": "l0", "choices": {"A": "True", "B": "False"}, "answer": "B"}]
        p_orig = str(tmp_path / "orig.jsonl")
        p_qa = str(tmp_path / "qa.jsonl")
        self._write_jsonl(p_orig, orig)
        self._write_jsonl(p_qa, qa)
        assert preprocess_redial_qa.check_qa_alignment(p_orig, p_qa, "logic") == 1

    def test_missing_unique_id(self, tmp_path):
        orig = [{"unique_id": "m0", "answer": 42}]
        qa = [{"unique_id": "m999", "choices": {"A": "42"}, "answer": "A"}]
        p_orig = str(tmp_path / "orig.jsonl")
        p_qa = str(tmp_path / "qa.jsonl")
        self._write_jsonl(p_orig, orig)
        self._write_jsonl(p_qa, qa)
        assert preprocess_redial_qa.check_qa_alignment(p_orig, p_qa, "math") == 1

    def test_planning_aligned(self, tmp_path):
        orig = [{"unique_id": "p0", "answer": [4800.0, 4800.0]}]
        qa = [{"unique_id": "p0", "choices": {"A": [4800.0, 4800.0], "B": [3600, 3600]}, "answer": "A"}]
        p_orig = str(tmp_path / "orig.jsonl")
        p_qa = str(tmp_path / "qa.jsonl")
        self._write_jsonl(p_orig, orig)
        self._write_jsonl(p_qa, qa)
        assert preprocess_redial_qa.check_qa_alignment(p_orig, p_qa, "planning") == 0
