"""Tests for scripts/ReDial/patch_benchmark_redial.py.

Verifies the helper functions for continuing and patching benchmark runs:
- Loading, saving, and indexing responses.
- Error detection in API responses.
- Crash recovery from temp + sidecar files.
- Continue and patch generation flows.
"""

import json
import os
import sys
from unittest.mock import patch, MagicMock

import pytest

# The script lives outside the installed package, so add it to sys.path.
SCRIPTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "scripts", "ReDial",
)
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))

# Mock load_config before importing (benchmark_redial calls it at module level).
_FAKE_CONFIG = {
    "directories": {"preprocessed": "/fake/preprocessed"},
    "keys": {"openrouter": "fake_key"},
}
with patch("dialecttax.utils.load_config", return_value=_FAKE_CONFIG):
    import patch_benchmark_redial as pbr  # noqa: E402


###########
# HELPERS #
###########

def _ok(content: str) -> dict:
    """Build a valid OpenRouter API response."""
    return {"choices": [{"message": {"content": content}}]}


def _err(msg: str = "timeout") -> dict:
    """Build an error response."""
    return {"error": msg}


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


_GEN_KWARGS = {
    "model": "test/model",
    "max_tokens_new": 32,
    "max_tokens_reasoning": None,
    "reasoning_effort": None,
    "temperature": 0.0,
    "max_workers": 1,
}


##########################
# LOAD_INDEXED_RESPONSES #
##########################

class TestLoadIndexedResponses:
    def test_uses_idx_field(self, tmp_path):
        """Responses with _idx field are keyed by that value."""
        records = [{**_ok("a"), "_idx": 5}, {**_ok("b"), "_idx": 10}]
        path = str(tmp_path / "responses.jsonl")
        _write_jsonl(path, records)

        result = pbr._load_indexed_responses(path)

        assert set(result.keys()) == {5, 10}
        assert result[5]["choices"][0]["message"]["content"] == "a"
        assert result[10]["choices"][0]["message"]["content"] == "b"

    def test_fallback_to_line_number(self, tmp_path):
        """Without _idx field, line number is used as key."""
        path = str(tmp_path / "responses.jsonl")
        _write_jsonl(path, [_ok("a"), _ok("b")])

        result = pbr._load_indexed_responses(path)

        assert set(result.keys()) == {0, 1}

    def test_nonexistent_file_returns_empty(self):
        assert pbr._load_indexed_responses("/nonexistent/path.jsonl") == {}

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines are skipped; line number still advances."""
        path = str(tmp_path / "responses.jsonl")
        with open(path, "w") as f:
            f.write(json.dumps(_ok("a")) + "\n")
            f.write("\n")
            f.write(json.dumps(_ok("b")) + "\n")

        result = pbr._load_indexed_responses(path)

        # Line 0 -> first, line 1 -> blank (skipped), line 2 -> second
        assert set(result.keys()) == {0, 2}


############
# IS_ERROR #
############

class TestIsError:
    def test_error_key(self):
        assert pbr.is_error({"error": "timeout"}) is True

    def test_empty_content(self):
        assert pbr.is_error({"choices": [{"message": {"content": ""}}]}) is True

    def test_none_content(self):
        assert pbr.is_error({"choices": [{"message": {"content": None}}]}) is True

    def test_missing_choices(self):
        assert pbr.is_error({}) is True

    def test_empty_choices_list(self):
        assert pbr.is_error({"choices": []}) is True

    def test_valid_response(self):
        assert pbr.is_error(_ok("hello")) is False


######################
# SAVE_ALL_RESPONSES #
######################

class TestSaveAllResponses:
    def test_saves_in_index_order(self, tmp_path):
        """Responses saved ordered by key, not insertion order."""
        responses = {2: _ok("c"), 0: _ok("a"), 1: _ok("b")}
        path = str(tmp_path / "sub" / "responses.jsonl")

        pbr._save_all_responses(responses, path)

        saved = _read_jsonl(path)
        assert [s["choices"][0]["message"]["content"] for s in saved] == [
            "a", "b", "c",
        ]

    def test_adds_idx_field(self, tmp_path):
        """Each response gets an _idx field matching its key."""
        responses = {3: _ok("x"), 7: _ok("y")}
        path = str(tmp_path / "responses.jsonl")

        pbr._save_all_responses(responses, path)

        saved = _read_jsonl(path)
        assert saved[0]["_idx"] == 3
        assert saved[1]["_idx"] == 7

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "a" / "b" / "responses.jsonl")
        pbr._save_all_responses({0: _ok("x")}, path)
        assert os.path.exists(path)


################
# RECOVER_TEMP #
################

class TestRecoverTemp:
    def test_recovers_partial_responses(self, tmp_path):
        """Merges temp responses into existing using sidecar index mapping."""
        resp_path = str(tmp_path / "responses.jsonl")

        # Sidecar says indices [5, 10, 15] were targeted
        with open(resp_path + ".idx", "w") as f:
            json.dump([5, 10, 15], f)

        # Only 2 of 3 completed before crash
        _write_jsonl(resp_path + ".tmp", [_ok("a"), _ok("b")])

        existing = {0: _ok("existing")}
        pbr._recover_temp(resp_path, existing)

        assert existing[5]["choices"][0]["message"]["content"] == "a"
        assert existing[10]["choices"][0]["message"]["content"] == "b"
        assert 15 not in existing

    def test_noop_when_no_temp_files(self, tmp_path):
        existing = {0: _ok("x")}
        pbr._recover_temp(str(tmp_path / "responses.jsonl"), existing)
        assert len(existing) == 1

    def test_noop_when_only_tmp_exists(self, tmp_path):
        """Both .tmp and .idx must exist for recovery."""
        resp_path = str(tmp_path / "responses.jsonl")
        _write_jsonl(resp_path + ".tmp", [_ok("a")])

        existing = {}
        pbr._recover_temp(resp_path, existing)
        assert existing == {}

    def test_cleans_up_temp_files(self, tmp_path):
        resp_path = str(tmp_path / "responses.jsonl")
        with open(resp_path + ".idx", "w") as f:
            json.dump([0], f)
        _write_jsonl(resp_path + ".tmp", [_ok("a")])

        pbr._recover_temp(resp_path, {})

        assert not os.path.exists(resp_path + ".tmp")
        assert not os.path.exists(resp_path + ".idx")


##########################
# GENERATE_WITH_RECOVERY #
##########################

class TestGenerateWithRecovery:
    def test_merges_new_into_existing(self, tmp_path):
        """New responses are merged into existing dict by target index."""
        resp_path = str(tmp_path / "responses.jsonl")
        existing = {0: _ok("keep")}

        with patch.object(pbr, "build_messages", return_value=[[], []]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("new_1"), _ok("new_2")]):
            pbr._generate_with_recovery(
                existing, [1, 2], [{}, {}, {}], resp_path,
                "math", "cot", "sae", "llama", "key", _GEN_KWARGS,
            )

        assert existing[0]["choices"][0]["message"]["content"] == "keep"
        assert existing[1]["choices"][0]["message"]["content"] == "new_1"
        assert existing[2]["choices"][0]["message"]["content"] == "new_2"

    def test_rewrites_main_file_with_idx(self, tmp_path):
        """Main responses.jsonl is rewritten with _idx fields in order."""
        resp_path = str(tmp_path / "responses.jsonl")
        existing = {0: _ok("a")}

        with patch.object(pbr, "build_messages", return_value=[[]]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("b")]):
            pbr._generate_with_recovery(
                existing, [1], [{}, {}], resp_path,
                "math", "cot", "sae", "llama", "key", _GEN_KWARGS,
            )

        saved = _read_jsonl(resp_path)
        assert len(saved) == 2
        assert saved[0]["_idx"] == 0
        assert saved[1]["_idx"] == 1

    def test_cleans_up_temp_files(self, tmp_path):
        resp_path = str(tmp_path / "responses.jsonl")

        with patch.object(pbr, "build_messages", return_value=[[]]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("a")]):
            pbr._generate_with_recovery(
                {}, [0], [{}], resp_path,
                "math", "cot", "sae", "llama", "key", _GEN_KWARGS,
            )

        assert not os.path.exists(resp_path + ".tmp")
        assert not os.path.exists(resp_path + ".idx")

    def test_passes_save_indices_to_generate(self, tmp_path):
        """Target indices are forwarded as save_indices to generate()."""
        resp_path = str(tmp_path / "responses.jsonl")

        with patch.object(pbr, "build_messages", return_value=[[]]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("x")]) as gen_mock:
            pbr._generate_with_recovery(
                {}, [42], [{}], resp_path,
                "math", "cot", "sae", "llama", "key", _GEN_KWARGS,
            )

        assert gen_mock.call_args.kwargs["save_indices"] == [42]


#########################
# CONTINUE_GENERATION #
#########################

class TestContinueGeneration:
    def _grade_mocks(self):
        """Context managers that mock grading/saving in _grade_and_save."""
        return (
            patch.object(pbr, "grade_task", return_value=[{"correct": True}]),
            patch.object(pbr, "save_results"),
            patch("dialecttax.endpoints.get_completions", return_value=["a"]),
        )

    def test_generates_missing_samples(self, tmp_path):
        """Only missing indices are sent to generate()."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}]
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_ok("b"), "_idx": 1},
        ])

        m1, m2, m3 = self._grade_mocks()
        with m1, m2, m3, \
             patch.object(pbr, "build_messages", return_value=[[]]) as bm, \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("c")]):
            pbr._continue_generation(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        assert bm.call_args.kwargs["indices"] == [2]

    def test_all_done_re_grades_without_generate(self, tmp_path):
        """When all samples exist, re-grades without calling generate()."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}]
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_ok("b"), "_idx": 1},
        ])

        m1, m2, m3 = self._grade_mocks()
        with m1 as grade_mock, m2, m3, \
             patch("dialecttax.endpoints.generate") as gen_mock:
            pbr._continue_generation(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        gen_mock.assert_not_called()
        grade_mock.assert_called_once()

    def test_temp_recovery_reduces_missing(self, tmp_path):
        """Temp recovery fills some indices, so fewer are generated."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}]

        # Only index 0 in main file
        _write_jsonl(resp_path, [{**_ok("a"), "_idx": 0}])

        # Index 1 recoverable from temp
        _write_jsonl(resp_path + ".tmp", [_ok("b")])
        with open(resp_path + ".idx", "w") as f:
            json.dump([1], f)

        m1, m2, m3 = self._grade_mocks()
        with m1, m2, m3, \
             patch.object(pbr, "build_messages", return_value=[[]]) as bm, \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("c")]):
            pbr._continue_generation(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        # Only index 2 should be generated (0 from main, 1 from temp)
        assert bm.call_args.kwargs["indices"] == [2]

    def test_continue_passes_save_indices_to_generate(self, tmp_path):
        """generate() receives save_indices matching the missing indices."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}, {}, {}]
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_ok("b"), "_idx": 1},
            {**_ok("c"), "_idx": 2},
        ])

        m1, m2, m3 = self._grade_mocks()
        with m1, m2, m3, \
             patch.object(pbr, "build_messages", return_value=[[], []]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("d"), _ok("e")]) as gen_mock:
            pbr._continue_generation(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        assert gen_mock.call_args.kwargs["save_indices"] == [3, 4]

    def test_continue_final_file_preserves_order(self, tmp_path):
        """After continue, responses.jsonl has all samples in index order."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}, {}, {}]
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_ok("b"), "_idx": 1},
            {**_ok("c"), "_idx": 2},
        ])

        m1, m2, m3 = self._grade_mocks()
        with m1, m2, m3, \
             patch.object(pbr, "build_messages", return_value=[[], []]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("d"), _ok("e")]):
            pbr._continue_generation(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        saved = _read_jsonl(resp_path)
        assert len(saved) == 5
        for i, expected in enumerate(["a", "b", "c", "d", "e"]):
            assert saved[i]["_idx"] == i
            assert saved[i]["choices"][0]["message"]["content"] == expected

    def test_continue_from_legacy_file_without_idx(self, tmp_path):
        """Continue works with old response files that lack _idx fields."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}, {}, {}]
        # Old file without _idx — line numbers used as indices
        _write_jsonl(resp_path, [_ok("a"), _ok("b"), _ok("c")])

        m1, m2, m3 = self._grade_mocks()
        with m1, m2, m3, \
             patch.object(pbr, "build_messages", return_value=[[], []]) as bm, \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("d"), _ok("e")]) as gen_mock:
            pbr._continue_generation(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        assert bm.call_args.kwargs["indices"] == [3, 4]
        assert gen_mock.call_args.kwargs["save_indices"] == [3, 4]

        saved = _read_jsonl(resp_path)
        assert len(saved) == 5
        for i, expected in enumerate(["a", "b", "c", "d", "e"]):
            assert saved[i]["_idx"] == i
            assert saved[i]["choices"][0]["message"]["content"] == expected

    def test_continue_grades_in_dataset_order(self, tmp_path):
        """Grading receives responses ordered by dataset index."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}, {}, {}]
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_ok("b"), "_idx": 1},
            {**_ok("c"), "_idx": 2},
        ])

        graded_contents = []

        def capture_completions(responses):
            for r in responses:
                try:
                    graded_contents.append(
                        r["choices"][0]["message"]["content"]
                    )
                except (KeyError, IndexError, TypeError):
                    graded_contents.append(None)
            return graded_contents

        with patch.object(
                 pbr, "grade_task",
                 return_value=[{"correct": True}] * 5), \
             patch.object(pbr, "save_results"), \
             patch("dialecttax.endpoints.get_completions",
                   side_effect=capture_completions), \
             patch.object(pbr, "build_messages", return_value=[[], []]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("d"), _ok("e")]):
            pbr._continue_generation(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        assert graded_contents == ["a", "b", "c", "d", "e"]


#################
# PATCH_SAMPLES #
#################

class TestPatchSamples:
    def _grade_mocks(self):
        return (
            patch.object(pbr, "grade_task", return_value=[{"correct": True}]),
            patch.object(pbr, "save_results"),
            patch("dialecttax.endpoints.get_completions", return_value=["a"]),
        )

    def test_patches_error_responses(self, tmp_path):
        """Only error indices are re-generated."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}]
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_err(), "_idx": 1},
            {**_ok("c"), "_idx": 2},
        ])

        m1, m2, m3 = self._grade_mocks()
        with m1, m2, m3, \
             patch.object(pbr, "build_messages", return_value=[[]]) as bm, \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("fixed")]):
            pbr._patch_samples(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        assert bm.call_args.kwargs["indices"] == [1]

    def test_no_errors_skips_generation(self, tmp_path):
        """When no errors exist, does not call generate()."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        _write_jsonl(resp_path, [{**_ok("a"), "_idx": 0}])

        with patch("dialecttax.endpoints.generate") as gen_mock:
            pbr._patch_samples(
                [{}], resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        gen_mock.assert_not_called()

    def test_no_existing_responses(self, tmp_path):
        """Returns early when no responses file exists."""
        with patch("dialecttax.endpoints.generate") as gen_mock:
            pbr._patch_samples(
                [{}], str(tmp_path / "nonexistent.jsonl"),
                str(tmp_path / "results.jsonl"),
                "math", "cot", "sae", "llama", "key", _GEN_KWARGS, "[test]",
            )

        gen_mock.assert_not_called()

    def test_ignores_out_of_range_indices(self, tmp_path):
        """Error responses with idx >= len(ds) are not patched."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}]  # only 1 sample
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_err(), "_idx": 5},  # out of range
        ])

        with patch("dialecttax.endpoints.generate") as gen_mock:
            pbr._patch_samples(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        gen_mock.assert_not_called()

    def test_patch_passes_save_indices_to_generate(self, tmp_path):
        """generate() receives save_indices matching the error indices."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}, {}, {}]
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_err(), "_idx": 1},
            {**_ok("c"), "_idx": 2},
            {**_err(), "_idx": 3},
            {**_ok("e"), "_idx": 4},
        ])

        m1, m2, m3 = self._grade_mocks()
        with m1, m2, m3, \
             patch.object(pbr, "build_messages", return_value=[[], []]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("b"), _ok("d")]) as gen_mock:
            pbr._patch_samples(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        assert gen_mock.call_args.kwargs["save_indices"] == [1, 3]

    def test_patch_final_file_preserves_order(self, tmp_path):
        """After patching, responses.jsonl has all samples in index order."""
        resp_path = str(tmp_path / "responses.jsonl")
        out_path = str(tmp_path / "results.jsonl")
        ds = [{}, {}, {}, {}, {}]
        _write_jsonl(resp_path, [
            {**_ok("a"), "_idx": 0},
            {**_err(), "_idx": 1},
            {**_ok("c"), "_idx": 2},
            {**_err(), "_idx": 3},
            {**_ok("e"), "_idx": 4},
        ])

        m1, m2, m3 = self._grade_mocks()
        with m1, m2, m3, \
             patch.object(pbr, "build_messages", return_value=[[], []]), \
             patch("dialecttax.endpoints.generate",
                   return_value=[_ok("b_fixed"), _ok("d_fixed")]):
            pbr._patch_samples(
                ds, resp_path, out_path, "math", "cot", "sae",
                "llama", "key", _GEN_KWARGS, "[test]",
            )

        saved = _read_jsonl(resp_path)
        assert len(saved) == 5
        expected = ["a", "b_fixed", "c", "d_fixed", "e"]
        for i, content in enumerate(expected):
            assert saved[i]["_idx"] == i
            assert saved[i]["choices"][0]["message"]["content"] == content
