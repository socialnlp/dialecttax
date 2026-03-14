"""Tests for scripts/logits/generate_logits.py."""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


###########
# IMPORTS #
###########

# generate_logits.py has module-level code that imports dialecttax and runs
# Hydra setup. We mock these heavy dependencies so the script's pure helper
# functions can be imported.

_mock_dialecttax = MagicMock()
_mock_dialecttax.utils.load_config.return_value = {
    "directories": {"experiments": "/tmp/experiments", "preprocessed": "/tmp", "datasets": "/tmp"},
    "keys": {},
}

_mock_redial = MagicMock()
_mock_redial.TASKS = ["algorithm", "logic", "math", "planning"]
_mock_redial.FILE_NAME_QA_FORMAT = "{task}_{dialect}_qa.jsonl"
_mock_redial.FILE_NAME_FORMAT = "{task}_{dialect}.jsonl"
_mock_redial.DIRECTORY_NAME = "ReDial"

_mock_parallelaave = MagicMock()
_mock_parallelaave.FILE_NAME_FORMAT = "{dialect}_samples.txt"
_mock_parallelaave.DIRECTORY_NAME = "parallelaave"
del _mock_parallelaave.TASKS
del _mock_parallelaave.FILE_NAME_QA_FORMAT

_mock_multivalue = MagicMock()
_mock_multivalue.FILE_NAME_FORMAT = "{dialect}_samples.txt"
_mock_multivalue.DIRECTORY_NAME = "multivalue"
del _mock_multivalue.TASKS
del _mock_multivalue.FILE_NAME_QA_FORMAT

_DATASET_MODULES = {
    "redial": _mock_redial,
    "parallelaave": _mock_parallelaave,
    "multivalue": _mock_multivalue,
}
_mock_dialecttax.data.DATASET_MODULES = _DATASET_MODULES

# Mock prompt constants
_mock_dialecttax.prompts.INSTS_MQA = {
    "math": {
        "naive": {"sae": "Pick the correct answer.\n{choices}", "aave": "Pick the correct answer.\n{choices}"},
        "cot": {"sae": "Think step by step.\n{choices}", "aave": "Think step by step.\n{choices}"},
    },
}
_mock_dialecttax.prompts.PROMPTS = {
    "math": {
        "naive": {"sae": "What is {problem}?", "aave": "What is {problem}?"},
        "cot": {"sae": "What is {problem}?", "aave": "What is {problem}?"},
    },
}
_mock_dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
    "math": MagicMock(return_value=lambda ds, i: f"What is {ds[i]['problem']}?"),
}
_mock_dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n\n{body}"
_mock_dialecttax.prompts.get_system_prompt = lambda dialect, **kwargs: f"System prompt for {dialect}"

_mock_dialecttax.logits = MagicMock()

_mock_mqa_grader = MagicMock()
_mock_dialecttax.data.graders = MagicMock()
_mock_dialecttax.data.graders.mqa = _mock_mqa_grader

_MOCK_MODULES = {
    "dialecttax": _mock_dialecttax,
    "dialecttax.utils": _mock_dialecttax.utils,
    "dialecttax.data": _mock_dialecttax.data,
    "dialecttax.data.graders": _mock_dialecttax.data.graders,
    "dialecttax.data.graders.mqa": _mock_mqa_grader,
    "dialecttax.models": _mock_dialecttax.models,
    "dialecttax.prompts": _mock_dialecttax.prompts,
    "dialecttax.logits": _mock_dialecttax.logits,
    "hydra": MagicMock(),
    "hydra.core": MagicMock(),
    "hydra.core.hydra_config": MagicMock(),
    "omegaconf": MagicMock(),
    "numpy": MagicMock(),
    "torch": MagicMock(),
}

_saved_modules = {}
for _name, _mock in _MOCK_MODULES.items():
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "scripts", "logits", "generate_logits.py",
    )
)
_spec = importlib.util.spec_from_file_location("generate_logits", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _orig

# Extract functions under test
_build_redial_prompt = _mod._build_redial_prompt
_build_samples = _mod._build_samples
_find_answer_step = _mod._find_answer_step
_answer_entropy_at_step = _mod._answer_entropy_at_step
_metadata_has_field = _mod._metadata_has_field


########################
# _BUILD_REDIAL_PROMPT #
########################


class TestBuildRedialPrompt:
    @pytest.fixture(autouse=True)
    def _setup_prompts(self):
        """Set up mock prompt functions."""
        _mod.dialecttax.prompts.INSTS_MQA = {
            "math": {
                "naive": {"sae": "Pick the correct answer.\n{choices}"},
                "cot": {"sae": "Think step by step.\n{choices}"},
            },
        }
        _mod.dialecttax.prompts.PROMPTS = {
            "math": {
                "naive": {"sae": "What is {problem}?"},
                "cot": {"sae": "What is {problem}?"},
            },
        }
        _mod.dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
            "math": MagicMock(return_value=lambda ds, i: f"What is {ds[i]['problem']}?"),
        }
        _mod.dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n\n{body}"
        _mod.dialecttax.prompts.get_system_prompt = lambda dialect, **kwargs: f"System prompt for {dialect}"

    def test_returns_system_and_user(self):
        """Returns a (system, user_prompt) tuple."""
        ds = [{"problem": "2+2", "choices": {"A": "3", "B": "4"}}]
        result = _build_redial_prompt(ds, 0, "math", "sae", reasoning="naive")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_prompt(self):
        """System prompt uses dialect."""
        ds = [{"problem": "2+2", "choices": {"A": "3", "B": "4"}}]
        system, _ = _build_redial_prompt(ds, 0, "math", "sae", reasoning="naive")
        assert "sae" in system

    def test_choices_in_prompt(self):
        """Choices are formatted into the instructions."""
        ds = [{"problem": "2+2", "choices": {"A": "3", "B": "4", "C": "5"}}]
        _, prompt = _build_redial_prompt(ds, 0, "math", "sae", reasoning="naive")
        assert "A. 3" in prompt
        assert "B. 4" in prompt
        assert "C. 5" in prompt

    def test_cot_reasoning(self):
        """Uses cot instructions when reasoning='cot'."""
        ds = [{"problem": "2+2", "choices": {"A": "3", "B": "4"}}]
        _, prompt = _build_redial_prompt(ds, 0, "math", "sae", reasoning="cot")
        assert "Think step by step" in prompt

    def test_naive_reasoning(self):
        """Uses naive instructions when reasoning='naive'."""
        ds = [{"problem": "2+2", "choices": {"A": "3", "B": "4"}}]
        _, prompt = _build_redial_prompt(ds, 0, "math", "sae", reasoning="naive")
        assert "Pick the correct answer" in prompt


##################
# _BUILD_SAMPLES #
##################


class TestBuildSamples:
    @pytest.fixture(autouse=True)
    def _setup_prompts(self):
        """Set up mock prompt functions."""
        _mod.dialecttax.prompts.INSTS_MQA = {
            "math": {
                "naive": {"sae": "Pick the correct answer.\n{choices}"},
                "cot": {"sae": "Think step by step.\n{choices}"},
            },
        }
        _mod.dialecttax.prompts.PROMPTS = {
            "math": {
                "naive": {"sae": "What is {problem}?"},
                "cot": {"sae": "What is {problem}?"},
            },
        }
        _mod.dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
            "math": MagicMock(return_value=lambda ds, i: f"What is {ds[i]['problem']}?"),
        }
        _mod.dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n\n{body}"
        _mod.dialecttax.prompts.get_system_prompt = lambda dialect, **kwargs: f"System prompt for {dialect}"
        _mod.dialecttax.data.DATASET_MODULES = _DATASET_MODULES

    def test_redial_has_prompt_key(self):
        """ReDial samples have system, prompt, answer keys."""
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": "B", "choices": {"A": "3", "B": "4"}}]
        samples = _build_samples(ds, "math", "sae", "redial", reasoning="naive")
        assert len(samples) == 1
        assert "prompt" in samples[0]
        assert "system" in samples[0]
        assert "answer" in samples[0]
        assert samples[0]["unique_id"] == "q1"

    def test_text_only_has_text_key(self):
        """Text-only datasets have text key, no prompt key."""
        ds = [{"unique_id": "t1", "text": "hello world"}]
        samples = _build_samples(ds, "none", "sae", "parallelaave")
        assert len(samples) == 1
        assert "text" in samples[0]
        assert "prompt" not in samples[0]

    def test_multivalue_text_only(self):
        """Multivalue is treated as text-only."""
        ds = [{"unique_id": "m1", "text": "some text"}]
        samples = _build_samples(ds, "none", "sae", "multivalue")
        assert "text" in samples[0]
        assert "prompt" not in samples[0]

    def test_answer_cast_to_string(self):
        """Numeric answers are converted to strings."""
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": 4, "choices": {"A": "3", "B": "4"}}]
        samples = _build_samples(ds, "math", "sae", "redial")
        assert samples[0]["answer"] == "4"

    def test_reasoning_passed_through(self):
        """Reasoning strategy is passed to prompt builder."""
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": "B", "choices": {"A": "3", "B": "4"}}]
        samples_naive = _build_samples(ds, "math", "sae", "redial", reasoning="naive")
        samples_cot = _build_samples(ds, "math", "sae", "redial", reasoning="cot")
        # cot and naive should produce different prompts
        assert "Think step by step" in samples_cot[0]["prompt"]
        assert "Pick the correct answer" in samples_naive[0]["prompt"]


##################
# MAX_TOKENS_NEW #
##################


class TestMaxTokensNewResolution:
    """Test the max_tokens_new cascade: model override > reasoning config default."""

    def _make_cfg(self, model_name, reasoning_name, model_max_tokens_new=None,
                  reasoning_base=256, reasoning_instruct=10):
        """Create a mock config for testing max_tokens_new resolution.

        Args:
            model_name: Model name (e.g. "qwen_8b_instruct").
            reasoning_name: Reasoning strategy name ("naive" or "cot").
            model_max_tokens_new: Dict keyed by reasoning name (e.g. {"cot": 2048}).
            reasoning_base: Default max_tokens_new for base models.
            reasoning_instruct: Default max_tokens_new for instruct models.
        """
        model_cfg = {"name": model_name}
        if model_max_tokens_new is not None:
            model_cfg["max_tokens_new"] = model_max_tokens_new

        # Use a real dict wrapped in a class so `key in cfg.model` works
        class DictLike(dict):
            def __getattr__(self, key):
                try:
                    return self[key]
                except KeyError:
                    raise AttributeError(key)

        cfg_model = DictLike(model_cfg)
        cfg_reasoning = DictLike({
            "name": reasoning_name,
            "max_tokens_new": DictLike({"base": reasoning_base, "instruct": reasoning_instruct}),
        })
        cfg = DictLike({"model": cfg_model, "reasoning": cfg_reasoning})
        return cfg

    def _resolve(self, cfg):
        """Replicate the resolution logic from generate_logits.py main."""
        instruct = cfg.model.name.endswith("_instruct")
        reasoning = cfg.reasoning.name
        variant = "instruct" if instruct else "base"
        max_tokens_new = int(cfg.reasoning.max_tokens_new[variant])
        if "max_tokens_new" in cfg.model and reasoning in cfg.model.max_tokens_new:
            max_tokens_new = int(cfg.model.max_tokens_new[reasoning])
        return max_tokens_new

    # --- Non-Qwen (no model override) ---

    def test_llama_naive_base(self):
        """Llama base + naive: uses reasoning default (256)."""
        cfg = self._make_cfg("llama_8b_base", "naive", reasoning_base=256, reasoning_instruct=10)
        assert self._resolve(cfg) == 256

    def test_llama_naive_instruct(self):
        """Llama instruct + naive: uses reasoning default (10)."""
        cfg = self._make_cfg("llama_8b_instruct", "naive", reasoning_base=256, reasoning_instruct=10)
        assert self._resolve(cfg) == 10

    def test_llama_cot_base(self):
        """Llama base + cot: uses reasoning default (1024)."""
        cfg = self._make_cfg("llama_8b_base", "cot", reasoning_base=1024, reasoning_instruct=1024)
        assert self._resolve(cfg) == 1024

    def test_llama_cot_instruct(self):
        """Llama instruct + cot: uses reasoning default (1024)."""
        cfg = self._make_cfg("llama_8b_instruct", "cot", reasoning_base=1024, reasoning_instruct=1024)
        assert self._resolve(cfg) == 1024

    # --- Qwen with model-level override ---

    def test_qwen_cot_base_override(self):
        """Qwen base + cot: model override (2048) takes precedence."""
        cfg = self._make_cfg(
            "qwen_8b_base", "cot",
            model_max_tokens_new={"cot": 2048},
            reasoning_base=1024, reasoning_instruct=1024,
        )
        assert self._resolve(cfg) == 2048

    def test_qwen_cot_instruct_override(self):
        """Qwen instruct + cot: model override (2048) takes precedence."""
        cfg = self._make_cfg(
            "qwen_8b_instruct", "cot",
            model_max_tokens_new={"cot": 2048},
            reasoning_base=1024, reasoning_instruct=1024,
        )
        assert self._resolve(cfg) == 2048

    def test_qwen_naive_base_no_override(self):
        """Qwen base + naive: no matching key, falls back to reasoning default (256)."""
        cfg = self._make_cfg(
            "qwen_8b_base", "naive",
            model_max_tokens_new={"cot": 2048},
            reasoning_base=256, reasoning_instruct=10,
        )
        assert self._resolve(cfg) == 256

    def test_qwen_naive_instruct_no_override(self):
        """Qwen instruct + naive: no matching key, falls back to reasoning default (10)."""
        cfg = self._make_cfg(
            "qwen_8b_instruct", "naive",
            model_max_tokens_new={"cot": 2048},
            reasoning_base=256, reasoning_instruct=10,
        )
        assert self._resolve(cfg) == 10

    # --- Gemma (no override, different model family) ---

    def test_gemma_cot_base(self):
        """Gemma base + cot: no model override, uses reasoning default."""
        cfg = self._make_cfg("gemma_12b_base", "cot", reasoning_base=1024, reasoning_instruct=1024)
        assert self._resolve(cfg) == 1024

    def test_gemma_naive_instruct(self):
        """Gemma instruct + naive: uses reasoning default (10)."""
        cfg = self._make_cfg("gemma_12b_instruct", "naive", reasoning_base=256, reasoning_instruct=10)
        assert self._resolve(cfg) == 10

    # --- 70B model ---

    def test_llama_70b_cot_instruct(self):
        """Llama 70B instruct + cot: uses reasoning default (1024)."""
        cfg = self._make_cfg("llama_70b_instruct", "cot", reasoning_base=1024, reasoning_instruct=1024)
        assert self._resolve(cfg) == 1024

    # --- Qwen 32B ---

    def test_qwen_32b_cot_instruct_override(self):
        """Qwen 32B instruct + cot: model override (2048)."""
        cfg = self._make_cfg(
            "qwen_32b_instruct", "cot",
            model_max_tokens_new={"cot": 2048},
            reasoning_base=1024, reasoning_instruct=1024,
        )
        assert self._resolve(cfg) == 2048


#####################
# _FIND_ANSWER_STEP #
#####################


class _FakeTokenizer:
    """Mock tokenizer that maps token IDs to predetermined strings."""

    def __init__(self, vocab):
        """Args: vocab: dict mapping token ID (int) to decoded string."""
        self._vocab = vocab

    def decode(self, ids, **kwargs):
        return "".join(self._vocab[i] for i in ids)


class TestFindAnswerStep:
    """Test _find_answer_step: locates the generation step that completes '#### <answer>'."""

    # --- Basic cases ---

    def test_clean_answer(self):
        """'#### B' as separate tokens → step of 'B'."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "B"})
        assert _find_answer_step("B", [0, 1, 2], tok) == 2

    def test_clean_answer_no_space(self):
        """'####B' without space → step of 'B'."""
        tok = _FakeTokenizer({0: "####", 1: "B"})
        assert _find_answer_step("B", [0, 1], tok) == 1

    def test_answer_with_trailing_tokens(self):
        """'#### B is correct' → step of first B."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "B", 3: " is", 4: " correct"})
        assert _find_answer_step("B", [0, 1, 2, 3, 4], tok) == 2

    # --- Spurious matches before #### ---

    def test_b_before_marker(self):
        """'B #### B' → finds B after ####, not before."""
        tok = _FakeTokenizer({0: "B", 1: " ", 2: "####", 3: " ", 4: "B"})
        assert _find_answer_step("B", [0, 1, 2, 3, 4], tok) == 4

    def test_b_before_marker_repeated(self):
        """'B #### B B' → finds first B after ####."""
        tok = _FakeTokenizer({0: "B", 1: " ", 2: "####", 3: " ", 4: "B", 5: " ", 6: "B"})
        assert _find_answer_step("B", [0, 1, 2, 3, 4, 5, 6], tok) == 4

    def test_multiple_b_after_marker(self):
        """'#### B B' → finds first B after ####."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "B", 3: " ", 4: "B"})
        assert _find_answer_step("B", [0, 1, 2, 3, 4], tok) == 2

    # --- Split tokens ---

    def test_hashes_split_across_tokens(self):
        """'##' + '##' + ' B' → finds B after split ####."""
        tok = _FakeTokenizer({0: "##", 1: "##", 2: " B"})
        assert _find_answer_step("B", [0, 1, 2], tok) == 2

    def test_marker_and_answer_same_token(self):
        """'#### B' as a single token → step 0."""
        tok = _FakeTokenizer({0: "#### B"})
        assert _find_answer_step("B", [0], tok) == 0

    def test_marker_answer_fused_no_space(self):
        """'####B' as a single token → step 0."""
        tok = _FakeTokenizer({0: "####B"})
        assert _find_answer_step("B", [0], tok) == 0

    def test_b_fused_before_hashes(self):
        """'B####' as single token + ' B' → finds B in second token."""
        tok = _FakeTokenizer({0: "B####", 1: " B"})
        assert _find_answer_step("B", [0, 1], tok) == 1

    def test_b_fused_before_hashes_no_later_b(self):
        """'B####' with no B after → returns None."""
        tok = _FakeTokenizer({0: "B####", 1: " C"})
        assert _find_answer_step("B", [0, 1], tok) is None

    # --- None / missing cases ---

    def test_predicted_none(self):
        """predicted=None → returns None immediately."""
        tok = _FakeTokenizer({0: "####", 1: " B"})
        assert _find_answer_step(None, [0, 1], tok) is None

    def test_no_marker_in_text(self):
        """No #### in generated text → returns None."""
        tok = _FakeTokenizer({0: "The", 1: " answer", 2: " is", 3: " B"})
        assert _find_answer_step("B", [0, 1, 2, 3], tok) is None

    def test_empty_gen_ids(self):
        """Empty generation → returns None."""
        tok = _FakeTokenizer({})
        assert _find_answer_step("B", [], tok) is None

    def test_marker_only_no_answer(self):
        """'####' with no answer token after → returns None."""
        tok = _FakeTokenizer({0: "####"})
        assert _find_answer_step("B", [0], tok) is None

    def test_marker_with_wrong_answer(self):
        """'#### C' but predicted='B' → returns None."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "C"})
        assert _find_answer_step("B", [0, 1, 2], tok) is None

    # --- Multi-char answers ---

    def test_numeric_answer(self):
        """'#### 42' → finds step of '42'."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "42"})
        assert _find_answer_step("42", [0, 1, 2], tok) == 2

    def test_numeric_split_across_tokens(self):
        """'#### 4' + '2' where predicted='42' → finds step that completes '42'."""
        tok = _FakeTokenizer({0: "####", 1: " 4", 2: "2"})
        assert _find_answer_step("42", [0, 1, 2], tok) == 2

    # --- Realistic CoT ---

    def test_cot_long_reasoning_then_answer(self):
        """Long CoT followed by '#### A' → finds A at the end."""
        tok = _FakeTokenizer({
            0: "Let", 1: " me", 2: " think", 3: ".", 4: " The",
            5: " answer", 6: " is", 7: " A", 8: ".", 9: " ",
            10: "####", 11: " ", 12: "A",
        })
        assert _find_answer_step("A", list(range(13)), tok) == 12

    def test_cot_answer_letter_in_reasoning(self):
        """CoT mentions 'A' in reasoning, then '#### A' → finds A after ####."""
        tok = _FakeTokenizer({
            0: "Option", 1: " A", 2: " is", 3: " correct", 4: " because",
            5: " ", 6: "####", 7: " ", 8: "A",
        })
        assert _find_answer_step("A", list(range(9)), tok) == 8

    # --- Regex special chars in answer ---

    def test_answer_with_regex_special_chars(self):
        """Answer containing regex special char (e.g. '$5') is escaped properly."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "$5"})
        assert _find_answer_step("$5", [0, 1, 2], tok) == 2

    def test_answer_with_parentheses(self):
        """Answer '(A)' is escaped properly."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "(A)"})
        assert _find_answer_step("(A)", [0, 1, 2], tok) == 2


###########################
# _ANSWER_ENTROPY_AT_STEP #
###########################


class TestAnswerEntropyAtStep:
    def test_returns_selected_entropy_as_float(self):
        assert _answer_entropy_at_step([0.1, 0.7, 0.3], 1) == pytest.approx(0.7)

    @pytest.mark.parametrize("step", [None, -1, 3])
    def test_missing_or_invalid_step_returns_none(self, step):
        assert _answer_entropy_at_step([0.1, 0.7, 0.3], step) is None

    def test_supports_tensor_like_scalar(self):
        scalar = MagicMock()
        scalar.item.return_value = 1.25
        assert _answer_entropy_at_step([scalar], 0) == 1.25


#######################
# _METADATA_HAS_FIELD #
#######################


class TestMetadataHasField:
    def test_requires_field_on_every_row(self, tmp_path):
        path = tmp_path / "metadata.jsonl"
        path.write_text('{"answer_entropy": 0.4}\n{"answer_entropy": null}\n')
        assert _metadata_has_field(path, "answer_entropy")

    def test_rejects_legacy_row_without_field(self, tmp_path):
        path = tmp_path / "metadata.jsonl"
        path.write_text('{"answer_entropy": 0.4}\n{"gen_mean_entropy": 0.8}\n')
        assert not _metadata_has_field(path, "answer_entropy")

    @pytest.mark.parametrize("contents", ["", "not json\n"])
    def test_rejects_empty_or_invalid_metadata(self, tmp_path, contents):
        path = tmp_path / "metadata.jsonl"
        path.write_text(contents)
        assert not _metadata_has_field(path, "answer_entropy")
