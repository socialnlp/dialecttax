"""Tests for scripts/rewards/benchmark_rewards.py."""

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


###########
# IMPORTS #
###########

# benchmark_rewards.py has module-level code that imports dialecttax and runs
# Hydra setup. We mock these heavy dependencies so the script's pure helper
# functions can be imported.

_mock_dialecttax = MagicMock()
_mock_dialecttax.utils.load_config.return_value = {
    "directories": {"experiments": "/tmp/experiments", "preprocessed": "/tmp", "datasets": "/tmp"},
    "keys": {},
}

_mock_redial = MagicMock()
_mock_redial.TASKS = ["algorithm", "logic", "math", "planning"]
_mock_redial.FILE_NAME_FORMAT = "{task}_{dialect}.jsonl"
_mock_redial.DIRECTORY_NAME = "ReDial"

_mock_parallelaave = MagicMock()
_mock_parallelaave.FILE_NAME_FORMAT = "{dialect}_samples.txt"
_mock_parallelaave.DIRECTORY_NAME = "parallelaave"
del _mock_parallelaave.TASKS  # no TASKS attribute

_DATASET_MODULES = {
    "redial": _mock_redial,
    "parallelaave": _mock_parallelaave,
}
_mock_dialecttax.data.DATASET_MODULES = _DATASET_MODULES

# Mock reward model classes
_mock_dialecttax.rewards.SkyworkRewardModel = MagicMock()
_mock_dialecttax.rewards.QRMRewardModel = MagicMock()
_mock_dialecttax.rewards.Ai2RewardModel = MagicMock()

# Mock prompt constants
_mock_dialecttax.prompts.PROMPT_REWARD_WORDS = "Score this word:"
_mock_dialecttax.prompts.PROMPT_REWARD_TOKENS = "Score this token:"
_mock_dialecttax.prompts.INSTS = {
    "math": {"naive": {"sae": "Solve the math problem.", "aave": "Solve the math problem."}},
}
_mock_dialecttax.prompts.PROMPTS = {
    "math": {"naive": {"sae": "What is {problem}?", "aave": "What is {problem}?"}},
}
_mock_dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
    "math": MagicMock(return_value=lambda ds, i: f"Problem: {ds[i]['problem']}"),
}
_mock_dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n{body}"

_MOCK_MODULES = {
    "dialecttax": _mock_dialecttax,
    "dialecttax.utils": _mock_dialecttax.utils,
    "dialecttax.data": _mock_dialecttax.data,
    "dialecttax.rewards": _mock_dialecttax.rewards,
    "dialecttax.prompts": _mock_dialecttax.prompts,
    "hydra": MagicMock(),
    "hydra.core": MagicMock(),
    "hydra.core.hydra_config": MagicMock(),
    "omegaconf": MagicMock(),
}

_saved_modules = {}
for _name, _mock in _MOCK_MODULES.items():
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "scripts", "rewards", "benchmark_rewards.py",
    )
)
_spec = importlib.util.spec_from_file_location("benchmark_rewards", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _orig

# Extract functions under test
load_reward_model = _mod.load_reward_model
_load_cache = _mod._load_cache
_save_cache = _mod._save_cache
score_items = _mod.score_items
score_samples = _mod.score_samples
_build_redial_prompt = _mod._build_redial_prompt
PROVIDER_CLASSES = _mod.PROVIDER_CLASSES


######################
# LOAD_REWARD_MODEL  #
######################


class TestLoadRewardModel:
    @pytest.fixture(autouse=True)
    def _reset_mocks(self):
        """Reset provider class mocks between tests."""
        for cls in PROVIDER_CLASSES.values():
            cls.reset_mock()

    def test_skywork_provider(self):
        """Derives 'skywork' provider from name prefix."""
        rm = load_reward_model("skywork_llama_8b")
        PROVIDER_CLASSES["skywork"].assert_called_with("skywork_llama_8b", device="auto")

    def test_qrm_provider(self):
        """Derives 'qrm' provider from name prefix."""
        load_reward_model("qrm_llama_8b", device="cuda:0")
        PROVIDER_CLASSES["qrm"].assert_called_with("qrm_llama_8b", device="cuda:0")

    def test_ai2_provider(self):
        """Derives 'ai2' provider from name prefix."""
        load_reward_model("ai2_llama_8b")
        PROVIDER_CLASSES["ai2"].assert_called_with("ai2_llama_8b", device="auto")

    def test_calls_load(self):
        """Calls .load() on the instantiated model."""
        rm = load_reward_model("skywork_llama_8b")
        rm.load.assert_called_once()

    def test_returns_model_instance(self):
        """Returns the loaded reward model instance."""
        rm = load_reward_model("skywork_llama_8b")
        assert rm is PROVIDER_CLASSES["skywork"].return_value

    def test_unknown_provider_raises(self):
        """Raises KeyError for unknown provider prefix."""
        with pytest.raises(KeyError):
            load_reward_model("unknown_model_8b")


##############
# LOAD_CACHE #
##############


class TestLoadCache:
    def test_missing_file_returns_empty_dict(self, tmp_path):
        """Returns {} when cache file does not exist."""
        result = _load_cache(os.path.join(str(tmp_path), "nonexistent.json"))
        assert result == {}

    def test_loads_existing_cache(self, tmp_path):
        """Loads and returns JSON contents from an existing file."""
        cache_path = os.path.join(str(tmp_path), "cache.json")
        data = {"hello": 1.5, "world": -0.3}
        with open(cache_path, "w") as f:
            json.dump(data, f)

        result = _load_cache(cache_path)
        assert result == data

    def test_loads_empty_cache(self, tmp_path):
        """Loads an empty JSON object correctly."""
        cache_path = os.path.join(str(tmp_path), "cache.json")
        with open(cache_path, "w") as f:
            json.dump({}, f)

        result = _load_cache(cache_path)
        assert result == {}


##############
# SAVE_CACHE #
##############


class TestSaveCache:
    def test_writes_json(self, tmp_path):
        """Saves a dict as valid JSON."""
        cache_path = os.path.join(str(tmp_path), "cache.json")
        data = {"token_a": 2.5, "token_b": -1.0}

        _save_cache(data, cache_path)

        with open(cache_path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        """Creates intermediate directories if missing."""
        cache_path = os.path.join(str(tmp_path), "a", "b", "cache.json")

        _save_cache({"x": 1.0}, cache_path)

        assert os.path.isfile(cache_path)

    def test_overwrites_existing(self, tmp_path):
        """Overwrites an existing cache file."""
        cache_path = os.path.join(str(tmp_path), "cache.json")
        _save_cache({"old": 1.0}, cache_path)

        _save_cache({"new": 2.0}, cache_path)

        with open(cache_path) as f:
            loaded = json.load(f)
        assert loaded == {"new": 2.0}

    def test_roundtrip(self, tmp_path):
        """Cache survives save then load."""
        cache_path = os.path.join(str(tmp_path), "cache.json")
        data = {"a": 0.1, "b": 0.2, "c": 0.3}

        _save_cache(data, cache_path)
        loaded = _load_cache(cache_path)

        assert loaded == data


###############
# SCORE_ITEMS #
###############


class TestScoreItems:
    def _make_rm(self, score_map):
        """Create a mock reward model that returns scores from a dict."""
        rm = MagicMock()
        rm.score.side_effect = lambda conv: score_map.get(conv[1]["content"], 0.0)
        return rm

    def test_scores_uncached_items(self):
        """Scores items not in cache and updates cache in place."""
        rm = self._make_rm({"hello": 1.0, "world": 2.0})
        cache = {}

        n_new = score_items(rm, "prompt", ["hello", "world"], cache, label="word")

        assert n_new == 2
        assert cache == {"hello": 1.0, "world": 2.0}

    def test_skips_cached_items(self):
        """Does not re-score items already in cache."""
        rm = self._make_rm({"new": 3.0})
        cache = {"existing": 1.5}

        n_new = score_items(rm, "prompt", ["existing", "new"], cache, label="word")

        assert n_new == 1
        assert rm.score.call_count == 1
        assert cache["existing"] == 1.5
        assert cache["new"] == 3.0

    def test_all_cached_returns_zero(self):
        """Returns 0 when all items are already cached."""
        rm = MagicMock()
        cache = {"a": 1.0, "b": 2.0}

        n_new = score_items(rm, "prompt", ["a", "b"], cache, label="token")

        assert n_new == 0
        rm.score.assert_not_called()

    def test_empty_items_returns_zero(self):
        """Returns 0 for empty items list."""
        rm = MagicMock()
        cache = {}

        n_new = score_items(rm, "prompt", [], cache, label="word")

        assert n_new == 0

    def test_builds_correct_conversation(self):
        """Passes correct conversation format to rm.score."""
        rm = MagicMock(score=MagicMock(return_value=1.0))
        cache = {}

        score_items(rm, "Rate this:", ["hello"], cache, label="word")

        rm.score.assert_called_once_with([
            {"role": "user", "content": "Rate this:"},
            {"role": "assistant", "content": "hello"},
        ])

    def test_preserves_existing_cache_entries(self):
        """Does not remove pre-existing cache entries."""
        rm = self._make_rm({"c": 3.0})
        cache = {"a": 1.0, "b": 2.0}

        score_items(rm, "prompt", ["c"], cache, label="word")

        assert "a" in cache
        assert "b" in cache
        assert "c" in cache


#################
# SCORE_SAMPLES #
#################


class TestScoreSamples:
    @pytest.fixture(autouse=True)
    def _setup_prompts(self):
        """Set up mock prompt functions for ReDial."""
        _mod.dialecttax.prompts.INSTS = {
            "math": {"naive": {"sae": "Solve the math problem."}},
        }
        _mod.dialecttax.prompts.PROMPTS = {
            "math": {"naive": {"sae": "What is {problem}?"}},
        }
        _mod.dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
            "math": MagicMock(return_value=lambda ds, i: f"Problem: {ds[i]['problem']}"),
        }
        _mod.dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n{body}"

    def test_returns_correct_format(self):
        """Returns list of dicts with unique_id, prompt, response, score."""
        rm = MagicMock(score=MagicMock(return_value=5.0))
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": "4"}]

        results = score_samples(rm, ds, "math", "sae")

        assert len(results) == 1
        assert results[0]["unique_id"] == "q1"
        assert results[0]["response"] == "4"
        assert results[0]["score"] == 5.0
        assert "prompt" in results[0]

    def test_scores_multiple_samples(self):
        """Scores each sample independently."""
        scores = iter([1.0, 2.0, 3.0])
        rm = MagicMock(score=MagicMock(side_effect=lambda _: next(scores)))
        ds = [
            {"unique_id": "q1", "problem": "1+1", "answer": "2"},
            {"unique_id": "q2", "problem": "2+2", "answer": "4"},
            {"unique_id": "q3", "problem": "3+3", "answer": "6"},
        ]

        results = score_samples(rm, ds, "math", "sae")

        assert len(results) == 3
        assert [r["score"] for r in results] == [1.0, 2.0, 3.0]
        assert [r["unique_id"] for r in results] == ["q1", "q2", "q3"]

    def test_response_is_string(self):
        """Converts numeric answers to strings."""
        rm = MagicMock(score=MagicMock(return_value=1.0))
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": 4}]

        results = score_samples(rm, ds, "math", "sae")

        assert results[0]["response"] == "4"

    def test_empty_dataset(self):
        """Returns empty list for empty dataset."""
        rm = MagicMock()

        results = score_samples(rm, [], "math", "sae")

        assert results == []

    def test_calls_rm_score_with_conversation(self):
        """Calls rm.score with user/assistant conversation."""
        rm = MagicMock(score=MagicMock(return_value=1.0))
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": "4"}]

        score_samples(rm, ds, "math", "sae")

        call_args = rm.score.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[0]["role"] == "user"
        assert call_args[1]["role"] == "assistant"
        assert call_args[1]["content"] == "4"


########################
# _BUILD_REDIAL_PROMPT #
########################


class TestBuildRedialPrompt:
    @pytest.fixture(autouse=True)
    def _setup_prompts(self):
        """Set up mock prompt functions."""
        _mod.dialecttax.prompts.INSTS = {
            "math": {"naive": {"sae": "Solve the math problem."}},
        }
        _mod.dialecttax.prompts.PROMPTS = {
            "math": {"naive": {"sae": "What is {problem}?"}},
        }
        _mod.dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
            "math": MagicMock(return_value=lambda ds, i: f"Problem: {ds[i]['problem']}"),
        }
        _mod.dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n{body}"

    def test_basic_prompt(self):
        """Builds a prompt with instructions and body."""
        ds = [{"problem": "2+2"}]

        result = _build_redial_prompt(ds, 0, "math", "naive", "sae")

        assert "Solve the math problem." in result
        assert "Problem: 2+2" in result

    def test_choices_formatting(self):
        """Formats {choices} placeholder in instructions."""
        _mod.dialecttax.prompts.INSTS = {
            "math": {"naive": {"sae": "Pick one:\n{choices}"}},
        }
        ds = [{"problem": "2+2", "choices": {"A": "3", "B": "4", "C": "5"}}]

        result = _build_redial_prompt(ds, 0, "math", "naive", "sae")

        assert "A. 3" in result
        assert "B. 4" in result
        assert "C. 5" in result

    def test_no_choices_placeholder(self):
        """Instructions without {choices} are used as-is."""
        ds = [{"problem": "2+2"}]

        result = _build_redial_prompt(ds, 0, "math", "naive", "sae")

        assert "Solve the math problem." in result
