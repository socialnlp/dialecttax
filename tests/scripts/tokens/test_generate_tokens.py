"""Tests for scripts/tokens/generate_tokens.py."""

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import json

import pytest


###########
# IMPORTS #
###########

# generate_tokens.py has module-level code that imports dialecttax (which
# requires the optional bts package) and runs Hydra setup.  We mock these
# heavy dependencies so the script's pure helper functions can be imported.

_mock_dialecttax = MagicMock()
_mock_dialecttax.utils.load_config.return_value = {
    "directories": {"experiments": "/tmp", "preprocessed": "/tmp", "datasets": "/tmp"},
    "keys": {},
}

_RESULTS_DISPATCH = {
    "bpe": _mock_dialecttax.tokenizers.bpe.bpe_results,
    "unigram": _mock_dialecttax.tokenizers.unigram.unigram_results,
    "wordpiece": _mock_dialecttax.tokenizers.wordpiece.wordpiece_results,
}


def _get_tokenizer_results(tokenizer_type):
    if tokenizer_type not in _RESULTS_DISPATCH:
        raise ValueError(f"Tokenizer type `{tokenizer_type}` is invalid.")
    return _RESULTS_DISPATCH[tokenizer_type]


_mock_dialecttax.tokenizers.get_tokenizer_results = _get_tokenizer_results
_mock_dialecttax.tokenizers.RESULTS_DISPATCH = _RESULTS_DISPATCH

_DATASET_MODULES = {
    "redial": _mock_dialecttax.data.redial,
    "parallelaave": _mock_dialecttax.data.parallelaave,
    "multivalue": _mock_dialecttax.data.multivalue,
}
_mock_dialecttax.data.DATASET_MODULES = _DATASET_MODULES

_MOCK_MODULES = {
    "dialecttax": _mock_dialecttax,
    "dialecttax.utils": _mock_dialecttax.utils,
    "dialecttax.data": _mock_dialecttax.data,
    "dialecttax.data.redial": _mock_dialecttax.data.redial,
    "dialecttax.data.parallelaave": _mock_dialecttax.data.parallelaave,
    "dialecttax.data.multivalue": _mock_dialecttax.data.multivalue,
    "dialecttax.tokenizers": _mock_dialecttax.tokenizers,
    "dialecttax.tokenizers.tokenization": MagicMock(
        TOKENIZER_NAME_TO_TYPE={
            "bpe": "bpe", "gpt2": "bpe", "gemma": "bpe",
            "llama": "bpe", "qwen": "bpe",
            "unigram": "unigram", "wordpiece": "wordpiece",
        },
    ),
    "dialecttax.tokenizers.bpe": _mock_dialecttax.tokenizers.bpe,
    "dialecttax.tokenizers.unigram": _mock_dialecttax.tokenizers.unigram,
    "dialecttax.tokenizers.wordpiece": _mock_dialecttax.tokenizers.wordpiece,
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
        "..", "..", "..", "scripts", "tokens", "generate_tokens.py",
    )
)
_spec = importlib.util.spec_from_file_location("generate_tokens", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _orig

get_tokenizer_results = _get_tokenizer_results
save_results = _mod.save_results
load_dataset = _mod.load_dataset
DATASET_MODULES = _DATASET_MODULES


########################
# GET_TOKENIZER_RESULTS #
########################


class TestGetTokenizerResults:
    def test_bpe(self):
        """Dispatches 'bpe' to bpe_results."""
        fn = get_tokenizer_results("bpe")
        assert fn is _RESULTS_DISPATCH["bpe"]

    def test_unigram(self):
        """Dispatches 'unigram' to unigram_results."""
        fn = get_tokenizer_results("unigram")
        assert fn is _RESULTS_DISPATCH["unigram"]

    def test_wordpiece(self):
        """Dispatches 'wordpiece' to wordpiece_results."""
        fn = get_tokenizer_results("wordpiece")
        assert fn is _RESULTS_DISPATCH["wordpiece"]

    def test_invalid_type_raises(self):
        """Raises ValueError for an unrecognized tokenizer type."""
        with pytest.raises(ValueError, match="invalid"):
            get_tokenizer_results("gpt2")


################
# SAVE_RESULTS #
################


class TestSaveResults:
    def test_writes_valid_jsonl(self, tmp_path):
        """Each line is a valid JSON object."""
        data = [
            {"unique_id": "a", "n_tokens": 1},
            {"unique_id": "b", "n_tokens": 2},
        ]
        path = os.path.join(str(tmp_path), "tokens.jsonl")

        save_results(data, path)

        with open(path) as f:
            loaded = [json.loads(line) for line in f]
        assert len(loaded) == 2
        assert loaded[0]["unique_id"] == "a"
        assert loaded[1]["n_tokens"] == 2

    def test_creates_parent_dirs(self, tmp_path):
        """Intermediate directories are created if missing."""
        path = os.path.join(str(tmp_path), "a", "b", "c", "tokens.jsonl")

        save_results([{"x": 1}], path)

        assert os.path.isfile(path)

    def test_empty_list(self, tmp_path):
        """Empty list produces an empty file."""
        path = os.path.join(str(tmp_path), "tokens.jsonl")

        save_results([], path)

        with open(path) as f:
            assert f.read().strip() == ""

    def test_roundtrip(self, tmp_path):
        """Results survive a save-then-load roundtrip."""
        data = [
            {"unique_id": "r0", "n_tokens": 2},
            {"unique_id": "r1", "n_tokens": 0},
            {"unique_id": "r2", "n_tokens": 1},
        ]
        path = os.path.join(str(tmp_path), "tokens.jsonl")

        save_results(data, path)

        with open(path) as f:
            loaded = [json.loads(line) for line in f]
        assert loaded == data

    def test_overwrites_existing_file(self, tmp_path):
        """Writing to an existing file replaces its contents."""
        path = os.path.join(str(tmp_path), "tokens.jsonl")
        save_results([{"old": True}], path)

        save_results([{"new": True}], path)

        with open(path) as f:
            loaded = [json.loads(line) for line in f]
        assert len(loaded) == 1
        assert loaded[0]["new"] is True

    def test_many_results(self, tmp_path):
        """Handles a large batch of results."""
        data = [{"i": i, "tok": f"t{i}"} for i in range(1000)]
        path = os.path.join(str(tmp_path), "tokens.jsonl")

        save_results(data, path)

        with open(path) as f:
            loaded = [json.loads(line) for line in f]
        assert len(loaded) == 1000
        assert loaded[0]["i"] == 0
        assert loaded[999]["i"] == 999

    def test_nan_written_as_null(self, tmp_path):
        """NaN float values are written as JSON null."""
        data = [{"n_tokens": float("nan"), "unique_id": "a"}]
        path = os.path.join(str(tmp_path), "tokens.jsonl")

        save_results(data, path)

        with open(path) as f:
            loaded = [json.loads(line) for line in f]
        assert loaded[0]["n_tokens"] is None


####################
# DATASET_MODULES  #
####################


class TestDatasetModules:
    def test_has_redial(self):
        """DATASET_MODULES contains 'redial' key."""
        assert "redial" in DATASET_MODULES

    def test_has_parallelaave(self):
        """DATASET_MODULES contains 'parallelaave' key."""
        assert "parallelaave" in DATASET_MODULES

    def test_has_multivalue(self):
        """DATASET_MODULES contains 'multivalue' key."""
        assert "multivalue" in DATASET_MODULES

    def test_exactly_three_entries(self):
        """DATASET_MODULES has exactly three entries."""
        assert len(DATASET_MODULES) == 3


################
# LOAD_DATASET #
################


class TestLoadDataset:
    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Point the mocked data module constants at real values."""
        redial = _mod.dialecttax.data.redial
        redial.FILE_NAME_FORMAT = "{task}_{dialect}.jsonl"
        redial.DIRECTORY_NAME = "ReDial"
        redial.TASKS = ["algorithm", "logic", "math", "planning"]
        redial.load_dataset.reset_mock()

        parallelaave = _mod.dialecttax.data.parallelaave
        parallelaave.FILE_NAME_FORMAT = "{dialect}_samples.txt"
        parallelaave.DIRECTORY_NAME = "parallelaave"
        parallelaave.load_dataset.reset_mock()
        # Ensure no TASKS attribute for non-task datasets
        if hasattr(parallelaave, "TASKS"):
            del parallelaave.TASKS

        multivalue = _mod.dialecttax.data.multivalue
        multivalue.FILE_NAME_FORMAT = "coqa_{dialect}.txt"
        multivalue.DIRECTORY_NAME = "multivalue"
        multivalue.load_dataset.reset_mock()
        if hasattr(multivalue, "TASKS"):
            del multivalue.TASKS

    # --- ReDial ---

    def test_redial_missing_file_returns_none(self, tmp_path):
        """Returns None when the ReDial dataset file does not exist."""
        result = load_dataset("redial", str(tmp_path), "math", "sae")
        assert result is None

    def test_redial_constructs_correct_path(self, tmp_path):
        """Calls redial.load_dataset with the correct relative path."""
        redial_dir = tmp_path / "ReDial"
        redial_dir.mkdir()
        (redial_dir / "logic_aave.jsonl").write_text("")

        redial_mock = _mod.dialecttax.data.redial
        redial_mock.load_dataset.return_value = []

        load_dataset("redial", str(tmp_path), "logic", "aave")

        redial_mock.load_dataset.assert_called_once_with(
            str(tmp_path), os.path.join("ReDial", "logic_aave.jsonl")
        )

    def test_redial_returns_loaded_data(self, tmp_path):
        """Returns the list of dicts from redial.load_dataset."""
        redial_dir = tmp_path / "ReDial"
        redial_dir.mkdir()
        (redial_dir / "math_sae.jsonl").write_text("")

        expected = [{"unique_id": "a"}, {"unique_id": "b"}]
        _mod.dialecttax.data.redial.load_dataset.return_value = expected

        result = load_dataset("redial", str(tmp_path), "math", "sae")
        assert result == expected

    # --- ParallelAAVE ---

    def test_parallelaave_missing_file_returns_none(self, tmp_path):
        """Returns None when the ParallelAAVE dataset file does not exist."""
        result = load_dataset("parallelaave", str(tmp_path), "math", "sae")
        assert result is None

    def test_parallelaave_constructs_correct_path(self, tmp_path):
        """Calls parallelaave.load_dataset with the correct relative path."""
        data_dir = tmp_path / "parallelaave"
        data_dir.mkdir()
        (data_dir / "aave_samples.txt").write_text("")

        pa_mock = _mod.dialecttax.data.parallelaave
        pa_mock.load_dataset.return_value = []

        load_dataset("parallelaave", str(tmp_path), "math", "aave")

        pa_mock.load_dataset.assert_called_once_with(
            str(tmp_path), os.path.join("parallelaave", "aave_samples.txt")
        )

    def test_parallelaave_ignores_task(self, tmp_path):
        """Task parameter does not affect the file path for ParallelAAVE."""
        data_dir = tmp_path / "parallelaave"
        data_dir.mkdir()
        (data_dir / "sae_samples.txt").write_text("")

        pa_mock = _mod.dialecttax.data.parallelaave
        pa_mock.load_dataset.return_value = []

        load_dataset("parallelaave", str(tmp_path), "logic", "sae")

        pa_mock.load_dataset.assert_called_once_with(
            str(tmp_path), os.path.join("parallelaave", "sae_samples.txt")
        )

    # --- MultiValue ---

    def test_multivalue_missing_file_returns_none(self, tmp_path):
        """Returns None when the MultiValue dataset file does not exist."""
        result = load_dataset("multivalue", str(tmp_path), "math", "sae")
        assert result is None

    def test_multivalue_constructs_correct_path(self, tmp_path):
        """Calls multivalue.load_dataset with the correct relative path."""
        data_dir = tmp_path / "multivalue"
        data_dir.mkdir()
        (data_dir / "coqa_chicano.txt").write_text("")

        mv_mock = _mod.dialecttax.data.multivalue
        mv_mock.load_dataset.return_value = []

        load_dataset("multivalue", str(tmp_path), "math", "chicano")

        mv_mock.load_dataset.assert_called_once_with(
            str(tmp_path), os.path.join("multivalue", "coqa_chicano.txt")
        )

    def test_multivalue_ignores_task(self, tmp_path):
        """Task parameter does not affect the file path for MultiValue."""
        data_dir = tmp_path / "multivalue"
        data_dir.mkdir()
        (data_dir / "coqa_indian.txt").write_text("")

        mv_mock = _mod.dialecttax.data.multivalue
        mv_mock.load_dataset.return_value = []

        load_dataset("multivalue", str(tmp_path), "algorithm", "indian")

        mv_mock.load_dataset.assert_called_once_with(
            str(tmp_path), os.path.join("multivalue", "coqa_indian.txt")
        )
