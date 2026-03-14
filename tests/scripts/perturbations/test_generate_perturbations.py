"""Tests for scripts/perturbations/generate_perturbations.py."""

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf


###########
# IMPORTS #
###########

# generate_perturbations.py has module-level code that imports dialecttax (which
# requires optional heavy packages) and runs Hydra setup.  We mock these
# dependencies so the script's functions can be imported and tested.
# omegaconf is kept real so we can pass proper DictConfig objects to main().

_mock_dialecttax = MagicMock()
_mock_dialecttax.utils.load_config.return_value = {
    "directories": {"experiments": "/tmp", "preprocessed": "/tmp", "datasets": "/tmp"},
    "keys": {},
}

# Load real perturbation functions for integration testing
_perturb_spec = importlib.util.spec_from_file_location(
    "dialecttax.perturbations",
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "src", "dialecttax", "perturbations.py"
    ),
)
_perturb_mod = importlib.util.module_from_spec(_perturb_spec)
_perturb_spec.loader.exec_module(_perturb_mod)
_mock_dialecttax.perturbations = _perturb_mod

_DATASET_MODULES = {
    "redial": _mock_dialecttax.data.redial,
    "parallelaave": _mock_dialecttax.data.parallelaave,
    "multivalue": _mock_dialecttax.data.multivalue,
}
_mock_dialecttax.data.DATASET_MODULES = _DATASET_MODULES

# Make @hydra.main a pass-through decorator so main() is directly callable
_mock_hydra = MagicMock()
_mock_hydra.main.return_value = lambda fn: fn

_mock_hydra_config_module = MagicMock()

_MOCK_MODULES = {
    "dialecttax": _mock_dialecttax,
    "dialecttax.utils": _mock_dialecttax.utils,
    "dialecttax.data": _mock_dialecttax.data,
    "dialecttax.data.redial": _mock_dialecttax.data.redial,
    "dialecttax.data.parallelaave": _mock_dialecttax.data.parallelaave,
    "dialecttax.data.multivalue": _mock_dialecttax.data.multivalue,
    "dialecttax.perturbations": _perturb_mod,
    "hydra": _mock_hydra,
    "hydra.core": MagicMock(),
    "hydra.core.hydra_config": _mock_hydra_config_module,
}

_saved_modules = {}
for _name, _mock in _MOCK_MODULES.items():
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "scripts", "perturbations", "generate_perturbations.py",
    )
)
_spec = importlib.util.spec_from_file_location("generate_perturbations", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _orig

load_dataset = _mod.load_dataset
main = _mod.main
DATASET_MODULES = _DATASET_MODULES


################
# LOAD_DATASET #
################


class TestLoadDataset:
    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Configure mocked data module attributes."""
        redial = _mod.dialecttax.data.redial
        redial.FILE_NAME_FORMAT = "{task}_{dialect}.jsonl"
        redial.DIRECTORY_NAME = "ReDial"
        redial.TASKS = ["algorithm", "logic", "math", "planning"]
        redial.load_dataset.reset_mock()

        parallelaave = _mod.dialecttax.data.parallelaave
        parallelaave.FILE_NAME_FORMAT = "{dialect}_samples.txt"
        parallelaave.DIRECTORY_NAME = "parallelaave"
        parallelaave.load_dataset.reset_mock()
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
        """Returns None when the dataset file does not exist."""
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
        """Returns None when the dataset file does not exist."""
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
        """Task parameter does not affect the file path."""
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


########
# MAIN #
########


def _make_cfg(
    perturbation_fn="swap",
    perturbation_name="swap-0.0",
    perturbation_kwargs=None,
    dataset_name="redial",
    dir_key="preprocessed",
    output_subdir="math/sae",
    dialects=None,
    tasks=None,
    task_name="math",
    dialect_name="sae",
    row_text="problem",
    rerun=False,
    seed=42,
):
    """Build a DictConfig mimicking the Hydra-resolved config."""
    cfg = {
        "dataset": {
            "name": dataset_name,
            "dir_key": dir_key,
            "dialects": dialects or ["sae", "aave"],
            "output_subdir": output_subdir,
        },
        "task": {
            "name": task_name,
            "row_text": row_text,
        },
        "dialect": {"name": dialect_name},
        "perturbation": {
            "name": perturbation_name,
            "fn": perturbation_fn,
            "kwargs": perturbation_kwargs if perturbation_kwargs is not None else {"p": 0.0},
        },
        "rerun": rerun,
        "seed": seed,
    }
    if tasks is not None:
        cfg["dataset"]["tasks"] = tasks
    return OmegaConf.create(cfg)


class TestMain:
    @pytest.fixture(autouse=True)
    def _setup_project_config(self, tmp_path):
        """Point preprocessed directory at tmp_path."""
        self.preprocessed_dir = str(tmp_path)
        self._orig_dirs = dict(_mod._project_config["directories"])
        _mod._project_config["directories"]["preprocessed"] = self.preprocessed_dir
        yield
        _mod._project_config["directories"].update(self._orig_dirs)

    def _out_path(self, perturbation_name="swap-0.0", dataset_name="redial", output_subdir="math/sae"):
        return os.path.join(self.preprocessed_dir, "perturbations", perturbation_name, dataset_name, f"{output_subdir}.jsonl")

    def _read_output(self, **kwargs):
        with open(self._out_path(**kwargs)) as f:
            return [json.loads(line) for line in f]

    # --- save ---

    def test_saves_jsonl(self):
        """Writes perturbed texts as JSONL."""
        cfg = _make_cfg()
        with patch.object(_mod, "load_dataset", return_value=["hello", "world"]):
            main(cfg)

        assert os.path.isfile(self._out_path())
        assert self._read_output() == ["hello", "world"]

    def test_output_is_valid_jsonl(self):
        """Output file contains one JSON string per line."""
        cfg = _make_cfg(perturbation_fn="drop", perturbation_name="drop-0.0")
        with patch.object(_mod, "load_dataset", return_value=["a", "b", "c"]):
            main(cfg)

        data = self._read_output(perturbation_name="drop-0.0")
        assert isinstance(data, list)
        assert all(isinstance(s, str) for s in data)
        assert len(data) == 3

    # --- rerun ---

    def test_skips_existing_results(self):
        """Does not overwrite when rerun=false and output exists."""
        out_path = self._out_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(json.dumps("original") + "\n")

        cfg = _make_cfg(rerun=False)
        with patch.object(_mod, "load_dataset") as mock_ld:
            main(cfg)
            mock_ld.assert_not_called()

        assert self._read_output() == ["original"]

    def test_rerun_overwrites_existing(self):
        """Regenerates output when rerun=true."""
        out_path = self._out_path()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            f.write(json.dumps("original") + "\n")

        cfg = _make_cfg(rerun=True)
        with patch.object(_mod, "load_dataset", return_value=["new"]):
            main(cfg)

        assert self._read_output() == ["new"]

    # --- skip invalid combos ---

    def test_skips_invalid_dialect(self):
        """Returns early when dialect is not in dataset.dialects."""
        cfg = _make_cfg(dialect_name="chicano", dialects=["sae", "aave"])
        with patch.object(_mod, "load_dataset") as mock_ld:
            main(cfg)
            mock_ld.assert_not_called()

        assert not os.path.exists(self._out_path())

    def test_skips_invalid_task(self):
        """Returns early when task is not in dataset.tasks."""
        cfg = _make_cfg(task_name="planning", tasks=["math", "algorithm"])
        with patch.object(_mod, "load_dataset") as mock_ld:
            main(cfg)
            mock_ld.assert_not_called()

        assert not os.path.exists(self._out_path())

    def test_no_task_check_without_tasks_key(self):
        """Proceeds when dataset config has no tasks key."""
        cfg = _make_cfg(task_name="planning", tasks=None)
        with patch.object(_mod, "load_dataset", return_value=["text"]) as mock_ld:
            main(cfg)
            mock_ld.assert_called_once()

    # --- missing dataset ---

    def test_returns_on_missing_dataset(self):
        """Returns early when load_dataset returns None."""
        cfg = _make_cfg()
        with patch.object(_mod, "load_dataset", return_value=None):
            main(cfg)

        assert not os.path.exists(self._out_path())

    # --- text extraction ---

    def test_extracts_text_from_dicts(self):
        """Extracts the row_text field when dataset returns dicts."""
        cfg = _make_cfg(row_text="problem")
        dataset = [
            {"problem": "What is 2+2?", "answer": "4"},
            {"problem": "What is 3+3?", "answer": "6"},
        ]
        with patch.object(_mod, "load_dataset", return_value=dataset):
            main(cfg)

        assert self._read_output() == ["What is 2+2?", "What is 3+3?"]

    def test_extracts_text_from_strings(self):
        """Uses strings directly when dataset returns strings."""
        cfg = _make_cfg()
        with patch.object(_mod, "load_dataset", return_value=["hello", "world"]):
            main(cfg)

        assert self._read_output() == ["hello", "world"]

    def test_uses_dataset_row_text_fallback(self):
        """Falls back to dataset.row_text when task config has no row_text."""
        cfg = OmegaConf.create({
            "dataset": {
                "name": "redial",
                "dir_key": "preprocessed",
                "dialects": ["sae", "aave"],
                "output_subdir": "math/sae",
                "row_text": "text",
            },
            "task": {"name": "math"},
            "dialect": {"name": "sae"},
            "perturbation": {"name": "swap-0.0", "fn": "swap", "kwargs": {"p": 0.0}},
            "rerun": True,
            "seed": 42,
        })
        dataset = [{"text": "hello", "other": "x"}, {"text": "world", "other": "y"}]
        with patch.object(_mod, "load_dataset", return_value=dataset):
            main(cfg)

        assert self._read_output() == ["hello", "world"]

    # --- perturbation dispatch ---

    def test_perturbation_kwargs_forwarded(self):
        """Perturbation kwargs from config are passed to the function."""
        cfg = _make_cfg(
            perturbation_fn="capitalize",
            perturbation_name="capitalize-alternating",
            perturbation_kwargs={"mode": "alternating"},
        )
        with patch.object(_mod, "load_dataset", return_value=["abcd"]):
            main(cfg)

        assert self._read_output(perturbation_name="capitalize-alternating") == ["aBcD"]

    def test_capitalize_random(self):
        """Capitalize random mode runs without error."""
        cfg = _make_cfg(
            perturbation_fn="capitalize",
            perturbation_name="capitalize-random",
            perturbation_kwargs={"mode": "random"},
        )
        with patch.object(_mod, "load_dataset", return_value=["abcd"]):
            main(cfg)

        data = self._read_output(perturbation_name="capitalize-random")
        assert len(data) == 1
        assert len(data[0]) == 4

    def test_drop_at_full_probability(self):
        """Drop at p=1.0 removes all non-space characters."""
        cfg = _make_cfg(perturbation_fn="drop", perturbation_name="drop-1.0", perturbation_kwargs={"p": 1.0})
        with patch.object(_mod, "load_dataset", return_value=["hello world"]):
            main(cfg)

        assert self._read_output(perturbation_name="drop-1.0") == [" "]

    def test_insert_at_zero_probability(self):
        """Insert at p=0.0 leaves text unchanged."""
        cfg = _make_cfg(perturbation_fn="insert", perturbation_name="insert-0.0", perturbation_kwargs={"p": 0.0})
        with patch.object(_mod, "load_dataset", return_value=["abc def"]):
            main(cfg)

        assert self._read_output(perturbation_name="insert-0.0") == ["abc def"]

    # --- seeding ---

    def test_seed_reproducibility(self):
        """Same seed produces identical perturbed output."""
        cfg = _make_cfg(
            seed=123,
            perturbation_fn="swap",
            perturbation_kwargs={"p": 0.3},
            rerun=True,
        )
        texts = ["the quick brown fox jumps over the lazy dog"] * 10

        with patch.object(_mod, "load_dataset", return_value=texts):
            main(cfg)
        run1 = self._read_output()

        with patch.object(_mod, "load_dataset", return_value=texts):
            main(cfg)
        run2 = self._read_output()

        assert run1 == run2

    def test_different_seeds_differ(self):
        """Different seeds produce different perturbed output."""
        texts = ["the quick brown fox jumps over the lazy dog"] * 20

        cfg1 = _make_cfg(seed=1, perturbation_fn="swap", perturbation_kwargs={"p": 0.5}, rerun=True)
        with patch.object(_mod, "load_dataset", return_value=texts):
            main(cfg1)
        run1 = self._read_output()

        cfg2 = _make_cfg(seed=2, perturbation_fn="swap", perturbation_kwargs={"p": 0.5}, rerun=True)
        with patch.object(_mod, "load_dataset", return_value=texts):
            main(cfg2)
        run2 = self._read_output()

        assert run1 != run2
