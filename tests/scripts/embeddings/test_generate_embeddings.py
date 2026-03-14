"""Tests for scripts/embeddings/generate_embeddings.py."""

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from omegaconf import OmegaConf


###########
# IMPORTS #
###########

# generate_embeddings.py has module-level code that imports dialecttax (which
# requires optional heavy packages) and runs Hydra setup.  We mock these
# dependencies so the script's functions can be imported and tested.
# omegaconf is kept real so we can pass proper DictConfig objects to main().

_mock_dialecttax = MagicMock()
_mock_dialecttax.utils.load_config.return_value = {
    "directories": {
        "experiments": "/tmp",
        "preprocessed": "/tmp/preprocessed",
        "datasets": "/tmp/datasets",
    },
    "keys": {},
}
_mock_dialecttax.perturbations.DIRECTORY_NAME = "perturbations"

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
    "dialecttax.perturbations": _mock_dialecttax.perturbations,
    "dialecttax.embeddings": _mock_dialecttax.embeddings,
    "hydra": _mock_hydra,
    "hydra.core": MagicMock(),
    "hydra.core.hydra_config": _mock_hydra_config_module,
    "torch": MagicMock(),
}

_saved_modules = {}
for _name, _mock in _MOCK_MODULES.items():
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "scripts", "embeddings", "generate_embeddings.py",
    )
)
_spec = importlib.util.spec_from_file_location("generate_embeddings", _script_path)
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

    # --- no perturbation: ReDial ---

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

    # --- no perturbation: ParallelAAVE ---

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

    # --- no perturbation: MultiValue ---

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

    # --- with perturbation: task dataset (ReDial) ---

    def test_perturbation_redial_constructs_correct_path(self, tmp_path):
        """Perturbation path includes task subdirectory for task datasets."""
        perturb_dir = tmp_path / "perturbations" / "swap-0.05" / "redial" / "math"
        perturb_dir.mkdir(parents=True)
        (perturb_dir / "sae.jsonl").write_text(json.dumps("hello") + "\n")

        result = load_dataset("redial", str(tmp_path), "math", "sae", perturbation_name="swap-0.05")
        assert result == ["hello"]

    def test_perturbation_redial_missing_file_returns_none(self, tmp_path):
        """Returns None when the perturbed dataset file does not exist."""
        result = load_dataset("redial", str(tmp_path), "math", "sae", perturbation_name="swap-0.05")
        assert result is None

    # --- with perturbation: taskless dataset (ParallelAAVE) ---

    def test_perturbation_parallelaave_constructs_correct_path(self, tmp_path):
        """Perturbation path has no task subdirectory for taskless datasets."""
        perturb_dir = tmp_path / "perturbations" / "drop-0.15" / "parallelaave"
        perturb_dir.mkdir(parents=True)
        (perturb_dir / "aave.jsonl").write_text(
            json.dumps("text one") + "\n" + json.dumps("text two") + "\n"
        )

        result = load_dataset("parallelaave", str(tmp_path), "math", "aave", perturbation_name="drop-0.15")
        assert result == ["text one", "text two"]

    def test_perturbation_parallelaave_missing_file_returns_none(self, tmp_path):
        """Returns None when the perturbed dataset file does not exist."""
        result = load_dataset("parallelaave", str(tmp_path), "math", "aave", perturbation_name="drop-0.15")
        assert result is None

    # --- with perturbation: translate ---

    def test_perturbation_translate_path(self, tmp_path):
        """Translate perturbation uses language-specific name in path."""
        perturb_dir = tmp_path / "perturbations" / "translate-chinese" / "redial" / "math"
        perturb_dir.mkdir(parents=True)
        (perturb_dir / "sae.jsonl").write_text(json.dumps("translated") + "\n")

        result = load_dataset("redial", str(tmp_path), "math", "sae", perturbation_name="translate-chinese")
        assert result == ["translated"]

    # --- with perturbation: does not call mod.load_dataset ---

    def test_perturbation_does_not_call_module_load_dataset(self, tmp_path):
        """Perturbation branch reads JSONL directly, not via mod.load_dataset."""
        perturb_dir = tmp_path / "perturbations" / "swap-0.05" / "redial" / "math"
        perturb_dir.mkdir(parents=True)
        (perturb_dir / "sae.jsonl").write_text(json.dumps("hello") + "\n")

        redial_mock = _mod.dialecttax.data.redial
        load_dataset("redial", str(tmp_path), "math", "sae", perturbation_name="swap-0.05")
        redial_mock.load_dataset.assert_not_called()


########
# MAIN #
########


def _make_cfg(
    dataset_name="parallelaave",
    dir_key="datasets",
    output_subdir="sae",
    dialects=None,
    tasks=None,
    task_name="math",
    dialect_name="sae",
    row_text="text",
    perturbation_name=None,
    dim=768,
    batch_size=32,
    rerun=False,
):
    """Build a DictConfig mimicking the Hydra-resolved config."""
    cfg = {
        "dataset": {
            "name": dataset_name,
            "dir_key": dir_key,
            "dialects": dialects or ["sae", "aave"],
            "output_subdir": output_subdir,
            "row_text": row_text,
        },
        "task": {
            "name": task_name,
            "row_text": row_text,
        },
        "dialect": {"name": dialect_name},
        "dim": dim,
        "batch_size": batch_size,
        "rerun": rerun,
    }
    if perturbation_name is not None:
        cfg["perturbation"] = {"name": perturbation_name}
    if tasks is not None:
        cfg["dataset"]["tasks"] = tasks
    return OmegaConf.create(cfg)


def _fake_encode(model, texts, dim=768, batch_size=256):
    """Return a deterministic float32 array of shape (len(texts), dim)."""
    return np.arange(len(texts) * dim, dtype=np.float32).reshape(len(texts), dim)


class TestMain:
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Point project dirs at tmp_path and mock GPU/model/HydraConfig."""
        self.out_dir = str(tmp_path / "output")
        self.preprocessed_dir = str(tmp_path / "preprocessed")
        self.datasets_dir = str(tmp_path / "datasets")

        self._orig_dirs = dict(_mod._project_config["directories"])
        _mod._project_config["directories"]["preprocessed"] = self.preprocessed_dir
        _mod._project_config["directories"]["datasets"] = self.datasets_dir

        # Mock HydraConfig to return our output directory
        _mock_hydra_config_module.HydraConfig.get.return_value.runtime.output_dir = self.out_dir

        # Mock torch.cuda to report no GPUs (forces device="cpu")
        _mod.torch.cuda.device_count.return_value = 0

        # Mock model loading and encoding
        _mod.dialecttax.embeddings.load_embedding_gemma.reset_mock()
        _mod.dialecttax.embeddings.load_embedding_gemma.return_value = MagicMock()
        _mod.dialecttax.embeddings.encode.reset_mock()
        _mod.dialecttax.embeddings.encode.side_effect = _fake_encode

        yield
        _mod._project_config["directories"].update(self._orig_dirs)

    def _out_path(self, dim=768, perturbation_name=None):
        if perturbation_name is not None:
            return os.path.join(self.out_dir, perturbation_name, f"embeddings-{dim}.npy")
        return os.path.join(self.out_dir, f"embeddings-{dim}.npy")

    # --- save ---

    def test_saves_npy(self):
        """Writes embeddings as .npy file."""
        cfg = _make_cfg()
        with patch.object(_mod, "load_dataset", return_value=["hello", "world"]):
            main(cfg)

        out = self._out_path()
        assert os.path.isfile(out)
        arr = np.load(out)
        assert arr.shape == (2, 768)

    def test_saves_correct_dim(self):
        """Output file name includes the dim."""
        cfg = _make_cfg(dim=256)
        with patch.object(_mod, "load_dataset", return_value=["a"]):
            main(cfg)

        assert os.path.isfile(self._out_path(dim=256))
        assert not os.path.exists(self._out_path(dim=768))

    def test_different_dims_do_not_overwrite(self):
        """Different dim values produce separate files."""
        with patch.object(_mod, "load_dataset", return_value=["a"]):
            main(_make_cfg(dim=768))
            main(_make_cfg(dim=256))

        assert os.path.isfile(self._out_path(dim=768))
        assert os.path.isfile(self._out_path(dim=256))

    # --- rerun ---

    def test_skips_existing_results(self):
        """Does not overwrite when rerun=false and output exists."""
        out = self._out_path()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        original = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.save(out, original)

        cfg = _make_cfg(rerun=False, dim=768)
        with patch.object(_mod, "load_dataset") as mock_ld:
            main(cfg)
            mock_ld.assert_not_called()

        np.testing.assert_array_equal(np.load(out), original)

    def test_rerun_overwrites_existing(self):
        """Regenerates output when rerun=true."""
        out = self._out_path()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, np.zeros((1, 768), dtype=np.float32))

        cfg = _make_cfg(rerun=True)
        with patch.object(_mod, "load_dataset", return_value=["new text"]):
            main(cfg)

        arr = np.load(out)
        assert arr.shape == (1, 768)
        assert not np.all(arr == 0)

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

        _mod.dialecttax.embeddings.encode.assert_called_once()
        texts_arg = _mod.dialecttax.embeddings.encode.call_args[0][1]
        assert texts_arg == ["What is 2+2?", "What is 3+3?"]

    def test_extracts_text_from_strings(self):
        """Uses strings directly when dataset returns strings."""
        cfg = _make_cfg()
        with patch.object(_mod, "load_dataset", return_value=["hello", "world"]):
            main(cfg)

        _mod.dialecttax.embeddings.encode.assert_called_once()
        texts_arg = _mod.dialecttax.embeddings.encode.call_args[0][1]
        assert texts_arg == ["hello", "world"]

    def test_uses_dataset_row_text_fallback(self):
        """Falls back to dataset.row_text when task config has no row_text."""
        cfg = OmegaConf.create({
            "dataset": {
                "name": "parallelaave",
                "dir_key": "datasets",
                "dialects": ["sae", "aave"],
                "output_subdir": "sae",
                "row_text": "text",
            },
            "task": {"name": "math"},
            "dialect": {"name": "sae"},
            "dim": 768,
            "batch_size": 32,
            "rerun": True,
        })
        dataset = [{"text": "hello", "other": "x"}]
        with patch.object(_mod, "load_dataset", return_value=dataset):
            main(cfg)

        texts_arg = _mod.dialecttax.embeddings.encode.call_args[0][1]
        assert texts_arg == ["hello"]

    # --- encode args ---

    def test_passes_dim_and_batch_size_to_encode(self):
        """dim and batch_size from config are forwarded to encode()."""
        cfg = _make_cfg(dim=512, batch_size=64)
        with patch.object(_mod, "load_dataset", return_value=["a"]):
            main(cfg)

        kwargs = _mod.dialecttax.embeddings.encode.call_args[1]
        assert kwargs["dim"] == 512
        assert kwargs["batch_size"] == 64

    # --- perturbation: output path ---

    def test_perturbation_saves_to_subdir(self):
        """With perturbation, saves to out_dir/perturbation_name/embeddings-{dim}.npy."""
        cfg = _make_cfg(perturbation_name="swap-0.05")
        with patch.object(_mod, "load_dataset", return_value=["hello"]):
            main(cfg)

        expected = self._out_path(perturbation_name="swap-0.05")
        assert os.path.isfile(expected)

    def test_perturbation_does_not_save_to_base_dir(self):
        """With perturbation, does not write embeddings-{dim}.npy at base out_dir."""
        cfg = _make_cfg(perturbation_name="drop-0.15")
        with patch.object(_mod, "load_dataset", return_value=["hello"]):
            main(cfg)

        base = self._out_path()
        assert not os.path.exists(base)

    def test_no_perturbation_saves_to_base_dir(self):
        """Without perturbation, saves embeddings-{dim}.npy at base out_dir."""
        cfg = _make_cfg()
        with patch.object(_mod, "load_dataset", return_value=["hello"]):
            main(cfg)

        assert os.path.isfile(self._out_path())

    def test_perturbation_different_dims_do_not_overwrite(self):
        """Different dims with same perturbation produce separate files."""
        with patch.object(_mod, "load_dataset", return_value=["a"]):
            main(_make_cfg(perturbation_name="swap-0.05", dim=768))
            main(_make_cfg(perturbation_name="swap-0.05", dim=256))

        assert os.path.isfile(self._out_path(dim=768, perturbation_name="swap-0.05"))
        assert os.path.isfile(self._out_path(dim=256, perturbation_name="swap-0.05"))

    def test_different_perturbations_do_not_overwrite(self):
        """Different perturbation names produce separate directories."""
        with patch.object(_mod, "load_dataset", return_value=["a"]):
            main(_make_cfg(perturbation_name="swap-0.05"))
            main(_make_cfg(perturbation_name="drop-0.15"))

        assert os.path.isfile(self._out_path(perturbation_name="swap-0.05"))
        assert os.path.isfile(self._out_path(perturbation_name="drop-0.15"))

    def test_perturbation_and_base_do_not_overwrite(self):
        """Perturbation and non-perturbation runs produce separate files."""
        with patch.object(_mod, "load_dataset", return_value=["a"]):
            main(_make_cfg())
            main(_make_cfg(perturbation_name="swap-0.05"))

        assert os.path.isfile(self._out_path())
        assert os.path.isfile(self._out_path(perturbation_name="swap-0.05"))

    # --- perturbation: skip existing ---

    def test_perturbation_skips_existing_results(self):
        """Does not overwrite perturbed embeddings when rerun=false."""
        out = self._out_path(perturbation_name="swap-0.05")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        original = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        np.save(out, original)

        cfg = _make_cfg(perturbation_name="swap-0.05", rerun=False)
        with patch.object(_mod, "load_dataset") as mock_ld:
            main(cfg)
            mock_ld.assert_not_called()

        np.testing.assert_array_equal(np.load(out), original)

    def test_perturbation_rerun_overwrites(self):
        """Regenerates perturbed embeddings when rerun=true."""
        out = self._out_path(perturbation_name="swap-0.05")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out, np.zeros((1, 768), dtype=np.float32))

        cfg = _make_cfg(perturbation_name="swap-0.05", rerun=True)
        with patch.object(_mod, "load_dataset", return_value=["new"]):
            main(cfg)

        arr = np.load(out)
        assert not np.all(arr == 0)

    # --- perturbation: dir_root ---

    def test_perturbation_uses_preprocessed_dir(self):
        """With perturbation, load_dataset is called with dir_preprocessed."""
        cfg = _make_cfg(perturbation_name="swap-0.05")
        with patch.object(_mod, "load_dataset", return_value=["text"]) as mock_ld:
            main(cfg)

        call_args = mock_ld.call_args
        dir_root_arg = call_args[0][1]
        assert dir_root_arg == self.preprocessed_dir

    def test_no_perturbation_uses_dataset_dir(self):
        """Without perturbation, load_dataset is called with dir_root from config."""
        cfg = _make_cfg()
        with patch.object(_mod, "load_dataset", return_value=["text"]) as mock_ld:
            main(cfg)

        call_args = mock_ld.call_args
        dir_root_arg = call_args[0][1]
        assert dir_root_arg == self.datasets_dir

    def test_perturbation_name_forwarded_to_load_dataset(self):
        """perturbation_name kwarg is passed through to load_dataset."""
        cfg = _make_cfg(perturbation_name="translate-chinese")
        with patch.object(_mod, "load_dataset", return_value=["text"]) as mock_ld:
            main(cfg)

        assert mock_ld.call_args[1]["perturbation_name"] == "translate-chinese"

    def test_no_perturbation_name_not_forwarded(self):
        """Without perturbation, perturbation_name is not passed to load_dataset."""
        cfg = _make_cfg()
        with patch.object(_mod, "load_dataset", return_value=["text"]) as mock_ld:
            main(cfg)

        assert "perturbation_name" not in mock_ld.call_args[1]
