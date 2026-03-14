"""Tests for scripts/gradients/generate_gradients.py."""

import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


###########
# IMPORTS #
###########

# generate_gradients.py has module-level code that imports dialecttax and runs
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

# Mock prompt constants
_mock_dialecttax.prompts.INSTS = {
    "math": {"naive": {"sae": "Solve the math problem.", "aave": "Solve the math problem."}},
}
_mock_dialecttax.prompts.PROMPTS = {
    "math": {"naive": {"sae": "Question: {question}\n", "aave": "Question: {question}\n"}},
}
_mock_dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
    "math": MagicMock(return_value=lambda ds, i: f"Question: {ds[i]['problem']}\n"),
}
_mock_dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n\n{body}"

# Mock dialecttax.gradients
_mock_gradients = MagicMock()
_mock_dialecttax.gradients = _mock_gradients

_MOCK_MODULES = {
    "dialecttax": _mock_dialecttax,
    "dialecttax.utils": _mock_dialecttax.utils,
    "dialecttax.data": _mock_dialecttax.data,
    "dialecttax.prompts": _mock_dialecttax.prompts,
    "dialecttax.gradients": _mock_gradients,
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
        "..", "..", "..", "scripts", "gradients", "generate_gradients.py",
    )
)
_spec = importlib.util.spec_from_file_location("generate_gradients", _script_path)
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
compute_all_gradients = _mod.compute_all_gradients
compute_cosine_similarity = _mod.compute_cosine_similarity


########################
# _BUILD_REDIAL_PROMPT #
########################

class TestBuildRedialPrompt:
    @pytest.fixture(autouse=True)
    def _setup_prompts(self):
        _mod.dialecttax.prompts.INSTS = {
            "math": {"naive": {"sae": "Solve the math problem."}},
        }
        _mod.dialecttax.prompts.PROMPTS = {
            "math": {"naive": {"sae": "Question: {question}\n"}},
        }
        _mod.dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
            "math": MagicMock(return_value=lambda ds, i: f"Question: {ds[i]['problem']}\n"),
        }
        _mod.dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n\n{body}"

    def test_basic_prompt(self):
        ds = [{"problem": "2+2"}]
        result = _build_redial_prompt(ds, 0, "math", "sae")
        assert "Solve the math problem." in result
        assert "2+2" in result

    def test_choices_formatting(self):
        _mod.dialecttax.prompts.INSTS = {
            "math": {"naive": {"sae": "Pick one:\n{choices}"}},
        }
        ds = [{"problem": "2+2", "choices": {"A": "3", "B": "4", "C": "5"}}]
        result = _build_redial_prompt(ds, 0, "math", "sae")
        assert "A. 3" in result
        assert "B. 4" in result


##################
# _BUILD_SAMPLES #
##################

class TestBuildSamples:
    @pytest.fixture(autouse=True)
    def _setup_prompts(self):
        _mod.dialecttax.prompts.INSTS = {
            "math": {"naive": {"sae": "Solve it."}},
        }
        _mod.dialecttax.prompts.PROMPTS = {
            "math": {"naive": {"sae": "Question: {question}\n"}},
        }
        _mod.dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {
            "math": MagicMock(return_value=lambda ds, i: f"Q: {ds[i]['problem']}\n"),
        }
        _mod.dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n\n{body}"

    def test_redial_has_text_key(self):
        """ReDial samples have text key with prompt + answer."""
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": "4"}]
        samples = _build_samples(ds, "math", "sae", "redial")
        assert len(samples) == 1
        assert "text" in samples[0]
        assert "unique_id" in samples[0]

    def test_redial_text_contains_answer(self):
        """ReDial text ends with the answer."""
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": "4"}]
        samples = _build_samples(ds, "math", "sae", "redial")
        assert samples[0]["text"].endswith("4")

    def test_redial_text_contains_prompt(self):
        """ReDial text contains the prompt content."""
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": "4"}]
        samples = _build_samples(ds, "math", "sae", "redial")
        assert "2+2" in samples[0]["text"]

    def test_redial_multiple_samples(self):
        ds = [
            {"unique_id": "q1", "problem": "1+1", "answer": "2"},
            {"unique_id": "q2", "problem": "3+3", "answer": "6"},
        ]
        samples = _build_samples(ds, "math", "sae", "redial")
        assert len(samples) == 2
        assert samples[0]["unique_id"] == "q1"
        assert samples[1]["unique_id"] == "q2"

    def test_redial_numeric_answer_to_string(self):
        """Numeric answers are converted to strings."""
        ds = [{"unique_id": "q1", "problem": "2+2", "answer": 4}]
        samples = _build_samples(ds, "math", "sae", "redial")
        assert samples[0]["text"].endswith("4")

    def test_text_only_dataset(self):
        """Text-only datasets use the text field directly."""
        ds = [
            {"unique_id": "p1", "text": "Hello world"},
            {"unique_id": "p2", "text": "Goodbye world"},
        ]
        samples = _build_samples(ds, "math", "sae", "parallelaave")
        assert len(samples) == 2
        assert samples[0]["text"] == "Hello world"
        assert samples[1]["text"] == "Goodbye world"

    def test_empty_dataset(self):
        samples = _build_samples([], "math", "sae", "redial")
        assert samples == []


#########################
# COMPUTE_ALL_GRADIENTS #
#########################

class TestComputeAllGradients:
    def _make_mock_model_and_tokenizer(self, projection_dim):
        """Create mock model and tokenizer for testing."""
        model = MagicMock()
        model.parameters.return_value = iter([torch.nn.Parameter(torch.randn(10))])
        device = torch.device("cpu")
        first_param = torch.nn.Parameter(torch.randn(1))
        model.parameters = MagicMock(return_value=iter([first_param]))
        # next(model.parameters()) should return a param on CPU
        model.parameters.return_value = iter([first_param])

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.randint(0, 100, (1, 5))}

        return model, tokenizer

    def test_output_shapes(self):
        """Returns projections array and metadata list with correct shapes."""
        projection_dim = 32
        model = MagicMock()
        first_param = torch.nn.Parameter(torch.randn(1))
        model.parameters.return_value = iter([first_param])

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.randint(0, 100, (1, 5))}

        # Mock the gradient computation
        mock_projected = torch.randn(projection_dim)
        _mod.dialecttax.gradients.compute_projected_gradient = MagicMock(
            return_value=(mock_projected, 2.5, 0.1),
        )

        samples = [
            {"unique_id": "q1", "text": "Hello"},
            {"unique_id": "q2", "text": "World"},
        ]
        projections, metadata = compute_all_gradients(model, tokenizer, samples, projection_dim, seed=42)

        assert projections.shape == (2, projection_dim)
        assert len(metadata) == 2

    def test_metadata_fields(self):
        """Metadata contains unique_id, loss, grad_norm, and n_tokens."""
        projection_dim = 16
        model = MagicMock()
        first_param = torch.nn.Parameter(torch.randn(1))
        model.parameters.return_value = iter([first_param])

        tokenizer = MagicMock()
        tokenizer.return_value = {"input_ids": torch.randint(0, 100, (1, 7))}

        _mod.dialecttax.gradients.compute_projected_gradient = MagicMock(
            return_value=(torch.randn(projection_dim), 3.14, 0.05),
        )

        samples = [{"unique_id": "q1", "text": "Test"}]
        _, metadata = compute_all_gradients(model, tokenizer, samples, projection_dim, seed=0)

        assert metadata[0]["unique_id"] == "q1"
        assert metadata[0]["loss"] == 3.14
        assert metadata[0]["grad_norm"] == 0.05
        assert metadata[0]["n_tokens"] == 7

    def test_empty_samples(self):
        """Returns empty arrays for empty samples."""
        model = MagicMock()
        first_param = torch.nn.Parameter(torch.randn(1))
        model.parameters.return_value = iter([first_param])
        tokenizer = MagicMock()

        projections, metadata = compute_all_gradients(model, tokenizer, [], 32, seed=0)

        assert projections.shape == (0, 32)
        assert metadata == []


###########################
# COMPUTE_COSINE_SIMILARITY #
###########################

class TestComputeCosineSimilarity:
    def test_identity(self):
        """Identical vectors have cosine similarity 1."""
        v = np.array([[1.0, 2.0, 3.0]])
        projections = np.vstack([v, v])
        sim = compute_cosine_similarity(projections)
        np.testing.assert_allclose(sim[0, 1], 1.0, atol=1e-6)

    def test_orthogonal(self):
        """Orthogonal vectors have cosine similarity 0."""
        projections = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        sim = compute_cosine_similarity(projections)
        np.testing.assert_allclose(sim[0, 1], 0.0, atol=1e-6)

    def test_opposite(self):
        """Opposite vectors have cosine similarity -1."""
        projections = np.array([
            [1.0, 2.0, 3.0],
            [-1.0, -2.0, -3.0],
        ])
        sim = compute_cosine_similarity(projections)
        np.testing.assert_allclose(sim[0, 1], -1.0, atol=1e-6)

    def test_diagonal_is_one(self):
        """Diagonal entries are all 1."""
        projections = np.random.randn(5, 10).astype(np.float32)
        sim = compute_cosine_similarity(projections)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-6)

    def test_symmetric(self):
        """Cosine similarity matrix is symmetric."""
        projections = np.random.randn(4, 8).astype(np.float32)
        sim = compute_cosine_similarity(projections)
        np.testing.assert_allclose(sim, sim.T, atol=1e-6)

    def test_output_shape(self):
        """Output is (n, n) for n input vectors."""
        projections = np.random.randn(3, 16).astype(np.float32)
        sim = compute_cosine_similarity(projections)
        assert sim.shape == (3, 3)

    def test_values_in_range(self):
        """All cosine similarities are in [-1, 1]."""
        projections = np.random.randn(10, 32).astype(np.float32)
        sim = compute_cosine_similarity(projections)
        assert np.all(sim >= -1.0 - 1e-6)
        assert np.all(sim <= 1.0 + 1e-6)

    def test_single_vector(self):
        """Single vector has similarity 1 with itself."""
        projections = np.array([[1.0, 2.0, 3.0]])
        sim = compute_cosine_similarity(projections)
        assert sim.shape == (1, 1)
        np.testing.assert_allclose(sim[0, 0], 1.0, atol=1e-6)

    def test_zero_vector_handled(self):
        """Zero vectors don't produce NaN (clamped by epsilon)."""
        projections = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ])
        sim = compute_cosine_similarity(projections)
        assert not np.any(np.isnan(sim))
