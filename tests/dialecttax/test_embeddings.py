"""Tests for dialecttax.embeddings (FAISS functions)."""

import importlib
import json
import os
from unittest.mock import MagicMock

import faiss
import numpy as np

# Import the module directly to avoid dialecttax.__init__ pulling in
# optional dependencies (bts) that may not be installed.
_spec = importlib.util.spec_from_file_location(
    "dialecttax.embeddings",
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "dialecttax", "embeddings.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

build_index = _mod.build_index
encode = _mod.encode
load_index = _mod.load_index
save_index = _mod.save_index
search_index = _mod.search_index


def _random_normalized(n, d, seed=42):
    """Generate random L2-normalized float32 vectors."""
    rng = np.random.RandomState(seed)
    emb = rng.randn(n, d).astype(np.float32)
    faiss.normalize_L2(emb)
    return emb


##########
# ENCODE #
##########


class TestEncode:
    def test_output_shape(self):
        """Returns (N, D) array matching model output dimension."""
        d = 16
        model = MagicMock()
        model.encode.return_value = np.random.randn(5, d).astype(np.float32)

        result = encode(model, ["a", "b", "c", "d", "e"])

        assert result.shape == (5, d)
        assert result.dtype == np.float32

    def test_l2_normalized(self):
        """Output vectors should have unit L2 norm."""
        d = 32
        model = MagicMock()
        model.encode.return_value = np.random.randn(10, d).astype(np.float32) * 5.0

        result = encode(model, [str(i) for i in range(10)])

        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_passes_batch_size(self):
        """batch_size is forwarded to model.encode()."""
        model = MagicMock()
        model.encode.return_value = np.random.randn(2, 8).astype(np.float32)

        encode(model, ["a", "b"], batch_size=64)

        model.encode.assert_called_once_with(
            ["a", "b"], batch_size=64, show_progress_bar=True
        )

    def test_casts_to_float32(self):
        """Float64 model output is cast to float32."""
        model = MagicMock()
        model.encode.return_value = np.random.randn(3, 8).astype(np.float64)

        result = encode(model, ["a", "b", "c"])

        assert result.dtype == np.float32


###############
# BUILD_INDEX #
###############


class TestBuildIndex:
    def test_returns_ivf_flat(self):
        """Returns a trained IndexIVFFlat instance."""
        emb = _random_normalized(500, 16)
        index = build_index(emb, nlist=5)

        assert isinstance(index, faiss.IndexIVFFlat)
        assert index.is_trained
        assert index.ntotal == 500

    def test_dimension_matches(self):
        """Index dimension matches input embedding dimension."""
        d = 32
        emb = _random_normalized(500, d)
        index = build_index(emb, nlist=5)

        assert index.d == d

    def test_inner_product_metric(self):
        """Index uses inner-product metric."""
        emb = _random_normalized(500, 16)
        index = build_index(emb, nlist=5)

        assert index.metric_type == faiss.METRIC_INNER_PRODUCT

    def test_nlist_respected(self):
        """Number of clusters matches nlist parameter."""
        emb = _random_normalized(500, 16)
        index = build_index(emb, nlist=10)

        assert index.nlist == 10


################
# SEARCH_INDEX #
################


class TestSearchIndex:
    def _build(self, n=1000, d=16, nlist=10):
        emb = _random_normalized(n, d)
        index = build_index(emb, nlist=nlist)
        return index, emb

    def test_output_shapes(self):
        """Returns (distances, indices) with shape (N, k)."""
        index, emb = self._build()
        queries = emb[:5]

        distances, indices = search_index(index, queries, k=3, nprobe=10)

        assert distances.shape == (5, 3)
        assert indices.shape == (5, 3)

    def test_self_is_top_match(self):
        """Each vector's nearest neighbor should be itself."""
        index, emb = self._build()
        queries = emb[:10]

        distances, indices = search_index(index, queries, k=1, nprobe=10)

        # With normalized vectors and IP metric, self-similarity should be ~1.0
        np.testing.assert_allclose(distances[:, 0], 1.0, atol=1e-5)
        np.testing.assert_array_equal(indices[:, 0], np.arange(10))

    def test_distances_sorted_descending(self):
        """Inner-product distances should be in descending order."""
        index, emb = self._build()
        queries = emb[:3]

        distances, _ = search_index(index, queries, k=5, nprobe=10)

        for row in distances:
            assert all(row[i] >= row[i + 1] - 1e-7 for i in range(len(row) - 1))


##############
# SAVE/LOAD  #
##############


class TestSaveLoadIndex:
    def test_roundtrip(self, tmp_path):
        """Index and metadata survive a save/load roundtrip."""
        emb = _random_normalized(500, 16)
        index = build_index(emb, nlist=5)
        metadata = [{"id": i, "text": f"doc_{i}"} for i in range(500)]
        path = os.path.join(str(tmp_path), "test_idx")

        save_index(index, metadata, path)
        loaded_index, loaded_meta = load_index(path)

        assert loaded_index.ntotal == index.ntotal
        assert loaded_index.d == index.d
        assert len(loaded_meta) == 500
        assert loaded_meta[0] == {"id": 0, "text": "doc_0"}
        assert loaded_meta[499] == {"id": 499, "text": "doc_499"}

    def test_files_created(self, tmp_path):
        """save_index creates .index and .jsonl files."""
        emb = _random_normalized(200, 8)
        index = build_index(emb, nlist=5)
        metadata = [{"i": i} for i in range(200)]
        path = os.path.join(str(tmp_path), "idx")

        save_index(index, metadata, path)

        assert os.path.isfile(f"{path}.index")
        assert os.path.isfile(f"{path}.jsonl")

    def test_jsonl_format(self, tmp_path):
        """Metadata file is valid JSONL (one JSON object per line)."""
        emb = _random_normalized(200, 8)
        index = build_index(emb, nlist=5)
        metadata = [{"key": "val", "n": i} for i in range(200)]
        path = os.path.join(str(tmp_path), "idx")

        save_index(index, metadata, path)

        with open(f"{path}.jsonl") as f:
            lines = f.readlines()
        assert len(lines) == 200
        for line in lines:
            parsed = json.loads(line)
            assert "key" in parsed

    def test_search_after_reload(self, tmp_path):
        """Loaded index produces the same search results as the original."""
        emb = _random_normalized(500, 16)
        index = build_index(emb, nlist=5)
        metadata = [{"id": i} for i in range(500)]
        path = os.path.join(str(tmp_path), "idx")

        save_index(index, metadata, path)
        loaded_index, _ = load_index(path)

        queries = emb[:3]
        d_orig, i_orig = search_index(index, queries, k=5, nprobe=5)
        d_loaded, i_loaded = search_index(loaded_index, queries, k=5, nprobe=5)

        np.testing.assert_array_equal(i_orig, i_loaded)
        np.testing.assert_allclose(d_orig, d_loaded, atol=1e-6)
