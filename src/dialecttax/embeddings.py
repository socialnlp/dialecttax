import json

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


# Recommended prompt for semantic similarity
# Source: https://huggingface.co/google/embeddinggemma-300m
PROMPT_SEMANTIC_SIMILARITY = "task: sentence similarity | query: {content}"


#############
# EMBEDDING #
#############

def load_embedding_gemma(model_name: str = "google/embeddinggemma-300m", device: str | None = None) -> SentenceTransformer:
    """Load an EmbeddingGemma model in bfloat16.

    Source: https://huggingface.co/google/embeddinggemma-300m
    Args:
        model_name: HuggingFace model ID.
        device: Device to load on. Defaults to CUDA if available.

    Returns:
        SentenceTransformer model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device=device)


def encode(model: SentenceTransformer, data: list[str], dim: int = 768, batch_size: int = 256) -> np.ndarray:
    """Encode texts and L2-normalize for cosine similarity via inner product.

    Args:
        model: A SentenceTransformer model.
        data: Strings to encode.
        batch_size: Batch size for encoding.

    Returns:
        L2-normalized float32 array of shape (N, D).
    """
    assert dim in (128, 256, 512, 768)

    # Add prompt for semantic similarity embedding task
    texts = [PROMPT_SEMANTIC_SIMILARITY.format(content=text) for text in data]

    # Generate embeddings
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    emb = np.asarray(emb, dtype=np.float32)

    # Truncate to embedding dimension
    emb = np.ascontiguousarray(emb[:, :dim])

    # L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(emb)
    return emb


##############
# SIMILARITY #
##############

def similarity_pairwise(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarities between two aligned NumPy arrays.

    We store L2-normalized arrays, so we only need the dot product.

    Args:
        model: A SentenceTransformer model.
        emb_a: First embedding, shape (N, M)
        emb_b: Second embedding, shape (N, M)

    Returns:
        1-D array of cosine similarities, shape (N,).
    """
    return np.sum(a * b, axis=1)


def similarity_baseline(emb_a: np.ndarray, emb_b: np.ndarray) -> tuple[float, float]:
    """Null baseline from unrelated pairs.

    Computes cosine similarities between all non-aligned pairs (i != j) across two embedding sets,
    excluding the diagonal (aligned pairs that share the same source sample).

    Args:
        emb_a: L2-normalized embeddings of shape (N, D).
        emb_b: L2-normalized embeddings of shape (N, D).

    Returns:
        (mu, sigma) of the null similarity distribution.
    """
    # Cross-similarity matrix (N, N)
    sim = emb_a @ emb_b.T

    # Exclude diagonal (aligned pairs)
    n = sim.shape[0]
    mask = ~np.eye(n, dtype=bool)
    off_diag = sim[mask]

    # Null distribution stats
    mu_null = off_diag.mean().item()
    sigma_null = off_diag.std().item()
    return mu_null, sigma_null


#########
# FAISS #
#########

def build_index(embeddings: np.ndarray, nlist: int = 100) -> faiss.IndexIVFFlat:
    """Build an IVF-Flat index with inner-product metric.

    Uses inverted file indexing for fast approximate nearest-neighbor search
    on L2-normalized vectors (inner product == cosine similarity).

    Args:
        embeddings: Float32 array of shape (N, D), must be L2-normalized.
        nlist: Number of Voronoi cells (clusters). Higher = more precise
            but slower to search.

    Returns:
        Trained FAISS IVF-Flat index with all embeddings added.
    """
    d = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    return index


def search_index(index: faiss.Index, queries: np.ndarray, k: int = 10, nprobe: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Search a FAISS index for top-k nearest neighbors.

    Args:
        index: A trained FAISS index.
        queries: Float32 query vectors of shape (N, D).
        k: Number of neighbors to retrieve per query.
        nprobe: Number of clusters to visit per query. Higher = more
            accurate but slower.

    Returns:
        Tuple of (distances, indices), each of shape (N, k).
    """
    # nprobe controls the speed/accuracy tradeoff at search time
    index.nprobe = nprobe
    distances, indices = index.search(queries, k)
    return distances, indices


def save_index(index: faiss.Index, metadata: list[dict], path: str) -> None:
    """Save a FAISS index and its metadata to disk.

    Args:
        index: A trained FAISS index.
        metadata: List of dicts mapping each vector ID to source info.
        path: Base path (without extension). Writes ``{path}.index``
            and ``{path}.jsonl``.
    """
    faiss.write_index(index, f"{path}.index")
    with open(f"{path}.jsonl", "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")


def load_index(path: str) -> tuple[faiss.Index, list[dict]]:
    """Load a FAISS index and its metadata from disk.

    Args:
        path: Base path (without extension). Reads ``{path}.index``
            and ``{path}.jsonl``.

    Returns:
        Tuple of (index, metadata).
    """
    index = faiss.read_index(f"{path}.index")
    with open(f"{path}.jsonl") as f:
        metadata = [json.loads(line) for line in f]
    return index, metadata
