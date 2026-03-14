"""Test that demo_indices are deterministic across runs when using the same seed."""

import numpy as np
import dialecttax.prompts


SEED = 42
DATASET_SIZE = 100
N_FEW_SHOT = 5


def test_same_seed_same_indices():
    """Same seed produces identical demo indices every time."""
    rng = np.random.default_rng(SEED)
    indices_1 = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT, rng=rng)

    rng = np.random.default_rng(SEED)
    indices_2 = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT, rng=rng)

    assert indices_1 == indices_2


def test_different_seed_different_indices():
    """Different seeds produce different demo indices."""
    rng1 = np.random.default_rng(SEED)
    indices_1 = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT, rng=rng1)

    rng2 = np.random.default_rng(SEED + 1)
    indices_2 = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT, rng=rng2)

    assert indices_1 != indices_2


def test_indices_stable_across_simulated_sweep():
    """Simulates multiple Hydra sweep jobs: each creates its own rng with
    the same seed, and all should get the same demo_indices."""
    all_indices = []
    for _ in range(5):  # simulate 5 sweep jobs
        rng = np.random.default_rng(SEED)
        indices = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT, rng=rng)
        all_indices.append(indices)

    for indices in all_indices[1:]:
        assert indices == all_indices[0]


def test_isolated_from_global_rng():
    """The rng object is not affected by global np.random calls."""
    rng = np.random.default_rng(SEED)
    indices_1 = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT, rng=rng)

    # Pollute global RNG state
    np.random.randn(1000)
    np.random.randint(0, 100, size=50)

    # Fresh rng with same seed still gives same result
    rng = np.random.default_rng(SEED)
    indices_2 = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT, rng=rng)

    assert indices_1 == indices_2


def test_default_seed_is_deterministic():
    """When no rng is passed, the default seed produces consistent results."""
    indices_1 = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT)
    indices_2 = dialecttax.prompts.get_demo_indices(DATASET_SIZE, N_FEW_SHOT)

    assert indices_1 == indices_2
