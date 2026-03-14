"""Per-document gradient projection for language models using CountSketch."""

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dialecttax.models import LANGUAGE_MODELS_BASE

log = logging.getLogger(__name__)

CHUNK_SIZE = 2**22  # 4M elements per chunk for CountSketch memory efficiency
SKETCH_REPLICAS = 64  # accumulator copies used to spread scatter_add atomic contention


###########
# LOADING #
###########

def load_model(model_id, device="auto"):
    """Load a causal LM with gradient checkpointing enabled.

    Args:
        model_id: HuggingFace model ID.
        device: Device string ("auto" for multi-GPU via accelerate).

    Returns:
        Tuple of (model, tokenizer).
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


################
# COUNTSKETCH  #
################

def _countsketch(grad_flat, projection_dim, seed):
    """Project a 1-D gradient vector using CountSketch.

    Processes in chunks of CHUNK_SIZE to limit memory usage. Each element is
    hashed to a random bucket with a random sign (+1/-1), then accumulated
    via scatter_add.

    A single projection_dim-wide accumulator makes scatter_add serialize on
    atomic collisions (4M elements into 8192 buckets), which measured 8x slower
    than the bandwidth the same traffic could reach. Elements are therefore
    striped across SKETCH_REPLICAS private copies of the accumulator, summed at
    the end. The random draws are unchanged, so this only reorders the float
    additions.

    Args:
        grad_flat: Flattened gradient tensor (any device).
        projection_dim: Target dimensionality.
        seed: Deterministic seed for hash functions.

    Returns:
        Projected vector of shape (projection_dim,) in float32 on the same device.
    """
    d = grad_flat.shape[0]
    device = grad_flat.device
    replicas = max(1, min(SKETCH_REPLICAS, d // projection_dim))
    projected = torch.zeros(replicas * projection_dim, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    lane = (torch.arange(min(d, CHUNK_SIZE), device=device) % replicas) * projection_dim

    for start in range(0, d, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, d)
        chunk = grad_flat[start:end].float()
        n = end - start
        buckets = torch.randint(0, projection_dim, (n,), generator=generator, device=device)
        signs = torch.randint(0, 2, (n,), generator=generator, device=device).float() * 2 - 1
        projected.scatter_add_(0, buckets + lane[:n], signs * chunk)

    return projected.view(replicas, projection_dim).sum(0)


##############
# PROJECTION #
##############

def project_gradient(model, projection_dim, seed):
    """Project all model parameter gradients into a single vector using CountSketch.

    Must be called after loss.backward(). Does not zero gradients.
    Each parameter uses a unique sub-seed (base seed + parameter index) so the
    per-parameter sketches are independent. Accumulates on the device holding the
    first gradient and copies to host once: a .cpu() per parameter would sync on
    every one of the model's few hundred gradient tensors.

    Args:
        model: Model with computed gradients (.grad attributes).
        projection_dim: Target dimensionality.
        seed: Base seed (each parameter uses seed + param_index).

    Returns:
        Projected gradient vector of shape (projection_dim,) on CPU in float32.
    """
    projected = None
    for idx, p in enumerate(model.parameters()):
        if p.grad is None:
            continue
        sketch = _countsketch(p.grad.detach().flatten(), projection_dim, seed + idx)
        # device_map="auto" can spread gradients over several devices.
        projected = sketch if projected is None else projected + sketch.to(projected.device)
    if projected is None:
        return torch.zeros(projection_dim, dtype=torch.float32)
    return projected.cpu()


def _gradient_norm(model):
    """Compute the L2 norm of all parameter gradients.

    Per-parameter norms stay on the device and are reduced in one step, so the
    model's few hundred gradient tensors cost a single host sync rather than one
    .item() each. vector_norm(dtype=float32) accumulates in float32 without
    materializing an upcast copy of the gradient.

    Args:
        model: Model with computed gradients (.grad attributes).

    Returns:
        Scalar gradient norm (float).
    """
    norms = [
        torch.linalg.vector_norm(p.grad.detach(), dtype=torch.float32)
        for p in model.parameters()
        if p.grad is not None
    ]
    if not norms:
        return 0.0
    device = norms[0].device
    return torch.linalg.vector_norm(torch.stack([n.to(device) for n in norms])).item()


def compute_projected_gradient(model, input_ids, projection_dim, seed):
    """Compute the CountSketch-projected gradient for a single document.

    Runs forward pass with causal LM cross-entropy loss over all tokens,
    then backpropagates and projects the full-parameter gradient.

    Args:
        model: CausalLM with gradient checkpointing enabled.
        input_ids: Token IDs of shape (1, seq_len) on the model's input device.
        projection_dim: Projection dimensionality.
        seed: Base seed for CountSketch.

    Returns:
        Tuple of (projected_gradient, loss_value, gradient_norm) where
        projected_gradient has shape (projection_dim,) on CPU in float32.
    """
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    projected = project_gradient(model, projection_dim, seed)
    grad_norm = _gradient_norm(model)
    model.zero_grad(set_to_none=True)
    return projected, loss.item(), grad_norm
