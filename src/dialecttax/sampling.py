"""
Entropy-Reweighted Sampling for LLM Reasoning

Generate K candidate sequences from a base model, compute the average
token-level entropy of each during generation (for free), then select
the candidate with probability proportional to exp(-beta * H(x)).

Inspired by:
  - RENT (Prabhudesai et al., 2025): entropy as a signal for reasoning quality
  - KL-Regularized RL (2510.20817): beta as the diversity lever
  - Power Sampling (Karan & Du, 2025): inference-time reasoning from base models

Cost: identical to best-of-K sampling. No MCMC, no training, no verifier.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass


@dataclass
class SamplingConfig:
    K: int = 8                  # number of candidate sequences
    beta: float = 1.0           # sharpening strength (higher = more confident picks)
    max_new_tokens: int = 1024  # max generation length
    temperature: float = 1.0    # sampling temperature for generation
    top_p: float = 1.0          # nucleus sampling threshold
    entropy_window: str = "all" # "all", "last_fraction", or "last_n"
    window_fraction: float = 0.5  # fraction of tokens to use if entropy_window="last_fraction"
    window_n: int = 64          # number of tokens if entropy_window="last_n"


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of the token distribution at each position.

    Args:
        logits: shape (seq_len, vocab_size)

    Returns:
        entropies: shape (seq_len,)
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def generate_with_entropy(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    config: SamplingConfig,
) -> tuple[list[torch.Tensor], list[float], list[float]]:
    """Generate K sequences and track per-token entropies during generation.

    Returns:
        sequences: list of K token tensors (full sequence including prompt)
        avg_entropies: list of K average entropy values
        log_likelihoods: list of K average log-likelihoods (for the hybrid variant)
    """
    device = input_ids.device
    prompt_len = input_ids.shape[1]
    sequences = []
    avg_entropies = []
    log_likelihoods = []

    for _ in range(config.K):
        current_ids = input_ids.clone()
        token_entropies = []
        token_log_probs = []

        for step in range(config.max_new_tokens):
            with torch.no_grad():
                outputs = model(current_ids)
                logits = outputs.logits[:, -1, :]  # (1, vocab_size)

            # Compute entropy of this distribution (free, we already have logits)
            entropy = compute_token_entropy(logits).item()
            token_entropies.append(entropy)

            # Apply temperature and sample
            scaled_logits = logits / config.temperature

            # Optional: nucleus sampling
            if config.top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= config.top_p
                sorted_logits[mask] = float("-inf")
                scaled_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Track log-likelihood of the chosen token (for hybrid variant)
            log_prob = F.log_softmax(logits, dim=-1)  # use unscaled logits
            token_log_probs.append(log_prob[0, next_token.item()].item())

            current_ids = torch.cat([current_ids, next_token], dim=-1)

            # Check for EOS
            if hasattr(model.config, "eos_token_id"):
                eos_id = model.config.eos_token_id
                if isinstance(eos_id, list):
                    if next_token.item() in eos_id:
                        break
                elif next_token.item() == eos_id:
                    break

        # Compute windowed average entropy (RENT insight: late tokens matter more)
        entropies_tensor = torch.tensor(token_entropies)
        if config.entropy_window == "last_fraction":
            n_keep = max(1, int(len(token_entropies) * config.window_fraction))
            entropies_tensor = entropies_tensor[-n_keep:]
        elif config.entropy_window == "last_n":
            n_keep = min(config.window_n, len(token_entropies))
            entropies_tensor = entropies_tensor[-n_keep:]

        avg_entropy = entropies_tensor.mean().item()
        avg_log_likelihood = sum(token_log_probs) / len(token_log_probs)

        sequences.append(current_ids[0])
        avg_entropies.append(avg_entropy)
        log_likelihoods.append(avg_log_likelihood)

    return sequences, avg_entropies, log_likelihoods


def select_sequence(
    sequences: list[torch.Tensor],
    avg_entropies: list[float],
    log_likelihoods: list[float],
    config: SamplingConfig,
    alpha: float | None = None,
) -> tuple[torch.Tensor, dict]:
    """Select a sequence using entropy-reweighted importance sampling.

    Two variants:
      1. Pure entropy:     w(x) ∝ exp(-beta * H(x))
      2. Hybrid (if alpha provided): w(x) ∝ exp(alpha * log π(x) - beta * H(x))

    Args:
        sequences: K candidate token sequences
        avg_entropies: K average entropy values
        log_likelihoods: K average log-likelihoods
        config: sampling configuration
        alpha: if provided, use hybrid weighting with power distribution

    Returns:
        selected: the chosen token sequence
        info: dict with diagnostics
    """
    entropies = torch.tensor(avg_entropies)

    # Compute log-weights
    log_weights = -config.beta * entropies
    if alpha is not None:
        ll = torch.tensor(log_likelihoods)
        log_weights = log_weights + alpha * ll

    # Normalize to probabilities
    weights = F.softmax(log_weights, dim=0)

    # Sample (or argmax for greedy selection)
    selected_idx = torch.multinomial(weights, num_samples=1).item()

    info = {
        "selected_idx": selected_idx,
        "weights": weights.tolist(),
        "entropies": avg_entropies,
        "log_likelihoods": log_likelihoods,
        "selected_entropy": avg_entropies[selected_idx],
        "selected_log_likelihood": log_likelihoods[selected_idx],
    }

    return sequences[selected_idx], info


def entropy_reweighted_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    config: SamplingConfig | None = None,
    alpha: float | None = None,
) -> tuple[str, dict]:
    """End-to-end entropy-reweighted generation.

    Args:
        model: a causal language model
        tokenizer: corresponding tokenizer
        prompt: the input prompt string
        config: sampling configuration (defaults provided)
        alpha: if set, use hybrid entropy + power distribution weighting

    Returns:
        response: the selected response string
        info: diagnostics dict
    """
    if config is None:
        config = SamplingConfig()

    device = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]

    sequences, avg_entropies, log_likelihoods = generate_with_entropy(
        model, input_ids, config
    )

    selected, info = select_sequence(
        sequences, avg_entropies, log_likelihoods, config, alpha=alpha
    )

    # Decode only the generated portion
    response = tokenizer.decode(selected[prompt_len:], skip_special_tokens=True)

    return response, info
