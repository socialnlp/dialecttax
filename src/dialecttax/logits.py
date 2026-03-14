"""Token-level and aggregate metrics for language model evaluation."""

import numpy as np
import torch
import torch.nn.functional as F


#####################
# TOKEN-LEVEL METRICS
#####################

def compute_log_probs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute log-probability of each token given its logits.

    Args:
        logits: (seq_len, vocab_size) logits at each position.
        token_ids: (seq_len,) actual token IDs at each position.

    Returns:
        (seq_len,) log-probabilities.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[torch.arange(len(token_ids)), token_ids]


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Predictive entropy of the distribution at each position.

    H(p) = -sum(p * log(p)) over the vocabulary.

    Args:
        logits: (seq_len, vocab_size) logits at each position.

    Returns:
        (seq_len,) entropy values (in nats).
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def last_token_hidden(hidden_states: torch.Tensor, token_ids: torch.Tensor, special_ids: set[int]) -> torch.Tensor:
    """Extract the hidden state of the last non-special token.

    Args:
        hidden_states: (seq_len, hidden_dim) hidden states from the last layer.
        token_ids: (seq_len,) token IDs for the sequence.
        special_ids: Set of special token IDs to skip.

    Returns:
        (hidden_dim,) hidden state vector.
    """
    token_list = token_ids.tolist()
    last_idx = len(token_list) - 1
    while last_idx >= 0 and token_list[last_idx] in special_ids:
        last_idx -= 1
    return hidden_states[last_idx]


def compute_input_metrics(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> list[dict]:
    """Compute token-level metrics for input (prompt) tokens via a forward pass.

    Args:
        model: the causal LM.
        input_ids: (batch_size, seq_len) input token IDs.
        attention_mask: (batch_size, seq_len) attention mask.

    Returns:
        List of dicts (one per sample) with keys:
            log_probs: (seq_len-1,) log-prob of each token given prefix
            entropy: (seq_len-1,) predictive entropy at each position
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    results = []
    for i in range(input_ids.shape[0]):
        mask = attention_mask[i].bool()
        seq_logits = logits[i, mask]  # (actual_len, vocab)
        seq_ids = input_ids[i, mask]  # (actual_len,)

        # Shift: logits at position t predict token at position t+1
        shifted_logits = seq_logits[:-1]  # (actual_len-1, vocab)
        shifted_ids = seq_ids[1:]         # (actual_len-1,)

        lp = compute_log_probs(shifted_logits, shifted_ids)
        results.append({
            "log_probs": lp.cpu(),
            "entropy": compute_entropy(shifted_logits).cpu(),
        })

    return results


def compute_generation_metrics(
    scores: tuple[torch.Tensor, ...],
    generated_ids: torch.Tensor,
    prompt_len: int,
    eos_token_id: int | None = None,
) -> list[dict]:
    """Compute token-level metrics for generated tokens.

    Args:
        scores: tuple of (batch_size, vocab_size) logits, one per generation step.
        generated_ids: (batch_size, prompt_len + gen_len) full sequence IDs.
        prompt_len: length of the input prompt (to slice generated token IDs).
        eos_token_id: if set, truncate metrics at the first EOS token.

    Returns:
        List of dicts (one per sample) with keys:
            log_probs, entropy: tensors of length = number of generated tokens.
    """
    # scores is a tuple of (batch_size, vocab_size), one per step
    # Stack into (gen_len, batch_size, vocab_size), then transpose to (batch_size, gen_len, vocab_size)
    stacked = torch.stack(scores, dim=0).transpose(0, 1)  # (batch, gen_len, vocab)
    gen_ids = generated_ids[:, prompt_len:]  # (batch, gen_len)

    results = []
    for i in range(stacked.shape[0]):
        seq_logits = stacked[i]  # (gen_len, vocab)
        seq_ids = gen_ids[i]     # (gen_len,)

        # Truncate at first EOS if present
        if eos_token_id is not None:
            eos_positions = (seq_ids == eos_token_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                end = eos_positions[0].item()
                seq_logits = seq_logits[:end]
                seq_ids = seq_ids[:end]

        if len(seq_ids) == 0:
            results.append({
                "log_probs": torch.tensor([]),
                "entropy": torch.tensor([]),
            })
            continue

        lp = compute_log_probs(seq_logits, seq_ids)
        results.append({
            "log_probs": lp.cpu(),
            "entropy": compute_entropy(seq_logits).cpu(),
        })

    return results


######################
# SAMPLE/RUN METRICS #
######################

def compute_sample_metrics(
    grading_result: dict,
    input_metrics: dict,
    generation_metrics: dict,
    completion: str,
) -> dict:
    """Aggregate token-level metrics into a per-sample summary.

    Args:
        grading_result: dict from grade_completions (completion, extracted, gold, correct).
        input_metrics: dict with log_probs, entropy tensors for the input.
        generation_metrics: dict with log_probs, entropy tensors for the generation.
        completion: the decoded completion string.

    Returns:
        Dict with per-sample summary metrics.
    """
    gen_lp = generation_metrics["log_probs"]
    gen_ent = generation_metrics["entropy"]
    inp_lp = input_metrics["log_probs"]
    inp_ent = input_metrics["entropy"]
    gen_len = len(gen_lp)
    inp_len = len(inp_lp)

    result = {
        # Grading
        "correct": grading_result["correct"],
        "extracted": grading_result["extracted"],
        "gold": grading_result["gold"],
        # Response
        "completion": completion,
        "response_length": gen_len,
        "accepted": grading_result["extracted"] is not None,
        # Input token metrics (per-token + summary)
        "input_length": inp_len,
        "input_log_probs": inp_lp.tolist(),
        "input_entropy": inp_ent.tolist(),
        "input_mean_log_prob": inp_lp.mean().item() if inp_len > 0 else 0.0,
        "input_mean_entropy": inp_ent.mean().item() if inp_len > 0 else 0.0,
        # Generation token metrics (per-token + summary)
        "gen_log_probs": gen_lp.tolist(),
        "gen_entropy": gen_ent.tolist(),
        "gen_mean_log_prob": gen_lp.mean().item() if gen_len > 0 else 0.0,
        "gen_mean_entropy": gen_ent.mean().item() if gen_len > 0 else 0.0,
    }
    return result


def compute_run_metrics(sample_metrics: list[dict]) -> dict:
    """Aggregate per-sample metrics into run-level summary.

    Args:
        sample_metrics: list of dicts from compute_sample_metrics.

    Returns:
        Dict with run-level aggregate metrics.
    """
    n = len(sample_metrics)
    if n == 0:
        return {}

    return {
        "n_samples": n,
        "accuracy": np.mean([s["correct"] for s in sample_metrics]),
        "acceptance_rate": np.mean([s["accepted"] for s in sample_metrics]),
        "mean_response_length": np.mean([s["response_length"] for s in sample_metrics]),
        "mean_input_length": np.mean([s["input_length"] for s in sample_metrics]),
        "mean_gen_log_prob": np.mean([s["gen_mean_log_prob"] for s in sample_metrics]),
        "mean_gen_entropy": np.mean([s["gen_mean_entropy"] for s in sample_metrics]),
        "mean_input_log_prob": np.mean([s["input_mean_log_prob"] for s in sample_metrics]),
        "mean_input_entropy": np.mean([s["input_mean_entropy"] for s in sample_metrics]),
    }
