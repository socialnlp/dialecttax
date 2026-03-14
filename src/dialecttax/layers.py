"""Per-layer hidden state extraction and pairwise cosine similarity."""

import logging

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

log = logging.getLogger(__name__)


###########
# LOADING #
###########

def load_model(model_id, device="auto"):
    """Load a causal LM for hidden state extraction.

    Args:
        model_id: HuggingFace model ID.
        device: Device string ("auto" for multi-GPU via accelerate).

    Returns:
        Tuple of (model, tokenizer).
    """
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map=device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


#################
# HIDDEN STATES #
#################

def extract_hidden_states(model, tokenizer, texts, batch_size=8):
    """Extract mean-pooled hidden states at every layer for a list of texts.

    Args:
        model: CausalLM.
        tokenizer: Corresponding tokenizer.
        texts: List of text strings.
        batch_size: Number of texts to process at once.

    Returns:
        ndarray of shape (n_samples, n_layers+1, hidden_dim) in float32.
        Layer 0 is the embedding layer; layers 1..N are transformer layers.
    """
    all_hidden = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states: tuple of (batch, seq_len, hidden_dim), one per layer
        attention_mask = inputs["attention_mask"]  # (batch, seq_len)
        mask_f = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        lengths = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1).float()  # (batch, 1, 1)

        for i in range(len(batch_texts)):
            sample_hidden = []
            for layer_states in outputs.hidden_states:
                # Mean-pool over non-padding tokens
                pooled = (layer_states[i] * mask_f[i]).sum(dim=0) / lengths[i].squeeze()
                sample_hidden.append(pooled.cpu().float().numpy())
            all_hidden.append(np.stack(sample_hidden))  # (n_layers+1, hidden_dim)

    return np.stack(all_hidden)  # (n_samples, n_layers+1, hidden_dim)


###############################
# PER-LAYER COSINE SIMILARITY #
###############################

def compute_pairwise_cosine(hidden_a, hidden_b):
    """Compute per-pair cosine similarity at each layer.

    Args:
        hidden_a: ndarray of shape (n_samples, n_layers, hidden_dim).
        hidden_b: ndarray of shape (n_samples, n_layers, hidden_dim), same n_samples.

    Returns:
        ndarray of shape (n_samples, n_layers) with cosine similarities.
    """
    # Dot product along hidden_dim: (n_samples, n_layers)
    dot = np.einsum("slh,slh->sl", hidden_a, hidden_b)
    norm_a = np.linalg.norm(hidden_a, axis=2)  # (n_samples, n_layers)
    norm_b = np.linalg.norm(hidden_b, axis=2)  # (n_samples, n_layers)
    return dot / np.maximum(norm_a * norm_b, 1e-8)
