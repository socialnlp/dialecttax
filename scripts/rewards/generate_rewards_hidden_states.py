"""
Extract reward-model hidden states at the score-head position on parallel
dialect samples, for downstream dialect-separability analysis.

For each parallel pair, applies the same chat template the RM uses for
scoring, runs a forward pass on the underlying backbone with
``output_hidden_states=True``, and saves the final-layer hidden state at
the last non-padding position (the vector the score head reads from).

Output structure mirrors ``generate_logits``:
    experiments/generate_rewards_hidden_states/{rm}/{dataset}/{task}/naive/{dialect}/
        hidden.npz       # archive keyed by unique_id, each value is (hidden_dim,)
        metadata.jsonl   # one row per sample

Usage:
    python scripts/rewards/generate_rewards_hidden_states.py
    python scripts/rewards/generate_rewards_hidden_states.py reward_model=qrm_llama_8b
    python scripts/rewards/generate_rewards_hidden_states.py --multirun \
        reward_model=skywork_llama_8b,qrm_llama_8b task=math,logic dialect=sae,aave
"""

import argparse
import gc
import json
import logging
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import dialecttax

# Reuse load_reward_model + _build_redial_prompt from the scoring pipeline
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_rewards import load_reward_model, _build_redial_prompt  # noqa: E402


#########
# SETUP #
#########

def setup_hydra():
    project_config = dialecttax.utils.load_config()
    OmegaConf.register_new_resolver(
        "project", lambda key: project_config["directories"][key],
        replace=True,
    )
    if hasattr(argparse.ArgumentParser, "_check_help"):
        _orig_check_help = argparse.ArgumentParser._check_help
        def _patched_check_help(self, action):
            if action.help is not None and not isinstance(action.help, str):
                action.help = repr(action.help)
            _orig_check_help(self, action)
        argparse.ArgumentParser._check_help = _patched_check_help
    return project_config


_project_config = setup_hydra()
log = logging.getLogger(__name__)


#########
# GEMMA #
#########

def _patch_gemma_attention(rm):
    """Reload Gemma 2 backbones with attn_implementation='eager'.

    Gemma 2 uses logit soft-capping and sliding-window attention, which
    interact poorly with sdpa/flash_attention_2 in bfloat16 — the last-token
    hidden state comes back as NaN. The eager attention path implements
    soft-capping correctly and is numerically stable. We pay for it in speed
    but Gemma 2 is the only family that needs this.

    Idempotent: returns early if already patched.
    """
    if "gemma" not in rm.model_id.lower():
        return rm
    cur = getattr(rm.model.config, "_attn_implementation", None)
    if cur == "eager":
        return rm

    from transformers import AutoModelForSequenceClassification
    log.info(f"Reloading {rm.model_id} with attn_implementation='eager' (Gemma 2 NaN fix)")
    del rm.model
    rm.model = None
    torch.cuda.empty_cache()
    gc.collect()

    kwargs = dict(
        dtype=torch.bfloat16,
        device_map=rm.device,
        attn_implementation="eager",
    )
    if isinstance(rm, dialecttax.rewards.SkyworkRewardModel):
        kwargs["num_labels"] = 1
    if isinstance(rm, dialecttax.rewards.QRMRewardModel):
        kwargs["trust_remote_code"] = True

    rm.model = AutoModelForSequenceClassification.from_pretrained(rm.model_id, **kwargs)

    # QRM moves its custom heads off the accelerate shard map manually
    if isinstance(rm, dialecttax.rewards.QRMRewardModel):
        last_device = rm.model.model.norm.weight.device
        rm.model.regression_layer.to(last_device)
        rm.model.gating.to(last_device)
        if rm.model.config.pad_token_id is None:
            rm.model.config.pad_token_id = rm.tokenizer.pad_token_id or rm.tokenizer.eos_token_id

    return rm


###################
# HIDDEN EXTRACT  #
###################

def _tokenize_for_rm(rm, conversation):
    """Apply the same chat-template tokenization the RM uses for scoring.

    Skywork/Ai2 use the base class _tokenize (apply_chat_template tokenize=False
    then tokenizer(text)). QRM uses apply_chat_template directly with tokenize=True.
    We match each provider's score path so the hidden state corresponds to the
    same input the score head reads.
    """
    if isinstance(rm, dialecttax.rewards.QRMRewardModel):
        input_ids = rm.tokenizer.apply_chat_template(
            conversation, return_tensors="pt", tokenize=True
        )["input_ids"]
        input_ids = input_ids.to(rm.input_device)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    inputs = rm._tokenize(conversation)
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return dict(inputs)


def _backbone(rm):
    """Return the transformer backbone that produces hidden states.

    Llama/Qwen/Gemma SequenceClassification models wrap the backbone at
    ``.model``. QRM's custom class follows the same convention.
    """
    inner = rm.model
    if hasattr(inner, "model"):
        return inner.model
    if hasattr(inner, "transformer"):
        return inner.transformer
    return inner


@torch.no_grad()
def extract_score_hidden(rm, conversation):
    """Return the last-layer hidden state at the score-head read position.

    Args:
        rm: Loaded RewardModel instance.
        conversation: Single conversation (list of role/content dicts).

    Returns:
        ndarray of shape (hidden_dim,) in float32.
    """
    inputs = _tokenize_for_rm(rm, conversation)
    input_ids = inputs["input_ids"]
    if input_ids.dim() != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected batch-1 2D input_ids, got shape {tuple(input_ids.shape)}")

    backbone = _backbone(rm)
    out = backbone(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"],
    )
    last_hidden = getattr(out, "last_hidden_state", None)
    if last_hidden is None:
        last_hidden = out[0]

    # Single-sample, no padding → the last token IS the score-head position.
    # Use input_ids.shape (host-side, deterministic) rather than .item() on a
    # CUDA reduction (which has hit garbage-index issues on some stacks).
    if last_hidden.shape[1] != input_ids.shape[1]:
        raise ValueError(
            f"hidden seq_len {last_hidden.shape[1]} != input seq_len {input_ids.shape[1]}"
        )
    last_idx = input_ids.shape[1] - 1
    vec = last_hidden[0, last_idx].detach().cpu().float().numpy()
    return vec


########
# MAIN #
########

@hydra.main(version_base=None, config_path="../../configs/generate_rewards_hidden_states",
            config_name="config")
def main(cfg: DictConfig):
    dataset_name = cfg.dataset.name
    task = cfg.task.name
    dialect = cfg.dialect.name

    valid_dialects = list(cfg.dataset.dialects)
    if dialect not in valid_dialects:
        log.info(f"Skipping (dialect '{dialect}' not in {valid_dialects} for dataset '{dataset_name}')")
        return
    if "tasks" in cfg.dataset and task not in list(cfg.dataset.tasks):
        log.info(f"Skipping (task '{task}' not in {list(cfg.dataset.tasks)} for dataset '{dataset_name}')")
        return

    mod = dialecttax.data.DATASET_MODULES[dataset_name]
    if not hasattr(mod, "TASKS"):
        log.info(f"Skipping (no TASKS attribute on dataset '{dataset_name}' -- sample-level only)")
        return

    out_dir = HydraConfig.get().runtime.output_dir
    hidden_path = os.path.join(out_dir, "hidden.npz")
    meta_path = os.path.join(out_dir, "metadata.jsonl")
    if not cfg.rerun and os.path.exists(hidden_path) and os.path.exists(meta_path):
        log.info(f"Skipping (outputs exist, rerun=false): {out_dir}")
        return

    log.info(
        f"\n{'=' * 60}\n"
        f"  Reward Model: {cfg.reward_model.name}\n"
        f"  Dataset:      {dataset_name}\n"
        f"  Task:         {task}\n"
        f"  Dialect:      {dialect}\n"
        f"{'=' * 60}"
    )

    # Load dataset
    dir_root = _project_config["directories"][cfg.dataset.dir_key]
    path_file = mod.FILE_NAME_FORMAT.format(task=task, dialect=dialect)
    path_file = os.path.join(mod.DIRECTORY_NAME, path_file)
    path = os.path.join(dir_root, path_file)
    if not os.path.exists(path):
        log.error(f"Dataset not found: {path}")
        return
    ds = mod.load_dataset(dir_root, path_file)
    log.info(f"Loaded {len(ds)} samples from {os.path.abspath(path)}")

    # Load reward model (control GPUs via CUDA_VISIBLE_DEVICES)
    rm = load_reward_model(cfg.reward_model.name, device=cfg.device)
    rm = _patch_gemma_attention(rm)

    # Extract hidden states one sample at a time (RMs are loaded with bfloat16,
    # batch sizes vary; sequential mirrors scoring and keeps memory predictable)
    hiddens = {}
    metadata = []
    reasoning = "naive"
    for i, sample in enumerate(ds):
        prompt = _build_redial_prompt(ds, i, task, reasoning, dialect)
        response = str(sample["answer"])
        conversation = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        try:
            vec = extract_score_hidden(rm, conversation)
        except Exception as e:
            log.warning(f"  [{i}] extraction failed ({sample['unique_id']}): {e}")
            continue
        uid = sample["unique_id"]
        hiddens[uid] = vec
        metadata.append({"unique_id": uid, "hidden_dim": int(vec.shape[0])})
        if (i + 1) % 50 == 0:
            log.info(f"  Extracted {i + 1}/{len(ds)} samples")

    if not hiddens:
        log.error("No hidden states extracted; skipping save.")
        return

    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(hidden_path, **hiddens)
    with open(meta_path, "w") as f:
        for m in metadata:
            f.write(json.dumps(m) + "\n")

    log.info(f"Saved {len(hiddens)} hidden vectors (dim={next(iter(hiddens.values())).shape[0]}) to: {hidden_path}")
    log.info(f"Saved {len(metadata)} metadata entries to: {meta_path}")

    # Free GPU between Hydra multirun iterations only if reused-by-name fails
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
