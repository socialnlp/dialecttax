import importlib
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    import flash_attn  # noqa: F401
    _HAS_FLASH_ATTN = True
except ImportError:
    _HAS_FLASH_ATTN = False


def _patch_llama_inputs_docstring():
    """Patch missing LLAMA_INPUTS_DOCSTRING for transformers>=5.x (used by QRM remote code)."""
    mod = importlib.import_module("transformers.models.llama.modeling_llama")
    if not hasattr(mod, "LLAMA_INPUTS_DOCSTRING"):
        mod.LLAMA_INPUTS_DOCSTRING = ""


_patch_llama_inputs_docstring()


REWARD_MODELS = {
    # Skywork
    # Link: https://huggingface.co/Skywork/Skywork-Reward-Gemma-2-27B
    "skywork_llama_3b": "Skywork/Skywork-Reward-V2-Llama-3.2-3B",
    "skywork_llama_8b": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
    "skywork_qwen_4b": "Skywork/Skywork-Reward-V2-Qwen3-4B",
    "skywork_qwen_8b": "Skywork/Skywork-Reward-V2-Qwen3-8B",
    "skywork_gemma_27b": "Skywork/Skywork-Reward-Gemma-2-27B",
    # QRM
    # Link: https://huggingface.co/nicolinho/QRM-Llama3.1-8B-v2
    "qrm_llama_8b": "nicolinho/QRM-Llama3.1-8B-v2",
    "qrm_gemma_27b": "nicolinho/QRM-Gemma-2-27B",
    # Ai2
    # Link: https://huggingface.co/allenai/Llama-3.1-70B-Instruct-RM-RB2
    "ai2_llama_8b_base": "allenai/Llama-3.1-8B-Base-RM-RB2",
    "ai2_llama_8b": "allenai/Llama-3.1-8B-Instruct-RM-RB2",
    "ai2_llama_70b": "allenai/Llama-3.1-70B-Instruct-RM-RB2",
}


class RewardModel:
    """Base class for reward models.

    Args:
        model_name: Key in REWARD_MODELS or a HuggingFace model ID.
        device: Device string (e.g. "cuda:0").
    """
    def __init__(self, model_name: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.model_id = REWARD_MODELS.get(model_name, model_name)
        self.model = None
        self.tokenizer = None

    @property
    def input_device(self):
        """Resolve the device for model inputs (handles device_map="auto")."""
        if self.device == "auto":
            return next(self.model.parameters()).device
        return self.device

    def load(self):
        """Load model and tokenizer onto device."""
        raise NotImplementedError

    def score(self, conversation: list[dict]) -> float:
        """Score a single conversation.

        Args:
            conversation: List of dicts with "role" and "content" keys.

        Returns:
            Scalar reward score (float).
        """
        raise NotImplementedError

    def score_batch(self, conversations: list[list[dict]], sequential: bool = True) -> list[float]:
        """Score a batch of conversations.

        Args:
            conversations: List of conversations, each a list of dicts with "role" and "content" keys.
            sequential: If True, score each conversation individually. If False, pad and batch.

        Returns:
            List of scalar reward scores (list[float]).
        """
        if sequential:
            return [self.score(c) for c in conversations]
        raise NotImplementedError

    def _tokenize(self, conversation):
        """Apply chat template and tokenize a single conversation.

        Args:
            conversation: List of dicts with "role" and "content" keys.

        Returns:
            Tokenized inputs on self.device.
        """
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        return self.tokenizer(text, return_tensors="pt").to(self.input_device)

    def _tokenize_batch(self, conversations):
        """Apply chat template and tokenize a batch of conversations with right padding.

        Args:
            conversations: List of conversations.

        Returns:
            Tokenized inputs on self.device with right padding and attention mask.
        """
        texts = [self.tokenizer.apply_chat_template(c, tokenize=False) for c in conversations]
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.input_device)
        self.tokenizer.padding_side = orig_padding_side
        return inputs


class SkyworkRewardModel(RewardModel):
    """Skywork reward models (AutoModelForSequenceClassification, num_labels=1)."""
    def load(self):
        attention_implementation = "flash_attention_2" if (_HAS_FLASH_ATTN and "cuda" in self.device) else "sdpa"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation=attention_implementation,
            num_labels=1,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

    def _tokenize(self, conversation):
        text = self.tokenizer.apply_chat_template(conversation, tokenize=False)
        return self.tokenizer(text, return_tensors="pt").to(self.input_device)

    def score(self, conversation):
        inputs = self._tokenize(conversation)
        with torch.no_grad():
            return self.model(**inputs).logits[0][0].item()

    def score_batch(self, conversations, sequential=True):
        if sequential:
            return super().score_batch(conversations, sequential=True)
        inputs = self._tokenize_batch(conversations)
        with torch.no_grad():
            return self.model(**inputs).logits[:, 0].cpu().float().tolist()


class QRMRewardModel(RewardModel):
    """QRM reward models by nicolinho (trust_remote_code, distributional rewards)."""
    def load(self):
        # Suppress load report and unsharded-layer warnings
        loggers = [
            logging.getLogger("transformers.modeling_utils"),
            logging.getLogger("transformers.integrations.tensor_parallel"),
        ]
        prev_levels = [l.level for l in loggers]
        for l in loggers:
            l.setLevel(logging.ERROR)

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )

        # Move custom layers (not in accelerate's shard map) to same device as final norm
        last_device = self.model.model.norm.weight.device
        self.model.regression_layer.to(last_device)
        self.model.gating.to(last_device)

        # Restore logging levels
        for l, prev in zip(loggers, prev_levels):
            l.setLevel(prev)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def score(self, conversation):
        input_ids = self.tokenizer.apply_chat_template(conversation, return_tensors="pt", tokenize=True)["input_ids"]
        input_ids = input_ids.to(self.input_device)
        with torch.no_grad():
            return self.model(input_ids).score.cpu().float().item()

    def score_batch(self, conversations, sequential=True):
        if sequential:
            return super().score_batch(conversations, sequential=True)
        input_ids_list = [
            self.tokenizer.apply_chat_template(c, return_tensors="pt", tokenize=True)["input_ids"].squeeze(0)
            for c in conversations
        ]
        max_len = max(t.size(0) for t in input_ids_list)
        pad_id = self.model.config.pad_token_id
        # Right-pad: QRM forward uses argmax on pad_token_id to find sequence lengths
        input_ids = torch.stack([
            torch.nn.functional.pad(t, (0, max_len - t.size(0)), value=pad_id) for t in input_ids_list
        ]).to(self.input_device)
        attention_mask = torch.stack([
            torch.nn.functional.pad(torch.ones(t.size(0), dtype=torch.long), (0, max_len - t.size(0)))
            for t in input_ids_list
        ]).to(self.input_device)
        with torch.no_grad():
            return self.model(input_ids, attention_mask=attention_mask).score.squeeze(-1).cpu().float().tolist()


class Ai2RewardModel(RewardModel):
    """Ai2 reward models (vanilla AutoModelForSequenceClassification)."""
    def load(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def score(self, conversation):
        inputs = self._tokenize(conversation)
        with torch.no_grad():
            return self.model(**inputs).logits[0][0].item()

    def score_batch(self, conversations, sequential=True):
        if sequential:
            return super().score_batch(conversations, sequential=True)
        inputs = self._tokenize_batch(conversations)
        with torch.no_grad():
            return self.model(**inputs).logits[:, 0].cpu().float().tolist()
