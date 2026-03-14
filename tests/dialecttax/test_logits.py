"""Tests for dialecttax.logits."""

import math
from unittest.mock import MagicMock

import numpy as np
import torch
import torch.nn.functional as F

from dialecttax.logits import (
    compute_entropy,
    compute_generation_metrics,
    compute_input_metrics,
    compute_log_probs,
    compute_run_metrics,
    compute_sample_metrics,
)


####################
# COMPUTE_LOG_PROBS #
####################

class TestComputeLogProbs:
    def test_uniform_logits(self):
        """Uniform logits ⇒ log(1/V) for every token."""
        vocab_size = 4
        seq_len = 3
        logits = torch.zeros(seq_len, vocab_size)
        token_ids = torch.tensor([0, 1, 2])

        result = compute_log_probs(logits, token_ids)

        expected = math.log(1.0 / vocab_size)
        assert result.shape == (seq_len,)
        torch.testing.assert_close(result, torch.full((seq_len,), expected))

    def test_one_hot_logits(self):
        """When logits strongly favor one token, that token gets ~0 log-prob."""
        logits = torch.tensor([[100.0, 0.0, 0.0],
                               [0.0, 100.0, 0.0]])
        token_ids = torch.tensor([0, 1])

        result = compute_log_probs(logits, token_ids)

        assert result.shape == (2,)
        assert result[0].item() > -1e-3  # close to 0
        assert result[1].item() > -1e-3

    def test_one_hot_logits_wrong_token(self):
        """Selecting the wrong token under a peaked distribution ⇒ very negative."""
        logits = torch.tensor([[100.0, 0.0, 0.0]])
        token_ids = torch.tensor([1])  # wrong token

        result = compute_log_probs(logits, token_ids)
        assert result[0].item() < -50.0

    def test_single_token(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        token_ids = torch.tensor([2])

        result = compute_log_probs(logits, token_ids)
        expected = F.log_softmax(logits, dim=-1)[0, 2]

        assert result.shape == (1,)
        torch.testing.assert_close(result[0], expected)

    def test_all_negative(self):
        """Log-probs are valid even when all logits are negative."""
        logits = torch.tensor([[-5.0, -3.0, -1.0]])
        token_ids = torch.tensor([2])

        result = compute_log_probs(logits, token_ids)
        assert result[0].item() < 0.0  # log-probs are always ≤ 0
        assert torch.isfinite(result).all()


###################
# COMPUTE_ENTROPY #
###################

class TestComputeEntropy:
    def test_uniform_distribution(self):
        """Uniform over V tokens ⇒ entropy = ln(V)."""
        vocab_size = 8
        logits = torch.zeros(1, vocab_size)

        result = compute_entropy(logits)

        expected = math.log(vocab_size)
        assert result.shape == (1,)
        assert abs(result[0].item() - expected) < 1e-5

    def test_peaked_distribution(self):
        """Peaked distribution ⇒ entropy near 0."""
        logits = torch.tensor([[100.0, 0.0, 0.0, 0.0]])

        result = compute_entropy(logits)

        assert result.shape == (1,)
        assert result[0].item() < 1e-3

    def test_entropy_nonnegative(self):
        """Entropy must be non-negative for any input."""
        torch.manual_seed(0)
        logits = torch.randn(10, 50)

        result = compute_entropy(logits)

        assert (result >= -1e-6).all()  # small tolerance for float

    def test_multiple_positions(self):
        """Returns one entropy value per position."""
        logits = torch.zeros(5, 4)

        result = compute_entropy(logits)

        assert result.shape == (5,)

    def test_entropy_increases_with_vocab(self):
        """Uniform entropy should increase with vocabulary size."""
        e_small = compute_entropy(torch.zeros(1, 4))[0].item()
        e_large = compute_entropy(torch.zeros(1, 64))[0].item()
        assert e_large > e_small


########################
# COMPUTE_INPUT_METRICS #
########################

class TestComputeInputMetrics:
    def _make_mock_model(self, logits: torch.Tensor):
        """Create a mock model returning the given logits."""
        model = MagicMock()
        output = MagicMock()
        output.logits = logits
        model.return_value = output
        return model

    def test_basic_single_sample(self):
        """Single sample, no padding."""
        batch, seq_len, vocab = 1, 4, 8
        logits = torch.randn(batch, seq_len, vocab)
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

        model = self._make_mock_model(logits)
        results = compute_input_metrics(model, input_ids, attention_mask)

        assert len(results) == 1
        assert results[0]["log_probs"].shape == (seq_len - 1,)
        assert results[0]["entropy"].shape == (seq_len - 1,)

    def test_batch_of_two(self):
        """Two samples in one batch."""
        batch, seq_len, vocab = 2, 5, 8
        logits = torch.randn(batch, seq_len, vocab)
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

        model = self._make_mock_model(logits)
        results = compute_input_metrics(model, input_ids, attention_mask)

        assert len(results) == 2
        for r in results:
            assert r["log_probs"].shape == (seq_len - 1,)
            assert r["entropy"].shape == (seq_len - 1,)

    def test_with_padding(self):
        """Padding tokens (mask=0) should be excluded."""
        batch, seq_len, vocab = 1, 6, 8
        logits = torch.randn(batch, seq_len, vocab)
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0]])  # 4 real tokens

        model = self._make_mock_model(logits)
        results = compute_input_metrics(model, input_ids, attention_mask)

        # With 4 real tokens, shifted metrics have length 3
        assert results[0]["log_probs"].shape == (3,)
        assert results[0]["entropy"].shape == (3,)

    def test_results_on_cpu(self):
        """Results should be on CPU regardless of input device."""
        batch, seq_len, vocab = 1, 3, 4
        logits = torch.randn(batch, seq_len, vocab)
        input_ids = torch.randint(0, vocab, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)

        model = self._make_mock_model(logits)
        results = compute_input_metrics(model, input_ids, attention_mask)

        assert results[0]["log_probs"].device == torch.device("cpu")
        assert results[0]["entropy"].device == torch.device("cpu")


##############################
# COMPUTE_GENERATION_METRICS #
##############################

class TestComputeGenerationMetrics:
    def test_basic(self):
        """Basic generation metrics without EOS truncation."""
        batch, gen_len, vocab = 1, 3, 8
        prompt_len = 2
        scores = tuple(torch.randn(batch, vocab) for _ in range(gen_len))
        generated_ids = torch.randint(0, vocab, (batch, prompt_len + gen_len))

        results = compute_generation_metrics(scores, generated_ids, prompt_len)

        assert len(results) == 1
        assert results[0]["log_probs"].shape == (gen_len,)
        assert results[0]["entropy"].shape == (gen_len,)

    def test_eos_truncation(self):
        """Metrics should be truncated at the first EOS token."""
        batch, gen_len, vocab = 1, 5, 8
        prompt_len = 2
        eos_id = 7

        scores = tuple(torch.randn(batch, vocab) for _ in range(gen_len))
        # Place EOS at generation position 2 (third generated token)
        gen_tokens = torch.tensor([3, 4, eos_id, 5, 6])
        prompt_tokens = torch.randint(0, vocab, (1, prompt_len))
        generated_ids = torch.cat([prompt_tokens, gen_tokens.unsqueeze(0)], dim=1)

        results = compute_generation_metrics(scores, generated_ids, prompt_len, eos_token_id=eos_id)

        # Should be truncated to 2 tokens (before EOS)
        assert results[0]["log_probs"].shape == (2,)
        assert results[0]["entropy"].shape == (2,)

    def test_eos_at_start(self):
        """EOS as first generated token ⇒ empty metrics."""
        batch, gen_len, vocab = 1, 3, 8
        prompt_len = 2
        eos_id = 7

        scores = tuple(torch.randn(batch, vocab) for _ in range(gen_len))
        gen_tokens = torch.tensor([eos_id, 1, 2])
        prompt_tokens = torch.randint(0, vocab, (1, prompt_len))
        generated_ids = torch.cat([prompt_tokens, gen_tokens.unsqueeze(0)], dim=1)

        results = compute_generation_metrics(scores, generated_ids, prompt_len, eos_token_id=eos_id)

        assert results[0]["log_probs"].shape == (0,)
        assert results[0]["entropy"].shape == (0,)

    def test_no_eos_in_sequence(self):
        """No EOS present ⇒ all generation steps kept."""
        batch, gen_len, vocab = 1, 4, 8
        prompt_len = 1
        eos_id = 7

        scores = tuple(torch.randn(batch, vocab) for _ in range(gen_len))
        generated_ids = torch.randint(0, 6, (batch, prompt_len + gen_len))  # no 7s

        results = compute_generation_metrics(scores, generated_ids, prompt_len, eos_token_id=eos_id)

        assert results[0]["log_probs"].shape == (gen_len,)

    def test_batch(self):
        """Multiple samples in a batch."""
        batch, gen_len, vocab = 3, 4, 8
        prompt_len = 2
        scores = tuple(torch.randn(batch, vocab) for _ in range(gen_len))
        generated_ids = torch.randint(0, vocab, (batch, prompt_len + gen_len))

        results = compute_generation_metrics(scores, generated_ids, prompt_len)

        assert len(results) == batch

    def test_results_on_cpu(self):
        batch, gen_len, vocab = 1, 2, 4
        prompt_len = 1
        scores = tuple(torch.randn(batch, vocab) for _ in range(gen_len))
        generated_ids = torch.randint(0, vocab, (batch, prompt_len + gen_len))

        results = compute_generation_metrics(scores, generated_ids, prompt_len)

        assert results[0]["log_probs"].device == torch.device("cpu")
        assert results[0]["entropy"].device == torch.device("cpu")


##########################
# COMPUTE_SAMPLE_METRICS #
##########################

class TestComputeSampleMetrics:
    def _make_grading_result(self, correct=True, extracted="42", gold="42"):
        return {"correct": correct, "extracted": extracted, "gold": gold}

    def _make_token_metrics(self, length, mean_lp=-1.0, mean_ent=0.5):
        return {
            "log_probs": torch.full((length,), mean_lp),
            "entropy": torch.full((length,), mean_ent),
        }

    def test_correct_sample(self):
        grading = self._make_grading_result(correct=True, extracted="42", gold="42")
        inp = self._make_token_metrics(10, mean_lp=-0.5, mean_ent=1.0)
        gen = self._make_token_metrics(5, mean_lp=-2.0, mean_ent=0.3)

        result = compute_sample_metrics(grading, inp, gen, "The answer is 42.")

        assert result["correct"] is True
        assert result["extracted"] == "42"
        assert result["gold"] == "42"
        assert result["completion"] == "The answer is 42."
        assert result["response_length"] == 5
        assert result["input_length"] == 10
        assert result["accepted"] is True
        assert abs(result["input_mean_log_prob"] - (-0.5)) < 1e-5
        assert abs(result["input_mean_entropy"] - 1.0) < 1e-5
        assert abs(result["gen_mean_log_prob"] - (-2.0)) < 1e-5
        assert abs(result["gen_mean_entropy"] - 0.3) < 1e-5
        # Per-token arrays
        assert len(result["input_log_probs"]) == 10
        assert len(result["input_entropy"]) == 10
        assert len(result["gen_log_probs"]) == 5
        assert len(result["gen_entropy"]) == 5
        torch.testing.assert_close(
            torch.tensor(result["input_log_probs"]), torch.full((10,), -0.5),
        )
        torch.testing.assert_close(
            torch.tensor(result["gen_entropy"]), torch.full((5,), 0.3),
        )

    def test_incorrect_sample(self):
        grading = self._make_grading_result(correct=False, extracted="7", gold="42")
        inp = self._make_token_metrics(3)
        gen = self._make_token_metrics(4)

        result = compute_sample_metrics(grading, inp, gen, "The answer is 7.")

        assert result["correct"] is False
        assert result["accepted"] is True

    def test_unextracted_answer(self):
        """When extracted is None, accepted should be False."""
        grading = self._make_grading_result(correct=False, extracted=None, gold="42")
        inp = self._make_token_metrics(3)
        gen = self._make_token_metrics(4)

        result = compute_sample_metrics(grading, inp, gen, "I don't know.")

        assert result["accepted"] is False
        assert result["extracted"] is None

    def test_empty_generation(self):
        """Zero-length generation ⇒ gen metrics default to 0."""
        grading = self._make_grading_result()
        inp = self._make_token_metrics(5)
        gen = self._make_token_metrics(0)

        result = compute_sample_metrics(grading, inp, gen, "")

        assert result["response_length"] == 0
        assert result["gen_mean_log_prob"] == 0.0
        assert result["gen_mean_entropy"] == 0.0
        assert result["gen_log_probs"] == []
        assert result["gen_entropy"] == []

    def test_empty_input(self):
        """Zero-length input ⇒ input metrics default to 0."""
        grading = self._make_grading_result()
        inp = self._make_token_metrics(0)
        gen = self._make_token_metrics(5)

        result = compute_sample_metrics(grading, inp, gen, "answer")

        assert result["input_length"] == 0
        assert result["input_mean_log_prob"] == 0.0
        assert result["input_mean_entropy"] == 0.0
        assert result["input_log_probs"] == []
        assert result["input_entropy"] == []


#######################
# COMPUTE_RUN_METRICS #
#######################

class TestComputeRunMetrics:
    def _make_sample(
        self,
        correct=True,
        accepted=True,
        response_length=10,
        input_length=5,
        gen_mean_log_prob=-1.0,
        gen_mean_entropy=0.5,
        input_mean_log_prob=-0.5,
        input_mean_entropy=1.0,
    ):
        return {
            "correct": correct,
            "accepted": accepted,
            "response_length": response_length,
            "input_length": input_length,
            "gen_mean_log_prob": gen_mean_log_prob,
            "gen_mean_entropy": gen_mean_entropy,
            "input_mean_log_prob": input_mean_log_prob,
            "input_mean_entropy": input_mean_entropy,
        }

    def test_empty_list(self):
        assert compute_run_metrics([]) == {}

    def test_single_sample(self):
        samples = [self._make_sample(correct=True, response_length=10)]
        result = compute_run_metrics(samples)

        assert result["n_samples"] == 1
        assert result["accuracy"] == 1.0
        assert result["acceptance_rate"] == 1.0
        assert result["mean_response_length"] == 10.0

    def test_mixed_accuracy(self):
        samples = [
            self._make_sample(correct=True),
            self._make_sample(correct=False),
            self._make_sample(correct=True),
            self._make_sample(correct=False),
        ]
        result = compute_run_metrics(samples)

        assert result["n_samples"] == 4
        assert abs(result["accuracy"] - 0.5) < 1e-7

    def test_mixed_acceptance(self):
        samples = [
            self._make_sample(accepted=True),
            self._make_sample(accepted=False),
        ]
        result = compute_run_metrics(samples)

        assert abs(result["acceptance_rate"] - 0.5) < 1e-7

    def test_aggregate_averages(self):
        samples = [
            self._make_sample(
                response_length=10,
                input_length=4,
                gen_mean_log_prob=-1.0,
                gen_mean_entropy=0.5,
                input_mean_log_prob=-0.3,
                input_mean_entropy=1.2,
            ),
            self._make_sample(
                response_length=20,
                input_length=6,
                gen_mean_log_prob=-3.0,
                gen_mean_entropy=1.5,
                input_mean_log_prob=-0.7,
                input_mean_entropy=0.8,
            ),
        ]
        result = compute_run_metrics(samples)

        assert abs(result["mean_response_length"] - 15.0) < 1e-7
        assert abs(result["mean_input_length"] - 5.0) < 1e-7
        assert abs(result["mean_gen_log_prob"] - (-2.0)) < 1e-7
        assert abs(result["mean_gen_entropy"] - 1.0) < 1e-7
        assert abs(result["mean_input_log_prob"] - (-0.5)) < 1e-7
        assert abs(result["mean_input_entropy"] - 1.0) < 1e-7

    def test_all_correct(self):
        samples = [self._make_sample(correct=True) for _ in range(5)]
        result = compute_run_metrics(samples)
        assert result["accuracy"] == 1.0

    def test_none_correct(self):
        samples = [self._make_sample(correct=False) for _ in range(5)]
        result = compute_run_metrics(samples)
        assert result["accuracy"] == 0.0
