"""Tests for scripts/characters/generate_characters.py."""

import importlib
import os
import sys
from unittest.mock import MagicMock

import pytest
import torch


###########
# IMPORTS #
###########

# generate_characters.py runs Hydra setup at import. Mock heavy/optional
# dependencies so the pure helpers can be exercised in isolation. torch and
# numpy are kept REAL so _pad_left can be tested end-to-end.

_mock_dialecttax = MagicMock()
_mock_dialecttax.utils.load_config.return_value = {
    "directories": {"experiments": "/tmp/experiments", "preprocessed": "/tmp", "datasets": "/tmp"},
    "keys": {},
}

_mock_redial = MagicMock()
_mock_redial.TASKS = ["algorithm", "logic", "math", "planning"]
_mock_redial.FILE_NAME_QA_FORMAT = "{task}_{dialect}_qa.jsonl"
_mock_redial.FILE_NAME_FORMAT = "{task}_{dialect}.jsonl"
_mock_redial.DIRECTORY_NAME = "ReDial"

_mock_dialecttax.data.DATASET_MODULES = {"redial": _mock_redial}
_mock_dialecttax.logits = MagicMock()

_mock_mqa_grader = MagicMock()
_mock_dialecttax.data.graders = MagicMock()
_mock_dialecttax.data.graders.mqa = _mock_mqa_grader

_MOCK_MODULES = {
    "dialecttax": _mock_dialecttax,
    "dialecttax.utils": _mock_dialecttax.utils,
    "dialecttax.characters": _mock_dialecttax.characters,
    "dialecttax.data": _mock_dialecttax.data,
    "dialecttax.data.graders": _mock_dialecttax.data.graders,
    "dialecttax.data.graders.mqa": _mock_mqa_grader,
    "dialecttax.models": _mock_dialecttax.models,
    "dialecttax.prompts": _mock_dialecttax.prompts,
    "dialecttax.logits": _mock_dialecttax.logits,
    "hydra": MagicMock(),
    "hydra.core": MagicMock(),
    "hydra.core.hydra_config": MagicMock(),
    "omegaconf": MagicMock(),
}

_saved_modules = {}
for _name, _mock in _MOCK_MODULES.items():
    _saved_modules[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "scripts", "characters", "generate_characters.py",
    )
)
_spec = importlib.util.spec_from_file_location("generate_characters", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name, _orig in _saved_modules.items():
    if _orig is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _orig

_find_answer_step = _mod._find_answer_step
_pad_left = _mod._pad_left


#####################
# _FIND_ANSWER_STEP #
#####################


class _FakeTokenizer:
    """Mock tokenizer that maps token IDs to predetermined strings."""

    def __init__(self, vocab):
        self._vocab = vocab

    def decode(self, ids, **kwargs):
        return "".join(self._vocab[i] for i in ids)


class TestFindAnswerStep:
    """Test _find_answer_step (binary-search variant) locates the step that completes '#### <answer>'."""

    ###############
    # BASIC CASES #
    ###############

    def test_clean_answer(self):
        """'#### B' as separate tokens -> step of 'B'."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "B"})
        assert _find_answer_step("B", [0, 1, 2], tok) == 2

    def test_clean_answer_no_space(self):
        """'####B' without space -> step of 'B'."""
        tok = _FakeTokenizer({0: "####", 1: "B"})
        assert _find_answer_step("B", [0, 1], tok) == 1

    def test_answer_with_trailing_tokens(self):
        """'#### B is correct' -> step of first B."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "B", 3: " is", 4: " correct"})
        assert _find_answer_step("B", [0, 1, 2, 3, 4], tok) == 2

    ##########################
    # SPURIOUS BEFORE MARKER #
    ##########################

    def test_b_before_marker(self):
        """'B #### B' -> finds B after ####, not before."""
        tok = _FakeTokenizer({0: "B", 1: " ", 2: "####", 3: " ", 4: "B"})
        assert _find_answer_step("B", [0, 1, 2, 3, 4], tok) == 4

    def test_b_before_marker_repeated(self):
        """'B #### B B' -> finds first B after ####."""
        tok = _FakeTokenizer({0: "B", 1: " ", 2: "####", 3: " ", 4: "B", 5: " ", 6: "B"})
        assert _find_answer_step("B", [0, 1, 2, 3, 4, 5, 6], tok) == 4

    def test_multiple_b_after_marker(self):
        """'#### B B' -> finds first B after ####."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "B", 3: " ", 4: "B"})
        assert _find_answer_step("B", [0, 1, 2, 3, 4], tok) == 2

    ################
    # SPLIT TOKENS #
    ################

    def test_hashes_split_across_tokens(self):
        """'##' + '##' + ' B' -> finds B after split ####."""
        tok = _FakeTokenizer({0: "##", 1: "##", 2: " B"})
        assert _find_answer_step("B", [0, 1, 2], tok) == 2

    def test_marker_and_answer_same_token(self):
        """'#### B' as a single token -> step 0."""
        tok = _FakeTokenizer({0: "#### B"})
        assert _find_answer_step("B", [0], tok) == 0

    def test_marker_answer_fused_no_space(self):
        """'####B' as a single token -> step 0."""
        tok = _FakeTokenizer({0: "####B"})
        assert _find_answer_step("B", [0], tok) == 0

    def test_b_fused_before_hashes(self):
        """'B####' as single token + ' B' -> finds B in second token."""
        tok = _FakeTokenizer({0: "B####", 1: " B"})
        assert _find_answer_step("B", [0, 1], tok) == 1

    def test_b_fused_before_hashes_no_later_b(self):
        """'B####' with no B after -> returns None (fallback finds B in first token)."""
        tok = _FakeTokenizer({0: "B####", 1: " C"})
        assert _find_answer_step("B", [0, 1], tok) == 0

    ###################
    # NONE OR MISSING #
    ###################

    def test_predicted_none(self):
        """predicted=None -> returns None immediately."""
        tok = _FakeTokenizer({0: "####", 1: " B"})
        assert _find_answer_step(None, [0, 1], tok) is None

    def test_no_marker_falls_back_to_last_word(self):
        """No #### -> fallback to last standalone <answer> match."""
        tok = _FakeTokenizer({0: "The", 1: " answer", 2: " is", 3: " B"})
        assert _find_answer_step("B", [0, 1, 2, 3], tok) == 3

    def test_empty_gen_ids(self):
        """Empty generation -> returns None."""
        tok = _FakeTokenizer({})
        assert _find_answer_step("B", [], tok) is None

    def test_marker_only_no_answer(self):
        """'####' with no answer token after -> returns None."""
        tok = _FakeTokenizer({0: "####"})
        assert _find_answer_step("B", [0], tok) is None

    def test_marker_with_wrong_answer(self):
        """'#### C' but predicted='B' (no fallback B in text) -> returns None."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "C"})
        assert _find_answer_step("B", [0, 1, 2], tok) is None

    ####################
    # MULTI-CHAR ANSWERS #
    ####################

    def test_numeric_answer(self):
        """'#### 42' -> finds step of '42'."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "42"})
        assert _find_answer_step("42", [0, 1, 2], tok) == 2

    def test_numeric_split_across_tokens(self):
        """'#### 4' + '2' where predicted='42' -> step that completes '42'."""
        tok = _FakeTokenizer({0: "####", 1: " 4", 2: "2"})
        assert _find_answer_step("42", [0, 1, 2], tok) == 2

    ##################
    # REALISTIC CASES #
    ##################

    def test_cot_long_reasoning_then_answer(self):
        """Long CoT followed by '#### A' -> finds A at the end."""
        tok = _FakeTokenizer({
            0: "Let", 1: " me", 2: " think", 3: ".", 4: " The",
            5: " answer", 6: " is", 7: " A", 8: ".", 9: " ",
            10: "####", 11: " ", 12: "A",
        })
        assert _find_answer_step("A", list(range(13)), tok) == 12

    def test_cot_answer_letter_in_reasoning(self):
        """CoT mentions 'A' in reasoning, then '#### A' -> finds A after ####."""
        tok = _FakeTokenizer({
            0: "Option", 1: " A", 2: " is", 3: " correct", 4: " because",
            5: " ", 6: "####", 7: " ", 8: "A",
        })
        assert _find_answer_step("A", list(range(9)), tok) == 8

    ###############
    # REGEX SAFETY #
    ###############

    def test_answer_with_regex_special_chars(self):
        """Answer containing regex special char (e.g. '$5') is escaped properly."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "$5"})
        assert _find_answer_step("$5", [0, 1, 2], tok) == 2

    def test_answer_with_parentheses(self):
        """Answer '(A)' is escaped properly."""
        tok = _FakeTokenizer({0: "####", 1: " ", 2: "(A)"})
        assert _find_answer_step("(A)", [0, 1, 2], tok) == 2

    ##################
    # BINARY-SEARCH  #
    ##################

    def test_single_token(self):
        """Length-1 gen_ids: lo=hi=0 path returns 0 when answer is in that token."""
        tok = _FakeTokenizer({0: "#### B"})
        assert _find_answer_step("B", [0], tok) == 0

    def test_answer_at_first_token(self):
        """Answer at the very start of the sequence (#### in token 0)."""
        tok = _FakeTokenizer({0: "#### B is", 1: " correct"})
        assert _find_answer_step("B", [0, 1], tok) == 0

    def test_long_sequence_answer_at_end(self):
        """50-token sequence; answer at the last token. Stress-tests binary search."""
        vocab = {i: f" w{i}" for i in range(50)}
        vocab[48] = "####"
        vocab[49] = " B"
        tok = _FakeTokenizer(vocab)
        assert _find_answer_step("B", list(range(50)), tok) == 49

    def test_long_sequence_answer_in_middle(self):
        """50-token sequence; answer mid-sequence."""
        vocab = {i: f" w{i}" for i in range(50)}
        vocab[20] = "####"
        vocab[21] = " B"
        tok = _FakeTokenizer(vocab)
        assert _find_answer_step("B", list(range(50)), tok) == 21

    def test_matches_linear_scan(self):
        """Binary search agrees with linear scan on a battery of random-ish inputs."""
        cases = [
            ({0: "####", 1: " ", 2: "B"}, "B", [0, 1, 2]),
            ({0: "##", 1: "##", 2: " B"}, "B", [0, 1, 2]),
            ({0: "Hi ", 1: "####", 2: " ", 3: "X"}, "X", [0, 1, 2, 3]),
            ({0: "#### A B C ", 1: "####", 2: " ", 3: "B"}, "B", [0, 1, 2, 3]),
        ]
        for vocab, pred, ids in cases:
            tok = _FakeTokenizer(vocab)
            full = tok.decode(ids)
            import re
            m = re.search(r"####\s*" + re.escape(pred), full, re.IGNORECASE)
            target = m.end() if m else None
            assert target is not None, f"setup error for case {vocab}"
            # Linear-scan reference.
            linear = next(
                (k for k in range(len(ids))
                 if len(tok.decode(ids[:k + 1])) >= target),
                None,
            )
            assert _find_answer_step(pred, ids, tok) == linear


#############
# _PAD_LEFT #
#############


class TestPadLeft:
    """Test _pad_left: left-pads variable-length id sequences for batched generate."""

    def test_uniform_lengths(self):
        """Equal-length inputs -> no padding."""
        ids, mask, real_lens, max_len = _pad_left([[1, 2, 3], [4, 5, 6]], pad_id=0, device="cpu")
        assert ids.shape == (2, 3)
        assert mask.shape == (2, 3)
        assert real_lens == [3, 3]
        assert max_len == 3
        assert torch.equal(ids, torch.tensor([[1, 2, 3], [4, 5, 6]]))
        assert torch.equal(mask, torch.ones(2, 3, dtype=torch.long))

    def test_left_padding_alignment(self):
        """Shorter sequences are padded on the LEFT (real tokens flush-right)."""
        ids, mask, real_lens, max_len = _pad_left([[1, 2], [3, 4, 5, 6]], pad_id=0, device="cpu")
        assert max_len == 4
        assert real_lens == [2, 4]
        # Row 0: 2 pad tokens then [1, 2].
        assert torch.equal(ids[0], torch.tensor([0, 0, 1, 2]))
        # Row 1: no padding.
        assert torch.equal(ids[1], torch.tensor([3, 4, 5, 6]))
        # Mask zeros out pad positions.
        assert torch.equal(mask[0], torch.tensor([0, 0, 1, 1]))
        assert torch.equal(mask[1], torch.tensor([1, 1, 1, 1]))

    def test_pad_id_used(self):
        """Padding uses the supplied pad_id, not zero."""
        ids, _, _, _ = _pad_left([[1, 2], [3, 4, 5]], pad_id=99, device="cpu")
        # Row 0 has 1 pad slot (max_len=3, real_len=2): [99, 1, 2].
        assert ids[0, 0].item() == 99
        assert ids[0, 1].item() == 1
        assert ids[0, 2].item() == 2
        assert torch.equal(ids[1], torch.tensor([3, 4, 5]))

    def test_singleton_batch(self):
        """Batch of 1 returns shapes (1, T) with mask all-ones."""
        ids, mask, real_lens, max_len = _pad_left([[7, 8, 9]], pad_id=0, device="cpu")
        assert ids.shape == (1, 3)
        assert mask.shape == (1, 3)
        assert real_lens == [3]
        assert max_len == 3
        assert torch.equal(mask, torch.ones(1, 3, dtype=torch.long))

    def test_dtypes(self):
        """input_ids and attention_mask are int64."""
        ids, mask, _, _ = _pad_left([[1, 2], [3]], pad_id=0, device="cpu")
        assert ids.dtype == torch.long
        assert mask.dtype == torch.long

    def test_real_lens_match_inputs(self):
        """real_lens reflects pre-padding lengths."""
        _, _, real_lens, _ = _pad_left([[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]], pad_id=0, device="cpu")
        assert real_lens == [1, 2, 3, 4]
