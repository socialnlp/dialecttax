"""Tests for scripts/logits/generate_logits_mca.py.

Covers the pure helpers — continuation-id probe, fallback teacher-forcing,
and input-text assembly — without requiring a real language model.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest


###########
# IMPORTS #
###########

# The script does module-level Hydra setup; mock the heavy dependencies so
# its pure helpers can be imported. numpy / torch are left real.

_mock_dialecttax = MagicMock()
_mock_dialecttax.utils.load_config.return_value = {
    "directories": {"experiments": "/tmp/experiments", "preprocessed": "/tmp", "datasets": "/tmp"},
    "keys": {},
}
_mock_dialecttax.prompts.INSTS_MQA = {
    "math": {"naive": {"sae": "Pick one.\n{choices}", "aave": "Pick one.\n{choices}"}}
}
_mock_dialecttax.prompts.PROMPTS = {"math": {"naive": {"sae": "{problem}", "aave": "{problem}"}}}
_mock_dialecttax.prompts.FORMAT_PROMPTS_REGISTRY = {"math": MagicMock()}
_mock_dialecttax.prompts.get_prompt = lambda body, instructions="": f"{instructions}\n\n{body}"
_mock_dialecttax.prompts.get_system_prompt = lambda dialect, **kwargs: f"System: {dialect}"
_mock_dialecttax.models.get_message = lambda user, system: [
    {"role": "system", "content": system},
    {"role": "user", "content": user},
]

_MOCK_MODULES = {
    "dialecttax": _mock_dialecttax,
    "dialecttax.utils": _mock_dialecttax.utils,
    "dialecttax.data": _mock_dialecttax.data,
    "dialecttax.models": _mock_dialecttax.models,
    "dialecttax.prompts": _mock_dialecttax.prompts,
    "dialecttax.logits": _mock_dialecttax.logits,
    "hydra": MagicMock(),
    "hydra.core": MagicMock(),
    "hydra.core.hydra_config": MagicMock(),
    "omegaconf": MagicMock(),
}

_saved = {}
for _name, _mock in _MOCK_MODULES.items():
    _saved[_name] = sys.modules.get(_name)
    sys.modules[_name] = _mock

_script_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..", "scripts", "logits", "generate_logits_mca.py",
    )
)
_spec = importlib.util.spec_from_file_location("generate_logits_mca", _script_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name, _orig in _saved.items():
    if _orig is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _orig

_letter_continuation_ids = _mod._letter_continuation_ids
_format_input_text = _mod._format_input_text
_align_template_to_generation = _mod._align_template_to_generation
ANSWER_PREFIX = _mod.ANSWER_PREFIX
SENTINEL_TOKEN = _mod.SENTINEL_TOKEN


####################################
# TESTS: _letter_continuation_ids  #
####################################

class _FakeTokenizer:
    """Minimal tokenizer that returns prescribed token-id sequences for
    specified strings. Any unrecognised input falls back to a char-code
    encoding so the strict-prefix logic can still be exercised."""

    def __init__(self, table):
        self._table = table

    def encode(self, text, add_special_tokens=False):
        if text in self._table:
            return list(self._table[text])
        # Fallback: one id per char. Not semantically meaningful but deterministic.
        return [ord(c) for c in text]


class TestLetterContinuationIds:
    def test_clean_single_token_tail(self):
        """The common case: tokenize(prefix) is a strict prefix of
        tokenize(prefix + ' A')."""
        tok = _FakeTokenizer({
            "some prefix ####":      [10, 20, 30, 827],
            "some prefix #### A":    [10, 20, 30, 827, 362],
            "some prefix #### B":    [10, 20, 30, 827, 426],
            "some prefix #### C":    [10, 20, 30, 827, 356],
        })
        cont, clean = _letter_continuation_ids(tok, "some prefix ####", ["A", "B", "C"])
        assert clean is True
        assert cont == {"A": [362], "B": [426], "C": [356]}

    def test_clean_multi_token_tail(self):
        """Some tokenizers split ' A' as two tokens — strict-prefix still holds."""
        tok = _FakeTokenizer({
            "some prefix ####":   [10, 20, 30, 827],
            "some prefix #### A": [10, 20, 30, 827, 220, 32],  # [' ', 'A']
            "some prefix #### B": [10, 20, 30, 827, 220, 33],
        })
        cont, clean = _letter_continuation_ids(tok, "some prefix ####", ["A", "B"])
        assert clean is True
        assert cont == {"A": [220, 32], "B": [220, 33]}

    def test_boundary_reshuffle_marked_unclean(self):
        """If the tokenizer's encoding of prefix is NOT a strict prefix of
        the encoding of prefix+' L', we flag the letter for fallback."""
        tok = _FakeTokenizer({
            "prefix####":     [1, 2, 900],         # "####" is its own token
            "prefix#### A":   [1, 2, 901, 362],    # "####" got re-tokenised
        })
        cont, clean = _letter_continuation_ids(tok, "prefix####", ["A"])
        assert clean is False
        assert cont["A"] is None

    def test_handles_multiple_letters_independently(self):
        """One letter being reshuffled doesn't corrupt the others."""
        tok = _FakeTokenizer({
            "pref####":     [1, 2, 100],
            "pref#### A":   [1, 2, 100, 362],             # clean
            "pref#### B":   [1, 2, 100, 220, 33],         # clean (two tokens)
            "pref#### C":   [1, 2, 101, 362],             # reshuffled
        })
        cont, clean = _letter_continuation_ids(tok, "pref####", ["A", "B", "C"])
        assert clean is False  # any letter being unclean flips the global flag
        assert cont["A"] == [362]
        assert cont["B"] == [220, 33]
        assert cont["C"] is None


#############################
# TESTS: _format_input_text #
#############################

class TestFormatInputText:
    def test_base_model_concatenates_system_and_prompt(self):
        sample = {"system": "You are a tutor.", "prompt": "What is 2+2?"}
        tokenizer = MagicMock()
        got = _format_input_text(sample, tokenizer, instruct=False)
        assert got == "You are a tutor.\n\nWhat is 2+2?\n####"

    def test_base_model_ends_exactly_with_answer_prefix(self):
        sample = {"system": "", "prompt": ""}
        tokenizer = MagicMock()
        got = _format_input_text(sample, tokenizer, instruct=False)
        assert got.endswith(ANSWER_PREFIX)

    def test_instruct_model_applies_chat_template_then_appends_prefix(self):
        sample = {"system": "sys", "prompt": "q"}
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "<chat>q</chat>"
        got = _format_input_text(sample, tokenizer, instruct=True)
        tokenizer.apply_chat_template.assert_called_once()
        _args, kwargs = tokenizer.apply_chat_template.call_args
        assert kwargs["tokenize"] is False
        assert kwargs["add_generation_prompt"] is True
        assert got == "<chat>q</chat>" + ANSWER_PREFIX

    def test_answer_prefix_is_four_hashes_no_trailing_space(self):
        # The letter's canonical token for Llama/Gemma/Qwen is " A" (merged
        # with its leading space). Teacher-forcing "#### " (trailing space)
        # would make us look up P(' A' | context ending in ' ') — double-space
        # semantics. "####" without trailing space is the correct choice.
        assert ANSWER_PREFIX == "####"


###########################################
# TESTS: CoT template-generation alignment #
###########################################

# Token-id legend used below:
#   body: 10, 11, 12, 13    "prompt body"
#   reasoning: 90, 91, 92   "some chain of thought"
#   ####: 827               "#### anchor"
#   " A": 362               "answer letter"
#   sentinel: 999999        (never appears in generation)

SENTINEL = 999999
HASH = 827


class TestAlignTemplateToGeneration:
    def test_naive_alignment_when_no_reasoning(self):
        """Generation matches the template exactly up through "####", then
        emits the real answer token. The sentinel's aligned position is the
        generation index where the answer token sits."""
        template = [10, 11, 12, 13, HASH, SENTINEL]
        gen = [10, 11, 12, 13, HASH, 362]
        m = _align_template_to_generation(template, gen)
        # Body + #### match one-for-one
        assert m[0] == 0 and m[1] == 1 and m[2] == 2 and m[3] == 3
        assert m[4] == 4                 # ####
        assert m[5] == 5                 # sentinel → answer-letter position

    def test_cot_alignment_with_reasoning_insertions(self):
        """Generation has reasoning tokens between body and "####"."""
        template = [10, 11, HASH, SENTINEL]
        gen = [10, 11, 90, 91, 92, HASH, 362]
        m = _align_template_to_generation(template, gen)
        assert m[0] == 0 and m[1] == 1
        assert m[2] == 5                 # #### skipped past reasoning
        assert m[3] == 6                 # sentinel → letter position

    def test_returns_none_when_anchor_never_appears(self):
        """No "####" in generation → alignment fails; caller falls back."""
        template = [10, 11, HASH, SENTINEL]
        gen = [10, 11, 90, 91, 92, 93]   # model wandered off
        m = _align_template_to_generation(template, gen)
        assert m is None

    def test_picks_first_anchor_match(self):
        """If the reasoning itself contains "####" (e.g. as a section
        heading), alignment stops at the first occurrence after the body
        prefix. That's a modelling choice — document it here."""
        template = [10, HASH, SENTINEL]
        gen = [10, 90, HASH, 91, HASH, 362]   # two "####"s in the generation
        m = _align_template_to_generation(template, gen)
        assert m[1] == 2                 # first #### after body
        assert m[2] == 3                 # sentinel → token right after first ####

    def test_sentinel_position_past_end_of_generation(self):
        """#### is emitted as the very last token — the sentinel has no
        corresponding generation position, so it maps to len(gen)."""
        template = [10, HASH, SENTINEL]
        gen = [10, 90, HASH]
        m = _align_template_to_generation(template, gen)
        assert m[1] == 2                 # #### position
        assert m[2] == 3                 # past end; caller treats as fallback


class TestSentinelConstant:
    def test_sentinel_has_recognisable_form(self):
        # Angle-bracket-pipe form so it's picked up as a special token and
        # can't collide with any ordinary vocab entry.
        assert SENTINEL_TOKEN.startswith("<|") and SENTINEL_TOKEN.endswith("|>")
