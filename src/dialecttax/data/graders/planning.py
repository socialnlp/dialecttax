"""
Grading utilities for planning questions.

Datasets: AsyncHow

Expects the model to produce answers in the format "#### {answer} seconds", where answer is a number.
"""

import re
from ast import literal_eval
from numbers import Number


def is_refusal(completion: str | None, extracted: str | None) -> bool:
    """Check if a completion is a model refusal."""
    return extracted is None and completion is not None and "I cannot " in completion


########
# DATA #
########

_TIME_PATTERN = re.compile(
    r"(\d+(?:[, ]+\d+)*(?:\.\d+)?)\s*"
    r"(years?|months?|weeks?|days?|hours?|hrs?|minutes?|mins?|seconds?|secs?|[dhmsw])"
    r"(?!\w)",  # word boundary: don't match "sand" from "3s and"
    re.IGNORECASE,
)


def extract_answer_fuzzy(text: str | None) -> int | None:
    """Extract the last contiguous time expression from free-form text.

    Scans for all ``<number> <unit>`` matches, groups adjacent ones
    (separated only by whitespace, commas, or "and"), and converts
    the last group to total seconds.

    Handles malformed outputs like ``"60 120 seconds"`` (reads as
    60120 seconds) and multi-part expressions like
    ``"3 minutes 20 seconds"`` (sums to 200).

    Args:
        text: Raw model completion text.

    Returns:
        Total seconds as an int, or None if no time expression found.
    """
    if text is None:
        return None

    matches = list(_TIME_PATTERN.finditer(text))
    if not matches:
        return None

    # Group consecutive matches separated by connectors only
    groups: list[list[re.Match]] = [[matches[0]]]
    for prev, curr in zip(matches, matches[1:]):
        gap = text[prev.end():curr.start()]
        if re.fullmatch(r"[\s,]*(?:and\s*)?", gap, re.IGNORECASE):
            groups[-1].append(curr)
        else:
            groups.append([curr])

    # Convert the last group to seconds
    last = groups[-1]
    span = text[last[0].start():last[-1].end()]
    result = time_str_to_int(span)
    return result if isinstance(result, int) else None


def extract_answer(text: str | None) -> str | None:
    """Extract the time answer from '#### {answer} second(s)'."""
    if text is None:
        return None

    match = re.search(r"####\s*(.+?)\s*seconds?", text)
    if not match:
        return None
    return match.group(1).strip()


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, removes commas, and converts to lowercase.
    """
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("_", "")
    answer = answer.lower()
    return answer


########
# TIME #
########

def time_str_to_int(answer: str) -> int | str:
    """Convert a time string like '12 hours 10 minutes' to seconds."""
    _UNIT_TO_SECS = {
        "year": 31536000, "month": 2592000, "week": 604800,
        "day": 86400, "d": 86400,
        "hour": 3600, "hr": 3600, "h": 3600,
        "minute": 60, "min": 60, "m": 60,
        "second": 1, "sec": 1, "s": 1, "w": 604800,
    }
    total = 0
    for match in _TIME_PATTERN.finditer(answer):
        value = float(match.group(1).replace(",", "").replace(" ", ""))
        unit = match.group(2).lower()
        unit = unit.rstrip("s") if len(unit) > 1 else unit
        total += value * _UNIT_TO_SECS[unit]
    return int(total) if total > 0 else answer


#########
# GRADE #
#########

def grade(predicted: int | None, gold: list[int]) -> bool:
    """Check whether the predicted answer matches the gold answer."""
    if predicted is None:
        return False

    if isinstance(predicted, Number):
        return gold[0] <= predicted <= gold[1]
    return False


def grade_completions(completions: str | list[str], gold_answers: str | list[str]) -> list[dict]:
    """Grade a list of completions against gold answers.

    Returns a list of dicts with keys: completion, extracted, gold, correct.
    """
    if isinstance(completions, str):
        assert isinstance(gold_answers, str)
        completions = [completions]
        gold_answers = [gold_answers]

    results = []
    for completion, gold in zip(completions, gold_answers):
        gold = literal_eval(gold)
        extracted = extract_answer(completion)
        result = {
            "completion": completion,
            "extracted": extracted,
            "refusal": is_refusal(completion, extracted),
            "gold": gold,
        }

        # Deal with non-compliant extractions
        predicted = normalize_answer(extracted)
        try:
            predicted = int(predicted)
            result["correct"] = grade(predicted, gold)
            result["delta"] = min(abs(predicted - gold[0]), abs(predicted - gold[1]))
        except ValueError:
            result["correct"] = False
            predicted = extract_answer_fuzzy(completion)
            if isinstance(predicted, int):
                result["calculated"] = predicted
                result["fuzzy"] = grade(predicted, gold)
                result["delta"] = min(abs(predicted - gold[0]), abs(predicted - gold[1]))

        results.append(result)
    return results
