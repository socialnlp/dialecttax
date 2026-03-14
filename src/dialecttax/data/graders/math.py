"""
Grading utilities for math questions.

Datasets: GSM8K, SVAMP

Expects the model to produce answers in the format "#### {answer}", where answer is a number.
"""

import re


def is_refusal(completion: str | None, extracted: str | None) -> bool:
    """Check if a completion is a model refusal."""
    return extracted is None and completion is not None and "I cannot " in completion


def extract_answer(text: str | None) -> str | None:
    """Extract the answer following the first '####' marker in the text."""
    if text is None:
        return None

    match = re.search(r"####\s*(.+)", text)
    if not match:
        return None
    return match.group(1).strip()


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, removes commas/dollar signs/$/%,
    and converts to lowercase.
    """
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace("$", "")
    answer = answer.replace("%", "")
    answer = answer.replace("_", "")
    answer = answer.lower()
    return answer


def grade(predicted: str | None, gold: str | int | float) -> bool:
    """Check whether the predicted answer matches the gold answer."""
    if predicted is None:
        return False
    return normalize_answer(predicted) == normalize_answer(gold)


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
        extracted = extract_answer(completion)
        correct = grade(extracted, gold)
        results.append({
            "completion": completion,
            "extracted": extracted,
            "refusal": is_refusal(completion, extracted),
            "gold": gold,
            "correct": correct,
        })
    return results
