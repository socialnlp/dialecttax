"""
Grading utilities for multiple-choice questions.

Datasets: ReDial-QA

Expects the model to produce answers in the format "#### {answer}",
where the answer is an upper case letter.
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
    # Take only the first token, strip punctuation, uppercase (e.g. "#### B." → "B")
    parts = match.group(1).strip().split()
    if not parts:
        return None
    token = re.sub(r"[^A-Za-z]", "", parts[0]).upper()
    return token if len(token) == 1 and token.isalpha() else None


def extract_answer_fuzzy(text: str | None) -> str | None:
    """Extract the last standalone uppercase letter (A-D) from the text."""
    if not text:
        return None
    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[-1]
    return None


def normalize_answer(answer: str) -> str:
    """Normalize an answer string for comparison.

    Strips whitespace, removes commas/periods/$,
    and converts to upper case.
    """
    if not isinstance(answer, str):
        answer = str(answer)

    answer = answer.strip()
    answer = answer.replace(",", "")
    answer = answer.replace(".", "")
    answer = answer.upper()
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
        result = {
            "completion": completion,
            "extracted": extracted,
            "refusal": is_refusal(completion, extracted),
            "gold": gold,
            "correct": grade(extracted, gold),
        }

        # Fallback: last standalone A-D letter when #### marker is missing
        if extracted is None:
            calculated = extract_answer_fuzzy(completion)
            if calculated is not None:
                result["calculated"] = calculated
                result["fuzzy"] = grade(calculated, gold)

        results.append(result)
    return results
