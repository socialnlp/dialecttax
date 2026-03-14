import re
from collections import Counter

WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def extract_words(ds: list[dict], row_text: str) -> dict[str, int]:
    """Extract word counts from all samples by splitting on whitespace and delimiters.

    Args:
        ds: List of sample dicts.
        row_text: Key for the text field in each sample.

    Returns:
        Dict mapping each word to its count, sorted by count descending.
    """
    counts = Counter()
    for sample in ds:
        text = sample[row_text]
        counts.update(WORD_PATTERN.findall(text))
    return dict(counts.most_common())
