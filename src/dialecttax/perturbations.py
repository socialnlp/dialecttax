"""
Transformations that perturb surface form while preserving semantic content.
"""

import random
import string


DIRECTORY_NAME = "perturbations"


##############
# CHARACTERS #
##############

def _perturb(texts, fn):
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    results = [fn(t) for t in texts]
    return results[0] if single else results


def swap(texts: str | list[str], p: float = 0.05):
    def _swap(text):
        chars = list(text)
        i = 0
        while i < len(chars) - 1:
            if chars[i] != " " and chars[i + 1] != " " and random.random() < p:
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
                i += 2  # skip the swapped pair
            else:
                i += 1
        return "".join(chars)

    return _perturb(texts, _swap)


def drop(texts: str | list[str], p: float = 0.15):
    def _drop(text):
        return "".join(c for c in text if c == " " or random.random() >= p)

    return _perturb(texts, _drop)


def insert(texts: str | list[str], p: float = 0.05):
    def _insert(text):
        out = []
        for c in text:
            out.append(c)
            if c != " " and random.random() < p:
                out.append(random.choice(string.ascii_lowercase))
        return "".join(out)

    return _perturb(texts, _insert)


def capitalize(texts: str | list[str], mode: str = "random"):
    def _random(text):
        return "".join(
            c.upper() if random.random() < 0.5 else c.lower() for c in text
        )

    def _alternating(text):
        out = []
        idx = 0
        for c in text:
            if c == " ":
                out.append(c)
            else:
                out.append(c.lower() if idx % 2 == 0 else c.upper())
                idx += 1
        return "".join(out)

    if mode == "random":
        return _perturb(texts, _random)
    else:
        return _perturb(texts, _alternating)


#############
# TRANSLATE #
#############

GTRANSLATE_LANGUAGE_CODES = {
    # High-resource
    "chinese": "zh-CN",
    "french": "fr",
    # Mid-resource
    "hindi": "hi",
    "polish": "pl",
    # Low-resource
    "khmer": "km",
    "yoruba": "yo",
}


def translate(texts: str | list[str], target_language: str, api_key: str):
    """Translate texts using Google Cloud Translate API.

    Beware of the 30K characters-per-request limit.
    """
    import html

    from google.auth.api_key import Credentials
    from google.cloud import translate_v2 as gtranslate

    single = isinstance(texts, str)
    if single:
        texts = [texts]

    # Chunk
    CHAR_LIMIT = 30_000
    SEGMENT_LIMIT = 128
    chunks = []
    current_chunk = []
    current_len = 0
    for t in texts:
        t_len = len(t)
        if current_chunk and (
            current_len + t_len > CHAR_LIMIT or len(current_chunk) >= SEGMENT_LIMIT
        ):
            chunks.append(current_chunk)
            current_chunk = [t]
            current_len = t_len
        else:
            current_chunk.append(t)
            current_len += t_len
    if current_chunk:
        chunks.append(current_chunk)

    # Translate
    client = gtranslate.Client(credentials=Credentials(api_key))
    results = []
    for chunk in chunks:
        responses = client.translate(chunk, target_language=target_language)
        results.extend(html.unescape(r["translatedText"]) for r in responses)

    return results[0] if single else results
