"""Shared Persian text helpers."""

from __future__ import annotations

import math
import re
from functools import lru_cache

# ASCII punctuation/symbols, Persian question mark, Persian and ASCII digits.
# (PerSent <= 1.3 used ``[!()-\[...]`` where ``)-\[`` was accidentally a character
# range that also deleted every upper-case Latin letter, e.g. "iPhone" -> "i hone".
# This class removes exactly the same characters minus A-Z.)
COMMENT_STRIP_RE = re.compile(r"[!-/:-@\[\]^_{}~؟۰-۹\d]+")
WHITESPACE_RE = re.compile(r"\s+")


def to_text(value: object) -> str:
    """Convert a cell value to text; ``None`` and NaN become an empty string."""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    try:
        import pandas as pd

        if value is pd.NA or value is pd.NaT:
            return ""
    except ImportError:  # pragma: no cover
        pass
    return str(value)


@lru_cache(maxsize=1)
def hazm_tools():
    """Lazily construct (and cache) hazm objects, which are slow to create."""
    from hazm import Lemmatizer, Normalizer, WordTokenizer, stopwords_list

    from ._stemmer import Stemmer

    return {
        # hazm >= 0.11's word_tokenize() builds a new WordTokenizer (re-reading its word list) on
        # every call, which is ~1000x slower; keep a single instance instead.
        "tokenizer": WordTokenizer(),
        "normalizer": Normalizer(),
        "stemmer": Stemmer(),
        "lemmatizer": Lemmatizer(),
        "stopwords": frozenset(stopwords_list()),
    }


def word_tokenize(text: str) -> list[str]:
    return hazm_tools()["tokenizer"].tokenize(text)
