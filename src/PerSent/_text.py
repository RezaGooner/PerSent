"""Shared Persian text preprocessing.

Two pipelines exist because the bundled models were trained with them:

* polarity (``PolarityClassifier``): hazm normalization, punctuation/digit removal,
  tokenization, stop-word removal, stemming.
* emotion (``EmotionAnalyzer``): half-space/punctuation/digit removal, tokenization,
  stop-word removal, stemming; words are then expanded to stem/lemma forms.

Changing either pipeline changes predictions of the pre-trained models.
"""

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

_ZWNJ_SPACE_RE = re.compile(r"[‌\s]+")
_NON_WORD_RE = re.compile(r"[^\w\s؀-ۿ]")
_DIGITS_RE = re.compile(r"\d+")


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


def _filter_and_stem(tokens: list[str]) -> list[str]:
    tools = hazm_tools()
    stemmer, stopwords = tools["stemmer"], tools["stopwords"]
    return [stemmer.stem(tok) for tok in tokens if tok not in stopwords and len(tok) > 1]


# ----------------------------------------------------------------------------- polarity
def clean_polarity_text(text: object) -> str:
    """Normalize and strip punctuation/digits; returns '' if nothing is left."""
    text = hazm_tools()["normalizer"].normalize(to_text(text))
    text = COMMENT_STRIP_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def polarity_tokens(text: object, *, cleaned: bool = False) -> list[str]:
    """Tokens fed to the polarity model (pass ``cleaned=True`` if already cleaned)."""
    text = to_text(text) if cleaned else clean_polarity_text(text)
    if not text:
        return []
    return _filter_and_stem(word_tokenize(text))


# ------------------------------------------------------------------------------ emotion
def normalize_emotion_text(text: object) -> str:
    """Collapse half-spaces/whitespace, drop punctuation and digits, lower-case."""
    text = _ZWNJ_SPACE_RE.sub(" ", to_text(text)).strip()
    text = _NON_WORD_RE.sub(" ", text)
    text = _DIGITS_RE.sub("", text)
    return text.lower()


def emotion_tokens(text: object) -> list[str]:
    text = normalize_emotion_text(text)
    if not text.strip():
        return []
    return _filter_and_stem(word_tokenize(text))


def word_forms(word: str) -> list[str]:
    """All surface forms of a word (word, stem, lemma and their combinations)."""
    tools = hazm_tools()
    stemmer, lemmatizer = tools["stemmer"], tools["lemmatizer"]
    stem = stemmer.stem(word)
    lemma = lemmatizer.lemmatize(word)
    forms = {
        word,
        stem,
        lemma,
        stemmer.stem(stem),
        lemmatizer.lemmatize(stem),
        stemmer.stem(lemma),
        lemmatizer.lemmatize(lemma),
    }
    return [f for f in forms if f and len(f) > 1]
