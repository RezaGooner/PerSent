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


# hazm's stop-word list contains words that carry opinion, intensity or negation. The
# newer pipelines keep them; the bundled 1.x models were trained without them, so the
# legacy pipelines still use hazm's full list.
OPINION_WORDS: frozenset[str] = frozenset(
    {
        "خوب", "خوبی", "بهتر", "بهترین", "عالی", "مناسب", "کافی", "کامل", "کاملا", "متاسفانه", "خطر",
        "جدی", "مهم", "بزرگ", "کوچک", "جدید", "خیلی", "بسیار", "زیاد", "زیادی", "کم", "کمی", "بیش",
        "بیشتر", "نه", "نیست", "نیستند", "نبود", "ندارد", "ندارند", "نباید", "نمی‌شود", "هیچ", "بدون",
        "عدم", "غیر",
    }
)  # fmt: skip

# Negation markers (after the emotion-normalization, where half-spaces became spaces,
# so "نمی‌ترسم" is tokenized as "نمی" + "ترسم").
NEGATION_BEFORE: frozenset[str] = frozenset({"نه", "نمی", "نمیخوام", "بدون", "عدم", "هیچ", "هرگز", "اصلا"})
NEGATION_AFTER: frozenset[str] = frozenset(
    {
        "نیست", "نیستم", "نیستی", "نیستیم", "نیستید", "نیستند", "نیس", "نبود", "نبودم", "نبودی", "نبودیم",
        "نبودید", "نبودند", "نشد", "نشدم", "نشدیم", "نشدند", "نمیشه", "نمیشم", "نمی", "ندارم", "ندارد",
        "نداره", "نداریم", "ندارند", "نکرد", "نکردم", "نکردیم",
    }
)  # fmt: skip


def stopwords(kind: str = "legacy") -> frozenset[str]:
    """``"legacy"``: hazm's full list; ``"opinion"``: hazm's list without :data:`OPINION_WORDS`."""
    sw = hazm_tools()["stopwords"]
    return sw if kind == "legacy" else _opinion_stopwords(sw)


@lru_cache(maxsize=1)
def _opinion_stopwords(sw: frozenset[str]) -> frozenset[str]:
    return frozenset(sw - OPINION_WORDS)


def _filter_and_stem(tokens: list[str], kind: str = "legacy") -> list[str]:
    stemmer, sw = hazm_tools()["stemmer"], stopwords(kind)
    return [stemmer.stem(tok) for tok in tokens if tok not in sw and len(tok) > 1]


def negated_mask(tokens: list[str], before: int = 2, after: int = 2) -> list[bool]:
    """``True`` for tokens that are negated by a nearby negation marker.

    A token counts as negated when one of the next ``after`` tokens is a copula/verb
    negation ("خوشحال نیستم") or one of the previous ``before`` tokens is a negating
    word ("نه خوشحالم", "اصلا نمی ترسم"). Negation markers themselves are not negated.
    """
    n = len(tokens)
    mask = [False] * n
    for i, tok in enumerate(tokens):
        if tok in NEGATION_BEFORE or tok in NEGATION_AFTER:
            continue
        nxt = tokens[i + 1 : i + 1 + after]
        prv = tokens[max(0, i - before) : i]
        if any(t in NEGATION_AFTER for t in nxt) or any(t in NEGATION_BEFORE for t in prv):
            mask[i] = True
    return mask


# -------------------------------------------------------------------------------- tf-idf
def tfidf_text(text: object) -> str:
    """Input of the TF-IDF models: hazm-normalized text (no stemming, no stop-word removal)."""
    text = hazm_tools()["normalizer"].normalize(to_text(text))
    return WHITESPACE_RE.sub(" ", text).strip()


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
