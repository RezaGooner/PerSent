"""Suffix-stripping Persian stemmer.

Vendored from hazm 0.9/0.10 (https://github.com/roshan-research/hazm, MIT License,
Copyright (c) Roshan) because hazm >= 0.11 ships a different stemmer. The bundled
PerSent models were trained with this one, so keeping it makes predictions identical
on every Python / hazm version.
"""

from __future__ import annotations

_ENDS = ("ات", "ان", "ترین", "تر", "م", "ت", "ش", "یی", "ی", "ها", "ٔ", "‌ا", "‌")


class Stemmer:
    """Strip common Persian suffixes (e.g. 'کتاب‌هایی' -> 'کتاب')."""

    ends = _ENDS

    def stem(self, word: str) -> str:
        for end in self.ends:
            if word.endswith(end):
                word = word[: -len(end)]
        if word.endswith("ۀ"):
            word = word[:-1] + "ه"
        return word
