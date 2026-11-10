"""Weighted, lexicon-based emotion analysis for Persian text.

Seven emotions: شادی (happiness), غم (sadness), ترس (fear), عصبانیت (anger),
تنفر (disgust), شگفتی (surprise), آرامش (calmness).
"""

from __future__ import annotations

import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ._models import EmotionLexicon
from ._text import emotion_tokens, normalize_emotion_text, to_text, word_forms
from ._utils import as_frame, bundled_model_path, logger, resolve_column, warn_legacy_pickle
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError
from .results import EMOTION_NAMES_EN, EmotionResult

#: Persian emotion name -> numeric id used inside model files.
EMOTION_MAP: dict[str, int] = {
    "شادی": 1,  # Happiness
    "غم": 2,  # Sadness
    "ترس": 3,  # Fear
    "عصبانیت": 4,  # Anger
    "تنفر": 5,  # Disgust
    "شگفتی": 6,  # Surprise
    "آرامش": 7,  # Calm
}
EMOTIONS: tuple[str, ...] = tuple(EMOTION_MAP)
MODEL_SUFFIX = ".json"
DEFAULT_MODEL_NAME = "weighted_sentiment_model"
BUNDLED_MODEL_FILENAME = "emotion_lexicon.json"
_EN_TO_FA = {v: k for k, v in EMOTION_NAMES_EN.items()}


def _to_weight(value: object) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if weight != weight else weight  # NaN -> default


def _resolve_model_file(path: str | os.PathLike) -> Path:
    path = Path(path)
    return path if path.suffix == MODEL_SUFFIX else path / f"{DEFAULT_MODEL_NAME}{MODEL_SUFFIX}"


class EmotionAnalyzer:
    """Score the seven basic emotions in Persian text with a weighted lexicon.

    Example:
        >>> emo = EmotionAnalyzer.pretrained()
        >>> result = emo.analyze("خیلی خوشحالم")
        >>> result.dominant, result.scores["شادی"]
        ('شادی', 100.0)

    Note:
        Negation is not handled yet: "خوشحال نیستم" is still scored as happiness.
    """

    def __init__(self, lexicon: EmotionLexicon | None = None) -> None:
        self.lexicon = lexicon

    def __repr__(self) -> str:
        if self.lexicon is None:
            return "EmotionAnalyzer(empty)"
        return f"EmotionAnalyzer(words={len(self.lexicon.word_weights)})"

    # ---------------------------------------------------------------- loading
    @classmethod
    def pretrained(cls) -> EmotionAnalyzer:
        """The lexicon bundled with PerSent."""
        return cls(EmotionLexicon.load(bundled_model_path(BUNDLED_MODEL_FILENAME)))

    @classmethod
    def load(cls, path: str | os.PathLike) -> EmotionAnalyzer:
        """Load a model saved with :meth:`save` (a ``.json`` file or a directory).

        PerSent <= 1.3 ``.joblib`` files are accepted with a ``DeprecationWarning``.
        """
        file = _resolve_model_file(path)
        if file.exists():
            return cls(EmotionLexicon.load(file))
        legacy = file.with_suffix(".joblib")
        if legacy.exists():
            return cls._load_legacy(legacy)
        raise ModelFileError(f"No PerSent emotion model found at {path}")

    @classmethod
    def _load_legacy(cls, path: Path, stacklevel: int = 5) -> EmotionAnalyzer:
        import joblib

        warn_legacy_pickle(path.parent, stacklevel=stacklevel)
        data = joblib.load(path)
        return cls(EmotionLexicon(data["emotion_map"], data["keywords"], data["word_weights"]))

    def save(self, path: str | os.PathLike) -> Path:
        """Save to a ``.json`` file, or to ``<dir>/weighted_sentiment_model.json``."""
        self._require_fitted()
        file = _resolve_model_file(path)
        self.lexicon.save(file)
        logger.info("Model saved to %s", file)
        return file

    # ---------------------------------------------------------------- state
    @property
    def is_fitted(self) -> bool:
        return self.lexicon is not None

    @property
    def emotions(self) -> tuple[str, ...]:
        return tuple(self.lexicon.emotion_map) if self.lexicon is not None else EMOTIONS

    def _require_fitted(self) -> None:
        if self.lexicon is None:
            raise ModelNotLoadedError(
                "Model not loaded. Use EmotionAnalyzer.pretrained(), EmotionAnalyzer.load(path), "
                "fit_csv() or add_lexicon()."
            )

    def _ensure_lexicon(self) -> EmotionLexicon:
        if self.lexicon is None:
            self.lexicon = EmotionLexicon(dict(EMOTION_MAP))
        return self.lexicon

    @staticmethod
    def _emotion_name(emotion: object) -> str | None:
        name = to_text(emotion).strip()
        if name in EMOTION_MAP:
            return name
        return _EN_TO_FA.get(name.lower())

    def _add(self, forms: Iterable[str], emotion: object, weight: float) -> bool:
        lex = self._ensure_lexicon()
        name = self._emotion_name(emotion)
        numeric = lex.emotion_map.get(name) if name else None
        if numeric is None:
            return False
        forms = list(forms)
        lex.keywords[numeric].extend(forms)
        for form in forms:
            if weight > lex.word_weights.get(form, {}).get(numeric, 0):
                lex.word_weights.setdefault(form, {})[numeric] = weight
        return True

    def _dedupe(self) -> None:
        lex = self.lexicon
        for emotion in list(lex.keywords):
            lex.keywords[emotion] = list(dict.fromkeys(lex.keywords[emotion]))

    # --------------------------------------------------------------- lexicon
    def add_word(self, word: str, emotion: str, weight: float = 1.0) -> None:
        """Add one lexicon entry. ``emotion`` may be Persian (``"شادی"``) or English (``"happiness"``)."""
        if self._emotion_name(emotion) is None:
            raise DataValidationError(f"Unknown emotion {emotion!r}; use one of {list(EMOTIONS)}")
        normalized = normalize_emotion_text(word).strip()
        if " " in normalized:
            warnings.warn(
                f"{word!r} is more than one word after normalization (half-spaces become spaces); "
                "only single words are matched at the moment, so this entry will have no effect.",
                UserWarning,
                stacklevel=2,
            )
        self._add(word_forms(normalized), emotion, float(weight))
        self._dedupe()

    def add_lexicon(
        self,
        data: str | os.PathLike | pd.DataFrame,
        word_column: int | str = 0,
        emotion_column: int | str = 1,
        weight_column: int | str | None = 2,
    ) -> int:
        """Add a word/emotion/weight lexicon and return the number of entries used.

        Columns may be 0-based indices or, for CSVs with a header row, names.
        Missing/invalid weights default to 1.0; unknown emotions are skipped.
        """
        if isinstance(data, pd.DataFrame):
            df = data.astype(str)
        else:
            if not os.path.exists(data):
                raise FileNotFoundError(f"Lexicon file {data} not found")
            by_name = any(isinstance(c, str) for c in (word_column, emotion_column, weight_column))
            df = pd.read_csv(data, header=0 if by_name else None, dtype=str, keep_default_na=False)
        word_c = resolve_column(df, word_column, "Word column")
        emotion_c = resolve_column(df, emotion_column, "Emotion column")
        weight_c = None
        if weight_column is not None:
            try:
                weight_c = resolve_column(df, weight_column, "Weight column")
            except DataValidationError:
                weight_c = None

        used = 0
        weights = df[weight_c] if weight_c is not None else [1.0] * len(df)
        for word, emotion, weight in zip(df[word_c], df[emotion_c], weights, strict=True):
            forms = word_forms(normalize_emotion_text(str(word).strip()))
            used += self._add(forms, emotion, _to_weight(weight))
        self._dedupe()
        return used

    def fit_csv(
        self,
        data: str | os.PathLike | pd.DataFrame,
        text_column: int | str = "text",
        emotion_column: int | str | None = None,
        weight_column: int | str | None = "weight",
        *,
        reset: bool = False,
        show_progress: bool = False,
    ) -> int:
        """Learn weighted emotion words from labelled texts; returns the number of rows used.

        ``emotion_column`` defaults to ``"sentiment"`` (or ``"emotion"``). Without a
        weight column every row has weight 1.0. New entries are merged into the
        current lexicon unless ``reset=True``.
        """
        df = as_frame(data, "Training")
        if emotion_column is None:
            emotion_column = next((c for c in ("sentiment", "emotion") if c in df.columns), "sentiment")
        text_c = resolve_column(df, text_column, "Text column")
        emotion_c = resolve_column(df, emotion_column, "Emotion column")
        weights = df[resolve_column(df, weight_column, "Weight column")] if weight_column is not None else None

        if reset:
            self.lexicon = None
        self._ensure_lexicon()
        rows = zip(df[text_c], df[emotion_c], weights if weights is not None else [1.0] * len(df), strict=True)
        used = 0
        for text, emotion, weight in tqdm(rows, total=len(df), desc="Training model", disable=not show_progress):
            forms: set[str] = set()
            for word in emotion_tokens(text):
                forms.update(word_forms(word))
            used += self._add(forms, emotion, _to_weight(weight))
        self._dedupe()
        return used

    # ------------------------------------------------------------ inference
    def analyze(self, text: str) -> EmotionResult:
        """Emotion percentages (0-100) and the lexicon words that matched."""
        self._require_fitted()
        weights = self.lexicon.word_weights
        reverse = {v: k for k, v in self.lexicon.emotion_map.items()}
        scores: defaultdict[int, float] = defaultdict(float)
        total = 0.0
        matched: list[str] = []
        for word in emotion_tokens(text):
            hit = False
            for form in word_forms(word):
                for emotion_id, weight in weights.get(form, {}).items():
                    scores[emotion_id] += weight
                    total += weight
                    hit = True
            if hit:
                matched.append(word)

        if total <= 0:
            pct = {name: 0.0 for name in self.lexicon.emotion_map}
        else:
            pct = {
                reverse[eid]: round(scores.get(eid, 0) / total * 100, 2) for eid in self.lexicon.emotion_map.values()
            }
        return EmotionResult(to_text(text), pct, tuple(dict.fromkeys(matched)))

    def scores(self, text: str) -> dict[str, float]:
        """Shortcut for ``analyze(text).scores``."""
        return self.analyze(text).scores

    def dominant(self, text: str) -> str | None:
        """Strongest emotion in ``text`` or ``None``."""
        return self.analyze(text).dominant

    def analyze_many(self, texts: Iterable[object], show_progress: bool = False) -> list[EmotionResult]:
        texts = list(texts)
        return [self.analyze(t) for t in tqdm(texts, desc="Analyzing", disable=not show_progress)]

    def analyze_csv(
        self,
        input_csv: str | os.PathLike | pd.DataFrame,
        output_csv: str | os.PathLike | None = None,
        *,
        text_column: int | str = "text",
        output_column: str | None = "emotions",
        expand: bool = True,
        lang: str = "fa",
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Analyze a CSV column and return (and optionally save) the result.

        With ``expand=True`` (default) one numeric column per emotion and a
        ``dominant_emotion`` column are added; ``output_column`` additionally
        stores the full score dict (set it to ``None`` to skip). ``lang="en"``
        uses English emotion names for the new columns. Raises on errors.
        """
        self._require_fitted()
        if lang not in ("fa", "en"):
            raise ValueError("lang must be 'fa' or 'en'")
        df = as_frame(input_csv).copy()
        column = resolve_column(df, text_column)
        results = self.analyze_many(df[column].tolist(), show_progress=show_progress)
        en = lang == "en"
        if output_column:
            df[output_column] = [r.scores_en() if en else r.scores for r in results]
        if expand:
            for emotion in self.emotions:
                name = EMOTION_NAMES_EN.get(emotion, emotion) if en else emotion
                df[name] = [r.scores.get(emotion, 0.0) for r in results]
            df["dominant_emotion"] = [(r.dominant_en if en else r.dominant) or "" for r in results]
        if output_csv is not None:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            logger.info("Results saved to %s", output_csv)
        return df


__all__ = ["EMOTIONS", "EMOTION_MAP", "EmotionAnalyzer"]
