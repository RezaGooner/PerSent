"""Weighted, lexicon-based emotion analyzer for Persian text.

Scores seven emotions: شادی (happiness), غم (sadness), ترس (fear),
عصبانیت (anger), تنفر (disgust), شگفتی (surprise), آرامش (calmness).
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ._models import EmotionLexicon
from ._text import hazm_tools, to_text, word_tokenize
from ._utils import bundled_model_path, logger, read_csv, resolve_column, warn_legacy_pickle
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError

DEFAULT_MODEL_NAME = "weighted_sentiment_model"
BUNDLED_MODEL_FILENAME = "emotion_lexicon.json"

EMOTION_MAP = {
    "شادی": 1,  # Happiness
    "غم": 2,  # Sadness
    "ترس": 3,  # Fear
    "عصبانیت": 4,  # Anger
    "تنفر": 5,  # Disgust
    "شگفتی": 6,  # Surprise
    "آرامش": 7,  # Calm
}

_ZWNJ_SPACE_RE = re.compile(r"[‌\s]+")
_NON_WORD_RE = re.compile(r"[^\w\s؀-ۿ]")
_DIGITS_RE = re.compile(r"\d+")


class SentimentAnalyzer:
    """Persian emotion analyzer with per-word weights.

    Args:
        model_dir: Directory used by :meth:`saveModel` / :meth:`loadModel` for your
            own models. If ``None`` (default), :meth:`loadModel` loads the
            pre-trained lexicon bundled with PerSent and :meth:`train` keeps the
            model in memory only.

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> analyzer.loadModel()
        >>> analyzer.analyzeText("خیلی خوشحالم")["شادی"]
        100.0

    Note:
        Negation is not handled yet: "خوشحال نیستم" is still scored as happiness.
    """

    def __init__(self, model_dir: str | os.PathLike | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.emotion_map: dict[str, int] = dict(EMOTION_MAP)
        self.keywords: defaultdict[int, list[str]] = defaultdict(list)
        self.word_weights: defaultdict[str, dict[int, float]] = defaultdict(dict)
        self.model_loaded = False

    # ------------------------------------------------------------------ text
    def _normalize_text(self, text: object) -> str:
        """Collapse half-spaces/whitespace, drop punctuation and digits."""
        text = _ZWNJ_SPACE_RE.sub(" ", to_text(text)).strip()
        text = _NON_WORD_RE.sub(" ", text)
        text = _DIGITS_RE.sub("", text)
        return text.lower()

    def _preprocess_text(self, text: object) -> list[str]:
        tools = hazm_tools()
        stemmer, stopwords = tools["stemmer"], tools["stopwords"]
        text = self._normalize_text(text)
        if not text.strip():
            return []
        return [stemmer.stem(t) for t in word_tokenize(text) if t not in stopwords and len(t) > 1]

    def _generate_word_forms(self, word: str) -> list[str]:
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

    def _add_entry(self, forms, emotion: str, weight: float) -> bool:
        numeric = self.emotion_map.get(emotion)
        if numeric is None:
            return False
        forms = list(forms)
        self.keywords[numeric].extend(forms)
        for form in forms:
            if weight > self.word_weights.get(form, {}).get(numeric, 0):
                self.word_weights.setdefault(form, {})[numeric] = weight
        return True

    def _dedupe_keywords(self) -> None:
        for emotion in list(self.keywords):
            self.keywords[emotion] = list(dict.fromkeys(self.keywords[emotion]))

    def _reset(self) -> None:
        self.keywords = defaultdict(list)
        self.word_weights = defaultdict(dict)

    @staticmethod
    def _to_weight(value: object) -> float:
        try:
            weight = float(value)
        except (TypeError, ValueError):
            return 1.0
        return 1.0 if weight != weight else weight  # NaN -> default

    # --------------------------------------------------------------- lexicon
    def loadLex(
        self,
        csv_file: str | os.PathLike,
        word_col: int | str = 0,
        emotion_col: int | str = 1,
        weight_col: int | str = 2,
    ) -> None:
        """Load a weighted lexicon CSV (word, emotion, weight) and add it to the model.

        Columns may be given as 0-based indices or, for CSVs with a header row,
        as column names. Missing/invalid weights default to 1.0. Rows whose
        emotion is not one of the seven supported labels are skipped.
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Lexicon file {csv_file} not found")
        by_name = any(isinstance(c, str) for c in (word_col, emotion_col, weight_col))
        df = pd.read_csv(csv_file, header=0 if by_name else None, dtype=str, keep_default_na=False)
        word_c = resolve_column(df, word_col, "Word column")
        emotion_c = resolve_column(df, emotion_col, "Emotion column")
        try:
            weight_c = resolve_column(df, weight_col, "Weight column")
        except DataValidationError:
            weight_c = None  # no weight column -> default weight

        for _, row in df.iterrows():
            word = self._normalize_text(str(row[word_c]).strip())
            emotion = str(row[emotion_c]).strip()
            weight = self._to_weight(row[weight_c]) if weight_c is not None else 1.0
            self._add_entry(self._generate_word_forms(word), emotion, weight)

        self._dedupe_keywords()
        self.model_loaded = True

    def train(
        self,
        train_csv: str | os.PathLike,
        text_col: int | str = "text",
        emotion_col: int | str | None = None,
        weight_col: int | str = "weight",
        reset: bool = False,
    ) -> dict:
        """Train (extend) the model from a CSV of texts, emotions and weights.

        ``emotion_col`` defaults to ``"sentiment"`` (the column used by the bundled
        dataset), falling back to ``"emotion"``. With ``reset=True`` the current
        model is discarded first; otherwise new entries are merged into it.
        If ``model_dir`` was given, the model is saved there afterwards.

        Returns:
            ``{"status": "success" | "error", "message": ...}``
        """
        try:
            df = read_csv(train_csv, "Training")
            if emotion_col is None:
                emotion_col = next((c for c in ("sentiment", "emotion") if c in df.columns), "sentiment")
            text_c = resolve_column(df, text_col, "Text column")
            emotion_c = resolve_column(df, emotion_col, "Emotion column")
            weight_c = resolve_column(df, weight_col, "Weight column")

            if reset:
                self._reset()
            for text, emotion, weight in tqdm(
                zip(df[text_c], df[emotion_c], df[weight_c], strict=True), total=len(df), desc="Training model"
            ):
                forms: set[str] = set()
                for word in self._preprocess_text(text):
                    forms.update(self._generate_word_forms(word))
                self._add_entry(forms, to_text(emotion).strip(), self._to_weight(weight))

            self._dedupe_keywords()
            self.model_loaded = True
            if self.model_dir is not None:
                self.saveModel()
            return {"status": "success", "message": "Model trained successfully"}
        except FileNotFoundError as exc:
            return {"status": "error", "message": f"File not found: {exc}"}
        except (DataValidationError, ValueError) as exc:
            return {"status": "error", "message": f"Data validation error: {exc}"}
        except Exception as exc:
            return {"status": "error", "message": f"Training failed: {exc}"}

    # ---------------------------------------------------------- persistence
    def _lexicon(self) -> EmotionLexicon:
        return EmotionLexicon(self.emotion_map, self.keywords, self.word_weights)

    def saveModel(self, model_name: str = DEFAULT_MODEL_NAME, model_dir: str | os.PathLike | None = None) -> Path:
        """Save the model as ``<model_dir>/<model_name>.json`` and return the path."""
        target = Path(model_dir) if model_dir is not None else self.model_dir
        if target is None:
            raise ValueError("No model_dir given: use SentimentAnalyzer(model_dir=...) or saveModel(model_dir=...)")
        path = target / f"{model_name}.json"
        self._lexicon().save(path)
        logger.info("Model saved to %s", path)
        return path

    def loadModel(self, model_name: str = DEFAULT_MODEL_NAME, model_dir: str | os.PathLike | None = None) -> None:
        """Load ``<model_dir>/<model_name>.json``, or the bundled pre-trained model.

        The bundled model is used when no directory is set or the file does not
        exist there (only for the default ``model_name``). The current working
        directory is never searched.
        """
        directory = Path(model_dir) if model_dir is not None else self.model_dir
        lexicon = None
        if directory is not None:
            path = directory / f"{model_name}.json"
            legacy = directory / f"{model_name}.joblib"
            if path.exists():
                lexicon = EmotionLexicon.load(path)
            elif legacy.exists():
                import joblib

                warn_legacy_pickle(directory, stacklevel=3)
                data = joblib.load(legacy)
                lexicon = EmotionLexicon(data["emotion_map"], data["keywords"], data["word_weights"])

        if lexicon is None:
            if model_name != DEFAULT_MODEL_NAME:
                where = directory if directory is not None else "(no model_dir set)"
                raise ModelFileError(f"Model '{model_name}' not found in {where}")
            if directory is not None:
                logger.warning("No model found in %s; loading the bundled pre-trained model.", directory)
            lexicon = EmotionLexicon.load(bundled_model_path(BUNDLED_MODEL_FILENAME))

        self.emotion_map = lexicon.emotion_map
        self.keywords = lexicon.keywords
        self.word_weights = lexicon.word_weights
        self.model_loaded = True

    # ------------------------------------------------------------ inference
    def analyzeText(self, text: str) -> dict[str, float]:
        """Return the percentage (0-100) of each emotion in ``text``.

        All emotions are 0.0 when no lexicon word is found.
        """
        if not self.model_loaded:
            raise ModelNotLoadedError("Model not loaded. Please load or train a model first.")

        scores: defaultdict[int, float] = defaultdict(float)
        total = 0.0
        for word in self._preprocess_text(text):
            for form in self._generate_word_forms(word):
                for emotion_id, weight in self.word_weights.get(form, {}).items():
                    scores[emotion_id] += weight
                    total += weight

        if total <= 0:
            return {emotion: 0.0 for emotion in self.emotion_map}
        return {emotion: round(scores.get(eid, 0) / total * 100, 2) for emotion, eid in self.emotion_map.items()}

    def analyzeCSV(
        self,
        input_csv: str | os.PathLike,
        output_csv: str | os.PathLike,
        text_col: int | str = "text",
        output_col: str = "sentiment_analysis",
    ) -> bool:
        """Analyze every text in a CSV and write the emotion scores to ``output_col``.

        Returns ``True`` on success, ``False`` on failure (the error is logged).
        """
        try:
            if not self.model_loaded:
                raise ModelNotLoadedError("Model not loaded. Please load or train a model first.")
            df = read_csv(input_csv)
            column = resolve_column(df, text_col)
            tqdm.pandas(desc="Analyzing sentiments")
            df[output_col] = df[column].progress_apply(lambda x: self.analyzeText(to_text(x)))
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            return True
        except FileNotFoundError as exc:
            logger.error("Input file not found - %s", exc)
        except (DataValidationError, ValueError) as exc:
            logger.error("Invalid column specification - %s", exc)
        except Exception as exc:
            logger.error("Error during CSV analysis: %s", exc)
        return False


__all__ = ["EMOTION_MAP", "SentimentAnalyzer"]
