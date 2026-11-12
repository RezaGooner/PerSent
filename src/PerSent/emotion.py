"""Emotion analysis for Persian text.

Seven emotions: شادی (happiness), غم (sadness), ترس (fear), عصبانیت (anger),
تنفر (disgust), شگفتی (surprise), آرامش (calmness).

Two backends:

* **lexicon** (default, bundled): weighted emotion words, with negation handling
  and multi-word entries.
* **classifier**: TF-IDF + logistic regression trained on labelled texts with
  :meth:`EmotionAnalyzer.fit_classifier` (no classifier is bundled yet).
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._models import EmotionLexicon
from ._text import (
    NEGATION_AFTER,
    NEGATION_BEFORE,
    hazm_tools,
    negated_mask,
    normalize_emotion_text,
    stopwords,
    to_text,
    word_forms,
    word_tokenize,
)
from ._tfidf import TfidfLinearModel, load_tfidf_model, npz_model_type
from ._utils import as_frame, bundled_model_path, logger, resolve_column, warn_legacy_pickle
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError
from .results import EMOTION_NAMES_EN, EmotionResult, EvaluationReport, TrainingReport

#: Persian emotion name -> numeric id used inside lexicon files.
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
NONE_LABEL = "_none"
#: Labels accepted for "no emotion / other" when training a classifier.
NONE_ALIASES = frozenset(
    {"other", "none", "neutral", "no_emotion", "سایر", "هیچ", "خنثی", "بدون_احساس", "بدون احساس", NONE_LABEL}
)

PRETRAINED: dict[str, str] = {"lexicon-v1": "emotion_lexicon.json"}
DEFAULT_PRETRAINED = "lexicon-v1"
DEFAULT_MODEL_NAME = "weighted_sentiment_model"
CLASSIFIER_FILENAME = "emotion_classifier.npz"
BUNDLED_MODEL_FILENAME = PRETRAINED[DEFAULT_PRETRAINED]
_EN_TO_FA = {v: k for k, v in EMOTION_NAMES_EN.items()}
_MAX_PHRASE = 3


def _to_weight(value: object) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if weight != weight else weight  # NaN -> default


def emotion_name(emotion: object) -> str | None:
    """Persian emotion name for a Persian/English label, ``NONE_LABEL`` for "other", else ``None``."""
    name = to_text(emotion).strip()
    if name in EMOTION_MAP:
        return name
    if name.lower() in NONE_ALIASES:
        return NONE_LABEL
    return _EN_TO_FA.get(name.lower())


class EmotionAnalyzer:
    """Score the seven basic emotions in Persian text.

    Args:
        lexicon: A weighted emotion lexicon (lexicon backend).
        classifier: A trained TF-IDF classifier (classifier backend); takes
            precedence over ``lexicon``.
        handle_negation: Lexicon backend: ignore emotion words that are negated
            ("خوشحال نیستم", "اصلا نمی‌ترسم").
        match_phrases: Lexicon backend: also match multi-word lexicon entries
            such as "وحشت‌زده" (which normalization splits into two words).
        stopwords: ``"opinion"`` keeps opinion words such as "خوب" that hazm's
            stop-word list would remove; ``"legacy"`` uses hazm's full list
            (behaviour of PerSent <= 1.5).

    Example:
        >>> emo = EmotionAnalyzer.pretrained()
        >>> result = emo.analyze("خیلی خوشحالم")
        >>> result.dominant, result.scores["شادی"]
        ('شادی', 100.0)
        >>> emo.analyze("اصلا خوشحال نیستم").dominant is None
        True
    """

    def __init__(
        self,
        lexicon: EmotionLexicon | None = None,
        *,
        classifier: TfidfLinearModel | None = None,
        handle_negation: bool = True,
        match_phrases: bool = True,
        stopwords: str = "opinion",
    ) -> None:
        if stopwords not in ("opinion", "legacy"):
            raise ValueError("stopwords must be 'opinion' or 'legacy'")
        self.lexicon = lexicon
        self.classifier = classifier
        self.handle_negation = handle_negation
        self.match_phrases = match_phrases
        self.stopwords = stopwords

    def __repr__(self) -> str:
        if self.classifier is not None:
            labels = [label for label in self.classifier.labels if label != NONE_LABEL]
            return f"EmotionAnalyzer(classifier, emotions={len(labels)}, features={self.classifier.n_features})"
        if self.lexicon is None:
            return "EmotionAnalyzer(empty)"
        return f"EmotionAnalyzer(lexicon, words={len(self.lexicon.word_weights)})"

    # ---------------------------------------------------------------- loading
    @staticmethod
    def available_models() -> tuple[str, ...]:
        """Names accepted by :meth:`pretrained`."""
        return tuple(PRETRAINED)

    @classmethod
    def pretrained(cls, name: str | None = None, **options) -> EmotionAnalyzer:
        """A bundled model (default: the ``lexicon-v1`` emotion lexicon)."""
        name = name or DEFAULT_PRETRAINED
        if name not in PRETRAINED:
            raise ModelFileError(f"Unknown pretrained emotion model {name!r}; available: {list(PRETRAINED)}")
        path = bundled_model_path(PRETRAINED[name])
        if path.suffix == ".npz":
            return cls(classifier=load_tfidf_model(path), **options)
        return cls(EmotionLexicon.load(path), **options)

    @classmethod
    def load(cls, path: str | os.PathLike, **options) -> EmotionAnalyzer:
        """Load a model saved with :meth:`save`: a lexicon ``.json``, a classifier ``.npz`` or a directory.

        PerSent <= 1.3 ``.joblib`` lexicons are accepted with a ``DeprecationWarning``.
        """
        path = Path(path)
        if path.is_dir():
            for candidate in (path / CLASSIFIER_FILENAME, path / f"{DEFAULT_MODEL_NAME}.json"):
                if candidate.exists():
                    return cls.load(candidate, **options)
            legacy = path / f"{DEFAULT_MODEL_NAME}.joblib"
            if legacy.exists():
                return cls._load_legacy(legacy, **options)
            raise ModelFileError(f"No PerSent emotion model found in {path}")
        if path.suffix == ".npz" and path.exists():
            if npz_model_type(path) != "tfidf_lr":
                raise ModelFileError(f"{path} is not an emotion classifier")
            return cls(classifier=load_tfidf_model(path), **options)
        if path.suffix == ".json" and path.exists():
            return cls(EmotionLexicon.load(path), **options)
        if path.with_suffix(".joblib").exists():
            return cls._load_legacy(path.with_suffix(".joblib"), **options)
        raise ModelFileError(f"No PerSent emotion model found at {path}")

    @classmethod
    def _load_legacy(cls, path: Path, stacklevel: int = 5, **options) -> EmotionAnalyzer:
        import joblib

        warn_legacy_pickle(path.parent, stacklevel=stacklevel)
        data = joblib.load(path)
        return cls(EmotionLexicon(data["emotion_map"], data["keywords"], data["word_weights"]), **options)

    def save(self, path: str | os.PathLike) -> Path:
        """Save the model.

        Lexicon: a ``.json`` file or ``<dir>/weighted_sentiment_model.json``.
        Classifier: a ``.npz`` file or ``<dir>/emotion_classifier.npz``.
        """
        self._require_fitted()
        path = Path(path)
        if self.classifier is not None:
            file = path if path.suffix == ".npz" else path / CLASSIFIER_FILENAME
            self.classifier.save(file)
        else:
            file = path if path.suffix == ".json" else path / f"{DEFAULT_MODEL_NAME}.json"
            self.lexicon.save(file)
        logger.info("Model saved to %s", file)
        return file

    # ---------------------------------------------------------------- state
    @property
    def backend(self) -> str | None:
        """``"classifier"``, ``"lexicon"`` or ``None`` (empty)."""
        if self.classifier is not None:
            return "classifier"
        return "lexicon" if self.lexicon is not None else None

    @property
    def is_fitted(self) -> bool:
        return self.backend is not None

    @property
    def emotions(self) -> tuple[str, ...]:
        if self.lexicon is not None and self.classifier is None:
            return tuple(self.lexicon.emotion_map)
        return EMOTIONS

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelNotLoadedError(
                "Model not loaded. Use EmotionAnalyzer.pretrained(), EmotionAnalyzer.load(path), "
                "fit_csv(), fit_classifier() or add_lexicon()."
            )

    def _ensure_lexicon(self) -> EmotionLexicon:
        if self.lexicon is None:
            self.lexicon = EmotionLexicon(dict(EMOTION_MAP))
        return self.lexicon

    @staticmethod
    def _emotion_name(emotion: object) -> str | None:
        name = emotion_name(emotion)
        return None if name == NONE_LABEL else name

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

    def _tokens(self, text: object) -> tuple[list[str], list[str | None], list[bool]]:
        """Raw tokens, their stems (``None`` for skipped tokens) and the negation mask."""
        normalized = normalize_emotion_text(text)
        raw = word_tokenize(normalized) if normalized.strip() else []
        stemmer, sw = hazm_tools()["stemmer"], stopwords(self.stopwords)
        stems: list[str | None] = [stemmer.stem(t) if t not in sw and len(t) > 1 else None for t in raw]
        if self.stopwords != "legacy":
            # PerSent <= 1.5 kept words that stem to "" (e.g. "ها"), which then matched junk
            # lexicon entries; the legacy mode keeps that for exact reproducibility.
            stems = [st or None for st in stems]
        if self.handle_negation:
            negated = negated_mask(raw)
            # negation markers carry no emotion themselves ("نمی" would otherwise stem to "نم")
            markers = NEGATION_BEFORE | NEGATION_AFTER
            stems = [None if t in markers else st for t, st in zip(raw, stems, strict=True)]
        else:
            negated = [False] * len(raw)
        return raw, stems, negated

    # --------------------------------------------------------------- lexicon
    def add_word(self, word: str, emotion: str, weight: float = 1.0) -> None:
        """Add one lexicon entry (a word or a short phrase such as "وحشت‌زده").

        ``emotion`` may be Persian (``"شادی"``) or English (``"happiness"``).
        """
        if self._emotion_name(emotion) is None:
            raise DataValidationError(f"Unknown emotion {emotion!r}; use one of {list(EMOTIONS)}")
        normalized = normalize_emotion_text(word).strip()
        forms = [normalized] if " " in normalized else word_forms(normalized)
        self._add(forms, emotion, float(weight))
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
        """Lexicon backend: learn weighted emotion words from labelled texts.

        Every (non-stop-word) word of a text is added to the text's emotion with the
        row's weight. Returns the number of rows used. ``emotion_column`` defaults
        to ``"sentiment"`` (or ``"emotion"``). New entries are merged into the
        current lexicon unless ``reset=True``. For better accuracy use
        :meth:`fit_classifier_csv` instead.
        """
        df = as_frame(data, "Training")
        if emotion_column is None:
            emotion_column = next((c for c in ("sentiment", "emotion") if c in df.columns), "sentiment")
        text_c = resolve_column(df, text_column, "Text column")
        emotion_c = resolve_column(df, emotion_column, "Emotion column")
        weights = df[resolve_column(df, weight_column, "Weight column")] if weight_column is not None else None

        if reset:
            self.lexicon = None
        self.classifier = None
        self._ensure_lexicon()
        rows = zip(df[text_c], df[emotion_c], weights if weights is not None else [1.0] * len(df), strict=True)
        used = 0
        for text, emotion, weight in tqdm(rows, total=len(df), desc="Training model", disable=not show_progress):
            forms: set[str] = set()
            for stem in self._tokens(text)[1]:
                if stem is not None:
                    forms.update(word_forms(stem))
            used += self._add(forms, emotion, _to_weight(weight))
        self._dedupe()
        return used

    # ------------------------------------------------------------ classifier
    def fit_classifier(
        self,
        texts: Sequence[object],
        emotions: Sequence[object],
        *,
        test_size: float = 0.2,
        random_state: int | None = None,
        class_weight: str | dict | None = "balanced",
        C: float = 4.0,
        **tfidf_options,
    ) -> TrainingReport:
        """Classifier backend: train TF-IDF + logistic regression on labelled texts.

        ``emotions`` are Persian or English emotion names; ``"other"``/``"none"``/
        ``"سایر"`` mark texts without one of the seven emotions (the analyzer then
        reports ``dominant=None`` for similar texts). Unknown labels raise. A
        stratified ``test_size`` split is held out for the returned report.
        Extra keyword arguments (``max_word_features``, ``char_ngrams``, ...) are
        passed to the TF-IDF model.
        """
        from sklearn.model_selection import train_test_split

        texts = [to_text(t) for t in texts]
        labels = [emotion_name(e) for e in emotions]
        if len(texts) != len(labels):
            raise DataValidationError("texts and emotions must have the same length")
        unknown = sorted({to_text(e) for e, n in zip(emotions, labels, strict=True) if n is None})
        if unknown:
            raise DataValidationError(f"Unknown emotions {unknown[:10]}; use one of {list(EMOTIONS)} or 'other'")
        if len(set(labels)) < 2:
            raise DataValidationError("Training data must contain at least two different emotions")

        X_test = y_test = None
        X_train, y_train = texts, labels
        if test_size and test_size > 0:
            counts = pd.Series(labels).value_counts()
            stratify = labels if counts.min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=test_size, random_state=random_state, stratify=stratify
            )
        self.classifier = TfidfLinearModel.train(
            X_train, y_train, C=C, class_weight=class_weight, **tfidf_options
        ).reorder([*EMOTIONS, NONE_LABEL])
        if X_test is None:
            return TrainingReport(n_train=len(y_train), n_test=0, labels=tuple(self.classifier.labels))
        ev = self._report(y_test, self._predict_labels(X_test))
        return TrainingReport(
            n_train=len(y_train),
            n_test=len(y_test),
            accuracy=ev.accuracy,
            macro_f1=ev.macro_f1,
            per_label=ev.per_label,
            labels=ev.labels,
        )

    def fit_classifier_csv(
        self,
        data: str | os.PathLike | pd.DataFrame,
        text_column: int | str = "text",
        emotion_column: int | str | None = None,
        **kwargs,
    ) -> TrainingReport:
        """:meth:`fit_classifier` on a CSV path or DataFrame."""
        df = as_frame(data, "Training")
        if emotion_column is None:
            emotion_column = next((c for c in ("sentiment", "emotion", "label") if c in df.columns), "sentiment")
        text_c = resolve_column(df, text_column, "Text column")
        emotion_c = resolve_column(df, emotion_column, "Emotion column")
        return self.fit_classifier(df[text_c].tolist(), df[emotion_c].tolist(), **kwargs)

    # ------------------------------------------------------------ evaluation
    def _predict_labels(self, texts: Sequence[object]) -> list[str]:
        return [r.dominant or NONE_LABEL for r in self.analyze_many(texts)]

    def evaluate(self, texts: Sequence[object], emotions: Sequence[object]) -> EvaluationReport:
        """Accuracy / macro-F1 of the dominant emotion against gold labels.

        Gold "other"/"none" labels (and texts the model finds no emotion in) count
        as the ``"_none"`` label.
        """
        self._require_fitted()
        gold = [emotion_name(e) or NONE_LABEL for e in emotions]
        return self._report(gold, self._predict_labels(texts))

    def evaluate_csv(
        self,
        data: str | os.PathLike | pd.DataFrame,
        text_column: int | str = "text",
        emotion_column: int | str | None = None,
    ) -> EvaluationReport:
        df = as_frame(data)
        if emotion_column is None:
            emotion_column = next((c for c in ("sentiment", "emotion", "label") if c in df.columns), "sentiment")
        return self.evaluate(
            df[resolve_column(df, text_column)].tolist(), df[resolve_column(df, emotion_column)].tolist()
        )

    @staticmethod
    def _report(y_true: list[str], y_pred: list[str]) -> EvaluationReport:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

        labels = [e for e in EMOTIONS if e in set(y_true) | set(y_pred)]
        if NONE_LABEL in set(y_true) | set(y_pred):
            labels.append(NONE_LABEL)
        gold_labels = [label for label in labels if label in set(y_true)]
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        per_label = {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)
        }
        return EvaluationReport(
            n=len(y_true),
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, labels=gold_labels, average="macro", zero_division=0)),
            per_label=per_label,
            confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            labels=tuple(labels),
        )

    # ------------------------------------------------------------ inference
    def analyze(self, text: str) -> EmotionResult:
        """Emotion percentages (0-100), the dominant emotion and the words behind it."""
        return self.analyze_many([text])[0]

    def analyze_many(self, texts: Iterable[object], show_progress: bool = False) -> list[EmotionResult]:
        self._require_fitted()
        texts = list(texts)
        if self.classifier is not None:
            return self._classify(texts, show_progress)
        return [self._lexicon_scores(t) for t in tqdm(texts, desc="Analyzing", disable=not show_progress)]

    def _classify(self, texts: list[object], show_progress: bool) -> list[EmotionResult]:
        clf = self.classifier
        results: list[EmotionResult] = []
        for start in tqdm(range(0, len(texts), 2048), desc="Analyzing", disable=not show_progress):
            chunk = [to_text(t) for t in texts[start : start + 2048]]
            proba = clf.predict_proba(chunk) if chunk else np.zeros((0, len(clf.labels)))
            for text, p in zip(chunk, proba, strict=True):
                if not text.strip():
                    results.append(EmotionResult(text, {e: 0.0 for e in EMOTIONS}))
                    continue
                by_label = dict(zip(clf.labels, p.tolist(), strict=True))
                scores = {e: round(by_label.get(e, 0.0) * 100, 2) for e in EMOTIONS}
                neutral = round(by_label.get(NONE_LABEL, 0.0) * 100, 2)
                result = EmotionResult(text, scores, (), neutral)
                if result.dominant:
                    result = EmotionResult(text, scores, tuple(clf.top_terms(text, result.dominant)), neutral)
                results.append(result)
        return results

    def _lexicon_scores(self, text: object) -> EmotionResult:
        lex = self.lexicon
        weights = lex.word_weights
        reverse = {v: k for k, v in lex.emotion_map.items()}
        scores: defaultdict[int, float] = defaultdict(float)
        total = 0.0
        matched: list[str] = []

        def add(entry: str) -> bool:
            nonlocal total
            hit = False
            for emotion_id, weight in weights.get(entry, {}).items():
                scores[emotion_id] += weight
                total += weight
                hit = True
            return hit

        raw, stems, negated = self._tokens(text)
        used = [False] * len(raw)
        if self.match_phrases:
            # hazm joins compound verbs ("زده شدم" -> "زده_شدم"); match phrases on the pieces
            pieces, owner = [], []
            for i, tok in enumerate(raw):
                for piece in tok.split("_"):
                    pieces.append(piece)
                    owner.append(i)
            taken = [False] * len(pieces)
            for size in range(_MAX_PHRASE, 1, -1):
                for j in range(len(pieces) - size + 1):
                    if any(taken[j : j + size]):
                        continue
                    phrase = " ".join(pieces[j : j + size])
                    if phrase in weights:
                        owners = set(owner[j : j + size])
                        if not any(negated[i] for i in owners) and add(phrase):
                            matched.append(phrase)
                        taken[j : j + size] = [True] * size
                        for i in owners:
                            used[i] = True
        for i, stem in enumerate(stems):
            if stem is None or used[i] or negated[i]:
                continue
            hit = False
            for form in word_forms(stem):
                hit = add(form) or hit
            if hit:
                matched.append(stem)

        if total <= 0:
            pct = {name: 0.0 for name in lex.emotion_map}
        else:
            pct = {reverse[eid]: round(scores.get(eid, 0) / total * 100, 2) for eid in lex.emotion_map.values()}
        return EmotionResult(to_text(text), pct, tuple(dict.fromkeys(matched)))

    def scores(self, text: str) -> dict[str, float]:
        """Shortcut for ``analyze(text).scores``."""
        return self.analyze(text).scores

    def dominant(self, text: str) -> str | None:
        """Strongest emotion in ``text`` or ``None``."""
        return self.analyze(text).dominant

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
            if self.backend == "classifier":
                df["no_emotion" if en else "بدون_احساس"] = [r.neutral for r in results]
            df["dominant_emotion"] = [(r.dominant_en if en else r.dominant) or "" for r in results]
        if output_csv is not None:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            logger.info("Results saved to %s", output_csv)
        return df


__all__ = ["EMOTIONS", "EMOTION_MAP", "NONE_LABEL", "EmotionAnalyzer", "emotion_name"]
