"""Result objects returned by :class:`~PerSent.PolarityClassifier` and :class:`~PerSent.EmotionAnalyzer`."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: Persian emotion names -> English names.
EMOTION_NAMES_EN: dict[str, str] = {
    "شادی": "happiness",
    "غم": "sadness",
    "ترس": "fear",
    "عصبانیت": "anger",
    "تنفر": "disgust",
    "شگفتی": "surprise",
    "آرامش": "calmness",
}

#: Polarity labels -> Persian names.
POLARITY_NAMES_FA: dict[str, str] = {
    "recommended": "پیشنهاد می‌کنم",
    "not_recommended": "پیشنهاد نمی‌کنم",
    "no_idea": "نظری ندارم",
}


@dataclass(frozen=True)
class PolarityResult:
    """Prediction for one text.

    Attributes:
        text: The input text.
        label: ``"recommended"``, ``"not_recommended"`` or ``"no_idea"``.
        scores: Probability of every label (sums to 1).
    """

    text: str
    label: str
    scores: dict[str, float]

    @property
    def confidence(self) -> float:
        """Probability of the predicted label."""
        return self.scores.get(self.label, 0.0)

    @property
    def label_fa(self) -> str:
        return POLARITY_NAMES_FA.get(self.label, self.label)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EmotionResult:
    """Emotion scores for one text.

    Attributes:
        text: The input text.
        scores: Percentage (0-100) of each emotion, keyed by Persian name.
        matched_words: Lexicon words found in the text (explains the scores).
    """

    text: str
    scores: dict[str, float]
    matched_words: tuple[str, ...] = ()

    @property
    def dominant(self) -> str | None:
        """Emotion with the highest score, or ``None`` if no emotion word was found."""
        if not self.scores or max(self.scores.values()) <= 0:
            return None
        return max(self.scores, key=self.scores.__getitem__)

    @property
    def dominant_en(self) -> str | None:
        return EMOTION_NAMES_EN.get(self.dominant) if self.dominant else None

    @property
    def has_emotion(self) -> bool:
        return self.dominant is not None

    def scores_en(self) -> dict[str, float]:
        """Scores keyed by English emotion names."""
        return {EMOTION_NAMES_EN.get(k, k): v for k, v in self.scores.items()}

    def to_dict(self, lang: str = "fa") -> dict:
        if lang not in ("fa", "en"):
            raise ValueError("lang must be 'fa' or 'en'")
        en = lang == "en"
        return {
            "text": self.text,
            "scores": self.scores_en() if en else dict(self.scores),
            "dominant": self.dominant_en if en else self.dominant,
            "matched_words": list(self.matched_words),
        }


@dataclass(frozen=True)
class TrainingReport:
    """Summary of a :meth:`PolarityClassifier.fit` run."""

    n_train: int
    n_test: int
    accuracy: float | None = None
    macro_f1: float | None = None
    per_label: dict[str, dict[str, float]] = field(default_factory=dict)
    labels: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        if self.accuracy is None:
            return f"Trained on {self.n_train} texts (no test split)"
        return (
            f"Trained on {self.n_train} texts, tested on {self.n_test}: "
            f"accuracy={self.accuracy:.4f}, macro-F1={self.macro_f1:.4f}"
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Metrics returned by :meth:`PolarityClassifier.evaluate`."""

    n: int
    accuracy: float
    macro_f1: float
    per_label: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    labels: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"n={self.n}: accuracy={self.accuracy:.4f}, macro-F1={self.macro_f1:.4f}"


__all__ = [
    "EMOTION_NAMES_EN",
    "POLARITY_NAMES_FA",
    "EmotionResult",
    "EvaluationReport",
    "PolarityResult",
    "TrainingReport",
]
