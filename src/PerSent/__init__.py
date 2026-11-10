"""PerSent: Persian sentiment and emotion analysis.

>>> from PerSent import PolarityClassifier, EmotionAnalyzer
>>> PolarityClassifier.pretrained().predict("کیفیت عالی داشت")
'recommended'
>>> EmotionAnalyzer.pretrained().analyze("خیلی خوشحالم").dominant
'شادی'
"""

from __future__ import annotations

from pathlib import Path

from ._version import __version__
from .CommentAnalyzer import CommentAnalyzer
from .emotion import EMOTIONS, EmotionAnalyzer
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError, PerSentError
from .polarity import LABELS, PolarityClassifier
from .results import EMOTION_NAMES_EN, EmotionResult, EvaluationReport, PolarityResult, TrainingReport
from .SentimentAnalyzer import SentimentAnalyzer

#: Directory containing the pre-trained models shipped with the package (read-only).
MODEL_DIR = Path(__file__).parent / "models"

# No NullHandler on purpose: without logging configuration, errors from the deprecated
# analyzeCSV() methods are still printed to stderr (as the print() calls of PerSent <= 1.3 did).

__all__ = [
    "EMOTIONS",
    "EMOTION_NAMES_EN",
    "LABELS",
    "MODEL_DIR",
    "CommentAnalyzer",
    "DataValidationError",
    "EmotionAnalyzer",
    "EmotionResult",
    "EvaluationReport",
    "ModelFileError",
    "ModelNotLoadedError",
    "PerSentError",
    "PolarityClassifier",
    "PolarityResult",
    "SentimentAnalyzer",
    "TrainingReport",
    "__version__",
]
