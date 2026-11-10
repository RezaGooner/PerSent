"""PerSent: Persian sentiment and emotion analysis.

>>> from PerSent import CommentAnalyzer, SentimentAnalyzer
"""

from __future__ import annotations

from pathlib import Path

from ._version import __version__
from .CommentAnalyzer import CommentAnalyzer
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError, PerSentError
from .SentimentAnalyzer import SentimentAnalyzer

#: Directory containing the pre-trained models shipped with the package (read-only).
MODEL_DIR = Path(__file__).parent / "models"

# No NullHandler on purpose: without logging configuration, errors from analyzeCSV()
# are still printed to stderr (as the print() calls of PerSent <= 1.3 did).

__all__ = [
    "MODEL_DIR",
    "CommentAnalyzer",
    "DataValidationError",
    "ModelFileError",
    "ModelNotLoadedError",
    "PerSentError",
    "SentimentAnalyzer",
    "__version__",
]
