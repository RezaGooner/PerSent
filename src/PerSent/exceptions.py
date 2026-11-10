"""Exceptions raised by PerSent.

All exceptions derive from :class:`PerSentError`, which itself derives from
:class:`Exception`, so existing ``except Exception`` handlers keep working.
"""

from __future__ import annotations


class PerSentError(Exception):
    """Base class for all PerSent errors."""


class ModelNotLoadedError(PerSentError, RuntimeError):
    """Raised when an analysis is requested before a model is trained or loaded."""


class ModelFileError(PerSentError, OSError):
    """Raised when model files are missing, unreadable or in an unknown format."""


class DataValidationError(PerSentError, ValueError):
    """Raised when input data (CSV columns, labels, ...) is invalid."""


__all__ = ["DataValidationError", "ModelFileError", "ModelNotLoadedError", "PerSentError"]
