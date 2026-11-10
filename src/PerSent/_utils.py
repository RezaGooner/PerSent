"""Internal helpers shared by the analyzers."""

from __future__ import annotations

import logging
import os
import warnings
from importlib import resources
from pathlib import Path

import pandas as pd

from .exceptions import DataValidationError

logger = logging.getLogger("PerSent")


def bundled_model_path(filename: str) -> Path:
    """Path of a pre-trained model file shipped inside the package."""
    return Path(str(resources.files("PerSent") / "models" / filename))


def resolve_column(df: pd.DataFrame, column: int | str, what: str = "Column") -> str:
    """Accept a column name or a (possibly negative) 0-based index and return the name."""
    if isinstance(column, bool):
        raise DataValidationError(f"{what} must be a column name or index, got {column!r}")
    if isinstance(column, int):
        n = len(df.columns)
        idx = column + n if column < 0 else column
        if not 0 <= idx < n:
            raise DataValidationError(f"{what} index {column} is out of range (CSV has {n} columns)")
        return df.columns[idx]
    if column not in df.columns:
        raise DataValidationError(f"{what} '{column}' not found in CSV file. Available: {list(df.columns)}")
    return column


def read_csv(path: str | os.PathLike, what: str = "Input") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} file {path} not found")
    return pd.read_csv(path)


def as_frame(data: str | os.PathLike | pd.DataFrame, what: str = "Input") -> pd.DataFrame:
    """Accept a CSV path or a DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data
    return read_csv(data, what)


def warn_deprecated(old: str, new: str, stacklevel: int = 3) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in PerSent 2.0; use {new} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def warn_legacy_pickle(path: Path, stacklevel: int = 4) -> None:
    warnings.warn(
        f"Loading legacy pickle-based model files from {path}. Pickle files can execute code: "
        "only load files you created yourself. Save the model again (save()/saveModel()) to convert "
        "it to the safe format used since PerSent 1.4.0.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
