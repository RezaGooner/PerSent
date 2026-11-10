"""Deprecated 1.x interface for review polarity — use :class:`PerSent.PolarityClassifier`.

Kept so that code written for PerSent <= 1.4 keeps working unchanged. It will be
removed in PerSent 2.0.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ._text import clean_polarity_text, polarity_tokens
from ._utils import logger, warn_deprecated
from .exceptions import DataValidationError, ModelNotLoadedError
from .polarity import EMPTY_TEXT_LABEL, LABELS, MODEL_FILENAME, PolarityClassifier

__all__ = ["EMPTY_TEXT_LABEL", "LABELS", "MODEL_FILENAME", "CommentAnalyzer"]


class CommentAnalyzer:
    """Deprecated: use :class:`PerSent.PolarityClassifier`.

    ==============================  ===================================================
    1.x                             1.5+
    ==============================  ===================================================
    ``CommentAnalyzer()``           ``PolarityClassifier()``
    ``loadModel()``                 ``PolarityClassifier.pretrained()`` / ``.load(dir)``
    ``saveModel()``                 ``save(dir)``
    ``train(csv)``                  ``fit_csv(csv)`` (returns a report)
    ``analyzeText(t)``              ``predict(t)`` / ``analyze(t)``
    ``analyzeCSV(i, o, s, col)``    ``analyze_csv(i, o, text_column=col, summary_csv=s)``
    ==============================  ===================================================
    """

    def __init__(self, model_dir: str | os.PathLike | None = None) -> None:
        warn_deprecated("CommentAnalyzer", "PerSent.PolarityClassifier")
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self._clf = PolarityClassifier()
        self.last_accuracy: float | None = None

    # 1.4 exposed these attributes
    @property
    def vectorizer(self):
        return self._clf.embeddings

    @vectorizer.setter
    def vectorizer(self, value) -> None:
        self._clf.embeddings = value

    @property
    def classifier(self):
        return self._clf.classifier

    @classifier.setter
    def classifier(self, value) -> None:
        self._clf.classifier = value

    @property
    def model(self) -> PolarityClassifier:
        """The underlying :class:`PolarityClassifier`."""
        return self._clf

    def _clean_text(self, text: object) -> str:
        return clean_polarity_text(text)

    def _preprocess_text(self, text: object) -> list[str]:
        return polarity_tokens(text)

    def _sentence_vector(self, tokens: list[str]) -> np.ndarray:
        self._require_model()
        return self._clf.embeddings.sentence_vector(tokens)

    def _require_model(self) -> None:
        if not self._clf.is_fitted:
            raise ModelNotLoadedError(
                "Model not trained! Call train() first or load a pretrained model with loadModel()."
            )

    def train(
        self,
        train_csv: str | os.PathLike,
        test_size: float = 0.2,
        vector_size: int = 100,
        window: int = 5,
        random_state: int | None = None,
        text_column: str = "body",
        label_column: str = "recommendation_status",
        workers: int = 4,
    ) -> float:
        """Train from a CSV; returns held-out accuracy (``nan`` without a test split)."""
        report = self._clf.fit_csv(
            train_csv,
            text_column=text_column,
            label_column=label_column,
            test_size=test_size,
            vector_size=vector_size,
            window=window,
            random_state=random_state,
            workers=workers,
            show_progress=True,
        )
        self.last_accuracy = report.accuracy if report.accuracy is not None else float("nan")
        if self.model_dir is not None:
            self.saveModel()
        else:
            logger.info("Model trained in memory; pass model_dir or call saveModel(path) to keep it.")
        return self.last_accuracy

    def analyzeText(self, text: str) -> str:
        self._require_model()
        return self._clf.predict(text)

    def analyzeTexts(self, texts, show_progress: bool = False) -> list[str]:
        self._require_model()
        return self._clf.predict_many(texts, show_progress=show_progress)

    def saveModel(self, model_dir: str | os.PathLike | None = None) -> Path:
        self._require_model()
        target = Path(model_dir) if model_dir is not None else self.model_dir
        if target is None:
            raise ValueError("No model_dir given: use CommentAnalyzer(model_dir=...) or saveModel(model_dir=...)")
        return self._clf.save(target)

    def loadModel(self, model_dir: str | os.PathLike | None = None) -> None:
        """Load from ``model_dir`` if it holds a model, else the bundled pre-trained one."""
        directory = Path(model_dir) if model_dir is not None else self.model_dir
        if directory is not None:
            if (directory / MODEL_FILENAME).exists():
                self._clf = PolarityClassifier.load(directory)
                return
            legacy_clf, legacy_w2v = directory / "classifier.joblib", directory / "word2vec.model"
            if legacy_clf.exists() and legacy_w2v.exists():
                self._clf = PolarityClassifier._load_legacy(legacy_clf, legacy_w2v, stacklevel=5)
                return
            logger.warning("No model found in %s; loading the bundled pre-trained model.", directory)
        self._clf = PolarityClassifier.pretrained()

    def analyzeCSV(
        self,
        input_csv: str | os.PathLike,
        output_path: str | os.PathLike,
        summary_path: str | os.PathLike | None = None,
        text_column: int | str = 0,
    ) -> pd.DataFrame | None:
        """Analyze a CSV column; returns the DataFrame or ``None`` on error (logged)."""
        try:
            self._require_model()
            df = self._clf.analyze_csv(input_csv, output_path, text_column=text_column, output_column="sentiment")
            if summary_path:
                self._generate_summary(df).to_csv(summary_path, index=False, encoding="utf-8-sig")
                logger.info("Summary report saved to %s", summary_path)
            return df
        except FileNotFoundError as exc:
            logger.error("File not found - %s", exc)
        except (DataValidationError, ValueError) as exc:
            logger.error("Invalid column specification - %s", exc)
        except Exception as exc:
            logger.error("Error during CSV analysis: %s", exc)
        return None

    def _generate_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        counts = df["sentiment"].value_counts().to_dict()
        total = len(df)

        def pct(n: int) -> str:
            return f"{100 * n / total:.2f}%" if total else "0.00%"

        rec, notrec, noidea = (counts.get(k, 0) for k in ("recommended", "not_recommended", "no_idea"))
        return pd.DataFrame(
            {
                "Category": ["Recommended", "Not Recommended", "No Idea", "Total", "Model Accuracy"],
                "Count": [rec, notrec, noidea, total, "N/A"],
                "Percentage": [pct(rec), pct(notrec), pct(noidea), "100%", "N/A"],
            }
        )
