"""Review polarity: ``recommended`` / ``not_recommended`` / ``no_idea``."""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._models import LinearClassifier, WordEmbeddings, load_comment_model, save_comment_model
from ._text import clean_polarity_text, polarity_tokens, to_text
from ._utils import as_frame, bundled_model_path, logger, resolve_column, warn_legacy_pickle
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError
from .results import EvaluationReport, PolarityResult, TrainingReport

LABELS: tuple[str, ...] = ("not_recommended", "recommended", "no_idea")
EMPTY_TEXT_LABEL = "no_idea"
MODEL_FILENAME = "comment_model.npz"
_LEGACY_LABELS = {0: "not_recommended", 1: "recommended", 2: "no_idea"}


def _resolve_model_file(path: str | os.PathLike) -> Path:
    path = Path(path)
    return path if path.suffix == ".npz" else path / MODEL_FILENAME


class PolarityClassifier:
    """Classify Persian reviews as ``recommended``, ``not_recommended`` or ``no_idea``.

    Word2Vec sentence embeddings (mean of word vectors) + multinomial logistic regression.

    Example:
        >>> clf = PolarityClassifier.pretrained()
        >>> clf.predict("کیفیت عالی داشت")
        'recommended'
        >>> clf.analyze("افتضاح بود").label
        'not_recommended'
    """

    def __init__(self, embeddings: WordEmbeddings | None = None, classifier: LinearClassifier | None = None) -> None:
        self.embeddings = embeddings
        self.classifier = classifier

    def __repr__(self) -> str:
        if not self.is_fitted:
            return "PolarityClassifier(not fitted)"
        return (
            f"PolarityClassifier(vocab={len(self.embeddings)}, vector_size={self.embeddings.vector_size}, "
            f"labels={self.labels})"
        )

    # ---------------------------------------------------------------- loading
    @classmethod
    def pretrained(cls) -> PolarityClassifier:
        """The model bundled with PerSent (~65 % accuracy on Digikala-style reviews)."""
        path = bundled_model_path(MODEL_FILENAME)
        if not path.exists():
            raise ModelFileError(f"Bundled model {path} is missing; reinstall PerSent")
        return cls(*load_comment_model(path))

    @classmethod
    def load(cls, path: str | os.PathLike) -> PolarityClassifier:
        """Load a model saved with :meth:`save` (a directory or a ``.npz`` file).

        Directories with PerSent <= 1.3 pickle files (``classifier.joblib`` +
        ``word2vec.model``) are also accepted, with a ``DeprecationWarning``.
        """
        file = _resolve_model_file(path)
        if file.exists():
            return cls(*load_comment_model(file))
        directory = Path(path)
        legacy_clf, legacy_w2v = directory / "classifier.joblib", directory / "word2vec.model"
        if legacy_clf.exists() and legacy_w2v.exists():
            return cls._load_legacy(legacy_clf, legacy_w2v)
        raise ModelFileError(f"No PerSent polarity model found at {path}")

    @classmethod
    def _load_legacy(cls, clf_path: Path, w2v_path: Path, stacklevel: int = 5) -> PolarityClassifier:
        import joblib
        from gensim.models import Word2Vec

        warn_legacy_pickle(clf_path.parent, stacklevel=stacklevel)
        clf = joblib.load(clf_path)
        w2v = Word2Vec.load(str(w2v_path))
        labels = [_LEGACY_LABELS.get(int(c), str(c)) if str(c).isdigit() else str(c) for c in clf.classes_]
        return cls(
            WordEmbeddings(list(w2v.wv.index_to_key), w2v.wv.vectors), LinearClassifier.from_sklearn(clf, labels)
        )

    def save(self, path: str | os.PathLike) -> Path:
        """Save to a directory (as ``comment_model.npz``) or to a ``.npz`` file path."""
        self._require_fitted()
        file = _resolve_model_file(path)
        save_comment_model(file, self.embeddings, self.classifier)
        logger.info("Model saved to %s", file)
        return file

    # ---------------------------------------------------------------- state
    @property
    def is_fitted(self) -> bool:
        return self.embeddings is not None and self.classifier is not None

    @property
    def labels(self) -> tuple[str, ...]:
        self._require_fitted()
        return tuple(self.classifier.labels)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelNotLoadedError(
                "Model not trained! Use PolarityClassifier.pretrained(), PolarityClassifier.load(path) or fit()."
            )

    # ------------------------------------------------------------- inference
    def _vectors(self, texts: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
        """Sentence vectors and a mask of texts that contain letters."""
        cleaned = [clean_polarity_text(t) for t in texts]
        has_text = np.array([bool(c) for c in cleaned], dtype=bool)
        size = self.embeddings.vector_size
        X = np.zeros((len(cleaned), size), dtype=np.float32)
        for i, c in enumerate(cleaned):
            if c:
                X[i] = self.embeddings.sentence_vector(polarity_tokens(c, cleaned=True))
        return X, has_text

    def _empty_scores(self) -> dict[str, float]:
        return {label: float(label == EMPTY_TEXT_LABEL) for label in self.classifier.labels}

    def predict(self, text: str) -> str:
        """Label for one text. Texts without letters return ``"no_idea"``."""
        return self.predict_many([text])[0]

    def predict_many(self, texts: Iterable[object], show_progress: bool = False) -> list[str]:
        self._require_fitted()
        texts = list(texts)
        out: list[str] = []
        for chunk in self._chunks(texts, show_progress):
            X, has_text = self._vectors(chunk)
            preds = self.classifier.predict(X) if len(chunk) else []
            out.extend(p if ok else EMPTY_TEXT_LABEL for p, ok in zip(preds, has_text, strict=True))
        return out

    def predict_proba(self, text: str) -> dict[str, float]:
        """Probability of each label for one text."""
        return self.analyze(text).scores

    def analyze(self, text: str) -> PolarityResult:
        """Label and per-label probabilities for one text."""
        return self.analyze_many([text])[0]

    def analyze_many(self, texts: Iterable[object], show_progress: bool = False) -> list[PolarityResult]:
        self._require_fitted()
        texts = list(texts)
        labels = self.classifier.labels
        results: list[PolarityResult] = []
        for chunk in self._chunks(texts, show_progress):
            X, has_text = self._vectors(chunk)
            proba = self.classifier.predict_proba(X) if len(chunk) else np.zeros((0, len(labels)))
            for text, p, ok in zip(chunk, proba, has_text, strict=True):
                if ok:
                    scores = {label: float(v) for label, v in zip(labels, p, strict=True)}
                    label = labels[int(np.argmax(p))]
                else:
                    scores, label = self._empty_scores(), EMPTY_TEXT_LABEL
                results.append(PolarityResult(to_text(text), label, scores))
        return results

    @staticmethod
    def _chunks(texts: list, show_progress: bool, size: int = 1024):
        bar = tqdm(total=len(texts), desc="Analyzing", disable=not show_progress)
        try:
            for start in range(0, max(len(texts), 1), size):
                chunk = texts[start : start + size]
                yield chunk
                bar.update(len(chunk))
        finally:
            bar.close()

    # -------------------------------------------------------------- training
    def fit(
        self,
        texts: Sequence[object],
        labels: Sequence[object],
        *,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 5,
        test_size: float = 0.2,
        random_state: int | None = None,
        workers: int = 4,
        C: float = 1.0,
        show_progress: bool = False,
    ) -> TrainingReport:
        """Train Word2Vec + logistic regression on ``texts`` / ``labels``.

        Labels must be ``recommended``, ``not_recommended`` or ``no_idea``
        (missing labels count as ``no_idea``). A stratified ``test_size`` split is
        held out for the returned :class:`~PerSent.results.TrainingReport`; the
        model itself is trained on the remaining data.
        """
        from gensim.models import Word2Vec
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        texts = list(texts)
        y = np.array([to_text(v).strip() or EMPTY_TEXT_LABEL for v in labels], dtype=object)
        if len(texts) != len(y):
            raise DataValidationError("texts and labels must have the same length")
        unknown = sorted(set(y) - set(LABELS))
        if unknown:
            raise DataValidationError(f"Unknown labels {unknown}; allowed labels are {list(LABELS)}")
        if len(set(y)) < 2:
            raise DataValidationError("Training data must contain at least two different labels")

        iterator = tqdm(texts, desc="Preprocessing", disable=not show_progress)
        tokens = [polarity_tokens(t) for t in iterator]
        w2v = Word2Vec(
            sentences=tokens,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            epochs=epochs,
            workers=workers,
            seed=random_state if random_state is not None else 1,
        )
        embeddings = WordEmbeddings(list(w2v.wv.index_to_key), w2v.wv.vectors)
        X = np.vstack([embeddings.sentence_vector(t) for t in tokens])

        X_test = y_test = None
        if test_size and test_size > 0:
            counts = pd.Series(y).value_counts()
            stratify = y if counts.min() >= 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
        else:
            X_train, y_train = X, y
        if len(set(y_train)) < 2:
            raise DataValidationError("The training split must contain at least two different labels")

        sk_model = LogisticRegression(max_iter=1000, C=C)
        sk_model.fit(X_train, y_train)
        self.embeddings = embeddings
        self.classifier = LinearClassifier.from_sklearn(sk_model)

        if X_test is None or not len(X_test):
            return TrainingReport(n_train=len(y_train), n_test=0, labels=self.labels)
        ev = self._evaluate_arrays(X_test, list(y_test))
        return TrainingReport(
            n_train=len(y_train),
            n_test=len(y_test),
            accuracy=ev.accuracy,
            macro_f1=ev.macro_f1,
            per_label=ev.per_label,
            labels=self.labels,
        )

    def fit_csv(
        self,
        data: str | os.PathLike | pd.DataFrame,
        text_column: int | str = "body",
        label_column: int | str = "recommendation_status",
        **kwargs,
    ) -> TrainingReport:
        """:meth:`fit` on a CSV file (or DataFrame) with text and label columns."""
        df = as_frame(data, "Training")
        try:
            text_c = resolve_column(df, text_column, "Text column")
            label_c = resolve_column(df, label_column, "Label column")
        except DataValidationError as exc:
            raise DataValidationError(
                f"CSV file must contain '{text_column}' and '{label_column}' columns ({exc})"
            ) from exc
        return self.fit(df[text_c].tolist(), df[label_c].tolist(), **kwargs)

    # ------------------------------------------------------------ evaluation
    def evaluate(self, texts: Sequence[object], labels: Sequence[object]) -> EvaluationReport:
        """Accuracy, macro-F1, per-label metrics and confusion matrix on labelled data."""
        self._require_fitted()
        y_true = [to_text(v).strip() or EMPTY_TEXT_LABEL for v in labels]
        y_pred = self.predict_many(texts)
        return self._report(y_true, y_pred)

    def evaluate_csv(
        self,
        data: str | os.PathLike | pd.DataFrame,
        text_column: int | str = "body",
        label_column: int | str = "recommendation_status",
    ) -> EvaluationReport:
        df = as_frame(data)
        return self.evaluate(
            df[resolve_column(df, text_column)].tolist(), df[resolve_column(df, label_column)].tolist()
        )

    def _evaluate_arrays(self, X: np.ndarray, y_true: list[str]) -> EvaluationReport:
        return self._report(y_true, self.classifier.predict(X))

    def _report(self, y_true: list[str], y_pred: list[str]) -> EvaluationReport:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

        labels = list(self.classifier.labels)
        p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        per_label = {
            lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, lab in enumerate(labels)
        }
        return EvaluationReport(
            n=len(y_true),
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
            per_label=per_label,
            confusion_matrix=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            labels=tuple(labels),
        )

    # ------------------------------------------------------------------ CSV
    def analyze_csv(
        self,
        input_csv: str | os.PathLike | pd.DataFrame,
        output_csv: str | os.PathLike | None = None,
        *,
        text_column: int | str = 0,
        output_column: str = "sentiment",
        include_scores: bool = False,
        summary_csv: str | os.PathLike | None = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Analyze a CSV column and return (and optionally save) the result.

        Adds ``output_column`` with the label and, with ``include_scores=True``,
        one ``score_<label>`` column per label. Files are written as UTF-8 with
        BOM so they open correctly in Excel. Raises on errors.
        """
        self._require_fitted()
        df = as_frame(input_csv).copy()
        column = resolve_column(df, text_column)
        results = self.analyze_many(df[column].tolist(), show_progress=show_progress)
        df[output_column] = [r.label for r in results]
        if include_scores:
            for label in self.classifier.labels:
                df[f"score_{label}"] = [round(r.scores[label], 6) for r in results]
        if output_csv is not None:
            df.to_csv(output_csv, index=False, encoding="utf-8-sig")
            logger.info("Results saved to %s", output_csv)
        if summary_csv is not None:
            self.summarize(df[output_column]).to_csv(summary_csv, index=False, encoding="utf-8-sig")
            logger.info("Summary report saved to %s", summary_csv)
        return df

    @staticmethod
    def summarize(labels: Iterable[str]) -> pd.DataFrame:
        """Count and percentage of each label."""
        series = pd.Series(list(labels), dtype=object)
        total = len(series)
        counts = series.value_counts()
        rows = [
            {
                "label": label,
                "count": int(counts.get(label, 0)),
                "percentage": round(100 * counts.get(label, 0) / total, 2) if total else 0.0,
            }
            for label in ("recommended", "not_recommended", "no_idea")
        ]
        return pd.DataFrame(rows)


__all__ = ["LABELS", "PolarityClassifier"]
