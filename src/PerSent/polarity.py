"""Review polarity: ``recommended`` / ``not_recommended`` / ``no_idea``.

Two model types:

* ``tfidf`` (default since 1.6): word + character n-gram TF-IDF and logistic regression.
* ``word2vec`` (PerSent <= 1.5): mean Word2Vec sentence vectors and logistic regression.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._models import LinearClassifier, WordEmbeddings, load_comment_model, save_comment_model
from ._text import clean_polarity_text, polarity_tokens, to_text
from ._tfidf import TfidfLinearModel, load_tfidf_model, npz_model_type
from ._utils import as_frame, bundled_model_path, logger, resolve_column, warn_legacy_pickle
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError
from .results import EvaluationReport, PolarityResult, TrainingReport

LABELS: tuple[str, ...] = ("not_recommended", "recommended", "no_idea")
EMPTY_TEXT_LABEL = "no_idea"
MODEL_FILENAME = "comment_model.npz"  # Word2Vec model file (PerSent 1.4/1.5 name)
TFIDF_FILENAME = "polarity_model.npz"
METHODS = ("tfidf", "word2vec")
#: Bundled models: name -> file inside PerSent/models.
PRETRAINED: dict[str, str] = {"tfidf-v2": "polarity_tfidf_v2.npz", "word2vec-v1": MODEL_FILENAME}
DEFAULT_PRETRAINED = "tfidf-v2"
_LEGACY_LABELS = {0: "not_recommended", 1: "recommended", 2: "no_idea"}
_W2V_OPTIONS = {"vector_size", "window", "min_count", "epochs", "workers"}
_TFIDF_OPTIONS = {"word_ngrams", "char_ngrams", "max_word_features", "max_char_features", "min_df", "class_weight"}


class PolarityClassifier:
    """Classify Persian reviews as ``recommended``, ``not_recommended`` or ``no_idea``.

    Example:
        >>> clf = PolarityClassifier.pretrained()          # TF-IDF model, ~75 % accuracy
        >>> clf.predict("کیفیت عالی داشت")
        'recommended'
        >>> clf.analyze("افتضاح بود").label
        'not_recommended'
        >>> PolarityClassifier.pretrained("word2vec-v1")   # the PerSent <= 1.5 model
    """

    def __init__(
        self,
        embeddings: WordEmbeddings | None = None,
        classifier: LinearClassifier | None = None,
        *,
        tfidf: TfidfLinearModel | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.classifier = classifier
        self.tfidf = tfidf
        self.name: str | None = None

    def __repr__(self) -> str:
        if self.method == "tfidf":
            return f"PolarityClassifier(method='tfidf', features={self.tfidf.n_features}, labels={self.labels})"
        if self.method == "word2vec":
            return (
                f"PolarityClassifier(method='word2vec', vocab={len(self.embeddings)}, "
                f"vector_size={self.embeddings.vector_size}, labels={self.labels})"
            )
        return "PolarityClassifier(not fitted)"

    # ---------------------------------------------------------------- loading
    @staticmethod
    def available_models() -> tuple[str, ...]:
        """Names accepted by :meth:`pretrained`."""
        return tuple(PRETRAINED)

    @classmethod
    def pretrained(cls, name: str | None = None) -> PolarityClassifier:
        """A model bundled with PerSent.

        ``"tfidf-v2"`` (default): TF-IDF + logistic regression trained on 300 000
        Digikala reviews (see ``docs/MODEL_CARD.md``). ``"word2vec-v1"``: the model of
        PerSent <= 1.5, kept for reproducibility.
        """
        name = name or DEFAULT_PRETRAINED
        if name not in PRETRAINED:
            raise ModelFileError(f"Unknown pretrained polarity model {name!r}; available: {list(PRETRAINED)}")
        path = bundled_model_path(PRETRAINED[name])
        if not path.exists():
            raise ModelFileError(f"Bundled model {path} is missing; reinstall PerSent")
        model = cls._from_file(path)
        model.name = name
        return model

    @classmethod
    def _from_file(cls, file: Path) -> PolarityClassifier:
        if npz_model_type(file) == "tfidf_lr":
            return cls(tfidf=load_tfidf_model(file))
        return cls(*load_comment_model(file))

    @classmethod
    def load(cls, path: str | os.PathLike) -> PolarityClassifier:
        """Load a model saved with :meth:`save` (a directory or a ``.npz`` file).

        Directories with PerSent <= 1.3 pickle files (``classifier.joblib`` +
        ``word2vec.model``) are also accepted, with a ``DeprecationWarning``.
        """
        path = Path(path)
        if path.suffix == ".npz":
            if not path.exists():
                raise ModelFileError(f"No PerSent polarity model found at {path}")
            return cls._from_file(path)
        for name in (TFIDF_FILENAME, MODEL_FILENAME):
            if (path / name).exists():
                return cls._from_file(path / name)
        legacy_clf, legacy_w2v = path / "classifier.joblib", path / "word2vec.model"
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
        embeddings = WordEmbeddings(list(w2v.wv.index_to_key), w2v.wv.vectors)
        return cls(embeddings, LinearClassifier.from_sklearn(clf, labels))

    def save(self, path: str | os.PathLike) -> Path:
        """Save to a ``.npz`` file, or into a directory.

        In a directory the file is ``polarity_model.npz`` (TF-IDF) or
        ``comment_model.npz`` (Word2Vec).
        """
        self._require_fitted()
        path = Path(path)
        if self.method == "tfidf":
            file = path if path.suffix == ".npz" else path / TFIDF_FILENAME
            self.tfidf.save(file)
        else:
            file = path if path.suffix == ".npz" else path / MODEL_FILENAME
            save_comment_model(file, self.embeddings, self.classifier)
        logger.info("Model saved to %s", file)
        return file

    # ---------------------------------------------------------------- state
    @property
    def method(self) -> str | None:
        """``"tfidf"``, ``"word2vec"`` or ``None`` if not fitted."""
        if self.tfidf is not None:
            return "tfidf"
        if self.embeddings is not None and self.classifier is not None:
            return "word2vec"
        return None

    @property
    def is_fitted(self) -> bool:
        return self.method is not None

    @property
    def labels(self) -> tuple[str, ...]:
        self._require_fitted()
        return tuple(self.tfidf.labels if self.method == "tfidf" else self.classifier.labels)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise ModelNotLoadedError(
                "Model not trained! Use PolarityClassifier.pretrained(), PolarityClassifier.load(path) or fit()."
            )

    # ------------------------------------------------------------- inference
    def _proba(self, texts: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
        """Label probabilities and a mask of texts that contain letters."""
        cleaned = [clean_polarity_text(t) for t in texts]
        has_text = np.array([bool(c) for c in cleaned], dtype=bool)
        n_labels = len(self.labels)
        proba = np.zeros((len(cleaned), n_labels))
        idx = np.flatnonzero(has_text)
        if len(idx):
            if self.method == "tfidf":
                proba[idx] = self.tfidf.predict_proba([to_text(texts[i]) for i in idx])
            else:
                size = self.embeddings.vector_size
                X = np.zeros((len(idx), size), dtype=np.float32)
                for row, i in enumerate(idx):
                    X[row] = self.embeddings.sentence_vector(polarity_tokens(cleaned[i], cleaned=True))
                proba[idx] = self.classifier.predict_proba(X)
        return proba, has_text

    def _empty_scores(self) -> dict[str, float]:
        return {label: float(label == EMPTY_TEXT_LABEL) for label in self.labels}

    def predict(self, text: str) -> str:
        """Label for one text. Texts without letters return ``"no_idea"``."""
        return self.predict_many([text])[0]

    def predict_many(self, texts: Iterable[object], show_progress: bool = False) -> list[str]:
        return [r.label for r in self.analyze_many(texts, show_progress=show_progress)]

    def predict_proba(self, text: str) -> dict[str, float]:
        """Probability of each label for one text."""
        return self.analyze(text).scores

    def analyze(self, text: str) -> PolarityResult:
        """Label and per-label probabilities for one text."""
        return self.analyze_many([text])[0]

    def analyze_many(self, texts: Iterable[object], show_progress: bool = False) -> list[PolarityResult]:
        self._require_fitted()
        texts = list(texts)
        labels = self.labels
        results: list[PolarityResult] = []
        for chunk in self._chunks(texts, show_progress):
            proba, has_text = self._proba(chunk)
            for text, p, ok in zip(chunk, proba, has_text, strict=True):
                if ok:
                    scores = {label: float(v) for label, v in zip(labels, p, strict=True)}
                    label = labels[int(np.argmax(p))]
                else:
                    scores, label = self._empty_scores(), EMPTY_TEXT_LABEL
                results.append(PolarityResult(to_text(text), label, scores))
        return results

    def explain(self, text: str, k: int = 5) -> list[str]:
        """Words/phrases of ``text`` that pushed the TF-IDF model most towards its prediction."""
        self._require_fitted()
        if self.method != "tfidf":
            return []
        return self.tfidf.top_terms(text, self.predict(text), k)

    @staticmethod
    def _chunks(texts: list, show_progress: bool, size: int = 2048):
        bar = tqdm(total=len(texts), desc="Analyzing", disable=not show_progress)
        try:
            for start in range(0, len(texts), size):
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
        method: str = "tfidf",
        test_size: float = 0.2,
        random_state: int | None = None,
        C: float | None = None,
        show_progress: bool = False,
        **options,
    ) -> TrainingReport:
        """Train a model on ``texts`` / ``labels``.

        Labels must be ``recommended``, ``not_recommended`` or ``no_idea`` (missing
        labels count as ``no_idea``). A stratified ``test_size`` split is held out
        for the returned :class:`~PerSent.results.TrainingReport`; the model is
        trained on the rest.

        ``method="tfidf"`` (default) options: ``word_ngrams=(1, 2)``,
        ``char_ngrams=(2, 5)``, ``max_word_features=200_000``,
        ``max_char_features=100_000``, ``min_df=3``, ``class_weight=None``; ``C=2.0``
        (the configuration of the bundled ``tfidf-v2`` model).

        ``method="word2vec"`` options: ``vector_size=100``, ``window=5``,
        ``min_count=1``, ``epochs=5``, ``workers=4``; ``C=1.0``.
        """
        from sklearn.model_selection import train_test_split

        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        allowed = _TFIDF_OPTIONS if method == "tfidf" else _W2V_OPTIONS
        unknown_opts = set(options) - allowed
        if unknown_opts:
            raise TypeError(f"Unknown options for method={method!r}: {sorted(unknown_opts)}")

        texts = list(texts)
        y = np.array([to_text(v).strip() or EMPTY_TEXT_LABEL for v in labels], dtype=object)
        if len(texts) != len(y):
            raise DataValidationError("texts and labels must have the same length")
        unknown = sorted(set(y) - set(LABELS))
        if unknown:
            raise DataValidationError(f"Unknown labels {unknown}; allowed labels are {list(LABELS)}")
        if len(set(y)) < 2:
            raise DataValidationError("Training data must contain at least two different labels")

        idx = np.arange(len(texts))
        test_idx = np.array([], dtype=int)
        if test_size and test_size > 0:
            counts = pd.Series(y).value_counts()
            stratify = y if counts.min() >= 2 else None
            idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=stratify)
        if len(set(y[idx])) < 2:
            raise DataValidationError("The training split must contain at least two different labels")

        if method == "tfidf":
            self._fit_tfidf([texts[i] for i in idx], list(y[idx]), C=2.0 if C is None else C, **options)
        else:
            self._fit_word2vec(texts, y, idx, random_state, C=1.0 if C is None else C, show=show_progress, **options)
        self.name = None

        if not len(test_idx):
            return TrainingReport(n_train=len(idx), n_test=0, labels=self.labels)
        ev = self.evaluate([texts[i] for i in test_idx], list(y[test_idx]))
        return TrainingReport(
            n_train=len(idx),
            n_test=len(test_idx),
            accuracy=ev.accuracy,
            macro_f1=ev.macro_f1,
            per_label=ev.per_label,
            labels=self.labels,
        )

    def _fit_tfidf(self, texts: list, labels: list, *, C: float, **options) -> None:
        self.tfidf = TfidfLinearModel.train(texts, labels, C=C, **options).reorder(LABELS)
        self.embeddings = self.classifier = None

    def _fit_word2vec(
        self,
        texts: list,
        y: np.ndarray,
        train_idx: np.ndarray,
        random_state: int | None,
        *,
        C: float,
        show: bool,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        epochs: int = 5,
        workers: int = 4,
    ) -> None:
        from gensim.models import Word2Vec
        from sklearn.linear_model import LogisticRegression

        # Word2Vec is unsupervised and (as in PerSent <= 1.5) sees every text
        tokens = [polarity_tokens(t) for t in tqdm(texts, desc="Preprocessing", disable=not show)]
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
        X = np.vstack([embeddings.sentence_vector(tokens[i]) for i in train_idx])
        sk_model = LogisticRegression(max_iter=1000, C=C).fit(X, y[train_idx])
        self.embeddings, self.classifier, self.tfidf = embeddings, LinearClassifier.from_sklearn(sk_model), None

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

    def _report(self, y_true: list[str], y_pred: list[str]) -> EvaluationReport:
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

        labels = list(self.labels)
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
            for label in self.labels:
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


__all__ = ["DEFAULT_PRETRAINED", "LABELS", "METHODS", "PRETRAINED", "PolarityClassifier"]
