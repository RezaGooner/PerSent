"""Recommendation / polarity classifier for Persian product and service reviews.

Predicts one of ``"recommended"``, ``"not_recommended"`` or ``"no_idea"``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ._models import LinearClassifier, WordEmbeddings, load_comment_model, save_comment_model
from ._text import COMMENT_STRIP_RE, WHITESPACE_RE, hazm_tools, to_text, word_tokenize
from ._utils import bundled_model_path, logger, read_csv, resolve_column, warn_legacy_pickle
from .exceptions import DataValidationError, ModelFileError, ModelNotLoadedError

LABELS = ("not_recommended", "recommended", "no_idea")
EMPTY_TEXT_LABEL = "no_idea"
MODEL_FILENAME = "comment_model.npz"
_LEGACY_LABELS = {0: "not_recommended", 1: "recommended", 2: "no_idea"}


class CommentAnalyzer:
    """Classify Persian comments as recommended / not_recommended / no_idea.

    Args:
        model_dir: Directory used by :meth:`saveModel` and :meth:`loadModel` for
            your own trained models. If ``None`` (default), :meth:`loadModel`
            loads the pre-trained model shipped with PerSent and :meth:`train`
            does not write anything to disk.

    Example:
        >>> analyzer = CommentAnalyzer()
        >>> analyzer.loadModel()
        >>> analyzer.analyzeText("کیفیت عالی داشت")
        'recommended'
    """

    def __init__(self, model_dir: str | os.PathLike | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir is not None else None
        self.vectorizer: WordEmbeddings | None = None
        self.classifier: LinearClassifier | None = None
        self.last_accuracy: float | None = None

    # ------------------------------------------------------------------ text
    def _clean_text(self, text: object) -> str:
        """Normalize and strip punctuation/digits; returns '' if nothing is left."""
        text = hazm_tools()["normalizer"].normalize(to_text(text))
        text = COMMENT_STRIP_RE.sub(" ", text)
        return WHITESPACE_RE.sub(" ", text).strip()

    def _preprocess_text(self, text: object) -> list[str]:
        """Normalize, clean, tokenize, drop stop-words and stem a Persian text."""
        text = self._clean_text(text)
        if not text:
            return []
        tools = hazm_tools()
        stemmer, stopwords = tools["stemmer"], tools["stopwords"]
        return [stemmer.stem(tok) for tok in word_tokenize(text) if tok not in stopwords and len(tok) > 1]

    def _sentence_vector(self, tokens: list[str]) -> np.ndarray:
        self._require_model()
        return self.vectorizer.sentence_vector(tokens)

    def _require_model(self) -> None:
        if self.classifier is None or self.vectorizer is None:
            raise ModelNotLoadedError(
                "Model not trained! Call train() first or load a pretrained model with loadModel()."
            )

    # -------------------------------------------------------------- training
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
        """Train a Word2Vec + logistic-regression model from a CSV file.

        The CSV must contain a text column (default ``body``) and a label column
        (default ``recommendation_status``) whose values are ``recommended``,
        ``not_recommended`` or ``no_idea``. Empty labels are treated as ``no_idea``.

        If ``model_dir`` was given, the trained model is saved there.

        Returns:
            Accuracy on the held-out test split (``nan`` if ``test_size`` is 0).
        """
        from gensim.models import Word2Vec
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        df = read_csv(train_csv, "Training")
        missing = [c for c in (text_column, label_column) if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"CSV file must contain '{text_column}' and '{label_column}' columns (missing: {missing})"
            )

        labels = df[label_column].fillna(EMPTY_TEXT_LABEL).astype(str).str.strip()
        unknown = sorted(set(labels) - set(LABELS))
        if unknown:
            raise DataValidationError(f"Unknown labels {unknown}; allowed labels are {list(LABELS)}")

        tokens = [self._preprocess_text(t) for t in tqdm(df[text_column], desc="Preprocessing")]
        w2v = Word2Vec(
            sentences=tokens,
            vector_size=vector_size,
            window=window,
            min_count=1,
            workers=workers,
            seed=random_state if random_state is not None else 1,
        )
        embeddings = WordEmbeddings(list(w2v.wv.index_to_key), w2v.wv.vectors)
        X = np.vstack([embeddings.sentence_vector(t) for t in tokens])
        y = labels.to_numpy()

        if test_size and test_size > 0:
            counts = pd.Series(y).value_counts()
            stratify = y if counts.min() >= 2 and len(counts) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify
            )
        else:
            X_train, y_train, X_test, y_test = X, y, None, None

        if len(set(y_train)) < 2:
            raise DataValidationError("Training data must contain at least two different labels")

        sk_model = LogisticRegression(max_iter=1000)
        sk_model.fit(X_train, y_train)

        self.vectorizer = embeddings
        self.classifier = LinearClassifier.from_sklearn(sk_model)

        accuracy = float("nan")
        if X_test is not None and len(X_test):
            accuracy = float(np.mean(np.array(self.classifier.predict(X_test)) == y_test))
        self.last_accuracy = accuracy

        if self.model_dir is not None:
            self.saveModel()
        else:
            logger.info("Model trained in memory; pass model_dir or call saveModel(path) to keep it.")
        return accuracy

    # ------------------------------------------------------------ inference
    def analyzeText(self, text: str) -> str:
        """Return ``"recommended"``, ``"not_recommended"`` or ``"no_idea"`` for one text.

        Texts without any letters (empty, only numbers/punctuation) return
        ``"no_idea"``. Note that hazm's stop-word list contains common opinion
        words such as "خوب" and "عالی"; texts made only of such words are still
        classified by the model (as in previous versions).
        """
        self._require_model()
        cleaned = self._clean_text(text)
        if not cleaned:
            return EMPTY_TEXT_LABEL
        tokens = self._preprocess_text(cleaned)
        return self.classifier.predict(self.vectorizer.sentence_vector(tokens))[0]

    def analyzeTexts(self, texts, show_progress: bool = False) -> list[str]:
        """Analyze an iterable of texts and return a list of labels."""
        iterator = tqdm(texts, desc="Analyzing comments") if show_progress else texts
        return [self.analyzeText(t) for t in iterator]

    # ---------------------------------------------------------- persistence
    def saveModel(self, model_dir: str | os.PathLike | None = None) -> Path:
        """Save the trained model to ``model_dir`` (or the analyzer's ``model_dir``).

        Returns the path of the written file.
        """
        self._require_model()
        target = Path(model_dir) if model_dir is not None else self.model_dir
        if target is None:
            raise ValueError("No model_dir given: use CommentAnalyzer(model_dir=...) or saveModel(model_dir=...)")
        path = target / MODEL_FILENAME
        save_comment_model(path, self.vectorizer, self.classifier)
        logger.info("Model saved to %s", path)
        return path

    def loadModel(self, model_dir: str | os.PathLike | None = None) -> None:
        """Load a model.

        Looks in ``model_dir`` (argument, else the analyzer's ``model_dir``). If no
        directory is set, or it contains no model, the pre-trained model bundled
        with PerSent is loaded. The current working directory is never searched.
        """
        directory = Path(model_dir) if model_dir is not None else self.model_dir
        if directory is not None:
            path = directory / MODEL_FILENAME
            if path.exists():
                self.vectorizer, self.classifier = load_comment_model(path)
                return
            legacy_clf, legacy_w2v = directory / "classifier.joblib", directory / "word2vec.model"
            if legacy_clf.exists() and legacy_w2v.exists():
                self._load_legacy(legacy_clf, legacy_w2v)
                return
            logger.warning("No model found in %s; loading the bundled pre-trained model.", directory)

        path = bundled_model_path(MODEL_FILENAME)
        if not path.exists():
            raise ModelFileError(f"Bundled model {path} is missing; reinstall PerSent")
        self.vectorizer, self.classifier = load_comment_model(path)

    def _load_legacy(self, clf_path: Path, w2v_path: Path) -> None:
        import joblib
        from gensim.models import Word2Vec

        warn_legacy_pickle(clf_path.parent)
        clf = joblib.load(clf_path)
        w2v = Word2Vec.load(str(w2v_path))
        labels = [_LEGACY_LABELS.get(int(c), str(c)) if str(c).isdigit() else str(c) for c in clf.classes_]
        self.vectorizer = WordEmbeddings(list(w2v.wv.index_to_key), w2v.wv.vectors)
        self.classifier = LinearClassifier.from_sklearn(clf, labels)

    # ------------------------------------------------------------------ CSV
    def analyzeCSV(
        self,
        input_csv: str | os.PathLike,
        output_path: str | os.PathLike,
        summary_path: str | os.PathLike | None = None,
        text_column: int | str = 0,
    ) -> pd.DataFrame | None:
        """Analyze every comment in a CSV file.

        Adds a ``sentiment`` column and writes the result to ``output_path``
        (UTF-8 with BOM, so it opens correctly in Excel). If ``summary_path`` is
        given, a per-label count/percentage report is written there too.

        ``text_column`` is a column name or a 0-based (optionally negative) index.

        Returns:
            The resulting DataFrame, or ``None`` if the analysis failed (the
            error is logged).
        """
        try:
            self._require_model()
            df = read_csv(input_csv)
            column = resolve_column(df, text_column)
            df["sentiment"] = self.analyzeTexts(df[column].tolist(), show_progress=True)
            df.to_csv(output_path, index=False, encoding="utf-8-sig")
            logger.info("Results saved to %s", output_path)
            if summary_path:
                self._generate_summary(df).to_csv(summary_path, index=False, encoding="utf-8-sig")
                logger.info("Summary report saved to %s", summary_path)
            return df
        except FileNotFoundError as exc:
            logger.error("File not found - %s", exc)
        except (DataValidationError, ValueError) as exc:
            logger.error("Invalid column specification - %s", exc)
        except Exception as exc:  # keep the historical "return None on failure" contract
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


__all__ = ["CommentAnalyzer"]
