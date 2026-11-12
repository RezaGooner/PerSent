"""TF-IDF (word + character n-grams) + multinomial logistic regression, stored without pickle.

The fitted scikit-learn vectorizers are saved as their vocabulary (ordered by feature
index) plus IDF weights and constructor parameters; at load time new vectorizers are
built with ``vocabulary=`` and ``idf_`` so ``transform`` is exactly scikit-learn's.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._text import tfidf_text
from .exceptions import ModelFileError

MODEL_TYPE = "tfidf_lr"
FORMAT_VERSION = 2

# constructor parameters we persist (everything else stays at scikit-learn's defaults)
_VECTORIZER_KEYS = ("analyzer", "ngram_range", "lowercase", "sublinear_tf", "norm", "use_idf", "smooth_idf")


def _make_vectorizer(params: dict, vocabulary: dict | None = None):
    from sklearn.feature_extraction.text import TfidfVectorizer

    kwargs = {k: (tuple(v) if k == "ngram_range" else v) for k, v in params.items() if k in _VECTORIZER_KEYS}
    return TfidfVectorizer(dtype=np.float32, vocabulary=vocabulary, **kwargs)


class TfidfLinearModel:
    """Text classifier: union of TF-IDF vectorizers followed by a linear softmax layer."""

    def __init__(self, vectorizers: list, coef: np.ndarray, intercept: np.ndarray, labels: Sequence[str]) -> None:
        self.vectorizers = list(vectorizers)
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.labels = [str(label) for label in labels]
        self.meta: dict = {}
        n_features = sum(len(v.vocabulary_) for v in self.vectorizers)
        if self.coef.shape != (len(self.labels), n_features):
            raise ValueError("coef shape does not match labels/vocabulary")

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        word_ngrams: tuple[int, int] = (1, 2),
        char_ngrams: tuple[int, int] | None = (2, 5),
        max_word_features: int | None = 200_000,
        max_char_features: int | None = 100_000,
        min_df: int = 3,
        C: float = 2.0,
        class_weight: str | dict | None = None,
        max_iter: int = 1000,
    ) -> TfidfLinearModel:
        from scipy.sparse import hstack
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        docs = [tfidf_text(t) for t in texts]
        min_df_eff = min_df if len(docs) >= 10 * min_df else 1
        vectorizers = [
            TfidfVectorizer(
                ngram_range=tuple(word_ngrams),
                min_df=min_df_eff,
                sublinear_tf=True,
                max_features=max_word_features,
                dtype=np.float32,
            )
        ]
        if char_ngrams:
            vectorizers.append(
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=tuple(char_ngrams),
                    min_df=min_df_eff,
                    sublinear_tf=True,
                    max_features=max_char_features,
                    dtype=np.float32,
                )
            )
        X = hstack([v.fit_transform(docs) for v in vectorizers]).tocsr()
        clf = LogisticRegression(max_iter=max_iter, C=C, class_weight=class_weight)
        clf.fit(X, list(labels))
        coef, intercept, classes = clf.coef_, clf.intercept_, [str(c) for c in clf.classes_]
        if coef.shape[0] == 1:  # binary -> two-row softmax form
            coef = np.vstack([-coef[0] / 2, coef[0] / 2])
            intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
        # drop fitting-only state (stop_words_ can be huge)
        for v in vectorizers:
            if hasattr(v, "stop_words_"):
                del v.stop_words_
        return cls(vectorizers, coef, intercept, classes)

    def reorder(self, labels: Sequence[str]) -> TfidfLinearModel:
        """Put the classes in ``labels`` order (labels not listed keep their relative order at the end)."""
        order = [self.labels.index(lab) for lab in labels if lab in self.labels]
        order += [i for i in range(len(self.labels)) if i not in order]
        self.coef = self.coef[order]
        self.intercept = self.intercept[order]
        self.labels = [self.labels[i] for i in order]
        return self

    # ----------------------------------------------------------------- inference
    @property
    def n_features(self) -> int:
        return int(self.coef.shape[1])

    def transform(self, texts: Sequence[str]):
        from scipy.sparse import hstack

        docs = [tfidf_text(t) for t in texts]
        return hstack([v.transform(docs) for v in self.vectorizers]).tocsr()

    def decision_function(self, texts: Sequence[str]) -> np.ndarray:
        X = self.transform(texts)
        return np.asarray(X @ self.coef.T, dtype=np.float64) + self.intercept

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        scores = self.decision_function(texts)
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def top_terms(self, text: str, label: str, k: int = 5) -> list[str]:
        """Word n-grams in ``text`` that push most towards ``label`` (simple explanation)."""
        if label not in self.labels or not self.vectorizers:
            return []
        word_vec = self.vectorizers[0]
        x = word_vec.transform([tfidf_text(text)])
        if x.nnz == 0:
            return []
        row = self.coef[self.labels.index(label), : len(word_vec.vocabulary_)]
        contrib = x.data * row[x.indices]
        order = np.argsort(-contrib)
        inverse = self._inverse_vocab(0)
        return [inverse[x.indices[i]] for i in order[:k] if contrib[i] > 0]

    def _inverse_vocab(self, i: int) -> list[str]:
        cache = self.__dict__.setdefault("_inverse", {})
        if i not in cache:
            vocab = self.vectorizers[i].vocabulary_
            inv = [""] * len(vocab)
            for term, idx in vocab.items():
                inv[idx] = term
            cache[i] = inv
        return cache[i]

    # -------------------------------------------------------------- persistence
    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "model_type": np.array(MODEL_TYPE),
            "format_version": np.array(FORMAT_VERSION),
            "n_vectorizers": np.array(len(self.vectorizers)),
            "coef": self.coef,
            "intercept": self.intercept,
            "labels": np.array(self.labels, dtype=np.str_),
            "meta": np.array(json.dumps(self.meta, ensure_ascii=False)),
        }
        for i, v in enumerate(self.vectorizers):
            params = {k: v.get_params()[k] for k in _VECTORIZER_KEYS}
            arrays[f"params_{i}"] = np.array(json.dumps(params))
            arrays[f"vocab_{i}"] = np.array(self._inverse_vocab(i), dtype=np.str_)
            arrays[f"idf_{i}"] = np.asarray(v.idf_, dtype=np.float32)
        with open(path, "wb") as fh:
            np.savez_compressed(fh, **arrays)

    @classmethod
    def from_npz(cls, data, path: str | os.PathLike = "") -> TfidfLinearModel:
        try:
            if int(data["format_version"]) > FORMAT_VERSION:
                raise ModelFileError(f"{path} uses a newer model format; upgrade PerSent to load it")
            vectorizers = []
            for i in range(int(data["n_vectorizers"])):
                params = json.loads(str(data[f"params_{i}"]))
                vocab = {term: idx for idx, term in enumerate(data[f"vocab_{i}"].tolist())}
                vec = _make_vectorizer(params, vocab)
                vec.idf_ = data[f"idf_{i}"]
                vectorizers.append(vec)
            model = cls(vectorizers, data["coef"], data["intercept"], data["labels"].tolist())
            model.meta = json.loads(str(data["meta"])) if "meta" in data.files else {}
            return model
        except ModelFileError:
            raise
        except (KeyError, ValueError, TypeError) as exc:
            raise ModelFileError(f"Could not read TF-IDF model {path}: {exc}") from exc


def npz_model_type(path: str | os.PathLike) -> str:
    """``"tfidf_lr"`` or ``"word2vec_lr"`` (files without a type are PerSent 1.4/1.5 Word2Vec models)."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return str(data["model_type"]) if "model_type" in data.files else "word2vec_lr"
    except (OSError, ValueError) as exc:
        raise ModelFileError(f"Could not read model file {path}: {exc}") from exc


def load_tfidf_model(path: str | os.PathLike) -> TfidfLinearModel:
    try:
        with np.load(path, allow_pickle=False) as data:
            return TfidfLinearModel.from_npz(data, path)
    except (OSError, ValueError) as exc:
        raise ModelFileError(f"Could not read model file {path}: {exc}") from exc
