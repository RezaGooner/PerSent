"""Pickle-free model containers and (de)serialization.

PerSent stores its models in plain numpy (``.npz`` loaded with ``allow_pickle=False``)
and JSON files, so loading a model can never execute code.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .exceptions import ModelFileError

FORMAT_VERSION = 1


class WordEmbeddings:
    """A minimal read-only word -> vector lookup table."""

    def __init__(self, words: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(words):
            raise ValueError("vectors must be a 2D array with one row per word")
        self.words = list(words)
        self.vectors = vectors
        self.index = {w: i for i, w in enumerate(self.words)}

    @property
    def vector_size(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: object) -> bool:
        return word in self.index

    def sentence_vector(self, tokens: Iterable[str]) -> np.ndarray:
        """Mean of the token vectors.

        Out-of-vocabulary tokens contribute a zero vector (this matches how the
        bundled model was trained). An empty token list yields a zero vector.
        """
        tokens = list(tokens)
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)
        total = np.zeros(self.vector_size, dtype=np.float64)
        for token in tokens:
            i = self.index.get(token)
            if i is not None:
                total += self.vectors[i]
        return (total / len(tokens)).astype(np.float32)


class LinearClassifier:
    """Multinomial logistic-regression inference (softmax over ``X @ coef.T + intercept``)."""

    def __init__(self, coef: np.ndarray, intercept: np.ndarray, labels: Sequence[str]) -> None:
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = np.asarray(intercept, dtype=np.float64)
        self.labels = [str(label) for label in labels]
        if self.coef.shape[0] != len(self.labels) or self.intercept.shape[0] != len(self.labels):
            raise ValueError("coef/intercept rows must match the number of labels")

    @classmethod
    def from_sklearn(cls, model, labels: Sequence[str] | None = None) -> LinearClassifier:
        classes = list(model.classes_)
        labels = [str(c) for c in classes] if labels is None else list(labels)
        coef, intercept = model.coef_, model.intercept_
        if coef.shape[0] == 1 and len(labels) == 2:  # binary sklearn model
            coef = np.vstack([-coef[0] / 2, coef[0] / 2])
            intercept = np.array([-intercept[0] / 2, intercept[0] / 2])
        return cls(coef, intercept, labels)

    @property
    def n_features(self) -> int:
        return int(self.coef.shape[1])

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return X @ self.coef.T + self.intercept

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> list[str]:
        idx = self.decision_function(X).argmax(axis=1)
        return [self.labels[i] for i in idx]


def save_comment_model(path: str | os.PathLike, embeddings: WordEmbeddings, classifier: LinearClassifier) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        np.savez_compressed(
            fh,
            format_version=np.array(FORMAT_VERSION),
            words=np.array(embeddings.words, dtype=np.str_),
            vectors=embeddings.vectors,
            coef=classifier.coef,
            intercept=classifier.intercept,
            labels=np.array(classifier.labels, dtype=np.str_),
        )


def load_comment_model(path: str | os.PathLike) -> tuple[WordEmbeddings, LinearClassifier]:
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version > FORMAT_VERSION:
                raise ModelFileError(f"{path} uses model format {version}; upgrade PerSent to load it")
            embeddings = WordEmbeddings(data["words"].tolist(), data["vectors"])
            classifier = LinearClassifier(data["coef"], data["intercept"], data["labels"].tolist())
    except (OSError, KeyError, ValueError) as exc:
        raise ModelFileError(f"Could not read comment model {path}: {exc}") from exc
    if classifier.n_features != embeddings.vector_size:
        raise ModelFileError(f"{path}: classifier and embedding sizes do not match")
    return embeddings, classifier


class EmotionLexicon:
    """Weighted emotion lexicon: ``word_weights[word][emotion_id] = weight``."""

    def __init__(
        self,
        emotion_map: Mapping[str, int],
        keywords: Mapping[int, Iterable[str]] | None = None,
        word_weights: Mapping[str, Mapping[int, float]] | None = None,
    ) -> None:
        # keep a stable order (by emotion id) regardless of how the file was written
        self.emotion_map = {k: int(v) for k, v in sorted(emotion_map.items(), key=lambda kv: int(kv[1]))}
        self.keywords: defaultdict[int, list[str]] = defaultdict(list)
        for k, v in (keywords or {}).items():
            self.keywords[int(k)] = list(v)
        self.word_weights: defaultdict[str, dict[int, float]] = defaultdict(dict)
        for word, weights in (word_weights or {}).items():
            self.word_weights[word] = {int(e): float(w) for e, w in weights.items()}

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "emotion_map": self.emotion_map,
            "keywords": {str(k): sorted(set(v)) for k, v in self.keywords.items()},
            "word_weights": {w: {str(e): v for e, v in ws.items()} for w, ws in self.word_weights.items()},
        }

    def save(self, path: str | os.PathLike) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    @classmethod
    def load(cls, path: str | os.PathLike) -> EmotionLexicon:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            if int(data.get("format_version", 1)) > FORMAT_VERSION:
                raise ModelFileError(f"{path} uses a newer model format; upgrade PerSent to load it")
            return cls(data["emotion_map"], data.get("keywords"), data["word_weights"])
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise ModelFileError(f"Could not read emotion model {path}: {exc}") from exc
