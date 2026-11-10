from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

import PerSent
from PerSent import MODEL_DIR, ModelFileError
from PerSent._models import EmotionLexicon, LinearClassifier, load_comment_model


def test_version():
    assert PerSent.__version__.count(".") == 2


def test_public_api():
    for name in ("CommentAnalyzer", "SentimentAnalyzer", "MODEL_DIR", "PerSentError"):
        assert hasattr(PerSent, name)


def test_bundled_models_are_pickle_free():
    npz = MODEL_DIR / "comment_model.npz"
    with zipfile.ZipFile(npz) as zf:
        for name in zf.namelist():
            with zf.open(name) as fh:
                np.lib.format.read_magic(fh)
    with np.load(npz, allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype != object
    json.loads((MODEL_DIR / "emotion_lexicon.json").read_text(encoding="utf-8"))


def test_corrupt_model_file(tmp_path):
    bad = tmp_path / "comment_model.npz"
    bad.write_bytes(b"garbage")
    with pytest.raises(ModelFileError):
        load_comment_model(bad)
    bad_json = tmp_path / "lex.json"
    bad_json.write_text("{", encoding="utf-8")
    with pytest.raises(ModelFileError):
        EmotionLexicon.load(bad_json)


def test_linear_classifier_matches_sklearn():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 5))
    y = np.array(["a", "b", "c"] * 30)
    X[y == "a"] += 2
    for labels in (y, np.where(y == "a", "a", "b")):  # multiclass and binary
        sk = LogisticRegression(max_iter=500).fit(X, labels)
        ours = LinearClassifier.from_sklearn(sk)
        assert ours.predict(X) == list(sk.predict(X))
        np.testing.assert_allclose(ours.predict_proba(X), sk.predict_proba(X), rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize(
    ("word", "stem"),
    [("کتابی", "کتاب"), ("کتاب‌ها", "کتاب"), ("کتاب‌هایی", "کتاب"), ("کتابهایشان", "کتاب"), ("اندیشه‌اش", "اندیشه")],
)
def test_vendored_stemmer(word, stem):
    from PerSent._stemmer import Stemmer

    assert Stemmer().stem(word) == stem
