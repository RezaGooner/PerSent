from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from PerSent import CommentAnalyzer, ModelNotLoadedError
from PerSent._text import COMMENT_STRIP_RE
from PerSent.exceptions import DataValidationError

LABELS = {"recommended", "not_recommended", "no_idea"}


def test_pretrained_predictions(comment_analyzer):
    assert comment_analyzer.analyzeText("کیفیت عالی داشت") == "recommended"
    assert comment_analyzer.analyzeText("اصلا خوب نبود، پولم رو هدر دادم") == "not_recommended"
    assert comment_analyzer.analyzeText("افتضاح بود") == "not_recommended"


@pytest.mark.parametrize("text", ["", "   ", "123", "!!!", None, float("nan")])
def test_text_without_letters_is_no_idea(comment_analyzer, text):
    assert comment_analyzer.analyzeText(text) == "no_idea"


def test_latin_uppercase_is_preserved():
    # PerSent <= 1.3 deleted A-Z because of an accidental regex range.
    assert COMMENT_STRIP_RE.sub(" ", "iPhone Pro") == "iPhone Pro"


def test_strip_regex_matches_legacy_minus_uppercase():
    legacy = r'[!()-\[\]{};:\'",؟<>./?@#$%^&*_~۰-۹\d]+'
    import re

    chars = [chr(c) for c in range(0x20, 0x7F)] + list("؟۰۹٣")
    for ch in chars:
        expected = bool(re.fullmatch(legacy, ch)) and not ("A" <= ch <= "Z")
        assert bool(COMMENT_STRIP_RE.fullmatch(ch)) == expected, ch


def test_analyze_before_load_raises():
    with pytest.raises(ModelNotLoadedError):
        CommentAnalyzer().analyzeText("سلام")


def test_init_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    CommentAnalyzer()
    assert os.listdir(tmp_path) == []


def test_load_ignores_current_directory(tmp_path, monkeypatch):
    # A planted file in ./PerSent/model must never be loaded.
    planted = tmp_path / "PerSent" / "model"
    planted.mkdir(parents=True)
    (planted / "classifier.joblib").write_bytes(b"not a pickle")
    (planted / "word2vec.model").write_bytes(b"not a model")
    monkeypatch.chdir(tmp_path)
    analyzer = CommentAnalyzer()
    analyzer.loadModel()
    assert analyzer.analyzeText("کیفیت عالی داشت") == "recommended"


@pytest.mark.parametrize("vector_size", [16, 100])
def test_train_save_load_roundtrip(comments_csv, tmp_path, vector_size):
    model_dir = tmp_path / "model"
    analyzer = CommentAnalyzer(model_dir=model_dir)
    acc = analyzer.train(comments_csv, vector_size=vector_size, random_state=0, workers=1)
    assert 0.0 <= acc <= 1.0
    assert (model_dir / "comment_model.npz").exists()
    assert analyzer.vectorizer.vector_size == vector_size

    reloaded = CommentAnalyzer(model_dir=model_dir)
    reloaded.loadModel()
    for text in ["خیلی خوب بود", "افتضاح بود", "نمیدونم"]:
        assert reloaded.analyzeText(text) == analyzer.analyzeText(text)
        assert reloaded.analyzeText(text) in LABELS


def test_train_without_model_dir_writes_nothing(comments_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(os.listdir(tmp_path))
    CommentAnalyzer().train(comments_csv, random_state=0, workers=1)
    assert set(os.listdir(tmp_path)) == before


def test_save_without_model_dir_raises(comments_csv):
    analyzer = CommentAnalyzer()
    analyzer.train(comments_csv, random_state=0, workers=1)
    with pytest.raises(ValueError):
        analyzer.saveModel()


def test_train_missing_labels_become_no_idea(tmp_path):
    path = tmp_path / "nan.csv"
    pd.DataFrame(
        {
            "body": ["خوب بود", "بد بود", "نمیدونم", "معمولی"] * 5,
            "recommendation_status": ["recommended", "not_recommended", None, "no_idea"] * 5,
        }
    ).to_csv(path, index=False)
    acc = CommentAnalyzer().train(path, random_state=0, workers=1)
    assert not np.isnan(acc)


def test_train_rejects_unknown_labels(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"body": ["a", "b"], "recommendation_status": ["recommended", "maybe"]}).to_csv(path, index=False)
    with pytest.raises(DataValidationError):
        CommentAnalyzer().train(path)


def test_train_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        CommentAnalyzer().train(path)


def test_train_missing_file():
    with pytest.raises(FileNotFoundError):
        CommentAnalyzer().train("does-not-exist.csv")


@pytest.mark.parametrize("column", [0, -1, "نظرات"])
def test_analyze_csv(comment_analyzer, tmp_path, column):
    src = tmp_path / "in.csv"
    pd.DataFrame({"نظرات": ["کیفیت عالی داشت", "افتضاح بود", ""]}).to_csv(src, index=False)
    out, summary = tmp_path / "out.csv", tmp_path / "summary.csv"
    df = comment_analyzer.analyzeCSV(src, out, summary, column)
    assert df is not None and list(df["sentiment"])[:2] == ["recommended", "not_recommended"]
    assert out.exists() and summary.exists()
    s = pd.read_csv(summary)
    assert s.loc[s.Category == "Total", "Count"].iloc[0] in (3, "3")


def test_analyze_csv_bad_column_returns_none(comment_analyzer, tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"a": ["x"]}).to_csv(src, index=False)
    assert comment_analyzer.analyzeCSV(src, tmp_path / "out.csv", text_column=5) is None
    assert comment_analyzer.analyzeCSV(src, tmp_path / "out.csv", text_column="zzz") is None
    assert comment_analyzer.analyzeCSV(tmp_path / "missing.csv", tmp_path / "out.csv") is None
