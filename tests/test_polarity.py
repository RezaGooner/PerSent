from __future__ import annotations

import math
import os
import warnings

import pandas as pd
import pytest

from PerSent import (
    LABELS,
    CommentAnalyzer,
    DataValidationError,
    ModelFileError,
    ModelNotLoadedError,
    PolarityClassifier,
    PolarityResult,
)

TRAIN_CSV = os.path.join(os.path.dirname(__file__), os.pardir, "dataset", "train.csv")


def test_pretrained_predict(polarity):
    assert polarity.predict("کیفیت عالی داشت") == "recommended"
    assert polarity.predict("افتضاح بود") == "not_recommended"
    assert polarity.labels == LABELS


def test_analyze_returns_probabilities(polarity):
    result = polarity.analyze("کیفیت عالی داشت")
    assert isinstance(result, PolarityResult)
    assert result.label == "recommended" == max(result.scores, key=result.scores.get)
    assert math.isclose(sum(result.scores.values()), 1.0, rel_tol=1e-6)
    assert set(result.scores) == set(LABELS)
    assert result.confidence == result.scores["recommended"]
    assert result.label_fa
    assert result.to_dict()["label"] == "recommended"
    assert polarity.predict_proba("کیفیت عالی داشت") == result.scores


@pytest.mark.parametrize("text", ["", "  ", "123", None, float("nan")])
def test_empty_text(polarity, text):
    result = polarity.analyze(text)
    assert result.label == "no_idea" and result.scores["no_idea"] == 1.0


def test_batch_matches_single(polarity):
    texts = ["کیفیت عالی داشت", "", "افتضاح بود", "نمیدونم", "خوب"]
    assert polarity.predict_many(texts) == [polarity.predict(t) for t in texts]
    assert [r.label for r in polarity.analyze_many(texts)] == polarity.predict_many(texts)
    assert polarity.predict_many([]) == []


def test_batch_matches_legacy_class(polarity, comment_analyzer):
    texts = ["کیفیت عالی داشت", "افتضاح بود", "", "iPhone 13 Pro عالی", "خوب بود", "نمیدونم هنوز"]
    if os.path.exists(TRAIN_CSV):
        texts += pd.read_csv(TRAIN_CSV, nrows=300)["body"].tolist()
    assert polarity.predict_many(texts) == [comment_analyzer.analyzeText(t) for t in texts]


def test_unfitted_raises():
    clf = PolarityClassifier()
    assert not clf.is_fitted and "not fitted" in repr(clf)
    with pytest.raises(ModelNotLoadedError):
        clf.predict("سلام")
    with pytest.raises(ModelNotLoadedError):
        clf.save("x")


def test_fit_save_load(comments_csv, tmp_path):
    clf = PolarityClassifier()
    report = clf.fit_csv(comments_csv, vector_size=16, random_state=0, workers=1)
    assert report.n_train + report.n_test == 30 and report.n_test == 6
    assert 0 <= report.accuracy <= 1 and 0 <= report.macro_f1 <= 1
    assert set(report.per_label) == set(LABELS)
    assert "accuracy" in str(report)

    saved = clf.save(tmp_path / "model")
    assert saved.name == "comment_model.npz"
    file = clf.save(tmp_path / "custom.npz")
    for path in (tmp_path / "model", file):
        other = PolarityClassifier.load(path)
        assert other.predict_many(["خیلی خوب بود", "افتضاح بود"]) == clf.predict_many(["خیلی خوب بود", "افتضاح بود"])


def test_fit_with_dataframe_and_no_test_split(comments_csv):
    df = pd.read_csv(comments_csv)
    report = PolarityClassifier().fit(df.body, df.recommendation_status, test_size=0, workers=1, random_state=0)
    assert report.accuracy is None and report.n_test == 0 and "no test split" in str(report)


def test_fit_validation():
    clf = PolarityClassifier()
    with pytest.raises(DataValidationError):
        clf.fit(["a", "b"], ["recommended", "maybe"])
    with pytest.raises(DataValidationError):
        clf.fit(["a", "b"], ["recommended"])
    with pytest.raises(DataValidationError):
        clf.fit(["a", "b"], ["recommended", "recommended"])
    with pytest.raises(DataValidationError):
        clf.fit_csv(pd.DataFrame({"x": ["a"]}))


def test_evaluate(polarity):
    texts = ["کیفیت عالی داشت", "افتضاح بود", "نمیدونم"]
    labels = ["recommended", "not_recommended", "no_idea"]
    report = polarity.evaluate(texts, labels)
    assert report.n == 3 and 0 <= report.accuracy <= 1
    assert len(report.confusion_matrix) == 3 and sum(map(sum, report.confusion_matrix)) == 3
    df = pd.DataFrame({"body": texts, "recommendation_status": labels})
    assert polarity.evaluate_csv(df).accuracy == report.accuracy


def test_load_errors(tmp_path):
    with pytest.raises(ModelFileError):
        PolarityClassifier.load(tmp_path)


def test_load_legacy_directory(tmp_path, polarity):
    joblib = pytest.importorskip("joblib")
    from gensim.models import Word2Vec
    from sklearn.linear_model import LogisticRegression

    texts = [["خوب"], ["بد"], ["هیچ"]] * 5
    w2v = Word2Vec(texts, vector_size=8, min_count=1, workers=1, seed=1)
    import numpy as np

    X = np.vstack([w2v.wv[t[0]] for t in texts])
    sk = LogisticRegression().fit(X, [1, 0, 2] * 5)
    joblib.dump(sk, tmp_path / "classifier.joblib")
    w2v.save(str(tmp_path / "word2vec.model"))
    with pytest.warns(DeprecationWarning, match="pickle"):
        clf = PolarityClassifier.load(tmp_path)
    assert set(clf.labels) == set(LABELS)


def test_analyze_csv(polarity, tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"id": [1, 2, 3], "نظرات": ["کیفیت عالی داشت", "افتضاح بود", ""]}).to_csv(src, index=False)
    out, summary = tmp_path / "out.csv", tmp_path / "summary.csv"
    df = polarity.analyze_csv(src, out, text_column="نظرات", include_scores=True, summary_csv=summary)
    assert list(df.sentiment) == ["recommended", "not_recommended", "no_idea"]
    assert {"score_recommended", "score_not_recommended", "score_no_idea"} <= set(df.columns)
    assert pd.read_csv(out, encoding="utf-8-sig").shape == df.shape
    s = pd.read_csv(summary, encoding="utf-8-sig")
    assert s["count"].sum() == 3 and math.isclose(s["percentage"].sum(), 100, abs_tol=0.1)
    # DataFrame input, no file written
    df2 = polarity.analyze_csv(pd.DataFrame({"t": ["خوب"]}), show_progress=False)
    assert df2.sentiment.tolist() == [polarity.predict("خوب")]


def test_analyze_csv_raises(polarity, tmp_path):
    with pytest.raises(FileNotFoundError):
        polarity.analyze_csv(tmp_path / "missing.csv")
    with pytest.raises(DataValidationError):
        polarity.analyze_csv(pd.DataFrame({"a": ["x"]}), text_column="zzz")


def test_legacy_class_warns_and_wraps(tmp_path):
    with pytest.warns(DeprecationWarning, match="PolarityClassifier"):
        legacy = CommentAnalyzer()
    legacy.loadModel()
    assert isinstance(legacy.model, PolarityClassifier)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert legacy.analyzeText("کیفیت عالی داشت") == "recommended"
