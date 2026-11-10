from __future__ import annotations

import os

import pandas as pd
import pytest

from PerSent import ModelFileError, ModelNotLoadedError, SentimentAnalyzer

EMOTIONS = {"شادی", "غم", "ترس", "عصبانیت", "تنفر", "شگفتی", "آرامش"}
DATASET = os.path.join(os.path.dirname(__file__), os.pardir, "dataset", "emotion_dataset.csv")


def test_pretrained_example(emotion_analyzer):
    result = emotion_analyzer.analyzeText("امتحانم رو خراب کردم. احساس می‌کنم یک شکست خورده‌ی تمام عیارم.")
    assert set(result) == EMOTIONS
    assert result["غم"] == 36.0 and result["عصبانیت"] == 36.0 and result["ترس"] == 28.0
    assert emotion_analyzer.analyzeText("خیلی خوشحالم")["شادی"] == 100.0


@pytest.mark.parametrize("text", ["", "سلام", "123", None])
def test_no_emotion_words(emotion_analyzer, text):
    assert emotion_analyzer.analyzeText(text) == {e: 0.0 for e in EMOTIONS}


def test_scores_sum_to_100(emotion_analyzer):
    result = emotion_analyzer.analyzeText("خوشحال و ناراحت و عصبانی هستم")
    assert abs(sum(result.values()) - 100) < 0.1


def test_analyze_before_load_raises():
    with pytest.raises(ModelNotLoadedError):
        SentimentAnalyzer().analyzeText("سلام")


def test_init_has_no_filesystem_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    SentimentAnalyzer()
    assert os.listdir(tmp_path) == []


@pytest.mark.skipif(not os.path.exists(DATASET), reason="dataset not available (sdist)")
def test_train_on_bundled_dataset_with_defaults():
    # PerSent <= 1.3 defaulted to emotion_col='emotion' and failed on its own dataset.
    analyzer = SentimentAnalyzer()
    assert analyzer.train(DATASET)["status"] == "success"
    assert analyzer.analyzeText("خیلی خوشحالم")["شادی"] > 0


def test_train_save_load_roundtrip(emotion_csv, tmp_path):
    model_dir = tmp_path / "m"
    analyzer = SentimentAnalyzer(model_dir=model_dir)
    assert analyzer.train(emotion_csv)["status"] == "success"
    assert (model_dir / "weighted_sentiment_model.json").exists()
    analyzer.saveModel("custom")
    other = SentimentAnalyzer(model_dir=model_dir)
    other.loadModel("custom")
    text = "از تاریکی می‌ترسم"
    assert other.analyzeText(text) == analyzer.analyzeText(text)
    assert other.analyzeText(text)["ترس"] == 100.0


def test_train_reset(emotion_analyzer, emotion_csv):
    analyzer = SentimentAnalyzer()
    analyzer.loadModel()
    analyzer.train(emotion_csv, reset=True)
    assert len(analyzer.word_weights) < len(emotion_analyzer.word_weights)


def test_train_errors_are_reported(tmp_path):
    analyzer = SentimentAnalyzer()
    assert analyzer.train(tmp_path / "missing.csv")["status"] == "error"
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"x": [1]}).to_csv(bad, index=False)
    assert analyzer.train(bad)["status"] == "error"


def test_load_unknown_custom_model_raises(tmp_path):
    with pytest.raises(ModelFileError):
        SentimentAnalyzer(model_dir=tmp_path).loadModel("nope")


@pytest.mark.parametrize(
    ("cols", "header"),
    [((0, 1, 2), False), (("word", "emotion", "weight"), True)],
)
def test_load_lexicon_by_index_and_name(tmp_path, cols, header):
    path = tmp_path / "lex.csv"
    df = pd.DataFrame({"word": ["خوشحال", "ترسناک"], "emotion": ["شادی", "ترس"], "weight": [2, "x"]})
    df.to_csv(path, index=False, header=header)
    analyzer = SentimentAnalyzer()
    analyzer.loadLex(path, *cols)
    assert analyzer.analyzeText("خوشحال")["شادی"] == 100.0
    assert analyzer.word_weights["ترسناک"][3] == 1.0  # invalid weight -> 1.0


def test_analyze_csv(emotion_analyzer, tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"text": ["خیلی خوشحالم", "سلام"]}).to_csv(src, index=False)
    out = tmp_path / "out.csv"
    assert emotion_analyzer.analyzeCSV(src, out) is True
    assert "sentiment_analysis" in pd.read_csv(out).columns
    assert emotion_analyzer.analyzeCSV(src, out, text_col="missing") is False
