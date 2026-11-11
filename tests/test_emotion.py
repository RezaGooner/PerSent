from __future__ import annotations

import os

import pandas as pd
import pytest

from PerSent import (
    EMOTION_NAMES_EN,
    EMOTIONS,
    DataValidationError,
    EmotionAnalyzer,
    EmotionResult,
    ModelFileError,
    ModelNotLoadedError,
    SentimentAnalyzer,
)

DATASET = os.path.join(os.path.dirname(__file__), os.pardir, "dataset", "emotion_dataset.csv")


def test_pretrained_analyze(emotions):
    result = emotions.analyze("امتحانم رو خراب کردم. احساس می‌کنم یک شکست خورده‌ی تمام عیارم.")
    assert isinstance(result, EmotionResult)
    assert tuple(result.scores) == EMOTIONS  # stable order
    assert result.scores["غم"] == 36.0 and result.dominant in ("غم", "عصبانیت")
    assert result.matched_words


def test_result_helpers(emotions):
    result = emotions.analyze("خیلی خوشحالم")
    assert result.dominant == "شادی" and result.dominant_en == "happiness" and result.has_emotion
    assert result.scores_en()["happiness"] == 100.0
    en = result.to_dict(lang="en")
    assert en["dominant"] == "happiness" and set(en["scores"]) == set(EMOTION_NAMES_EN.values())
    with pytest.raises(ValueError):
        result.to_dict(lang="de")
    assert emotions.dominant("خیلی خوشحالم") == "شادی"
    assert emotions.scores("خیلی خوشحالم") == result.scores


@pytest.mark.parametrize("text", ["", "سلام", None])
def test_no_emotion(emotions, text):
    result = emotions.analyze(text)
    assert result.dominant is None and not result.has_emotion and result.matched_words == ()
    assert set(result.scores.values()) == {0.0}


def test_matches_legacy_class(emotions, emotion_analyzer):
    for text in ["خیلی خوشحالم", "از تاریکی می‌ترسم و ناراحتم", "سلام"]:
        assert emotions.analyze(text).scores == emotion_analyzer.analyzeText(text)


def test_unfitted():
    emo = EmotionAnalyzer()
    assert not emo.is_fitted and "empty" in repr(emo)
    with pytest.raises(ModelNotLoadedError):
        emo.analyze("سلام")


def test_add_word_persian_and_english():
    emo = EmotionAnalyzer()
    emo.add_word("خوشحال", "شادی", 2)
    emo.add_word("ترسناک", "fear")
    assert emo.analyze("خوشحال").dominant == "شادی"
    assert emo.analyze("ترسناک").dominant == "ترس"
    with pytest.raises(DataValidationError):
        emo.add_word("x", "boredom")
    emo.add_word("ذوق‌زده", "شادی")  # multi-word entries are matched since 1.6
    assert emo.analyze("خیلی ذوق‌زده شدم").dominant == "شادی"


def test_add_lexicon_dataframe():
    emo = EmotionAnalyzer()
    used = emo.add_lexicon(pd.DataFrame({"w": ["خوشحال", "x"], "e": ["شادی", "unknown"], "wt": ["1.5", "a"]}))
    assert used == 1 and emo.analyze("خوشحال").dominant == "شادی"


@pytest.mark.skipif(not os.path.exists(DATASET), reason="dataset not available (sdist)")
def test_fit_csv_dataset():
    emo = EmotionAnalyzer()
    assert emo.fit_csv(DATASET) > 900
    assert emo.analyze("خیلی خوشحالم").dominant == "شادی"


def test_fit_csv_reset_and_weightless(emotion_csv):
    emo = EmotionAnalyzer.pretrained()
    before = len(emo.lexicon.word_weights)
    emo.fit_csv(emotion_csv, reset=True)
    assert len(emo.lexicon.word_weights) < before
    df = pd.read_csv(emotion_csv).drop(columns="weight")
    emo2 = EmotionAnalyzer()
    assert emo2.fit_csv(df, weight_column=None) == 3
    with pytest.raises(DataValidationError):
        EmotionAnalyzer().fit_csv(pd.DataFrame({"text": ["a"]}))


def test_save_load(emotion_csv, tmp_path):
    emo = EmotionAnalyzer()
    emo.fit_csv(emotion_csv)
    p1 = emo.save(tmp_path)
    assert p1.name == "weighted_sentiment_model.json"
    p2 = emo.save(tmp_path / "custom.json")
    for p in (tmp_path, p2):
        assert EmotionAnalyzer.load(p).analyze("از تاریکی می‌ترسم").scores == emo.analyze("از تاریکی می‌ترسم").scores
    with pytest.raises(ModelFileError):
        EmotionAnalyzer.load(tmp_path / "missing.json")


def test_analyze_csv(emotions, tmp_path):
    src = tmp_path / "in.csv"
    pd.DataFrame({"text": ["خیلی خوشحالم", "سلام"]}).to_csv(src, index=False)
    out = tmp_path / "out.csv"
    df = emotions.analyze_csv(src, out, show_progress=False)
    assert set(EMOTIONS) <= set(df.columns) and df.dominant_emotion.tolist() == ["شادی", ""]
    assert out.exists()
    en = emotions.analyze_csv(src, lang="en", output_column=None, show_progress=False)
    assert "happiness" in en.columns and "emotions" not in en.columns
    with pytest.raises(ValueError):
        emotions.analyze_csv(src, lang="xx")


def test_legacy_class_warns():
    with pytest.warns(DeprecationWarning, match="EmotionAnalyzer"):
        legacy = SentimentAnalyzer()
    legacy.loadModel()
    assert isinstance(legacy.model, EmotionAnalyzer)
