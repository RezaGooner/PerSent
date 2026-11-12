"""TF-IDF backends, named pretrained models, negation and phrase matching (PerSent 1.6)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from PerSent import (
    CommentAnalyzer,
    DataValidationError,
    EmotionAnalyzer,
    ModelFileError,
    PolarityClassifier,
    SentimentAnalyzer,
)
from PerSent._text import OPINION_WORDS, negated_mask, stopwords
from PerSent._tfidf import TfidfLinearModel, load_tfidf_model, npz_model_type

REVIEWS = [
    ("کیفیت عالی داشت و راضی هستم", "recommended"),
    ("خیلی خوب بود پیشنهاد میکنم", "recommended"),
    ("ارزش خرید داره عالیه", "recommended"),
    ("افتضاح بود پولم هدر رفت", "not_recommended"),
    ("خراب شد اصلا نخرید", "not_recommended"),
    ("کیفیت بد و بی ارزش", "not_recommended"),
    ("هنوز استفاده نکردم", "no_idea"),
    ("نظری ندارم فعلا", "no_idea"),
    ("باید بیشتر کار کنم تا بگم", "no_idea"),
] * 8

EMOTION_TEXTS = [
    ("امروز خیلی خوشحال و شادم", "happiness"),
    ("چه روز شاد و قشنگی", "شادی"),
    ("از تاریکی می‌ترسم", "fear"),
    ("خیلی وحشت کردم ترسیدم", "ترس"),
    ("دلم گرفته و غمگینم", "sadness"),
    ("خیلی ناراحت و غصه دارم", "غم"),
    ("فردا جلسه داریم", "other"),
    ("ساعت پنج میام", "other"),
] * 6


# ------------------------------------------------------------------- pretrained models
def test_available_models():
    assert set(PolarityClassifier.available_models()) == {"tfidf-v2", "word2vec-v1"}
    assert EmotionAnalyzer.available_models() == ("lexicon-v1",)


def test_default_pretrained_is_tfidf(polarity):
    assert polarity.method == "tfidf" and polarity.name == "tfidf-v2"
    assert polarity.predict("کیفیت عالی داشت") == "recommended"
    assert polarity.predict("افتضاح بود پولم هدر رفت") == "not_recommended"
    assert polarity.predict("بد بود") == "not_recommended"  # was no_idea with the 1.x model
    assert polarity.predict("") == "no_idea"
    assert "tfidf" in repr(polarity)
    meta = polarity.tfidf.meta
    assert meta["name"] == "tfidf-v2" and meta["metrics"]["test"]["accuracy"] > 0.7


def test_word2vec_v1_is_the_old_model(comment_analyzer):
    old = PolarityClassifier.pretrained("word2vec-v1")
    assert old.method == "word2vec"
    texts = ["کیفیت عالی داشت", "افتضاح بود", "نمیدونم", "خوب", "iPhone 13 Pro عالی"]
    assert old.predict_many(texts) == [comment_analyzer.analyzeText(t) for t in texts]


def test_unknown_pretrained_name():
    with pytest.raises(ModelFileError):
        PolarityClassifier.pretrained("nope")
    with pytest.raises(ModelFileError):
        EmotionAnalyzer.pretrained("nope")


def test_explain(polarity):
    terms = polarity.explain("کیفیت عالی داشت ولی دیر رسید")
    assert terms and all(isinstance(t, str) for t in terms)
    assert PolarityClassifier.pretrained("word2vec-v1").explain("خوب") == []


# ------------------------------------------------------------------------- TF-IDF core
def test_tfidf_roundtrip_is_exact(tmp_path):
    texts, labels = zip(*REVIEWS, strict=True)
    model = TfidfLinearModel.train(texts, labels, max_word_features=500, max_char_features=500)
    model.meta = {"hello": "سلام"}
    path = tmp_path / "m.npz"
    model.save(path)
    assert npz_model_type(path) == "tfidf_lr"
    again = load_tfidf_model(path)
    probe = ["کیفیت عالی", "خراب شد", "", "متن کاملا جدید"]
    np.testing.assert_allclose(again.predict_proba(probe), model.predict_proba(probe), rtol=1e-6)
    assert again.meta == {"hello": "سلام"}
    with np.load(path, allow_pickle=False) as data:  # pickle-free
        assert all(data[k].dtype != object for k in data.files)


def test_tfidf_binary_and_word_only():
    texts = ["خوب", "عالی", "بد", "افتضاح"] * 5
    labels = ["recommended", "recommended", "not_recommended", "not_recommended"] * 5
    model = TfidfLinearModel.train(texts, labels, char_ngrams=None)
    assert len(model.vectorizers) == 1 and model.coef.shape[0] == 2
    assert model.labels[int(np.argmax(model.predict_proba(["عالی"])[0]))] == "recommended"


# ------------------------------------------------------------------ PolarityClassifier
@pytest.mark.parametrize("method", ["tfidf", "word2vec"])
def test_fit_methods_save_load(method, tmp_path):
    texts, labels = zip(*REVIEWS, strict=True)
    clf = PolarityClassifier()
    opts = {"workers": 1, "vector_size": 16} if method == "word2vec" else {"max_word_features": 1000}
    report = clf.fit(list(texts), list(labels), method=method, random_state=0, **opts)
    assert clf.method == method and report.n_test > 0
    saved = clf.save(tmp_path)
    assert saved.name == ("polarity_model.npz" if method == "tfidf" else "comment_model.npz")
    again = PolarityClassifier.load(tmp_path)
    assert again.method == method
    probe = ["کیفیت عالی داشت", "خراب شد", "هنوز استفاده نکردم"]
    assert again.predict_many(probe) == clf.predict_many(probe)


def test_fit_rejects_bad_options():
    with pytest.raises(ValueError):
        PolarityClassifier().fit(["a", "b"], ["recommended", "no_idea"], method="bert")
    with pytest.raises(TypeError):
        PolarityClassifier().fit(["a", "b"], ["recommended", "no_idea"], method="tfidf", vector_size=10)


def test_legacy_comment_analyzer_trains_word2vec(comments_csv, tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = CommentAnalyzer(model_dir=tmp_path)
    legacy.train(comments_csv, vector_size=8, random_state=0, workers=1)
    assert legacy.model.method == "word2vec" and (tmp_path / "comment_model.npz").exists()


# ------------------------------------------------------------- negation & stop-words
def test_opinion_stopwords():
    assert "خوب" in stopwords("legacy") and "خوب" not in stopwords("opinion")
    assert OPINION_WORDS & stopwords("legacy")
    assert "از" in stopwords("opinion")


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (["خوشحال", "نیستم"], [True, False]),
        (["نه", "خوشحالم", "نه", "ناراحت"], [False, True, False, True]),
        (["اصلا", "نمی", "ترسم"], [False, False, True]),
        (["خیلی", "خوشحالم"], [False, False]),
    ],
)
def test_negated_mask(tokens, expected):
    assert negated_mask(tokens) == expected


@pytest.mark.parametrize(
    ("text", "dominant"),
    [
        ("خیلی خوشحالم", "شادی"),
        ("اصلا خوشحال نیستم", None),
        ("از تاریکی نمی‌ترسم", None),
        ("نه خوشحالم نه ناراحت", None),
        ("خوشحال نیستم ولی خیلی عصبانی هستم", "عصبانیت"),
    ],
)
def test_lexicon_negation(emotions, text, dominant):
    assert emotions.analyze(text).dominant == dominant


def test_legacy_sentiment_analyzer_ignores_negation(emotion_analyzer):
    assert emotion_analyzer.analyzeText("اصلا خوشحال نیستم")["شادی"] == 100.0


def test_phrase_entries():
    emo = EmotionAnalyzer()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no "more than one word" warning any more
        emo.add_word("وحشت‌زده", "ترس")
    assert emo.analyze("دیشب وحشت‌زده شدم").dominant == "ترس"
    assert "وحشت زده" in emo.analyze("دیشب وحشت‌زده شدم").matched_words
    assert emo.analyze("وحشت‌زده نبودم").dominant is None
    assert EmotionAnalyzer(emo.lexicon, match_phrases=False).analyze("وحشت‌زده").dominant is None


def test_invalid_stopwords_option():
    with pytest.raises(ValueError):
        EmotionAnalyzer(stopwords="none")


# ------------------------------------------------------------- emotion classifier
def test_emotion_classifier(tmp_path):
    texts, labels = zip(*EMOTION_TEXTS, strict=True)
    emo = EmotionAnalyzer()
    report = emo.fit_classifier(texts, labels, random_state=0, max_word_features=2000, max_char_features=2000)
    assert emo.backend == "classifier" and report.n_test > 0 and "classifier" in repr(emo)
    result = emo.analyze("خیلی خوشحال و شادم")
    assert result.dominant == "شادی" and result.matched_words
    assert set(result.scores) == {"شادی", "غم", "ترس", "عصبانیت", "تنفر", "شگفتی", "آرامش"}
    assert 0 <= result.neutral <= 100 and "neutral" in result.to_dict()
    assert emo.analyze("فردا جلسه داریم ساعت پنج").dominant is None  # trained "other" class
    assert emo.analyze("").dominant is None

    path = emo.save(tmp_path)
    assert path.name == "emotion_classifier.npz"
    again = EmotionAnalyzer.load(tmp_path)
    assert again.backend == "classifier"
    assert again.analyze("از تاریکی می‌ترسم").scores == emo.analyze("از تاریکی می‌ترسم").scores
    ev = again.evaluate(texts, labels)
    assert ev.accuracy > 0.8 and "_none" in ev.labels

    df = again.analyze_csv(pd.DataFrame({"text": ["خیلی خوشحالم", "ساعت پنج میام"]}), show_progress=False)
    assert "بدون_احساس" in df.columns


def test_emotion_classifier_validation():
    with pytest.raises(DataValidationError):
        EmotionAnalyzer().fit_classifier(["a", "b"], ["شادی", "boredom"])
    with pytest.raises(DataValidationError):
        EmotionAnalyzer().fit_classifier(["a", "b"], ["شادی", "شادی"])


def test_emotion_classifier_csv(tmp_path):
    df = pd.DataFrame(EMOTION_TEXTS, columns=["text", "sentiment"])
    emo = EmotionAnalyzer()
    emo.fit_classifier_csv(df, test_size=0, max_word_features=500, char_ngrams=None)
    assert emo.evaluate_csv(df).accuracy > 0.8


def test_lexicon_evaluate(emotions):
    report = emotions.evaluate(["خیلی خوشحالم", "سلام"], ["شادی", "other"])
    assert report.accuracy == 1.0


def test_legacy_wrapper_unchanged_on_old_files(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = SentimentAnalyzer(model_dir=tmp_path)
    legacy.loadModel()
    assert legacy.model.handle_negation is False and legacy.model.stopwords == "legacy"


def test_emotion_none_aliases_include_analyze_csv_column_name():
    from PerSent.emotion import NONE_LABEL, emotion_name

    for alias in ("بدون_احساس", "بدون احساس", "other", "None", "خنثی"):
        assert emotion_name(alias) == NONE_LABEL
