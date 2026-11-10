from __future__ import annotations

import warnings

import pandas as pd
import pytest

from PerSent import CommentAnalyzer, EmotionAnalyzer, PolarityClassifier, SentimentAnalyzer


@pytest.fixture(scope="session")
def polarity() -> PolarityClassifier:
    return PolarityClassifier.pretrained()


@pytest.fixture(scope="session")
def emotions() -> EmotionAnalyzer:
    return EmotionAnalyzer.pretrained()


@pytest.fixture(scope="session")
def comment_analyzer() -> CommentAnalyzer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        analyzer = CommentAnalyzer()
    analyzer.loadModel()
    return analyzer


@pytest.fixture(scope="session")
def emotion_analyzer() -> SentimentAnalyzer:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        analyzer = SentimentAnalyzer()
    analyzer.loadModel()
    return analyzer


@pytest.fixture
def comments_csv(tmp_path):
    rows = {
        "body": ["خیلی خوب بود و راضی هستم", "کیفیتش افتضاح بود پولم هدر رفت", "نمیدونم هنوز استفاده نکردم"] * 10,
        "recommendation_status": ["recommended", "not_recommended", "no_idea"] * 10,
    }
    path = tmp_path / "comments.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def emotion_csv(tmp_path):
    rows = {
        "text": ["امروز خیلی خوشحال و شاد هستم", "از تاریکی می‌ترسم", "خیلی غمگین و ناراحتم"],
        "sentiment": ["شادی", "ترس", "غم"],
        "weight": [1.0, 1.5, 2.0],
    }
    path = tmp_path / "emotions.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
