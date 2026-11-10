# PerSent (Persian Sentiment Analyzer)

[![PyPI](https://img.shields.io/pypi/v/PerSent.svg)](https://pypi.org/project/PerSent/)
[![Python](https://img.shields.io/pypi/pyversions/PerSent.svg)](https://pypi.org/project/PerSent/)
[![CI](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Persian](https://img.shields.io/badge/Persian-فارسی-blue.svg)](README.fa.md)

![PerSent Logo](https://github.com/user-attachments/assets/6bb1633b-6ed3-47fa-aae2-f97886dc4e22)

PerSent is a lightweight Python library and command-line tool for **sentiment and emotion analysis of Persian (Farsi) text**.

- **`PolarityClassifier`** labels product/service reviews as `recommended`, `not_recommended` or `no_idea`, with probabilities.
- **`EmotionAnalyzer`** scores seven emotions with a weighted lexicon:
  شادی (happiness), غم (sadness), ترس (fear), عصبانیت (anger), تنفر (disgust), شگفتی (surprise), آرامش (calmness).
- Pre-trained models included (no download at runtime), CSV in/out, train and evaluate your own models.

## Installation

```bash
pip install PerSent
```

Requires Python 3.10–3.13.

## Quick start

### Review polarity

```python
from PerSent import PolarityClassifier

clf = PolarityClassifier.pretrained()

clf.predict("کیفیت عالی داشت")                 # 'recommended'
clf.predict_proba("کیفیت عالی داشت")           # {'not_recommended': 0.13, 'recommended': 0.74, 'no_idea': 0.13}

result = clf.analyze("افتضاح بود")
result.label, result.confidence                # ('not_recommended', 0.9998)

clf.predict_many(["خوب بود", "پولم هدر رفت"])    # batch
```

### Emotions

```python
from PerSent import EmotionAnalyzer

emo = EmotionAnalyzer.pretrained()
result = emo.analyze("امتحانم رو خراب کردم. احساس می‌کنم یک شکست خورده‌ی تمام عیارم.")

result.scores          # {'شادی': 0.0, 'غم': 36.0, 'ترس': 28.0, 'عصبانیت': 36.0, ...}
result.dominant        # 'غم'
result.matched_words   # lexicon words that produced the scores
result.scores_en()     # {'happiness': 0.0, 'sadness': 36.0, 'fear': 28.0, 'anger': 36.0, ...}
result.to_dict(lang="en")
```

### CSV files

```python
clf.analyze_csv("comments.csv", "results.csv", text_column="نظرات",   # name or 0-based / negative index
                include_scores=True, summary_csv="summary.csv")

emo.analyze_csv("texts.csv", "emotions.csv", text_column="text", lang="en")  # one column per emotion
```

Both methods also accept a `pandas.DataFrame` and return the resulting DataFrame. Output files are UTF-8 with BOM, so
they open correctly in Excel.

### Command line

```bash
persent polarity "کیفیت عالی داشت" "افتضاح بود" --proba
echo "خیلی خوشحالم" | persent emotion --lang en
persent emotion --file texts.txt --json                    # one JSON object per line
persent csv comments.csv -o results.csv -c نظرات --scores --summary summary.csv
persent csv texts.csv -o emotions.csv --task emotion -c text
persent train polarity train.csv -o my_model/ --seed 42
persent evaluate test.csv -m my_model/
```

`python -m PerSent ...` works as well. Run `persent <command> --help` for all options.

## Training, evaluating and saving models

```python
from PerSent import PolarityClassifier, EmotionAnalyzer

clf = PolarityClassifier()
report = clf.fit_csv("dataset/train.csv", random_state=42)   # columns: body, recommendation_status
print(report)                  # Trained on ... texts, tested on ...: accuracy=..., macro-F1=...
clf.save("my_model/")          # -> my_model/comment_model.npz
clf = PolarityClassifier.load("my_model/")
print(clf.evaluate_csv("test.csv"))

emo = EmotionAnalyzer.pretrained()          # or EmotionAnalyzer() to start empty
emo.fit_csv("dataset/emotion_dataset.csv")  # columns: text, sentiment, weight
emo.add_lexicon("my_words.csv")             # word, emotion, weight
emo.add_word("سرمست", "happiness", weight=1.5)   # single words only
emo.save("my_emotions.json")
emo = EmotionAnalyzer.load("my_emotions.json")
```

Models are stored as plain `.npz` / `.json` files — loading them never executes code.

## API reference

### `PolarityClassifier`

| Member | Description |
|---|---|
| `PolarityClassifier.pretrained()` | Bundled model. |
| `PolarityClassifier.load(path)` / `save(path)` | Directory (`comment_model.npz`) or `.npz` file. 1.x pickle folders load with a warning. |
| `predict(text)` / `predict_many(texts)` | Label(s). Texts without letters return `no_idea`. |
| `predict_proba(text)` | `{label: probability}`. |
| `analyze(text)` / `analyze_many(texts)` | `PolarityResult(text, label, scores)` with `.confidence`, `.label_fa`, `.to_dict()`. |
| `fit(texts, labels, *, vector_size=100, window=5, min_count=1, epochs=5, test_size=0.2, random_state=None, C=1.0, ...)` | Train Word2Vec + logistic regression. Returns a `TrainingReport` (accuracy, macro-F1, per-label metrics on a stratified hold-out split). |
| `fit_csv(data, text_column="body", label_column="recommendation_status", **fit_kwargs)` | `fit` on a CSV path or DataFrame. |
| `evaluate(texts, labels)` / `evaluate_csv(data, ...)` | `EvaluationReport` with accuracy, macro-F1, per-label metrics and confusion matrix. |
| `analyze_csv(input, output=None, *, text_column=0, output_column="sentiment", include_scores=False, summary_csv=None)` | Analyze a column; returns a DataFrame. |

### `EmotionAnalyzer`

| Member | Description |
|---|---|
| `EmotionAnalyzer.pretrained()` | Bundled lexicon. |
| `EmotionAnalyzer.load(path)` / `save(path)` | `.json` file or directory. |
| `analyze(text)` / `analyze_many(texts)` | `EmotionResult(text, scores, matched_words)` with `.dominant`, `.dominant_en`, `.scores_en()`, `.to_dict(lang)`. |
| `scores(text)` / `dominant(text)` | Shortcuts. |
| `fit_csv(data, text_column="text", emotion_column=None, weight_column="weight", *, reset=False)` | Learn emotion words from labelled texts (merged into the current lexicon unless `reset=True`). |
| `add_lexicon(data, word_column=0, emotion_column=1, weight_column=2)` | Add a word/emotion/weight table. Emotions may be Persian or English names. |
| `add_word(word, emotion, weight=1.0)` | Add a single word. |
| `analyze_csv(input, output=None, *, text_column="text", expand=True, lang="fa")` | One column per emotion plus `dominant_emotion`. |

Errors derive from `PerSent.PerSentError`: `ModelNotLoadedError`, `ModelFileError`, `DataValidationError`
(which is also a `ValueError`). Messages are logged with the standard `logging` module (logger `"PerSent"`).

## Model quality and limitations

- The bundled polarity model reaches about **65 % accuracy** (macro-F1 ≈ 0.64) on a 10 000-review sample of
  [`dataset/train.csv`](dataset/train.csv) — three balanced classes, so chance level is 33 %. Better models are planned.
- The emotion analyzer is lexicon-based and does **not handle negation** yet ("خوشحال نیستم" is scored as happiness).
- hazm's stop-word list contains opinion words such as "خوب", "بد" and "عالی", which are removed before
  classification; short reviews made only of such words are therefore unreliable.

A larger training set is available [here](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train).

## Upgrading from 1.x

`CommentAnalyzer` and `SentimentAnalyzer` still work exactly as before but emit a `DeprecationWarning`; they will be
removed in PerSent 2.0.

| 1.x | 1.5+ |
|---|---|
| `a = CommentAnalyzer(); a.loadModel()` | `clf = PolarityClassifier.pretrained()` |
| `a.analyzeText(t)` | `clf.predict(t)` (or `clf.analyze(t)` for probabilities) |
| `a.analyzeTexts(ts)` | `clf.predict_many(ts)` |
| `a.train(csv)` → accuracy | `clf.fit_csv(csv)` → `TrainingReport` |
| `a.saveModel()` / `a.loadModel()` with `model_dir` | `clf.save(dir)` / `PolarityClassifier.load(dir)` |
| `a.analyzeCSV(i, o, s, col)` → DataFrame or `None` | `clf.analyze_csv(i, o, text_column=col, summary_csv=s)` (raises on error) |
| `s = SentimentAnalyzer(); s.loadModel()` | `emo = EmotionAnalyzer.pretrained()` |
| `s.analyzeText(t)` | `emo.analyze(t).scores` |
| `s.train(csv)` → `{"status": ...}` | `emo.fit_csv(csv)` (raises on error) |
| `s.loadLex(csv)` | `emo.add_lexicon(csv)` |
| `s.saveModel(name)` / `s.loadModel(name)` | `emo.save(path)` / `EmotionAnalyzer.load(path)` |
| `s.analyzeCSV(i, o)` → `True` / `False` | `emo.analyze_csv(i, o)` |

See [CHANGELOG.md](CHANGELOG.md) for all changes.

## Development

```bash
git clone https://github.com/RezaGooner/PerSent.git
cd PerSent
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

## Contributing

Bug reports, ideas and pull requests are welcome — see [CONTRIBUTING](.github/CONTRIBUTING.md).

- [Open an issue](https://github.com/RezaGooner/PerSent/issues/new/choose)
- [Fork and send a pull request](https://github.com/RezaGooner/PerSent/fork)
- Email: `RezaAsadiProgrammer@gmail.com` · Telegram: `@RezaGooner`

The project grew out of [Sentiment-Survey-Analyzer](https://github.com/RezaGooner/Sentiment-Survey-Analyzer).

## License

[MIT](LICENSE)
