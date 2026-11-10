# PerSent (Persian Sentiment Analyzer)

[![PyPI](https://img.shields.io/pypi/v/PerSent.svg)](https://pypi.org/project/PerSent/)
[![Python](https://img.shields.io/pypi/pyversions/PerSent.svg)](https://pypi.org/project/PerSent/)
[![CI](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Persian](https://img.shields.io/badge/Persian-فارسی-blue.svg)](README.fa.md)

![PerSent Logo](https://github.com/user-attachments/assets/6bb1633b-6ed3-47fa-aae2-f97886dc4e22)

PerSent is a lightweight Python library for **sentiment and emotion analysis of Persian (Farsi) text**.

- **`CommentAnalyzer`** classifies product/service reviews as `recommended`, `not_recommended` or `no_idea`
  (Word2Vec sentence embeddings + logistic regression).
- **`SentimentAnalyzer`** scores seven emotions with a weighted lexicon:
  شادی (happiness), غم (sadness), ترس (fear), عصبانیت (anger), تنفر (disgust), شگفتی (surprise), آرامش (calmness).
- Single texts or whole CSV files, pre-trained models included, train your own models from CSV.

## Installation

```bash
pip install PerSent
```

Requires Python 3.10–3.13. Pre-trained models are bundled with the package, so no download is needed at runtime.

## Quick start

### Review / comment polarity

```python
from PerSent import CommentAnalyzer

analyzer = CommentAnalyzer()
analyzer.loadModel()                         # bundled pre-trained model

analyzer.analyzeText("کیفیت عالی داشت")       # 'recommended'
analyzer.analyzeText("افتضاح بود")            # 'not_recommended'
analyzer.analyzeTexts(["خوب بود", "بد بود"])   # list of labels
```

Batch-process a CSV file (adds a `sentiment` column):

```python
analyzer.analyzeCSV("comments.csv", "results.csv")                 # first column
analyzer.analyzeCSV("comments.csv", "results.csv", None, -1)       # last column
analyzer.analyzeCSV("comments.csv", "results.csv", None, "نظرات")  # column by name
analyzer.analyzeCSV("comments.csv", "results.csv", "summary.csv")  # + summary report
```

### Emotions

```python
from PerSent import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.loadModel()

result = analyzer.analyzeText("امتحانم رو خراب کردم. احساس می‌کنم یک شکست خورده‌ی تمام عیارم.")
for emotion, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {score:.2f}%")
```

```text
غم: 36.00%
عصبانیت: 36.00%
ترس: 28.00%
شادی: 0.00%
...
```

```python
analyzer.analyzeCSV("input.csv", "output.csv")   # text column 'text' -> 'sentiment_analysis'
```

## Training and saving your own models

Pass a `model_dir` to keep your own models. Without it, `loadModel()` always loads the bundled
pre-trained model and `train()` keeps the result in memory only.

```python
from PerSent import CommentAnalyzer, SentimentAnalyzer

comments = CommentAnalyzer(model_dir="my_models")
accuracy = comments.train("train.csv", random_state=42)   # saved to my_models/comment_model.npz
comments.loadModel()                                        # later: loads from my_models/

emotions = SentimentAnalyzer(model_dir="my_models")
emotions.train("dataset/emotion_dataset.csv")              # saved to my_models/weighted_sentiment_model.json
emotions.saveModel("custom")                                # my_models/custom.json
emotions.loadModel("custom")
```

## API reference

### `CommentAnalyzer(model_dir=None)`

| Method | Description |
|---|---|
| `train(train_csv, test_size=0.2, vector_size=100, window=5, random_state=None, text_column="body", label_column="recommendation_status", workers=4)` | Train from a CSV. Labels must be `recommended`, `not_recommended` or `no_idea`; empty labels count as `no_idea`. Returns held-out accuracy. Saves to `model_dir` if set. |
| `analyzeText(text)` | Label for one text. Texts without letters (empty, digits only, …) return `no_idea`. |
| `analyzeTexts(texts, show_progress=False)` | Labels for many texts. |
| `analyzeCSV(input_csv, output_path, summary_path=None, text_column=0)` | Analyze a CSV column (name or 0-based / negative index). Returns a DataFrame, or `None` on error. |
| `saveModel(model_dir=None)` / `loadModel(model_dir=None)` | Save / load `comment_model.npz`. `loadModel` falls back to the bundled model. |

### `SentimentAnalyzer(model_dir=None)`

| Method | Description |
|---|---|
| `train(train_csv, text_col="text", emotion_col=None, weight_col="weight", reset=False)` | Learn weighted emotion words from a CSV. `emotion_col` defaults to `sentiment` (or `emotion`). Merges into the current model unless `reset=True`. Returns `{"status", "message"}`. |
| `loadLex(csv_file, word_col=0, emotion_col=1, weight_col=2)` | Add a word/emotion/weight lexicon. Columns by index, or by name for CSVs with a header. Invalid weights default to 1.0. |
| `analyzeText(text)` | `{emotion: percentage}` for the seven emotions (all 0.0 if no lexicon word is found). |
| `analyzeCSV(input_csv, output_csv, text_col="text", output_col="sentiment_analysis")` | Analyze a CSV. Returns `True` / `False`. |
| `saveModel(model_name="weighted_sentiment_model", model_dir=None)` / `loadModel(...)` | Save / load `<model_name>.json`. |

Errors derive from `PerSent.PerSentError` (`ModelNotLoadedError`, `ModelFileError`, `DataValidationError`).
Progress and status messages use the standard `logging` module (logger name `"PerSent"`).

## Model quality and limitations

- The bundled comment model reaches about **65 % accuracy** (macro-F1 ≈ 0.64) on a 10 000-review sample of
  [`dataset/train.csv`](dataset/train.csv) (three balanced classes, so chance level is 33 %). Better models are planned.
- The emotion analyzer is lexicon-based and does **not handle negation** yet ("خوشحال نیستم" is scored as happiness).
- hazm's stop-word list contains common opinion words such as "خوب" and "عالی", which are removed before
  classification.

A larger training set is available [here](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train).

## Upgrading from 1.3.x

- Models are stored in safe, pickle-free formats (`.npz` / `.json`). Models you saved with 1.3.x can still be loaded
  with `CommentAnalyzer(model_dir=...)` / `SentimentAnalyzer(model_dir=...)` (a `DeprecationWarning` is shown);
  call `saveModel()` once to convert them. `scripts/convert_legacy_models.py` converts a folder in one go.
- PerSent no longer creates or reads a `PerSent/model` folder in the current working directory. Pass `model_dir`
  explicitly to save and load your own models.

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
