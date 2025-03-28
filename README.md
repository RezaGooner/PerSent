# PerSent (Persian Sentiment Analyzer) 

[![Persian](https://img.shields.io/badge/Persian-فارسی-blue.svg)](README.fa.md)

![PerSent Logo](https://github.com/user-attachments/assets/6bb1633b-6ed3-47fa-aae2-f97886dc4e22)

## Introduction
PerSent is a practical Python library designed for Persian sentiment analysis. The name stands for "Persian Sentiment Analyzer". Currently in its early testing phase, the library provides basic functionality and is available on [PyPI](https://pypi.org/project/PerSent/). Install it using:

```bash
pip install PerSent
```

Current capabilities include:

- Sentiment analysis of opinions/comments

- Emotion analysis of texts (happiness, sadness, anger, surprise, fear, disgust, calmness)

- Analysis of product/service reviews (recommended/not recommended/no idea)

- Both single-text and batch CSV processing

- Output displayed in terminal or saved to CSV with summary statistics

Initial repository that evolved into this library:
[Click here](https://github.com/RezaGooner/Sentiment-Survey-Analyzer)

We welcome user testing and feedback to improve the library. If you encounter bugs or have suggestions, please:
[Contribute](#Contribute)

For installation issues due to dependency conflicts (especially with mingw-w64), consider using online platforms like DeepNote.com.

## Structure
### Comment Analysis Functions

$$train(train_csv, test_size=0.2, vector_size=100, window=5)$$

Trains the model using a CSV file with columns:

1- body (text content)

2- recommendation_status (must be one of:)

- no_idea

- recommended

- not_recommended

Null/NaN values are converted to no_idea, affecting model accuracy. Optional parameters:

- test_size: Test data proportion (default 0.2)

- vector_size: Word vector dimensions (default 100)

- window: Context window size (default 5)

Returns test accuracy score.

---

  $$predict(text)$$

The core function that analyzes a text and returns one of:
"not_recommended", "recommended", or "no_idea".

---

  $$save_model()$$
  
  $$load_model()$$

  Model persistence functions. Models are saved in the model directory.

---

  $$csvPredict(input_csv, output_path, summary_path=None, text_column=0)$$


Batch processes comments from a CSV file. For single-column files, text_column isn't needed. Otherwise specify column name/index (0-based, negative indices supported). Output contains:

1- Original text

2- Recommendation status
Optional summary_path generates statistics:

- Total count

- Recommended count

- Not recommended count

- No idea count

- Model accuracy (not implemented in current version)

Returns a DataFrame and saves results.

---

### Emotion Analysis Functions

$$load_weighted_lexicon(csv_file, word_col=0, emotion_col=1, weight_col=2)$$

Loads a CSV with three columns:

1- Keywords

2- Emotion (happiness, sadness, anger, fear, disgust, calmness)

3- Emotion weight (defaults to 1 if unspecified, affecting accuracy)

Column indices are optional.

---

$$train_from_csv(train_csv, text_col='text', emotion_col='sentiment', weight_col='weight')$$

Trains the emotion model using a CSV with specified column names (optional).

---

$$save_model(model_name='weighted_sentiment_model')$$

$$load_model(model_name='weighted_sentiment_model')$$

Model persistence functions (saved in model directory).

---

$$analyze_text(text)$$

Analyzes a single text, returning percentage scores for each emotion.

---

$$analyze_csv(input_csv, output_csv, text_col='text', output_col='sentiment_analysis')$$

Batch processes texts from CSV. Returns True on success. Requires:

- input_csv path

- output_csv path
Optional column names.

---

## Installation
Install via pip:

```bash
pip install PerSent
```

For specific versions:

```bash
pip install PerSent==<VERSION_NUMBER>
```

## Usage
- Comment Analysis

Basic single-text analysis:

```bash
from PerSent import CommentAnalyzer

analyzer = CommentAnalyzer()

'''
Training (if you have data):
Requires CSV with comments and recommendation status columns
Status must be: recommended/not_recommended/no_idea
'''
analyzer.train("train.csv")

# Load pre-trained model
analyzer.load_model()

# Predict
text = "کیفیت عالی داشت" # "Excellent quality"
result = analyzer.predict(text)
print(f"Sentiment: {result}")  # Output: Sentiment: recommended
```

The included pre-trained model has ~70% accuracy. For better results, you can train with larger datasets. I've prepared a split dataset (due to size):

[Download Here](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train)

---

Batch CSV processing:

```bash
from PerSent import CommentAnalyzer
analyzer = CommentAnalyzer()
analyzer.load_model()

# Basic usage (single-column CSV)
analyzer.csvPredict(
    input_csv="comments.csv",
    output_path="results.csv"
)

# Alternative usage patterns:
# 1. Using column index (0-based)
analyzer.csvPredict("comments.csv", "results.csv", None, 0)

# 2. Negative indices (count from end)
analyzer.csvPredict("comments.csv", "results.csv", None, -1)

# 3. Column name
analyzer.csvPredict("comments.csv", "results.csv", None, "نظرات") # "Comments" column

# 4. With summary (single-column)
analyzer.csvPredict("comments.csv", "results.csv", "summary.csv")

# 5. With summary and column specification
analyzer.csvPredict("comments.csv", "results.csv", "summary.csv", 2)
```

- Emotion Analysis

Single text analysis with pre-trained model:

```bash
from PerSent import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.load_model()

sample_text = "امتحانم رو خراب کردم. احساس می‌کنم یک شکست خورده‌ی تمام عیارم."
# "I failed my exam. I feel like a complete failure."

result = analyzer.analyze_text(sample_text)
for emotion, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {score:.2f}%")
```

output :

```bash
غم: 36.00%                     #Sadness
عصبانیت: 36.00%                 #anger
ترس: 28.00%                    #fear
شادی: 0.00%                     #happiness
تنفر: 0.00%                      #disgust
شگفتی: 0.00%                    #surprise
آرامش: 0.00%                    #calmness
```

To train your own model:

``` bash
analyzer.train_from_csv('emotion_dataset.csv')
```

Required CSV columns:

1- Keywords

2- Emotion (happiness, sadness, anger, disgust, fear, calmness)

3- Emotion weight

Model persistence:

```bash
analyzer.save_model("custom_model_name")
analyzer.load_model("custom_model_name")
```

Batch CSV processing:

```bash
analyzer.analyze_csv("input.csv", "output.csv")
```

## Contribution
As mentioned, this library needs community collaboration. Please share suggestions, bugs, or feedback via:

- [Fork Repository & Pull Request](https://github.com/RezaGooner/PerSent/fork)

- [Create Issue](https://github.com/RezaGooner/PerSent/issues/new)

- Email: ```RezaAsadiProgrammer@gmail.com```

- Telegram: ```@RezaGooner```
