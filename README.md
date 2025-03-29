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

- [Contribute](#contribution)

For installation issues due to dependency conflicts (especially with mingw-w64), consider using online platforms like DeepNote.com.

## Structure
### Comment Analysis Functions

```train(train_csv, test_size=0.2, vector_size=100, window=5)```

| Parameter     | Data Type | Default Value | Description                                                                 | Optional/Required |
|--------------|-----------|---------------|-----------------------------------------------------------------------------|-------------------|
| `train_csv`  | str       | -             | Path to CSV file containing training data with `body` and `recommendation_status` columns | Required          |
| `test_size`  | float     | 0.2           | Proportion of test data (between 0.0 and 1.0)                               | Optional          |
| `vector_size`| int       | 100           | Output vector dimension for Word2Vec model (embedding size)                 | Optional          |
| `window`     | int       | 5             | Context window size for Word2Vec model                                      | Optional          |

recommendation_status must be one of:

- no_idea

- recommended

- not_recommended

Null/NaN values are converted to no_idea, affecting model accuracy.

- Returns test accuracy score.

---

  ```analyzeText(text)```
  
| Parameter | Data Type | Description                          | Optional/Required |
|-----------|-----------|--------------------------------------|-------------------|
| `text`    | str       | The Persian text to be analyzed      | Required          |

The core function that analyzes a text and returns one of:
"not_recommended", "recommended", or "no_idea".

---

```saveModel()```

```loadModel()```
  
  Model persistence functions. Models are saved in the model directory.

---

```analyzeCSV(input_csv, output_path, summary_path=None, text_column=0)```

| Parameter      | Data Type       | Default Value | Description                                                                 | Optional/Required |
|----------------|-----------------|---------------|-----------------------------------------------------------------------------|-------------------|
| `input_csv`    | str             | -             | Path to input CSV file containing comments to analyze                       | Required          |
| `output_path`  | str             | -             | Path where analyzed results CSV will be saved                               | Required          |
| `summary_path` | str or None     | None          | Optional path to save summary statistics CSV                                | Optional          |
| `text_column`  | int or str      | 0             | Column index (int) or name (str) containing the text to analyze             | Optional          |
  

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

```loadLex(csv_file, word_col=0, emotion_col=1, weight_col=2)```

| Parameter      | Data Type       | Default Value | Description                                                                 | Optional/Required |
|----------------|-----------------|---------------|-----------------------------------------------------------------------------|-------------------|
| `csv_file`     | str             | -             | Path to CSV lexicon file                                                    | Required          |
| `word_col`     | int or str      | 0             | Column index (int) or name (str) containing words                           | Optional          |
| `emotion_col`  | int or str      | 1             | Column index (int) or name (str) containing emotion labels                  | Optional          |
| `weight_col`   | int or str      | 2             | Column index (int) or name (str) containing weight values                   | Optional          |

Loads a CSV with three columns:

1- Keywords

2- Emotion (happiness, sadness, anger, fear, disgust, calmness)

3- Emotion weight (defaults to 1 if unspecified, affecting accuracy)

Column indices are optional.

---

```train(train_csv, text_col='text', emotion_col='sentiment', weight_col='weight')```

| Parameter      | Data Type       | Default Value | Description                                                                 | Optional/Required |
|----------------|-----------------|---------------|-----------------------------------------------------------------------------|-------------------|
| `train_csv`    | str             | -             | Path to training CSV file                                                   | Required          |
| `text_col`     | str or int      | 'text'        | Column name/index containing text data                                      | Optional          |
| `emotion_col`  | str or int      | 'emotion'     | Column name/index containing emotion labels                                 | Optional          |
| `weight_col`   | str or int      | 'weight'      | Column name/index containing weight values                                  | Optional          |

Trains the emotion model using a CSV with specified column names (optional).

---

```saveModel(model_name='weighted_sentiment_model')```

| Parameter     | Type  | Default Value               | Description                                                                 | Optional/Required |
|--------------|-------|-----------------------------|-----------------------------------------------------------------------------|-------------------|
| `model_name` | str   | 'weighted_sentiment_model'  | Base filename for saving model (without extension)                          | Optional          |


```loadModel(model_name='weighted_sentiment_model')```

| Parameter     | Type  | Default Value               | Description                                                                 | Optional/Required |
|--------------|-------|-----------------------------|-----------------------------------------------------------------------------|-------------------|
| `model_name` | str   | 'weighted_sentiment_model'  | Base filename of model to load (without extension)                         | Optional          |


Model persistence functions (saved in model directory).

---

```analyzeText(text)```

| Parameter | Type | Description                          | Optional/Required |
|-----------|------|--------------------------------------|-------------------|
| `text`    | str  | Persian text to analyze              | Required          |

Analyzes a single text, returning percentage scores for each emotion.

---

```analyzeCSV(input_csv, output_csv, text_col='text', output_col='sentiment_analysis')```

| Parameter           | Type          | Default Value          | Description                                                                 | Optional/Required |
|---------------------|---------------|------------------------|-----------------------------------------------------------------------------|-------------------|
| `input_csv`         | str           | -                      | Path to input CSV file containing text to analyze                           | Required          |
| `output_csv`        | str           | -                      | Path to save analyzed results                                               | Required          |
| `text_col`          | str/int       | 'text'                 | Column name/index containing text to analyze                                | Optional          |
| `output_col`        | str           | 'sentiment_analysis'   | Column name for output results                                              | Optional          |

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
analyzer.loadModel()

# Predict
text = "کیفیت عالی داشت" # "Excellent quality"
result = analyzer.analyzeText(text)
print(f"Sentiment: {result}")  # Output: Sentiment: recommended
```

The included pre-trained model has ~70% accuracy. For better results, you can train with larger datasets. I've prepared a split dataset (due to size):

[Download Here](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train)

---

Batch CSV processing:

```bash
from PerSent import CommentAnalyzer
analyzer = CommentAnalyzer()
analyzer.loadModel()

# Basic usage (single-column CSV)
analyzer.analyzeCSV(
    input_csv="comments.csv",
    output_path="results.csv"
)

# Alternative usage patterns:
# 1. Using column index (0-based)
analyzer.analyzeCSV("comments.csv", "results.csv", None, 0)

# 2. Negative indices (count from end)
analyzer.analyzeCSV("comments.csv", "results.csv", None, -1)

# 3. Column name
analyzer.analyzeCSV("comments.csv", "results.csv", None, "نظرات") # "Comments" column

# 4. With summary (single-column)
analyzer.analyzeCSV("comments.csv", "results.csv", "summary.csv")

# 5. With summary and column specification
analyzer.analyzeCSV("comments.csv", "results.csv", "summary.csv", 2)
```

- Emotion Analysis

Single text analysis with pre-trained model:

```bash
from PerSent import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.loadModel()

sample_text = "امتحانم رو خراب کردم. احساس می‌کنم یک شکست خورده‌ی تمام عیارم."
# "I failed my exam. I feel like a complete failure."

result = analyzer.analyzeText(sample_text)
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
analyzer.train('emotion_dataset.csv')
```

Required CSV columns:

1- Keywords

2- Emotion (happiness, sadness, anger, disgust, fear, calmness)

3- Emotion weight

Model persistence:

```bash
analyzer.saveModel("custom_model_name")
analyzer.loadModel("custom_model_name")
```

Batch CSV processing:

```bash
analyzer.analyzeCSV("input.csv", "output.csv")
```

## Contribution
As mentioned, this library needs community collaboration. Please share suggestions, bugs, or feedback via:

- [Fork Repository & Pull Request](https://github.com/RezaGooner/PerSent/fork)

- [Create Issue](https://github.com/RezaGooner/PerSent/issues/new)

- Email: ```RezaAsadiProgrammer@gmail.com```

- Telegram: ```@RezaGooner```
