# PerSent - Persian Sentiment Analyzer
[![فارسی](https://img.shields.io/badge/Persian-فارسی-blue.svg)](README.fa.md)


![PerSent Logo](https://github.com/user-attachments/assets/6bb1633b-6ed3-47fa-aae2-f97886dc4e22)

## Introduction
PerSent is a Python library designed for Persian sentiment analysis. The name stands for "Persian Sentiment Analyzer". Currently in its early testing phase, PerSent provides tools for analyzing sentiment in Persian text, particularly useful for product reviews and service feedback.

## Features
- Sentiment classification into three categories:
  - `recommended`
  - `not_recommended` 
  - `no_idea`
- Single text analysis
- Batch processing from CSV files
- Summary report generation

## Installation
Install the latest version using pip:

```bash
pip install PerSent
```
For a specific version:

``` bash
pip install PerSent==<VERSION_NUMBER>
```

## Basic Usage
### Single Text Analysis
``` bash
from PerSent import CommentAnalyzer

# Initialize analyzer
analyzer = CommentAnalyzer()

# Load pre-trained model
analyzer.load_model()

# Analyze text
text = "کیفیت عالی داشت"
result = analyzer.predict(text)
print(f"Sentiment: {result}")
# Output: Sentiment: recommended
```

### Training Your Own Model
``` bash
'''
Train the model using a CSV file containing:
- Comments
- Recommendation status (recommended/not_recommended/no_idea)
'''
analyzer.train("train.csv")
```

## Batch Processing
### CSV Processing

``` bash
analyzer.csvPredict(
    input_csv="comments.csv",
    output_path="results.csv"
)
```

### Advanced CSV Processing Options
``` bash
# Using column index
analyzer.csvPredict("comments.csv", "results.csv", None, 0)

# Using column name  
analyzer.csvPredict("comments.csv", "results.csv", None, "Comments")

# With summary report
analyzer.csvPredict("comments.csv", "results.csv", "summary.csv")
```

## Dataset
A sample training dataset is available:
[Download Dataset](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train)

## Contribution
We welcome contributions and feedback:

- [Fork Repository & Pull Request](https://github.com/RezaGooner/PerSent/fork)
- [Make Issue](https://github.com/RezaGooner/PerSent/issues/new)
- E-Mail : ```RezaAsadiProgrammer@gmail.com```
- Telegram : ```@RezaGooner```
