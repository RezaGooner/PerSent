---
title: "PerSent: A Python Library for Persian Sentiment and Emotion Analysis"
tags:
  - Python
  - NLP
  - Sentiment Analysis
  - Emotion Analysis
  - Persian Language Processing
authors:
  - name: "Reza Asadi"
    orcid: "0009-0005-6852-5756"
    affiliation: 1
affiliations:
  - name: "Department of Computer Engineering, Yasuj University, Yasuj, Iran"
    index: 1
date: 2025-09-05
bibliography: paper.bib
---

# Summary

**PerSent** is an open-source Python library designed for sentiment and emotion analysis in Persian texts.  
By combining machine learning models and lexical resources, it provides researchers and developers with an easy-to-use toolkit for:

- Sentiment classification of user opinions and product reviews (`recommended`, `not_recommended`, `no_idea`).
- Emotion detection across seven categories: happiness, sadness, anger, fear, disgust, surprise, and calmness.
- Batch analysis of large datasets with CSV input/output.
- Model training and persistence for customized use.
- provide pre-trained model


The library is available on [PyPI](https://pypi.org/project/PerSent/) under the MIT License and is intended to fill the gap in public, reusable resources for **Persian Natural Language Processing** (NLP).

# Statement of need

While sentiment analysis tools are abundant in English and other widely spoken languages, resources for **Persian** remain scarce, fragmented, and often proprietary.  
This lack hinders both academic research and industrial applications in NLP for Persian text.

PerSent addresses this gap by offering:

1. **Unified functionality** — a single Python interface for both sentiment and emotion analysis in Persian.
2. **Flexibility** — supports both pre-trained models (~80% baseline accuracy) and custom dataset training.
3. **Batch and interactive modes** — suitable for both large-scale corpus analysis and real-time processing.
4. **Ease of integration** — lightweight dependencies and clear API make it easy to embed into websites, chatbots, or research workflows.

By publishing PerSent as an open-source, PyPI-installable package, we enable NLP researchers, computational linguists, and developers in Persian-speaking contexts to prototype and deploy faster.

# Functionality

The library provides two main analyzers:

- **`CommentAnalyzer`** — trains and applies models for recommendation-based sentiment.
- **`SentimentAnalyzer`** — lexicon-based or trained models for multi-label emotion detection.

Example usage:

```python
from PerSent import CommentAnalyzer

analyzer = CommentAnalyzer()
analyzer.loadModel()
print(analyzer.analyzeText("کیفیت عالی داشت"))  # recommended
```

Batch CSV analysis:

```python
analyzer.analyzeCSV("comments.csv", "results.csv", "summary.csv")
```

Emotion analysis:

```python
from PerSent import SentimentAnalyzer
analyzer = SentimentAnalyzer()
analyzer.loadModel()
print(analyzer.analyzeText("امتحانم رو خراب کردم.")) # The output includes percentages of the seven emotions.
```

# Acknowledgements

I acknowledge the support of the Persian NLP community, contributors to open-source Python libraries such as `gensim` and `scikit-learn`, and dataset providers whose corpora were invaluable for testing.

# References
