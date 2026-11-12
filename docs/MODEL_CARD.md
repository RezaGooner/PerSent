# PerSent model card

## `tfidf-v2` — review polarity (default since PerSent 1.6)

`PolarityClassifier.pretrained()` / `PolarityClassifier.pretrained("tfidf-v2")`
File: `src/PerSent/models/polarity_tfidf_v2.npz` (4.7 MB, pickle-free)

### Task
Classify a Persian product/service review as `recommended`, `not_recommended` or `no_idea`
(the three values of Digikala's "do you recommend this product?" field). Texts without any
letters return `no_idea`.

### Model
* Input: hazm-normalized text (no stemming, no stop-word removal).
* Features: TF-IDF of word 1–2-grams (200 000 most frequent, `min_df=3`) and character
  2–5-grams within word boundaries (100 000, `min_df=3`), sublinear TF, L2-normalized.
* Classifier: multinomial logistic regression, `C=2.0`.
* The configuration was chosen on the validation split among five candidates (word-only and
  word+char variants, `C` = 2 or 4; validation accuracy 0.753–0.762). It is within 0.05 points
  of the best one; character n-grams were kept for robustness to spelling variants.

### Training data
[RadeAI "Digikala (Comments & Products)"](https://huggingface.co/datasets/RadeAI/Digikala_comments_products)
(MIT licence per its dataset card): 6 156 289 Digikala comments. Cleaned with
`scripts/build_polarity_dataset.py`:

* review body only (titles often repeat the label);
* the label must agree with the star rating — `recommended`: rate > 3.5,
  `not_recommended`: 0 < rate ≤ 2.5, `no_idea`: 1.5 < rate ≤ 3.5;
* exact duplicates removed, texts seen with conflicting labels removed;
* **split by product** (80 / 10 / 10 % of products): no product appears in two splits;
* balanced: 100 000 / 10 000 / 10 000 reviews per class for train / validation / test.

Reproduce: `python scripts/build_polarity_dataset.py big_train.csv data/polarity` then
`python scripts/train_polarity_model.py data/polarity src/PerSent/models/polarity_tfidf_v2.npz`.

### Evaluation

Held-out test split (30 000 reviews of products never seen in training, balanced):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| **`tfidf-v2`** | **0.755** | **0.755** |
| `word2vec-v1` (PerSent ≤ 1.5) | 0.653 | 0.649 |

| `tfidf-v2` per class | Precision | Recall | F1 |
|---|---|---|---|
| recommended | 0.841 | 0.829 | 0.835 |
| not_recommended | 0.773 | 0.790 | 0.781 |
| no_idea | 0.652 | 0.647 | 0.649 |

Out-of-domain check — SentiPers/DeepSentiPers test set (1 854 human-labelled sentences about
digital products; −2/−1 → not_recommended, 0 → no_idea, 1/2 → recommended):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| `tfidf-v2` | 0.533 | 0.455 |
| `word2vec-v1` | 0.461 | 0.366 |

### Limitations
* Trained on e-commerce reviews; accuracy drops on other domains and on single sentences
  (see SentiPers above). `no_idea` is the hardest class.
* Labels come from users' own recommendation choice, filtered by star rating; they are noisy.
* Bag-of-n-grams: long-range negation, sarcasm and comparisons are not modelled.
* For higher accuracy fine-tuned transformers (e.g. ParsBERT, macro-F1 ≈ 0.82 on a Digikala
  benchmark) are stronger but much larger.

## `word2vec-v1` — review polarity (PerSent ≤ 1.5)

`PolarityClassifier.pretrained("word2vec-v1")` — mean Word2Vec (100-d) sentence vectors +
logistic regression, trained on the 149 400-review `dataset/train.csv`. Kept unchanged for
reproducibility and used by the deprecated `CommentAnalyzer`.

## `lexicon-v1` — emotions (default)

`EmotionAnalyzer.pretrained()` — weighted lexicon for seven emotions (670 base words plus their
stems/lemmas), built from `dataset/emotion_dataset.csv`. Since 1.6 negated words are ignored
("خوشحال نیستم"), multi-word entries are matched and opinion words such as "خوب" are no longer
removed as stop-words. On a merged ArmanEmo + ShortPersianEmo + EmoPars test set
(1 906 texts, 6 emotions) it reaches accuracy 0.267 / macro-F1 0.273 (majority class: 0.262);
a TF-IDF classifier trained on the same sources with `EmotionAnalyzer.fit_classifier()` reaches
0.529 / 0.477.

No emotion classifier is bundled:

* ArmanEmo (CC BY-NC-SA 4.0) — the authors allow non-commercial use only, which is
  incompatible with shipping it in an MIT package.
* EmoPars — the authors kindly allowed publishing a model trained on it (2026-11-12). However, a
  classifier trained on EmoPars alone does not beat the lexicon on independent data
  (ArmanEmo test macro-F1 0.278 vs 0.306; 0.18 on EmoPars's own test, near chance), so it is not
  shipped.
* ShortPersianEmo (GPL-3.0) — permission requested. Trained alone it scores macro-F1 0.666 on its
  own test set but 0.279 on ArmanEmo, so single-dataset TF-IDF classifiers do not generalize.

`scripts/build_emotion_dataset.py` builds the merged evaluation set.
