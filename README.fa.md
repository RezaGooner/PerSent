<div dir="rtl">

# PerSent پرسنت

[![PyPI](https://img.shields.io/pypi/v/PerSent.svg)](https://pypi.org/project/PerSent/)
[![Python](https://img.shields.io/pypi/pyversions/PerSent.svg)](https://pypi.org/project/PerSent/)
[![CI](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml)
[![English](https://img.shields.io/badge/English-انگلیسی-blue.svg)](README.md)

![PerSent Logo](https://github.com/user-attachments/assets/6bb1633b-6ed3-47fa-aae2-f97886dc4e22)

پرسنت (Persian Sentiment Analyzer) یک کتابخانهٔ سبک پایتون و ابزار خط فرمان برای **تحلیل احساسات و هیجان‌های متن فارسی** است.

- **`PolarityClassifier`**: نظر کاربران دربارهٔ کالا یا خدمات را همراه با احتمال هر دسته در یکی از سه دستهٔ `recommended` (پیشنهاد می‌کنم)، `not_recommended` (پیشنهاد نمی‌کنم) و `no_idea` (نظری ندارم) قرار می‌دهد. دقت مدل آمادهٔ آن روی محصولاتی از دیجی‌کالا که در آموزش دیده نشده‌اند **۷۵.۵٪** است.
- **`EmotionAnalyzer`**: درصد هفت حس شادی، غم، ترس، عصبانیت، تنفر، شگفتی و آرامش را محاسبه می‌کند. این کار یا با یک واژه‌نامهٔ وزن‌دار انجام می‌شود که منفی‌سازی را می‌فهمد، یا با دسته‌بندی که خودتان روی داده‌هایتان آموزش می‌دهید.
- مدل‌های آماده داخل خود بسته هستند، بدون pickle ذخیره شده‌اند و هنگام اجرا چیزی دانلود نمی‌شود. ورودی و خروجی CSV، آموزش و ارزیابی مدل با داده‌های خودتان و ابزار خط فرمان هم پشتیبانی می‌شود.

## نصب

</div>

```bash
pip install PerSent
```

<div dir="rtl">

به پایتون ۳.۱۰ تا ۳.۱۳ نیاز دارد.

## شروع سریع

### تحلیل نظرات

</div>

```python
from PerSent import PolarityClassifier

clf = PolarityClassifier.pretrained()           # مدل "tfidf-v2"، جزئیات در docs/MODEL_CARD.md

clf.predict("کیفیت عالی داشت")                 # 'recommended'
clf.predict_proba("کیفیت عالی داشت")           # {'not_recommended': 0.0, 'recommended': 0.99, 'no_idea': 0.01}

result = clf.analyze("افتضاح بود")
result.label, result.confidence                # ('not_recommended', 1.0)
result.label_fa                                # 'پیشنهاد نمی‌کنم'

clf.predict_many(["خوب بود", "بد بود"])          # ['recommended', 'not_recommended']
clf.explain("کیفیت عالی داشت")                 # کلمه‌هایی که در پیش‌بینی بیشترین اثر را داشتند

PolarityClassifier.pretrained("word2vec-v1")   # مدل نسخه‌های 1.5 و قبل‌تر
```

<div dir="rtl">

### تحلیل احساس

</div>

```python
from PerSent import EmotionAnalyzer

emo = EmotionAnalyzer.pretrained()
result = emo.analyze("امتحانم رو خراب کردم. احساس می‌کنم یک شکست خورده‌ی تمام عیارم.")

result.scores          # {'شادی': 0.0, 'غم': 36.0, 'ترس': 28.0, 'عصبانیت': 36.0, ...}
result.dominant        # 'غم'
result.matched_words   # کلمه‌هایی از واژه‌نامه که در متن پیدا شدند
result.scores_en()     # همان امتیازها با نام انگلیسی احساس‌ها

emo.analyze("اصلا خوشحال نیستم").dominant     # None، چون کلمه‌های منفی‌شده حساب نمی‌شوند
```

<div dir="rtl">

### فایل CSV

</div>

```python
clf.analyze_csv("comments.csv", "results.csv", text_column="نظرات",   # نام ستون یا شمارهٔ آن (از ۰)
                include_scores=True, summary_csv="summary.csv")

emo.analyze_csv("texts.csv", "emotions.csv", text_column="text")      # یک ستون برای هر احساس
```

<div dir="rtl">

هر دو متد به‌جای مسیر فایل، `DataFrame` پانداس هم می‌پذیرند و نتیجه را به شکل DataFrame برمی‌گردانند. فایل‌های خروجی با کدگذاری UTF-8 همراه BOM ذخیره می‌شوند تا در Excel درست نمایش داده شوند.

### خط فرمان

</div>

```bash
persent polarity "کیفیت عالی داشت" "افتضاح بود" --proba
echo "خیلی خوشحالم" | persent emotion
persent emotion --file texts.txt --json
persent csv comments.csv -o results.csv -c نظرات --scores --summary summary.csv
persent csv texts.csv -o emotions.csv --task emotion -c text
persent train polarity train.csv -o my_model/ --seed 42   # --method tfidf (پیش‌فرض) یا word2vec
persent train emotion emotions.csv -o emo.npz --method classifier
persent evaluate test.csv -m my_model/
persent polarity "خوب بود" -m word2vec-v1                 # نام هر مدل آماده
```

<div dir="rtl">

دستور `python -m PerSent ...` هم همین کار را می‌کند. برای دیدن همهٔ گزینه‌ها `persent <command> --help` را اجرا کنید.

## آموزش، ارزیابی و ذخیرهٔ مدل

</div>

```python
from PerSent import PolarityClassifier, EmotionAnalyzer

clf = PolarityClassifier()
report = clf.fit_csv("dataset/train.csv", random_state=42)   # ستون‌ها: body و recommendation_status
print(report)                  # دقت و macro-F1 روی دادهٔ آزمون
clf.save("my_model/")
clf = PolarityClassifier.load("my_model/")
print(clf.evaluate_csv("test.csv"))
# با method="word2vec" معماری نسخه‌های 1.5 و قبل‌تر آموزش داده می‌شود

# احساس با دسته‌بند: به متن‌های برچسب‌دار نیاز دارد و خیلی دقیق‌تر از واژه‌نامه است
emo = EmotionAnalyzer()
print(emo.fit_classifier_csv("emotions.csv"))   # ستون‌ها: text و sentiment (برچسب "other" یعنی بدون احساس)
emo.save("my_emotions.npz")

# احساس با واژه‌نامه
lex = EmotionAnalyzer.pretrained()
lex.fit_csv("dataset/emotion_dataset.csv")   # ستون‌ها: text، sentiment و weight
lex.add_lexicon("my_words.csv")              # کلمه، حس، وزن
lex.add_word("وحشت‌زده", "ترس", weight=1.5)    # کلمه یا عبارت کوتاه
lex.save("my_lexicon.json")
```

<div dir="rtl">

مدل‌ها به شکل فایل‌های ساده `.npz` و `.json` ذخیره می‌شوند و بارگذاری آن‌ها هیچ کدی اجرا نمی‌کند.

- برچسب‌های آموزش `PolarityClassifier` یکی از `recommended`، `not_recommended` یا `no_idea` هستند و برچسب خالی `no_idea` در نظر گرفته می‌شود.
- نام احساس‌ها در `EmotionAnalyzer` را می‌توانید فارسی (`شادی`) یا انگلیسی (`happiness`) بنویسید.

جدول کامل متدها در [README انگلیسی](README.md#api-reference) آمده است.

## کیفیت مدل و محدودیت‌ها

جزئیات داده‌ها و روش ارزیابی در [docs/MODEL_CARD.md](docs/MODEL_CARD.md) آمده است.

</div>

| مدل | دادهٔ آزمون | Accuracy | Macro-F1 |
|---|---|---|---|
| نظرات `tfidf-v2` (پیش‌فرض) | ۳۰٬۰۰۰ نظر دیجی‌کالا دربارهٔ محصولاتی که در آموزش نبودند | **۰.۷۵۵** | **۰.۷۵۵** |
| نظرات `word2vec-v1` (نسخهٔ 1.5 و قبل‌تر) | همان | ۰.۶۵۳ | ۰.۶۴۹ |
| احساس، واژه‌نامهٔ `lexicon-v1` | ArmanEmo + ShortPersianEmo + EmoPars (۱٬۹۰۶ متن) | ۰.۲۶۷ | ۰.۲۷۳ |
| احساس، دسته‌بند آموزش‌دیده با `fit_classifier` | همان | ۰.۵۲۹ | ۰.۴۷۷ |

<div dir="rtl">

- مدل نظرات روی نظرهای فروشگاه اینترنتی آموزش دیده و روی متن‌های حوزه‌های دیگر و جمله‌های تکی ضعیف‌تر است.
- واژه‌نامهٔ احساس فقط یک مبنای ساده است. برای کار واقعی، روی دادهٔ برچسب‌دار یک دسته‌بند آموزش دهید. دسته‌بند احساس هنوز همراه کتابخانه منتشر نشده، چون مجوز دیتاست‌های عمومی احساس فارسی اجازهٔ بازنشر با مجوز MIT را نمی‌دهد.

مجموعه‌دادهٔ بزرگ‌تر برای آموزش: [اینجا](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train)

## ارتقا از نسخهٔ 1.x

`CommentAnalyzer` و `SentimentAnalyzer` دقیقاً مثل قبل کار می‌کنند، با همان مدل‌ها و همان نتیجه‌ها. فقط یک `DeprecationWarning` نشان می‌دهند و در نسخهٔ 2.0 حذف خواهند شد.

**در نسخهٔ 1.6 پیش‌فرض کلاس‌های جدید عوض شده است:**
- `PolarityClassifier.pretrained()` حالا مدل دقیق‌تر `tfidf-v2` را برمی‌گرداند و `fit()` به‌طور پیش‌فرض مدل TF-IDF آموزش می‌دهد.
- `EmotionAnalyzer` کلمه‌های منفی‌شده را حساب نمی‌کند.

برای گرفتن همان نتیجه‌های نسخهٔ 1.5:
- `PolarityClassifier.pretrained("word2vec-v1")`
- `fit(..., method="word2vec")`
- `EmotionAnalyzer.pretrained(handle_negation=False, match_phrases=False, stopwords="legacy")`

</div>

| 1.x | 1.5+ |
|---|---|
| `a = CommentAnalyzer(); a.loadModel()` | `clf = PolarityClassifier.pretrained()` |
| `a.analyzeText(t)` | `clf.predict(t)` / `clf.analyze(t)` |
| `a.train(csv)` | `clf.fit_csv(csv)` |
| `a.saveModel()` / `a.loadModel()` | `clf.save(dir)` / `PolarityClassifier.load(dir)` |
| `a.analyzeCSV(i, o, s, col)` | `clf.analyze_csv(i, o, text_column=col, summary_csv=s)` |
| `s = SentimentAnalyzer(); s.loadModel()` | `emo = EmotionAnalyzer.pretrained()` |
| `s.analyzeText(t)` | `emo.analyze(t).scores` |
| `s.train(csv)` / `s.loadLex(csv)` | `emo.fit_csv(csv)` / `emo.add_lexicon(csv)` |
| `s.saveModel(name)` / `s.loadModel(name)` | `emo.save(path)` / `EmotionAnalyzer.load(path)` |
| `s.analyzeCSV(i, o)` | `emo.analyze_csv(i, o)` |

<div dir="rtl">

متدهای جدید در صورت خطا exception می‌دهند، نه `None` یا `False`. فهرست کامل تغییرات در [CHANGELOG.md](CHANGELOG.md) است.

## توسعه

</div>

```bash
git clone https://github.com/RezaGooner/PerSent.git
cd PerSent
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

<div dir="rtl">

## مشارکت

خوشحال می‌شوم پیشنهادها، گزارش باگ و Pull Requestهایتان را ببینم. راهنمای مشارکت در [CONTRIBUTING](.github/CONTRIBUTING.md) است.

- [ثبت Issue](https://github.com/RezaGooner/PerSent/issues/new/choose)
- [Fork و ارسال Pull Request](https://github.com/RezaGooner/PerSent/fork)
- ایمیل: `RezaAsadiProgrammer@gmail.com` · تلگرام: `@RezaGooner`

این پروژه از مخزن [Sentiment-Survey-Analyzer](https://github.com/RezaGooner/Sentiment-Survey-Analyzer) شکل گرفته است.

## مجوز

[MIT](LICENSE)

</div>
