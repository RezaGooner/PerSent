<div dir="rtl">

# PerSent پرسنت

[![PyPI](https://img.shields.io/pypi/v/PerSent.svg)](https://pypi.org/project/PerSent/)
[![Python](https://img.shields.io/pypi/pyversions/PerSent.svg)](https://pypi.org/project/PerSent/)
[![CI](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml)
[![English](https://img.shields.io/badge/English-انگلیسی-blue.svg)](README.md)

![PerSent Logo](https://github.com/user-attachments/assets/6bb1633b-6ed3-47fa-aae2-f97886dc4e22)

پرسنت (Persian Sentiment Analyzer) یک کتابخانهٔ سبک پایتون برای **تحلیل احساسات و هیجان‌های متن فارسی** است.

- **`CommentAnalyzer`**: نظر کاربران دربارهٔ کالا یا خدمات را در یکی از سه دستهٔ `recommended` (پیشنهاد می‌کند)، `not_recommended` (پیشنهاد نمی‌کند) و `no_idea` (نظری ندارد) قرار می‌دهد.
- **`SentimentAnalyzer`**: درصد هفت حس شادی، غم، ترس، عصبانیت، تنفر، شگفتی و آرامش را در متن، با یک واژه‌نامهٔ وزن‌دار محاسبه می‌کند.
- تحلیل تک‌متن یا کل یک فایل CSV، همراه با مدل‌های ازپیش‌آموزش‌دیده و امکان آموزش مدل با داده‌های خودتان.

## نصب

</div>

```bash
pip install PerSent
```

<div dir="rtl">

به پایتون ۳.۱۰ تا ۳.۱۳ نیاز دارد. مدل‌های ازپیش‌آموزش‌دیده داخل خود بسته هستند و هنگام اجرا چیزی دانلود نمی‌شود.

## شروع سریع

### تحلیل نظرات

</div>

```python
from PerSent import CommentAnalyzer

analyzer = CommentAnalyzer()
analyzer.loadModel()                         # مدل آمادهٔ همراه کتابخانه

analyzer.analyzeText("کیفیت عالی داشت")       # 'recommended'
analyzer.analyzeText("افتضاح بود")            # 'not_recommended'
analyzer.analyzeTexts(["خوب بود", "بد بود"])   # فهرست برچسب‌ها
```

<div dir="rtl">

تحلیل یک فایل CSV (یک ستون `sentiment` به خروجی اضافه می‌شود):

</div>

```python
analyzer.analyzeCSV("comments.csv", "results.csv")                 # ستون اول
analyzer.analyzeCSV("comments.csv", "results.csv", None, -1)       # ستون آخر
analyzer.analyzeCSV("comments.csv", "results.csv", None, "نظرات")  # با نام ستون
analyzer.analyzeCSV("comments.csv", "results.csv", "summary.csv")  # همراه با فایل خلاصه
```

<div dir="rtl">

فایل خلاصه تعداد و درصد نظرات هر دسته و تعداد کل نظرات را نشان می‌دهد.

### تحلیل احساس

</div>

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
analyzer.analyzeCSV("input.csv", "output.csv")   # ستون متن: text ← ستون خروجی: sentiment_analysis
```

<div dir="rtl">

## آموزش و ذخیرهٔ مدل‌های خودتان

برای نگه داشتن مدل‌های خودتان، پارامتر `model_dir` را بدهید. اگر ندهید، `loadModel()` همیشه مدل آمادهٔ کتابخانه را بارگذاری می‌کند و `train()` چیزی روی دیسک نمی‌نویسد.

</div>

```python
from PerSent import CommentAnalyzer, SentimentAnalyzer

comments = CommentAnalyzer(model_dir="my_models")
accuracy = comments.train("train.csv", random_state=42)   # ذخیره در my_models/comment_model.npz
comments.loadModel()                                        # بعداً: بارگذاری از my_models

emotions = SentimentAnalyzer(model_dir="my_models")
emotions.train("dataset/emotion_dataset.csv")              # ذخیره در my_models/weighted_sentiment_model.json
emotions.saveModel("custom")                                # my_models/custom.json
emotions.loadModel("custom")
```

<div dir="rtl">

- فایل آموزش `CommentAnalyzer` باید ستون‌های `body` (متن) و `recommendation_status` (برچسب) را داشته باشد. برچسب‌ها یکی از `recommended`، `not_recommended` یا `no_idea` هستند و برچسب خالی `no_idea` در نظر گرفته می‌شود. خروجی `train` دقت مدل روی دادهٔ آزمون است.
- فایل آموزش `SentimentAnalyzer` ستون‌های `text`، `sentiment` (یکی از هفت حس) و `weight` (وزن) دارد. نمونه‌اش در [`dataset/emotion_dataset.csv`](dataset/emotion_dataset.csv) است. با `reset=True` مدل قبلی پاک می‌شود. در غیر این صورت داده‌های جدید به آن اضافه می‌شوند.
- با `loadLex(csv_file, word_col=0, emotion_col=1, weight_col=2)` می‌توانید یک واژه‌نامهٔ «کلمه، حس، وزن» اضافه کنید. ستون‌ها را با شماره یا، اگر فایل سرستون دارد، با نام مشخص کنید.

جدول کامل پارامترها در [README انگلیسی](README.md#api-reference) آمده است.

## کیفیت مدل و محدودیت‌ها

- دقت مدل آمادهٔ تحلیل نظرات روی نمونه‌ای ۱۰٬۰۰۰تایی از [`dataset/train.csv`](dataset/train.csv) حدود **۶۵٪** است (macro-F1 ≈ ۰.۶۴). سه دسته متوازن‌اند و دقت تصادفی ۳۳٪ است. بهبود مدل در برنامهٔ نسخه‌های بعدی است.
- تحلیلگر احساس بر پایهٔ واژه‌نامه کار می‌کند و **هنوز منفی‌سازی را تشخیص نمی‌دهد**. مثلاً «خوشحال نیستم» شادی حساب می‌شود.
- لیست ایست‌واژه‌های hazm کلمه‌هایی مثل «خوب» و «عالی» را هم شامل می‌شود و این کلمه‌ها پیش از دسته‌بندی حذف می‌شوند.

مجموعه‌دادهٔ بزرگ‌تر برای آموزش: [اینجا](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train)

## ارتقا از نسخهٔ 1.3

- مدل‌ها حالا با قالب‌های امن و بدون pickle (`.npz` و `.json`) ذخیره می‌شوند. مدل‌هایی را که با نسخهٔ 1.3 ذخیره کرده‌اید هنوز می‌توانید با `model_dir` بارگذاری کنید (یک `DeprecationWarning` نمایش داده می‌شود). یک بار `saveModel()` را صدا بزنید تا به قالب جدید تبدیل شوند.
- کتابخانه دیگر پوشهٔ `PerSent/model` را در پوشهٔ جاری نمی‌سازد و از آنجا هم نمی‌خواند. برای مدل‌های خودتان `model_dir` را صریحاً بدهید.

فهرست کامل تغییرات در [CHANGELOG.md](CHANGELOG.md) است.

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
