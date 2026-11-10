<div dir="rtl">

# PerSent پرسنت

[![PyPI](https://img.shields.io/pypi/v/PerSent.svg)](https://pypi.org/project/PerSent/)
[![Python](https://img.shields.io/pypi/pyversions/PerSent.svg)](https://pypi.org/project/PerSent/)
[![CI](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml/badge.svg)](https://github.com/RezaGooner/PerSent/actions/workflows/ci.yml)
[![English](https://img.shields.io/badge/English-انگلیسی-blue.svg)](README.md)

![PerSent Logo](https://github.com/user-attachments/assets/6bb1633b-6ed3-47fa-aae2-f97886dc4e22)

پرسنت (Persian Sentiment Analyzer) یک کتابخانهٔ سبک پایتون و ابزار خط فرمان برای **تحلیل احساسات و هیجان‌های متن فارسی** است.

- **`PolarityClassifier`**: نظر کاربران دربارهٔ کالا یا خدمات را همراه با احتمال هر دسته در یکی از سه دستهٔ `recommended` (پیشنهاد می‌کنم)، `not_recommended` (پیشنهاد نمی‌کنم) و `no_idea` (نظری ندارم) قرار می‌دهد.
- **`EmotionAnalyzer`**: درصد هفت حس شادی، غم، ترس، عصبانیت، تنفر، شگفتی و آرامش را با یک واژه‌نامهٔ وزن‌دار محاسبه می‌کند.
- مدل‌های آماده داخل خود بسته هستند و هنگام اجرا چیزی دانلود نمی‌شود. ورودی و خروجی CSV، و آموزش و ارزیابی مدل با داده‌های خودتان هم پشتیبانی می‌شود.

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

clf = PolarityClassifier.pretrained()

clf.predict("کیفیت عالی داشت")                 # 'recommended'
clf.predict_proba("کیفیت عالی داشت")           # {'not_recommended': 0.13, 'recommended': 0.74, 'no_idea': 0.13}

result = clf.analyze("افتضاح بود")
result.label, result.confidence                # ('not_recommended', 0.9998)
result.label_fa                                # 'پیشنهاد نمی‌کنم'

clf.predict_many(["خوب بود", "پولم هدر رفت"])    # چند متن با هم
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
persent train polarity train.csv -o my_model/ --seed 42
persent evaluate test.csv -m my_model/
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

emo = EmotionAnalyzer.pretrained()          # یا EmotionAnalyzer() برای شروع از صفر
emo.fit_csv("dataset/emotion_dataset.csv")  # ستون‌ها: text، sentiment و weight
emo.add_lexicon("my_words.csv")             # کلمه، حس، وزن
emo.add_word("سرمست", "شادی", weight=1.5)    # فقط تک‌کلمه
emo.save("my_emotions.json")
emo = EmotionAnalyzer.load("my_emotions.json")
```

<div dir="rtl">

مدل‌ها به شکل فایل‌های ساده `.npz` و `.json` ذخیره می‌شوند و بارگذاری آن‌ها هیچ کدی اجرا نمی‌کند.

- برچسب‌های آموزش `PolarityClassifier` یکی از `recommended`، `not_recommended` یا `no_idea` هستند و برچسب خالی `no_idea` در نظر گرفته می‌شود.
- نام احساس‌ها در `EmotionAnalyzer` را می‌توانید فارسی (`شادی`) یا انگلیسی (`happiness`) بنویسید.

جدول کامل متدها در [README انگلیسی](README.md#api-reference) آمده است.

## کیفیت مدل و محدودیت‌ها

- دقت مدل آمادهٔ تحلیل نظرات روی نمونه‌ای ۱۰٬۰۰۰تایی از [`dataset/train.csv`](dataset/train.csv) حدود **۶۵٪** است (macro-F1 ≈ ۰.۶۴). سه دسته متوازن‌اند و دقت تصادفی ۳۳٪ است. بهبود مدل در برنامهٔ نسخه‌های بعدی است.
- تحلیلگر احساس بر پایهٔ واژه‌نامه کار می‌کند و **هنوز منفی‌سازی را تشخیص نمی‌دهد**. مثلاً «خوشحال نیستم» شادی حساب می‌شود.
- لیست ایست‌واژه‌های hazm کلمه‌هایی مثل «خوب»، «بد» و «عالی» را هم شامل می‌شود و این کلمه‌ها پیش از دسته‌بندی حذف می‌شوند. به همین دلیل نتیجه برای نظرهای خیلی کوتاه قابل اعتماد نیست.

مجموعه‌دادهٔ بزرگ‌تر برای آموزش: [اینجا](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train)

## ارتقا از نسخهٔ 1.x

`CommentAnalyzer` و `SentimentAnalyzer` دقیقاً مثل قبل کار می‌کنند، ولی یک `DeprecationWarning` نشان می‌دهند و در نسخهٔ 2.0 حذف خواهند شد.

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
