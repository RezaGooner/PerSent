# PerSent پرسنت

[![English](https://img.shields.io/badge/English-انگلیسی-blue.svg)](README.md)

![1743009344748](https://github.com/user-attachments/assets/6bb1633b-6ed3-47fa-aae2-f97886dc4e22)

## مقدمه
پرسنت PerSent به عنوان یک کتابخانه کاربردی برای زبان پایتون نوشته شده که به کاربران در زمینه تحلیل احساسات زبان فارسی کمک میکنه. PerSent مخفف Persian Sentiment Analyzer هست که تصمیم گرفتم این اسم رو بر این اساس براش انتخاب کنم. این کتابخانه در حال حاضر خیلی ابزار نداره و در مراحل تست اولیه قرار داره و نسخه آماده استفاده اون در [Pypi](https://pypi.org/project/PerSent/) هم انتشار داده شده که میتونید با دستور زیر اون رو برای خودتون نصب و آماده استفاده کنید :
``` bash
pip install PerSent
```
استفاده ای که در حال حاضر داره تحلیل احساسات نظرات و همچنین تحلیل احساس جملات (شادی، غم، عصبانیت، تعجب، ترس ، نفرت و آرامش) هست. میتونید یک دیدگاه رو از منظر احساس کاربر نسبت به خرید یا خدمات تحلیل کنید (recommended / not recommended / no idea) و یک متن رو از نظر اینکه چند درصد احساس شادی داره ، چند دردصد مخاطب پیام نفرت انگیز انتقال داده و ...

میتونید هم روی یک متن به تنهایی و هم روی مجموعه ای از نظرات درون فایل  csv تحلیل رو انجام بدید. خروجی روی ترمینال نمایش داده میشه و اگه ورودی به صورت مجموعه ای باشه روی یک فایل csv دیگه به همراه خلاصه ای از نتایج کلی ذخیره میشه . ریپازیتوری اولیه ای که اون رو به این کتابخانه تبدیل کردم:

[کلیک کنید](https://github.com/RezaGooner/Sentiment-Survey-Analyzer)


کتابخانه نیاز داره تا توسط کاربران تست بشه و باگ های خودش را توی مراحل استفاده نشون بده. اگر هنگام استفاده از اون به مشکل یا باگی خوردید ، حس کردید تغییری میتونه اون رو بهبوده ببخشه یا پیشنهادی نسبت به تغییر جایی از کتابخانه داشتید حتما اون رو با من در میون بذارید :

- [مشارکت](#مشارکت)


  همچنین به دلیل استفاده شدن از کتابخانه های به خصوص در این پروژه ، چون نیاز به هماهنگی بین نسخه های بخش های مختلف مثل mingw-w64 و ... هست، اگه توی نصب هر بخشی مشکل داشتید و نتونستید بخش های مورد نیاز رو همگام کنید از پلتفرم های آنلاین مثل ```DeepNote.com``` استفاده کنید.

## ساختار 
### **توابع تحلیل نظرات**

```train(train_csv, test_size=0.2, vector_size=100, window=5)```

| Parameter     | Data Type | Default Value | Description                                                                 | Optional/Required |
|--------------|-----------|---------------|-----------------------------------------------------------------------------|-------------------|
| `train_csv`  | str       | -             | Path to CSV file containing training data with `body` and `recommendation_status` columns | Required          |
| `test_size`  | float     | 0.2           | Proportion of test data (between 0.0 and 1.0)                               | Optional          |
| `vector_size`| int       | 100           | Output vector dimension for Word2Vec model (embedding size)                 | Optional          |
| `window`     | int       | 5             | Context window size for Word2Vec model                                      | Optional          |

 ستون دوم باید یکی از مقادیر زیر رو داشته باشه :
  - no_idea
  - recommended
  - not_recommended

 و اگه غیر این null و NaN  باشه تبدیل به no_idea میشه و روی دقت مدل آموزشی شما تاثیر میذاره. 
 

خروجی این تابع ، دقت به دست آمده از داده های تست هست.

  ---

  ```analyzeText(text)```
  
| Parameter | Data Type | Description                          | Optional/Required |
|-----------|-----------|--------------------------------------|-------------------|
| `text`    | str       | The Persian text to be analyzed      | Required          |

  تابع اصلی که باهاش کار داریم که یک text میگیره و یکی از مقادیر "not_recommended", "recommended", "no_idea" رو با توجه به تحلیلی که روی متن انجام داده برمیگردونه.

  ---
  
```saveModel()```

```loadModel()```

  این دو تابع برای بارگیری و بارگذاری مدل هست که مدل ها توی پوشه model ذخیره میشن و هروقت مدل جدیدی رو آموزش بدیم ، اونجا میتونیم پیداش کنیم.

  ---

```analyzeCSV(input_csv, output_path, summary_path=None, text_column=0)```

| Parameter      | Data Type       | Default Value | Description                                                                 | Optional/Required |
|----------------|-----------------|---------------|-----------------------------------------------------------------------------|-------------------|
| `input_csv`    | str             | -             | Path to input CSV file containing comments to analyze                       | Required          |
| `output_path`  | str             | -             | Path where analyzed results CSV will be saved                               | Required          |
| `summary_path` | str or None     | None          | Optional path to save summary statistics CSV                                | Optional          |
| `text_column`  | int or str      | 0             | Column index (int) or name (str) containing the text to analyze             | Optional          |
  

  تابعی که برای پردازش گروهی از نظرات ذخیره شده توی یک فایل csv به کار میره. اگه فایل تک ستونه باشه نیازی به مشخص کردن ستون text_column نداریم وگرنه باید اسم ستون یا اندیس ستون (شروع از صفر و حتی اعداد منفی ) رو به عنوان آرگومان ارسال کنیم. ورودی ما input_csv و مسیر ذخیره خروجی output_path هست که خروجی شامل دو ستون هست که ستون اول نظرات و ستون دوم وضعیت پیشنهاد شدن recommendation اون نظر هست. اگر خواستیم یه خلاصه کلی از نظرات مثل تعداد کل ، تعداد پیشنهاد شده ها ، تعداد پیشنهاد نشده ها و تعداد بدون ایده ها و همچنین دقت مدل رو داشته باشیم باید به summary_path هم مسیر بدیم. این تابع پردازش و تحلیل رو انجام و فایل نهایی رو ذخیره  میکنه ، علاوه بر اون دیتافریمی رو هم برمیگردونه.


  ---
  
### **توابع تحلیل احساسات**


```loadLex(csv_file, word_col=0, emotion_col=1, weight_col=2)```

| Parameter      | Data Type       | Default Value | Description                                                                 | Optional/Required |
|----------------|-----------------|---------------|-----------------------------------------------------------------------------|-------------------|
| `csv_file`     | str             | -             | Path to CSV lexicon file                                                    | Required          |
| `word_col`     | int or str      | 0             | Column index (int) or name (str) containing words                           | Optional          |
| `emotion_col`  | int or str      | 1             | Column index (int) or name (str) containing emotion labels                  | Optional          |
| `weight_col`   | int or str      | 2             | Column index (int) or name (str) containing weight values                   | Optional          |

برای خوندن فایل csv حاوی سه ستون کلیدواژه ها ، حس (شادی ، غم ، عصبانیت ، ترس ، نفرت و آرامش) و ستون وزن حس هست. که شماره ستون ها اختیاری هست. و شما باید مسیر فایل رو به عنوان آرگومان مشخص کنید. کلمات باید نسبت به حسی که دارن وزن بگیرن وگرنه وزن 1 میگیرن و این روی دقت مدل تون تاثیر میذاره.

---

```train(train_csv, text_col='text', emotion_col='sentiment', weight_col='weight')```

| Parameter      | Data Type       | Default Value | Description                                                                 | Optional/Required |
|----------------|-----------------|---------------|-----------------------------------------------------------------------------|-------------------|
| `train_csv`    | str             | -             | Path to training CSV file                                                   | Required          |
| `text_col`     | str or int      | 'text'        | Column name/index containing text data                                      | Optional          |
| `emotion_col`  | str or int      | 'emotion'     | Column name/index containing emotion labels                                 | Optional          |
| `weight_col`   | str or int      | 'weight'      | Column name/index containing weight values                                  | Optional          |

برای آموزش مدل از این تابع استفاده میکنیم و ساختار فایل csv مثل بخش قبل هست که گفتم و باید اسم ستون ها رو هم مشخص کنیم ( در صورت نیاز چون اختیاری هست)

---

```saveModel(model_name='weighted_sentiment_model')```

| Parameter     | Type  | Default Value               | Description                                                                 | Optional/Required |
|--------------|-------|-----------------------------|-----------------------------------------------------------------------------|-------------------|
| `model_name` | str   | 'weighted_sentiment_model'  | Base filename for saving model (without extension)                          | Optional          |


```loadModel(model_name='weighted_sentiment_model')```

| Parameter     | Type  | Default Value               | Description                                                                 | Optional/Required |
|--------------|-------|-----------------------------|-----------------------------------------------------------------------------|-------------------|
| `model_name` | str   | 'weighted_sentiment_model'  | Base filename of model to load (without extension)                         | Optional          |

ذخیره و بازیابی مدل که مثل همون لود و سیوی هست که بالاتر گفتم و مدل ها هم توی همون پوشه model ذخیره میشن.

---

```analyzeText(text)```

| Parameter | Type | Description                          | Optional/Required |
|-----------|------|--------------------------------------|-------------------|
| `text`    | str  | Persian text to analyze              | Required          |

تابع تحلیل تک ورودی هست که متنی رو میگیره و با توجه آموزش های مدل ، متن رو تحلیل میکنه و درصدی از حس های گفته شده رو برمیگردونه. کاربردش هم پایین تر گفته شده.

---

```analyzeCSV(input_csv, output_csv, text_col='text', output_col='sentiment_analysis')```

| Parameter           | Type          | Default Value          | Description                                                                 | Optional/Required |
|---------------------|---------------|------------------------|-----------------------------------------------------------------------------|-------------------|
| `input_csv`         | str           | -                      | Path to input CSV file containing text to analyze                           | Required          |
| `output_csv`        | str           | -                      | Path to save analyzed results                                               | Required          |
| `text_col`          | str/int       | 'text'                 | Column name/index containing text to analyze                                | Optional          |
| `output_col`        | str           | 'sentiment_analysis'   | Column name for output results                                              | Optional          |

متدی که برای تحلیل مجموعه ای از متون ذخیره شده توی یک فایل csv به کار میره و مسیر فایل ورودی و همچنین مسیر ذخیره خروجی رو میگیره. و اسم ستون ها هم اختیاری هستند. در صورت موفق بودن True و در غیر این صورت False برمیگردونه.

---
## نصب 
برای نصب کتابخانه میتونید مثل نصب کتابخانه های دیگه از دستور pip استفاده کنید :
``` bash
pip install PerSent
```
اگر هم خواستید نسخه خاصی از اون رو نصب کنید از دستور زیر استفاده کنید :
```bash
pip install PerSent==<VERSION_NUMBER>
```
به جای $$<VERSION_NUMBER>$$ نسخه مورد نیاز رو وارد کنید.

## استفاده 
- **تحلیل نظرات**

بعد از نصب موفق کتابخانه حالا آماده استفاده از اون میشیم . ابتدا میخوایم یه نظر رو به تنهایی تحلیل احساسات کنیم.
``` bash
from PerSent import CommentAnalyzer

# ایجاد نمونه شی از کلاس
analyzer = CommentAnalyzer()

'''
 آموزش مدل (اگر داده دارید)

میتوانید یک مجموعه داده csv حاوی دو ستون نظرات و وضعیت پیشنهاد را به مدل بدهید تا آموزش لازم را ببیند و در استفاده های بعدی مورد استفاده قرار دهد
وضعیت پیشنهاد باید شامل سه مقدار زیر باشد :
1 recommended
2 not_recommended
3 no_idea
'''
analyzer.train("train.csv")

#  بارگذاری مدل از پیش آموزش دیده
analyzer.loadModel()

# پیش‌بینی متن
text = "کیفیت عالی داشت"
result = analyzer.analyzeText(text)
print(f"احساس متن: {result}")

# خروجی: احساس متن: recommended
```
برای بخش یادگیری، روی یک مجموعه داده آموزش انجام دادم و مدلی رو با دقت حدودی 70 درصد آماده کردم. چون سخت افرار لازم رو نداشتم نتونستم مدلی رو با مجموعه داده بزرگتر آموزش بدم. اگه بستر لازم برای این کار رو دارید میتونید از مجموعه داده ای که آماده کردم (یا هر مدل مناسب دیگه ای که دارید) استفاده کنید و مدل مناسبی رو آموزش بدید و چه خوب که اون رو با من به اشتراک بگذارید تا اون رو همراه کتابخانه انتشار بدم. به دلیل حجیم بودن مجموعه داده اون رو تقسیم کردم و شما باید rar ها رو دانلود و اکسترکت کنید تا به فایل اصلی دسترسی پیدا کنید :
[کلیک کنید](https://github.com/RezaGooner/Sentiment-Survey-Analyzer/tree/main/Dataset/big_train)

---

حالا میخوایم یک سری نظرات رو که توی یک فایل csv قرار داره تحلیل کنیم. شکل کلی تابع به این صورت هست :
``` bash
analyzeCSV(input_csv, output_path, summary_path=None, text_column=0)
```
که مسیر خلاصه (summary_path) و ستون نظرات (text_column) اختیاری هستند. 
برای مثال ما میخوایم یک فایل csv تک ستونه رو تحلیل کنیم :
``` bash
from PerSent import CommentAnalyzer
# نمونه‌سازی تحلیلگر
analyzer = CommentAnalyzer()
analyzer.loadModel()

# تحلیل فایل 
analyzer.analyzeCSV(
    input_csv="comments.csv",
    output_path="resultsss.csv"
)
```
نظرات در مسیر input_csv قرار داره و نتیجه پردازش در مسیر output_path قرار میگیره. اگر لازم بود خلاصه ای از پردازش و تحلیل هم داشته باشید باید آرگومان summary_path رو مقداردهی کنید. 
خلاصه یک فایل csv شامل موارد زیر هست :

1. تعداد نظرات پیشنهاد دهنده (recommended)
2. تعداد نظرات پیشنهاد نداده (not_recommended)
3. تعداد نظرات بدون ایده (no_idea)
4. تعداد کل نظرات
5. دقت مدل (که توی این نسخه محاسبه نمیشه)


برای حالت های دیگه :
``` bash
from PerSent import CommentAnalyzer

# نمونه‌سازی تحلیلگر
analyzer = CommentAnalyzer()
analyzer.loadModel()

# روش‌های مختلف فراخوانی تابع:

# 1. استفاده از اندیس ستون (شمارش از 0)
analyzer.analyzeCSV("comments.csv", "results.csv", None , 0)  # ستون اول

# 2. استفاده از اندیس منفی (شمارش از آخر)
analyzer.analyzeCSV("comments.csv", "results.csv", None , -1)  # ستون آخر

# 3. استفاده از نام ستون
analyzer.analyzeCSV("comments.csv", "results.csv", None , "نظرات") # ستون با سرستون نظرات

# 4. استفاده از خلاصه سازی بدون مشخص کردن ستون (فایل تک ستون)
analyzer.analyzeCSV("comments.csv", "results.csv" , "summary_path.csv")

# 5. استفاده از خلاصه سازی به همراه مشخص کردن ستون
analyzer.analyzeCSV("comments.csv", "results.csv" , "summary_path.csv", 2)

```

- **تحلیل احساسی**

  حالا نوبت اینه که جمله های فارسی مون رو بر اساس احساسات جمله یعنی حس شادی ، غم ، عصبانیت ، تعجب ، نفرت و آرامش تحلیل کنیم . میتونیم از مدل از پیش ساخته ای که توی کتابخانه قرار دادم استفاده کنید یا csv مد نظر خودتونو بهش بدید تا مدل آموزش دیده خودتونو داشته باشید.

  ابتدا میخوایم یک جمله رو تحلیل کنیم با مدل از پیش ساخته :
```bash
from PerSent import SentimentAnalyzer

analyzer = SentimentAnalyzer()
analyzer.loadModel()

sample_text = "امتحانم رو خراب کردم. احساس می‌کنم یک شکست خورده‌ی تمام عیارم."

result = analyzer.analyzeText(text)
for emotion, score in sorted(result.items(), key=lambda x: x[1], reverse=True):
    print(f"{emotion}: {score:.2f}%")

```
خروجی :
```bash
غم: 36.00%
عصبانیت: 36.00%
ترس: 28.00%
شادی: 0.00%
تنفر: 0.00%
شگفتی: 0.00%
آرامش: 0.00%
```

اگر بخوایم مدل آموزش دیده خودمون رو داشته باشیم از دستور زیر استفاده میکنیم :
``` bash
analyzer.train('emotion_dataset.csv)
```
فایل باید حاوی سه ستون زیر باشه :
1- ستون کلیدواژه
2- ستون حس (شادی ، غم ، عصبانیت ، نفرت ، ترس ، آرامش)
3- ستون وزن حس

و دو دستور زیر برای بارگیری و بارگذاری مدل هست :
``` bash
analyzer.saveModel("weighted_sentiment_model_name")
analyzer.loadModel("weighted_sentiment_model_name")
```

و همچنین برای تحلیل یک فایل csv پر از جملات فارسی از این بخش استفاده می کنیم :
``` bash
analyzer.analyzeCSV("input_csv" , "output_csv")
```


## مشارکت

همونظور که گفتم این کتابخانه نیاز به همکاری و مشارکت کاربران داره. پیشنهاد ، باگ یا هر نظری نسبت به پروژه دارید رو از راه های زیر با من در میون بگذارید:

- [Fork Repository & Pull Request](https://github.com/RezaGooner/PerSent/fork)
- [Make Issue](https://github.com/RezaGooner/PerSent/issues/new)
- E-Mail : ```RezaAsadiProgrammer@gmail.com```
- Telegram : ```@RezaGooner```
