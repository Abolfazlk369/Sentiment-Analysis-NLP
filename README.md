# 🎭 Persian Sentiment Analysis using ParsBERT & PyTorch

سیستم تحلیل احساسات و تشخیص هوشمند نظرات کاربران (مثبت/منفی) به زبان فارسی با استفاده از مدل ترنسفورمر **ParsBERT** و فریم‌ورک **PyTorch**.

An Intelligent Persian Sentiment Analysis system leveraging Fine-Tuned **ParsBERT** (Transformer architecture) and **PyTorch** for Natural Language Processing (NLP).

---

## 📌 درباره پروژه (About The Project)

در این پروژه، از تکنیک **Fine-Tuning** روی مدل پیش‌آموزش‌دیده **ParsBERT** (`bert-fa-base-uncased`) برای پردازش متن و تشخیص نظر مثبت یا منفی کاربران (مثلاً در فروشگاه‌های اینترنتی یا شبکه‌های اجتماعی) استفاده شده است. داده‌ها ابتدا توسط Tokenizer مربوط به BERT توکنایز شده و سپس لایه طبقه‌بندی‌کننده (Classifier) سفارشی روی داده‌های فارسی آموزش داده می‌شود.

This project utilizes Fine-Tuning techniques on the pre-trained **ParsBERT** model to classify Persian user comments into **Positive** or **Negative** categories. It leverages Hugging Face Transformers for tokenization and PyTorch for custom classifier architecture.

---

## ✨ ویژگی‌های کلیدی (Key Features)

* **معماری پیشرفته NLP:** استفاده از مدل ترنسفورمر ParsBERT قدرتمند مخصوص زبان فارسی.
* **پیش‌پردازش و توکنایزیشن (Tokenization):** بسته‌بندی داده‌ها با Attention Masks و Padding استاندارد.
* **لایه‌بندی سفارشی در PyTorch:** افزودن لایه‌های Dropout و Linear بر روی خروجی Pooled_Output مدل BERT جهت جلوگیری از Overfitting.
* **ارزیابی دقیق:** محاسبه Precision, Recall و F1-Score برای بررسی عملکرد مدل.
* **تابع پیش‌بینی آنلاین (Inference Function):** قابلیت دریافت متن‌های جدید و پیش‌بینی سریع احساسات به همراه ایموجی.

---

## 🛠 پیش‌نیازها و نصب (Requirements & Installation)

کتابخانه‌های مورد نیاز را می‌توانید با دستور زیر نصب کنید:

pip install -r requirements.txt

---

🚀 نحوه اجرا (How to Run)
کافیست فایل اصلی پروژه را اجرا کنید:

python main.py



---

🏗 معماری مدل (Model Architecture)
Input Text ➡️ ParsBERT Tokenizer ➡️ ParsBERT Transformer Encoder ➡️ Dropout (0.3) ➡️ Dense Linear Layer (2 classes) ➡️ Softmax / Prediction
📝 لایسنس (License)
این پروژه تحت لایسنس MIT منتشر شده است. استفاده و تغییر در کد آزاد است.
