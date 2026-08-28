import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ۱. تنظیم دانه تصادفی جهت تکرارپذیری
np.random.seed(42)
torch.manual_seed(42)

# تعیین سخت‌افزار (GPU در صورت وجود، در غیر این صورت CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"اجرا روی دستگاه: {device}")

# ---------------------------------------------------------
# ۲. شبیه‌سازی داده‌های متنی (Sentiment Dataset)
# ---------------------------------------------------------
print("\nدر حال ساخت داده‌های نمونه متنی...")
raw_data = {
    "text": [
        "این محصول عالی بود، کیفیت فوق‌العاده‌ای داشت و کاملاً راضی هستم",
        "اصلا پیشنهاد نمی‌کنم، کیفیت بد و ارسال بسیار کند بود",
        "بسیار عالی و کاربردی، بسته بندی هم خیلی خوب بود",
        "افتضاح بود، بعد از دو روز خراب شد و مرجوع کردم",
        "طراحی بسیار زیبا و عملکرد روان، ارزش خرید بالایی دارد",
        "کیفیت ساخت پایینی داره و اصلاً مثل عکسش نیست",
        "خریدمش و واقعا راضیم، به همه پیشنهاد میکنم",
        "بسیار بد و بی‌کیفیت، پولتون رو دور نریزید",
        "سرعت عالی و کارکرد بدون نقص، حتما باز هم میخرم",
        "اصلا خوب نبود، پشتیبانی هم پاسخگو نیست"
    ] * 20,  # تکثیر داده‌ها برای آموزش بهتر
    "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 20  # ۱: مثبت | ۰: منفی
}

df = pd.DataFrame(raw_data)

# ---------------------------------------------------------
# ۳. آماده‌سازی Tokenizer با ParsBERT (مدل NLP فارسی)
# ---------------------------------------------------------
MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"
print(f"در حال بارگذاری Tokenizer مدل {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# تقسیم داده‌ها به آموزش و تست (80% Train, 20% Test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'].values, df['label'].values, test_size=0.2, random_state=42
)

train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ---------------------------------------------------------
# ۴. تعریف معماری مدل Classifier با PyTorch + BERT
# ---------------------------------------------------------
class BERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)


model = BERTClassifier(MODEL_NAME).to(device)

# ---------------------------------------------------------
# ۵. تنظیمات آموزش (Optimizer & Loss)
# ---------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# ---------------------------------------------------------
# ۶. حلقه آموزش (Training Loop)
# ---------------------------------------------------------
print("\nشروع آموزش مدل تحلیل احساسات...")
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

# ---------------------------------------------------------
# ۷. ارزیابی مدل و تست روی داده جدید
# ---------------------------------------------------------
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.tolist())

print("\n--- گزارش ارزیابی مدل (Classification Report) ---")
print(classification_report(true_labels, predictions, target_names=['منفی (0)', 'مثبت (1)']))


# ---------------------------------------------------------
# ۸. تابع پیش‌بینی برای متن ورودی دلخواه
# ---------------------------------------------------------
def predict_sentiment(text):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, pred = torch.max(outputs, dim=1)

    sentiment = "مثبت 😊" if pred.item() == 1 else "منفی 😡"
    print(f"\nمتن: '{text}'")
    print(f"نتیجه تحلیل: {sentiment}")


# تست چند متن جدید
print("\n--- تست هوشمند با متن‌های جدید ---")
predict_sentiment("ارسالشون خیلی سریع بود و جنسش حرف نداشت")
predict_sentiment("اصلا کیفیت نداشت و الکی پولم هدر رفت")