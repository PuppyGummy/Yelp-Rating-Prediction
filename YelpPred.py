import json
import random
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from imblearn.over_sampling import SMOTE
import numpy as np

# nltk.download('stopwords')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def load_data():
    random.seed(42)
    num_lines = sum(1 for l in open('yelp_academic_dataset_review.json'))
    size = 10000
    keep_idx = set(random.sample(range(num_lines), size))
    data = []
    with open('yelp_academic_dataset_review.json') as f:
        for i, line in enumerate(f):
            if i in keep_idx:
                data.append(json.loads(line))
    df = pd.DataFrame(data)
    df = df.drop(['review_id', 'user_id', 'business_id', 'date'], axis=1)
    df['text'] = df['text'].apply(clean_text)
    return df

def data_preprocessing():
    df = load_data()
    df['stars'] = df['stars'].astype(int)

    # Drop any missing values
    df = df.dropna()
    
    X = df['text']
    y = df[['stars', 'useful', 'funny', 'cool']]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
    
    # smote = SMOTE(random_state=42)
    # X_train_res, y_train_res = smote.fit_resample(pd.DataFrame(X_train), y_train[['stars']])
    
    # train_data_resampled = pd.concat([X_train_res, y_train[['useful', 'funny', 'cool']].reset_index(drop=True)], axis=1)
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

class YelpReviewDataset(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        targets = self.targets.iloc[idx]
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
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'stars': torch.tensor(targets['stars'], dtype=torch.long),
            'regression_targets': torch.tensor(targets[['useful', 'funny', 'cool']].values, dtype=torch.float)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = YelpReviewDataset(
        texts=df['text'],
        targets=df[['stars', 'useful', 'funny', 'cool']],
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4
    )

class BERTMultiTaskModel(nn.Module):
    def __init__(self, n_classes):
        super(BERTMultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 3)
    
    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = bert_outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        classification_logits = self.classifier(pooled_output)
        regression_outputs = self.regressor(pooled_output)
        return classification_logits, regression_outputs

def train_epoch(model, data_loader, classification_loss_fn, regression_loss_fn, optimizer, device, scheduler):
    model.train()
    total_classification_loss = 0
    total_regression_loss = 0
    for d in data_loader:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        stars = d['stars'].to(device)
        regression_targets = d['regression_targets'].to(device)
        classification_logits, regression_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        classification_loss = classification_loss_fn(classification_logits, stars)
        regression_loss = regression_loss_fn(regression_outputs, regression_targets)
        loss = classification_loss + regression_loss
        total_classification_loss += classification_loss.item()
        total_regression_loss += regression_loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return total_classification_loss / len(data_loader), total_regression_loss / len(data_loader)

def eval_model(model, data_loader, classification_loss_fn, regression_loss_fn, device):
    model.eval()
    total_classification_loss = 0
    total_regression_loss = 0
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            stars = d['stars'].to(device)
            regression_targets = d['regression_targets'].to(device)
            classification_logits, regression_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            classification_loss = classification_loss_fn(classification_logits, stars)
            regression_loss = regression_loss_fn(regression_outputs, regression_targets)
            total_classification_loss += classification_loss.item()
            total_regression_loss += regression_loss.item()
    return total_classification_loss / len(data_loader), total_regression_loss / len(data_loader)

def predict(model, data_loader, device):
    model = model.eval()
    reviews = []
    stars = []
    regression_outputs = []
    predictions = []
    regression_predictions = []
    with torch.no_grad():
        for d in data_loader:
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            stars_batch = d['stars'].to(device)
            regression_targets_batch = d['regression_targets'].to(device)
            classification_logits, regression_output = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(classification_logits, dim=1)
            reviews.extend(d['text'])
            stars.extend(stars_batch.cpu().numpy())
            regression_outputs.extend(regression_output.cpu().numpy())
            predictions.extend(preds.cpu().numpy())
            regression_predictions.extend(regression_output.cpu().numpy())
    return reviews, stars, regression_outputs, predictions, regression_predictions

if __name__ == '__main__':
    X_train, X_valid, X_test, y_train, y_valid, y_test = data_preprocessing()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    MAX_LEN = 160
    BATCH_SIZE = 16

    # 将数据框合并以创建数据加载器
    train_data = pd.concat([X_train, y_train], axis=1)
    val_data = pd.concat([X_valid, y_valid], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    train_data_loader = create_data_loader(train_data, tokenizer, MAX_LEN, BATCH_SIZE)
    val_data_loader = create_data_loader(val_data, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(test_data, tokenizer, MAX_LEN, BATCH_SIZE)

    model = BERTMultiTaskModel(n_classes=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    EPOCHS = 10
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    classification_loss_fn = nn.CrossEntropyLoss().to(device)
    regression_loss_fn = nn.MSELoss().to(device)

    for epoch in range(EPOCHS):
        train_classification_loss, train_regression_loss = train_epoch(
            model,
            train_data_loader,
            classification_loss_fn,
            regression_loss_fn,
            optimizer,
            device,
            scheduler
        )
        val_classification_loss, val_regression_loss = eval_model(
            model,
            val_data_loader,
            classification_loss_fn,
            regression_loss_fn,
            device
        )
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print(f'Train classification loss: {train_classification_loss}, Train regression loss: {train_regression_loss}')
        print(f'Validation classification loss: {val_classification_loss}, Validation regression loss: {val_regression_loss}')

    reviews, true_stars, true_regression, pred_stars, pred_regression = predict(model, test_data_loader, device)

    accuracy = accuracy_score(true_stars, pred_stars)
    precision, recall, f1, _ = precision_recall_fscore_support(true_stars, pred_stars, average='weighted')

    print(f'Classification Task:')
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')

    true_regression = np.array(true_regression)
    pred_regression = np.array(pred_regression)

    mse = mean_squared_error(true_regression, pred_regression)
    rmse = np.sqrt(mse)

    print(f'Regression Task:')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
