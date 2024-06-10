import json
import random
import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from nltk.corpus import stopwords
from torch.optim import lr_scheduler

# Constants
MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 10000
N_SAMPLES = 20000

# Clean the text data
def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def load_data(n_samples=10000, file_path='yelp_academic_dataset_review.json'):
    random.seed(42)
    num_lines = sum(1 for l in open(file_path))
    keep_idx = set(random.sample(range(num_lines), n_samples))
    data = []
    with open(file_path) as f:
        for i, line in enumerate(f):
            if i in keep_idx:
                data.append(json.loads(line))
    df = pd.DataFrame(data)
    df = df.drop(['review_id', 'user_id', 'business_id', 'date'], axis=1)
    df['text'] = df['text'].apply(clean_text)
    return df


def data_preprocessing():
    df = load_data(N_SAMPLES)
    df['stars'] = df['stars'].astype(int)

    # Drop any missing values
    df = df.dropna()

    # Prepare the texts and the labels
    texts = df['text'].values
    stars = df['stars'].values - 1  # Convert 1-5 to 0-4
    funny = df['funny'].values
    useful = df['useful'].values
    cool = df['cool'].values

    tokenizer = get_tokenizer("basic_english")

    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(df['text']), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1  # Convert 1-5 to 0-4

    # Split the data
    X_train_texts, X_temp_texts, y_train_stars, y_temp_stars = train_test_split(texts, stars, test_size=0.4, random_state=42)
    X_valid_texts, X_test_texts, y_valid_stars, y_test_stars = train_test_split(X_temp_texts, y_temp_stars, test_size=0.5, random_state=42)

    y_train_funny, y_temp_funny = train_test_split(funny, test_size=0.4, random_state=42)
    y_valid_funny, y_test_funny = train_test_split(y_temp_funny, test_size=0.5, random_state=42)

    y_train_useful, y_temp_useful = train_test_split(useful, test_size=0.4, random_state=42)
    y_valid_useful, y_test_useful = train_test_split(y_temp_useful, test_size=0.5, random_state=42)

    y_train_cool, y_temp_cool = train_test_split(cool, test_size=0.4, random_state=42)
    y_valid_cool, y_test_cool = train_test_split(y_temp_cool, test_size=0.5, random_state=42)

    def preprocess_batch(batch_texts, batch_stars, batch_funny, batch_useful, batch_cool):
        token_ids = [text_pipeline(text) for text in batch_texts]
        token_ids = [item[:MAX_SEQUENCE_LENGTH] for item in token_ids]
        token_ids = [pad(item) for item in token_ids]
        return (
            torch.tensor(token_ids, dtype=torch.long), 
            torch.tensor(batch_stars, dtype=torch.long),
            torch.tensor(batch_funny, dtype=torch.float),
            torch.tensor(batch_useful, dtype=torch.float),
            torch.tensor(batch_cool, dtype=torch.float)
        )

    def pad(sequence):
        return sequence + [0] * (MAX_SEQUENCE_LENGTH - len(sequence))

    X_train, y_train_stars, y_train_funny, y_train_useful, y_train_cool = preprocess_batch(X_train_texts, y_train_stars, y_train_funny, y_train_useful, y_train_cool)
    X_valid, y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool = preprocess_batch(X_valid_texts, y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool)
    X_test, y_test_stars, y_test_funny, y_test_useful, y_test_cool = preprocess_batch(X_test_texts, y_test_stars, y_test_funny, y_test_useful, y_test_cool)

    return X_train, X_valid, X_test, (y_train_stars, y_train_funny, y_train_useful, y_train_cool), (y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool), (y_test_stars, y_test_funny, y_test_useful, y_test_cool), vocab


class YelpReviewDataset(Dataset):
    def __init__(self, texts, stars, funny, useful, cool):
        self.texts = texts
        self.stars = stars
        self.funny = funny
        self.useful = useful
        self.cool = cool

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.stars[idx], self.funny[idx], self.useful[idx], self.cool[idx]

def create_data_loader(texts, stars, funny, useful, cool, batch_size):
    dataset = YelpReviewDataset(texts, stars, funny, useful, cool)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

class MultiTaskCNNTextModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, num_filters, filter_sizes):
        super(MultiTaskCNNTextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (fs, embed_size)) for fs in filter_sizes])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(num_filters) for _ in filter_sizes])
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.regressor = nn.Linear(len(filter_sizes) * num_filters, 3)  # 3 regression targets: funny, useful, cool

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add a channel dimension: (batch_size, 1, sequence_length, embed_size)
        x = [self.batch_norms[i](F.relu(conv(x))).squeeze(3) for i, conv in enumerate(self.convs)]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        classification_logits = self.classifier(x)
        regression_outputs = self.regressor(x)
        return classification_logits, regression_outputs

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    model.train()
    total_classification_loss = 0
    total_regression_loss = 0
    total_correct = 0
    total = 0

    for texts, stars, funny, useful, cool in tqdm(data_loader, desc="Training"):
        texts = texts.to(device)
        stars = stars.to(device)
        funny = funny.to(device)
        useful = useful.to(device)
        cool = cool.to(device)

        optimizer.zero_grad()
        classification_logits, regression_outputs = model(texts)
        classification_loss = F.cross_entropy(classification_logits, stars)
        regression_loss = F.mse_loss(regression_outputs[:, 0], funny) + \
                          F.mse_loss(regression_outputs[:, 1], useful) + \
                          F.mse_loss(regression_outputs[:, 2], cool)
        
        loss = classification_loss + regression_loss
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_classification_loss += classification_loss.item() * texts.size(0)
        total_regression_loss += regression_loss.item() * texts.size(0)
        total_correct += (classification_logits.argmax(1) == stars).sum().item()
        total += stars.size(0)

    return total_correct / total, total_classification_loss / total, total_regression_loss / total

def eval_model(model, data_loader, device):
    model.eval()
    total_classification_loss = 0
    total_regression_loss = 0
    total_correct = 0
    total = 0
    mse_funny, mse_useful, mse_cool = 0, 0, 0

    with torch.no_grad():
        for texts, stars, funny, useful, cool in tqdm(data_loader, desc="Evaluating"):
            texts = texts.to(device)
            stars = stars.to(device)
            funny = funny.to(device)
            useful = useful.to(device)
            cool = cool.to(device)

            classification_logits, regression_outputs = model(texts)
            classification_loss = F.cross_entropy(classification_logits, stars)
            regression_loss_funny = F.mse_loss(regression_outputs[:, 0], funny)
            regression_loss_useful = F.mse_loss(regression_outputs[:, 1], useful)
            regression_loss_cool = F.mse_loss(regression_outputs[:, 2], cool)

            regression_loss = regression_loss_funny + regression_loss_useful + regression_loss_cool

            total_classification_loss += classification_loss.item() * texts.size(0)
            total_regression_loss += regression_loss.item() * texts.size(0)
            total_correct += (classification_logits.argmax(1) == stars).sum().item()
            total += stars.size(0)

            mse_funny += regression_loss_funny.item() * texts.size(0)
            mse_useful += regression_loss_useful.item() * texts.size(0)
            mse_cool += regression_loss_cool.item() * texts.size(0)

    classification_accuracy = total_correct / total
    avg_classification_loss = total_classification_loss / total
    avg_regression_loss = total_regression_loss / total
    mse_funny /= total
    mse_useful /= total
    mse_cool /= total
    rmse_funny = mse_funny ** 0.5
    rmse_useful = mse_useful ** 0.5
    rmse_cool = mse_cool ** 0.5

    return (classification_accuracy, avg_classification_loss, avg_regression_loss,
            mse_funny, mse_useful, mse_cool, rmse_funny, rmse_useful, rmse_cool)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), 'checkpoint.pt')

early_stopping = EarlyStopping(patience=5, delta=0.01)

if __name__ == '__main__':
    X_train, X_valid, X_test, y_train, y_valid, y_test, vocab = data_preprocessing()
    y_train_stars, y_train_funny, y_train_useful, y_train_cool = y_train
    y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool = y_valid
    y_test_stars, y_test_funny, y_test_useful, y_test_cool = y_test
    
    BATCH_SIZE = 32
    EMBED_SIZE = 100
    NUM_CLASSES = 5
    NUM_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]

    train_loader = create_data_loader(X_train, y_train_stars, y_train_funny, y_train_useful, y_train_cool, BATCH_SIZE)
    valid_loader = create_data_loader(X_valid, y_valid_stars, y_valid_funny, y_valid_useful, y_valid_cool, BATCH_SIZE)
    test_loader = create_data_loader(X_test, y_test_stars, y_test_funny, y_test_useful, y_test_cool, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiTaskCNNTextModel(vocab_size=len(vocab), embed_size=EMBED_SIZE, num_classes=NUM_CLASSES,
                                  num_filters=NUM_FILTERS, filter_sizes=FILTER_SIZES)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    EPOCHS = 20
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print('-' * 10)
        train_acc, train_class_loss, train_reg_loss = train_epoch(model, train_loader, optimizer, device, scheduler=None)
        print(f"Train classification loss: {train_class_loss:.4f}, Train regression loss: {train_reg_loss:.4f}, Train accuracy: {train_acc:.4f}")

        (val_acc, val_class_loss, val_reg_loss,
         mse_funny, mse_useful, mse_cool,
         rmse_funny, rmse_useful, rmse_cool) = eval_model(model, valid_loader, device)
        
        print(f"Validation classification loss: {val_class_loss:.4f}, Validation regression loss: {val_reg_loss:.4f}, Validation accuracy: {val_acc:.4f}")
        print(f"Validation MSE - Funny: {mse_funny:.4f}, Useful: {mse_useful:.4f}, Cool: {mse_cool:.4f}")
        print(f"Validation RMSE - Funny: {rmse_funny:.4f}, Useful: {rmse_useful:.4f}, Cool: {rmse_cool:.4f}")
        
        # Pass validation loss to scheduler
        scheduler.step(val_class_loss + val_reg_loss)

        early_stopping(val_class_loss + val_reg_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    model.load_state_dict(torch.load('checkpoint.pt'))

    (test_acc, test_class_loss, test_reg_loss,
     mse_funny, mse_useful, mse_cool,
     rmse_funny, rmse_useful, rmse_cool) = eval_model(model, test_loader, device)

    print(f"Test classification loss: {test_class_loss:.4f}, Test regression loss: {test_reg_loss:.4f}, Test accuracy: {test_acc:.4f}")
    print(f"Test MSE - Funny: {mse_funny:.4f}, Useful: {mse_useful:.4f}, Cool: {mse_cool:.4f}")
    print(f"Test RMSE - Funny: {rmse_funny:.4f}, Useful: {rmse_useful:.4f}, Cool: {rmse_cool:.4f}")