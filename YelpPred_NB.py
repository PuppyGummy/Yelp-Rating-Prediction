import pandas as pd
import json
import random
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
from sklearn import preprocessing
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 10000
N_SAMPLES = 80000
NUM_CLASSES = 5

# Clean the text data
def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    # Convert to lowercase
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    # Remove stop words
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

df = load_data(N_SAMPLES)
# Drop any missing values
df = df.dropna()
#print(df.head())

# Split the data
X = df['text']
y_stars = df['stars']
y_useful = df['useful']
y_funny = df['funny']
y_cool = df['cool']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

scaler = preprocessing.MaxAbsScaler()
X_scaler = scaler.fit_transform(X_tfidf)

'''
scaler = preprocessing.MaxAbsScaler()
X_scaler = scaler.fit_transform(X)
'''

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_word_counts = None
        self.class_totals = None
        self.class_priors = None
        self.vocab = None
        self.vocab_size = 0
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_word_counts = {cls: np.zeros(X.shape[1]) for cls in self.classes}
        self.class_totals = {cls: 0 for cls in self.classes}
        self.class_priors = {cls: 0 for cls in self.classes}
        
        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_word_counts[cls] = np.sum(X_cls, axis=0)
            self.class_totals[cls] = np.sum(self.class_word_counts[cls])
            self.class_priors[cls] = X_cls.shape[0] / X.shape[0]
        
        self.vocab_size = X.shape[1]
    
    def predict(self, X):
        y_pred = []
        for x in X:
            class_probs = {}
            for cls in self.classes:
                log_prior = np.log(self.class_priors[cls])
                log_likelihood = np.sum(np.log((self.class_word_counts[cls] + 1) / (self.class_totals[cls] + self.vocab_size)) * x)
                class_probs[cls] = log_prior + log_likelihood
            y_pred.append(max(class_probs, key=class_probs.get))
        return np.array(y_pred)

# Convert sparse matrix to dense matrix
X_tfidf_dense = X_scaler.toarray()

# Split the data into training and testing sets
X_train, X_test, y_stars_train, y_stars_test = train_test_split(X_tfidf_dense, y_stars, test_size=0.2, random_state=42)

'''
pipe_nb = make_pipeline(
    TfidfVectorizer(),
    NaiveBayes()
)

scores = cross_validate(pipe_nb, X_train, y_stars_train, return_train_score=True)
print(scores)
'''
# Initialize and train the Naive Bayes model
nb_model = NaiveBayes()
nb_model.fit(X_train, y_stars_train)

# Predict on the test set
y_stars_pred = nb_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_stars_test, y_stars_pred)
mae = mean_absolute_error(y_stars_test, y_stars_pred)

print(f'Accuracy: {accuracy}')
print(f'Mean Absolute Error: {mae}')
print(classification_report(y_stars_test, y_stars_pred))

'''
#Hyperparameter Tuning to improve Accuracy
cv_method = RepeatedStratifiedKFold(n_splits=5,  n_repeats=3, random_state=999)
'''

# Train for 'useful'
X_train, X_test, y_useful_train, y_useful_test = train_test_split(X_tfidf_dense, y_useful, test_size=0.2, random_state=42)
nb_model_useful = NaiveBayes()
nb_model_useful.fit(X_train, y_useful_train)
y_useful_pred = nb_model_useful.predict(X_test)
mae_useful = mean_absolute_error(y_useful_test, y_useful_pred)
print(f'Mean Absolute Error for Useful: {mae_useful}')

# Train for 'funny'
X_train, X_test, y_funny_train, y_funny_test = train_test_split(X_tfidf_dense, y_funny, test_size=0.2, random_state=42)
nb_model_funny = NaiveBayes()
nb_model_funny.fit(X_train, y_funny_train)
y_funny_pred = nb_model_funny.predict(X_test)
mae_funny = mean_absolute_error(y_funny_test, y_funny_pred)
print(f'Mean Absolute Error for Funny: {mae_funny}')

# Train for 'cool'
X_train, X_test, y_cool_train, y_cool_test = train_test_split(X_tfidf_dense, y_cool, test_size=0.2, random_state=42)
nb_model_cool = NaiveBayes()
nb_model_cool.fit(X_train, y_cool_train)
y_cool_pred = nb_model_cool.predict(X_test)
mae_cool = mean_absolute_error(y_cool_test, y_cool_pred)
print(f'Mean Absolute Error for Cool: {mae_cool}')
