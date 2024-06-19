import pandas as pd
import json
import random
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, mean_squared_error
from joblib import dump, load
import os
from sklearn import preprocessing

N_SAMPLES = 80000

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.class_prior = None
        self.feature_prob = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        word_counts = np.array([np.sum(y == class_label) for class_label in self.classes])
        self.class_prior = np.log(word_counts / len(y))

        feature_counts = np.array([X[y == class_label].sum(axis=0) for class_label in self.classes])
        feature_counts += 1

        self.feature_prob = np.log(feature_counts / feature_counts.sum(axis=1, keepdims=True))

    def predict(self, X):
        log_probs = X @ self.feature_prob.T + self.class_prior
        return self.classes[np.argmax(log_probs, axis=1)]
    
# Clean the text data
def clean_text(text, additional_stop_words=set()):
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

def load_data(n_samples=10000, file_path='../yelp_academic_dataset_review.json'):
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
    
def load_model(feature, X_train_dense, y_train):
    path = '../Models/naive_bayes_model_cv_'+ feature +'.joblib'
    model = NaiveBayes()
    if os.path.isfile(path):
        model = load(path)
        print("Load pre-trained " + feature +" model.")
    else:
        model.fit(X_train_dense, y_train)
        dump(model, path)
    return model

def normalization(X_train, X_test, X_val):
    scaler = preprocessing.MaxAbsScaler()
    X_scaler = scaler.fit_transform(X_train)
    X_sc_test = scaler.transform(X_test)
    X_sc_val = scaler.transform(X_val)
    return X_scaler, X_sc_test, X_sc_val

def cv_vectorizer(X_train, X_test, X_val):
    vectorizer = CountVectorizer(max_features=5000)
    X_cv_train = vectorizer.fit_transform(X_train)
    X_cv_test = vectorizer.transform(X_test)
    X_cv_val = vectorizer.transform(X_val)
    return X_cv_train, X_cv_test, X_cv_val

def train_split_save_data_cv():
    df = load_data(N_SAMPLES)
    # Drop any missing values
    df = df.dropna()
    #df.head()
    
    # Split the data
    X = df['text']
    y_stars = df['stars']
    y_useful = df['useful']
    y_funny = df['funny']
    y_cool = df['cool']
    
    # Split the data into training and testing sets
    X_train, X_temp, y_stars_train, y_stars_temp, y_useful_train, y_useful_temp, y_funny_train, y_funny_temp, y_cool_train, y_cool_temp = train_test_split(
    X, y_stars, y_useful, y_funny, y_cool, test_size=0.4, random_state=42)
     
    # Split temp into validation and test sets
    X_val, X_test, y_stars_val, y_stars_test, y_useful_val, y_useful_test, y_funny_val, y_funny_test, y_cool_val, y_cool_test = train_test_split(
    X_temp, y_stars_temp, y_useful_temp, y_funny_temp, y_cool_temp, test_size=0.5, random_state=42)
    
    # Initialize Count Vectorizer
    X_cv_train, X_cv_test, X_cv_val = cv_vectorizer(X_train, X_test, X_val)
    
    # Normalization the data 
    X_scaler, X_sc_test, X_sc_val = normalization(X_cv_train, X_cv_test, X_cv_val)
    
    # Convert sparse matrix to dense matrix
    X_train_dense = X_scaler.toarray()
    X_val_dense = X_sc_val.toarray()
    X_test_dense = X_sc_test.toarray()
    
    # Initialize and train the Naive Bayes model
    nb_model_stars = load_model('stars', X_train_dense, y_stars_train)
        
    # Validate the model
    y_stars_val_pred = nb_model_stars.predict(X_val_dense)
    mae_stars_val = mean_absolute_error(y_stars_val, y_stars_val_pred)
    print(f'Mean Absolute Error for Stars on Validation Set: {mae_stars_val}')
    
    # Predict on the test set
    y_stars_test_pred = nb_model_stars.predict(X_test_dense)

    # Evaluate the model
    accuracy_stars = accuracy_score(y_stars_test, y_stars_test_pred)
    mae_stars_test = mean_absolute_error(y_stars_test, y_stars_test_pred)
    report = classification_report(y_stars_test, y_stars_test_pred)
    print(f'Accuracy for Stars: {accuracy_stars}')
    print(f'Mean Absolute Error for Stars: {mae_stars_test}')
    print(f'Report for Naive Bayes\n {report}')
    
    
   # Train for 'useful'
    nb_model_useful = load_model('useful', X_train_dense, y_useful_train)

    # Validate the model
    y_useful_val_pred = nb_model_useful.predict(X_val_dense)
    mae_useful_val = mean_absolute_error(y_useful_val, y_useful_val_pred)
    acc_useful_val = accuracy_score(y_useful_val, y_useful_val_pred)
    print(f'Mean Absolute Error for Useful on Validation Set: {mae_useful_val}')

    # Predict on the test set
    y_useful_test_pred = nb_model_useful.predict(X_test_dense)
    mae_useful_test = mean_absolute_error(y_useful_test, y_useful_test_pred)
    acc_useful_test = accuracy_score(y_useful_val, y_useful_val_pred)
    print(f'Mean Absolute Error for Useful: {mae_useful_test}')

    
    # Train for 'funny'
    nb_model_funny = load_model('funny', X_train_dense, y_funny_train)

    # Validate the model
    y_funny_val_pred = nb_model_funny.predict(X_val_dense)
    mae_funny_val = mean_absolute_error(y_funny_val, y_funny_val_pred)
    print(f'Mean Absolute Error for Funny on Validation Set: {mae_funny_val}')

    # Predict on the test set
    y_funny_test_pred = nb_model_funny.predict(X_test_dense)
    mae_funny_test = mean_absolute_error(y_funny_test, y_funny_test_pred)
    print(f'Mean Absolute Error for Funny: {mae_funny_test}')

    
    # Train for 'cool'
    nb_model_cool = load_model('cool', X_train_dense, y_cool_train)

    # Validate the model
    y_cool_val_pred = nb_model_cool.predict(X_val_dense)
    mae_cool_val = mean_absolute_error(y_cool_val, y_cool_val_pred)
    print(f'Mean Absolute Error for Cool on Validation Set: {mae_cool_val}')

    # Predict on the test set
    y_cool_test_pred = nb_model_cool.predict(X_test_dense)
    mae_cool_test = mean_absolute_error(y_cool_test, y_cool_test_pred)
    print(f'Mean Absolute Error for Cool: {mae_cool_test}')
    
if __name__ == '__main__':

    train_split_save_data_cv()
