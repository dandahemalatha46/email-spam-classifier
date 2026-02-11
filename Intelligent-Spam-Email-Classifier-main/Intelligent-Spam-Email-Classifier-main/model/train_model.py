import pandas as pd
import numpy as np
import os
import joblib
import string
import requests
import zipfile
import io
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level from 'model' folder to 'spam_classifier' root
PROJECT_ROOT = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")
CSV_FILE = os.path.join(DATA_DIR, "spam.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")

# URL for SMS Spam Collection (Raw text, perfect for Naive Bayes text classification)
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

def extract_local_zip():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    for file in os.listdir(DATA_DIR):
        if file.endswith(".zip"):
            print(f"Found zip file: {file}. Extracting...")
            try:
                with zipfile.ZipFile(os.path.join(DATA_DIR, file), 'r') as zip_ref:
                    zip_ref.extractall(DATA_DIR)
                print("Extraction complete.")
                return True
            except Exception as e:
                print(f"Error extracting zip: {e}")
    return False

def download_data():
    # First check if we need to extract a local zip
    extract_local_zip()

    if os.path.exists(DATA_FILE) or os.path.exists(CSV_FILE):
        print("Dataset found.")
        return

    print("Dataset not found. Downloading SMS Spam Collection...")
    try:
        r = requests.get(DATASET_URL)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(DATA_DIR)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading data: {e}")
        print("Please manually place 'SMSSpamCollection' or 'spam.csv' in the 'data' folder.")

def load_data():
    # Try loading CSV first (common format)
    if os.path.exists(CSV_FILE):
        try:
            # Attempt to read standard spam CSVs (often encoding is latin-1)
            df = pd.read_csv(CSV_FILE, encoding='latin-1')
            # Look for common column names
            if 'v1' in df.columns and 'v2' in df.columns:
                df = df.rename(columns={'v1': 'label', 'v2': 'text'})
                df = df[['label', 'text']]
            elif 'Category' in df.columns and 'Message' in df.columns:
                 df = df.rename(columns={'Category': 'label', 'Message': 'text'})
            
            # Map labels to binary
            df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
            return df
        except Exception as e:
            print(f"Error reading CSV: {e}")

    # Try loading raw SMSSpamCollection (tab separated)
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE, sep='\t', names=['label', 'text'], encoding='utf-8')
            df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
            return df
        except Exception as e:
            print(f"Error reading SMSSpamCollection: {e}")

    return None

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def train():
    download_data()
    df = load_data()
    
    if df is None:
        print("Error: No valid dataset found. Please check 'data' folder.")
        return

    print(f"Dataset loaded: {len(df)} samples.")
    print("Preprocessing data...")
    df['clean_text'] = df['text'].apply(preprocess_text)

    # Feature Extraction
    print("Extracting features (Bag of Words)...")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['clean_text'])
    y = df['label_num']

    # Handle missing values if any
    if y.isnull().any():
        print("Dropping NaN labels...")
        # Align X and y by index if dropping rows (a bit complex with sparse matrix)
        # Re-doing simplified:
        df = df.dropna(subset=['label_num'])
        df['clean_text'] = df['text'].apply(preprocess_text)
        X = vectorizer.fit_transform(df['clean_text'])
        y = df['label_num']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    print("Training Naïve Bayes Model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    
    # Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_DIR}")

if __name__ == "__main__":
    train()
