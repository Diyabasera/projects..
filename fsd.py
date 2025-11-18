
import os
import glob
import re
import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# ---------------------------
# 1) Helper: find dataset(s)
# ---------------------------
def find_csv_files():
    # look for common names
    cwd = os.getcwd()
    files = os.listdir(cwd)
    csvs = [f for f in files if f.lower().endswith('.csv')]
    return csvs

# ---------------------------
# 2) Load dataset(s)
# ---------------------------
def load_data():
    csvs = find_csv_files()
    csvs_lower = [c.lower() for c in csvs]

    # preferred: Fake.csv + True.csv
    if 'fake.csv' in csvs_lower and 'true.csv' in csvs_lower:
        fake_path = csvs[csvs_lower.index('fake.csv')]
        true_path = csvs[csvs_lower.index('true.csv')]
        print(f"Loading {fake_path} and {true_path} ...")
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
        df_fake['label'] = 'FAKE'
        df_true['label'] = 'REAL'
        data = pd.concat([df_fake, df_true], ignore_index=True)
        return data

    # else if single combined file exists with label column
    # look for common combined names
    combined_candidates = ['mixed.csv', 'news.csv', 'dataset.csv', 'data.csv', 'combined.csv']
    for name in combined_candidates:
        if name in csvs_lower:
            path = csvs[csvs_lower.index(name)]
            print(f"Loading combined dataset {path} ...")
            df = pd.read_csv(path)
            return df

    # if only one csv exists (take it)
    if len(csvs) == 1:
        print(f"Only one CSV found: {csvs[0]} — attempting to load it.")
        df = pd.read_csv(csvs[0])
        return df

    # fallback: no suitable CSV found
    raise FileNotFoundError("No suitable dataset found. Put Fake.csv & True.csv (or a combined file) in this folder.")

# ---------------------------
# 3) Text cleaning
# ---------------------------
def ensure_nltk():
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK stopwords (only once)...")
        nltk.download('stopwords')

def clean_text(text, stop_words_set):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)           # remove urls
    text = re.sub(r'<.*?>', ' ', text)                      # remove html tags
    text = re.sub(r'[^a-z0-9\s]', ' ', text)                # keep alphanum whitespace
    text = re.sub(r'\s+', ' ', text).strip()                # normalize spaces
    tokens = [w for w in text.split() if w not in stop_words_set]
    return ' '.join(tokens)

# ---------------------------
# 4) Main pipeline
# ---------------------------
def main():
    start_time = time.time()
    print("Starting Fake News Detector pipeline...")

    # load
    try:
        data = load_data()
    except Exception as e:
        print("ERROR loading datasets:", e)
        print("Make sure your CSV files (Fake.csv and True.csv) are in this folder.")
        return

    print("Columns found in dataset:", data.columns.tolist())

    # If 'title' and 'text' columns exist (Kaggle dataset), combine them
    if 'title' in data.columns and 'text' in data.columns:
        data['content'] = data['title'].fillna('') + ' ' + data['text'].fillna('')
    elif 'text' in data.columns:
        data['content'] = data['text'].fillna('')
    else:
        # try to find a text-like column
        text_cols = [c for c in data.columns if data[c].dtype == object]
        if len(text_cols) >= 1:
            # use the longest text-like column
            data['content'] = data[text_cols[0]].fillna('')
            print(f"No 'text' column found — using column '{text_cols[0]}' as content.")
        else:
            print("No text column available in the CSV — cannot proceed.")
            return

    # label handling: if no label column, try to auto-detect
    if 'label' not in data.columns:
        possible_label_cols = ['label', 'truth', 'target', 'class', 'category']
        found = None
        for col in possible_label_cols:
            if col in data.columns:
                found = col
                break
        if found:
            data.rename(columns={found: 'label'}, inplace=True)
        else:
            print("No 'label' column found. You must have one column that tells 'FAKE' or 'REAL'.")
            print("If you have separate Fake.csv and True.csv, put both files in the folder with those exact names.")
            return

    # ensure label values are strings and normalized
    data['label'] = data['label'].astype(str).str.upper().str.strip()
    # some label encodings might be numeric: map 0/1 to FAKE/REAL if necessary
    unique_labels = set(data['label'].unique())
    if set(['0','1']).issubset(unique_labels) or set([0,1]).issubset(unique_labels):
        # assume 0 -> FAKE, 1 -> REAL
        data['label'] = data['label'].replace({'0':'FAKE','1':'REAL', 0:'FAKE', 1:'REAL'})

    print("Labels distribution:")
    print(data['label'].value_counts())

    # ---------- preprocessing ----------
    ensure_nltk()
    stop_words_set = set(stopwords.words('english'))

    print("Cleaning text (this may take a while for large datasets)...")
    data['cleaned'] = data['content'].apply(lambda x: clean_text(x, stop_words_set))
    # quick check
    print("Sample cleaned text:", data['cleaned'].iloc[0][:200])

    # ---------- vectorize ----------
    print("Vectorizing text with TF-IDF ...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = tfidf.fit_transform(data['cleaned'])
    le = LabelEncoder()
    y = le.fit_transform(data['label'])  # FAKE/REAL -> 0/1

    # ---------- train-test split ----------
    print("Train-test split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # ---------- model training ----------
    print("Training models ...")
    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression accuracy: {acc_lr:.4f}")

    # Multinomial Naive Bayes (fast & common for text)
    mnb = MultinomialNB()
    mnb.fit(X_train, y_train)
    y_pred_mnb = mnb.predict(X_test)
    acc_mnb = accuracy_score(y_test, y_pred_mnb)
    print(f"MultinomialNB accuracy: {acc_mnb:.4f}")

    # ---------- evaluation ----------
    print("\nClassification report (Logistic Regression):")
    print(classification_report(y_test, y_pred_lr, target_names=le.classes_))

    # confusion matrix (LogReg)
    cm = confusion_matrix(y_test, y_pred_lr)
    print("Confusion Matrix (Logistic Regression):")
    print(cm)

    # plot confusion matrix
    try:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title("Confusion Matrix (Logistic Regression)")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()
    except Exception as e:
        print("Could not show plot (maybe running in terminal-only). Error:", e)

    # ---------- save model & vectorizer ----------
    print("Saving model and vectorizer to disk (logreg_model.joblib, tfidf_vectorizer.joblib, label_encoder.joblib) ...")
    joblib.dump(logreg, "logreg_model.joblib")
    joblib.dump(tfidf, "tfidf_vectorizer.joblib")
    joblib.dump(le, "label_encoder.joblib")

    # ---------- interactive predict ----------
    def predict_text(text, model=logreg, vectorizer=tfidf, label_encoder=le):
        c = clean_text(text, stop_words_set)
        v = vectorizer.transform([c])
        p = model.predict(v)
        return label_encoder.inverse_transform(p)[0]

    # try a sample interactive loop
    print("\nYou can now type any news text to predict (empty line to quit). Example:")
    while True:
        try:
            s = input("\nEnter news text (or press Enter to exit):\n").strip()
        except KeyboardInterrupt:
            print("\nExiting.")
            break
        if s == "":
            print("Done.")
            break
        pred = predict_text(s)
        print("PREDICTION ->", pred)

    print(f"All done. Time elapsed: {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
