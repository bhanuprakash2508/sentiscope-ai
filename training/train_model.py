import sys
import os

# Fix import issue
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils.preprocess import clean_text

print("=" * 60)
print("IMDb SENTIMENT MODEL TRAINING")
print("=" * 60)

# Load Dataset
print("\n[1] Loading dataset...")

df = pd.read_csv("dataset/imdb_reviews.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

# Preprocess Text
print("\n[2] Cleaning reviews...")

df["text"] = df["text"].astype(str)

df["text"] = df["text"].apply(clean_text)

print("Text cleaning completed")


# Features and Labels
print("\n[3] Preparing features...")

X = df["text"]
y = df["target"]

# Train Test Split
print("\n[4] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.20,random_state=42
)

print("Training:", len(X_train))
print("Testing:", len(X_test))

# TF-IDF Vectorization
print("\n[5] Vectorizing text...")

vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("Vocabulary size:", len(vectorizer.vocabulary_))

# Train Model
print("\n[6] Training model...")

model = LogisticRegression(max_iter=2000)

model.fit(X_train,y_train)

print("Model training completed")

# Evaluate
print("\n[7] Evaluating model...")

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test,predictions)

print("\nAccuracy Score:")
print(accuracy)

print("\nClassification Report:")
print(classification_report(y_test,predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test,predictions))

# Save Model
print("\n[8] Saving model...")

os.makedirs(
    "models",
    exist_ok=True
)

with open(
    "models/sentiment_model.pkl",
    "wb"
) as file:

    pickle.dump(model,file)

with open(
    "models/vectorizer.pkl",
    "wb"
) as file:

    pickle.dump(vectorizer,file)

print("Model saved successfully")

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)