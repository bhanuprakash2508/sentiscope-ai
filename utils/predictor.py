import pickle
import os

from utils.preprocess import clean_text


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "vectorizer.pkl"
)


# Load trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# Load vectorizer
with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)


def predict_sentiment(text):

    # Clean input text
    cleaned_text = clean_text(text)

    # Convert text to vector
    vector = vectorizer.transform(
        [cleaned_text]
    )

    # Prediction
    prediction = model.predict(
        vector
    )[0]

    # Probability score
    probabilities = model.predict_proba(
        vector
    )[0]

    # Confidence percentage
    confidence = float(
        max(probabilities) * 100
    )

    # Final sentiment
    sentiment = (
        "Positive"
        if prediction == 1
        else "Negative"
    )

    return {
        "sentiment": sentiment,
        "confidence": round(
            confidence,
            2
        )
    }