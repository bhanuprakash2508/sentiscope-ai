import pickle
import os

from utils.preprocess import clean_text

# paths
MODEL_PATH = os.path.join(
    "models",
    "sentiment_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    "models",
    "vectorizer.pkl"
)

# load trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# load vectorizer
with open(VECTORIZER_PATH, "rb") as file:
    vectorizer = pickle.load(file)

def predict_sentiment(text):

    # clean input text
    cleaned_text = clean_text(text)

    # convert text → vector
    vector = vectorizer.transform(
        [cleaned_text]
    )

    # prediction
    prediction = model.predict(
        vector
    )[0]

    # probability
    probabilities = model.predict_proba(
        vector
    )[0]

    # convert numpy float to normal float
    confidence = float(
        max(probabilities) * 100
    )

    # sentiment label
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