from flask import Flask, render_template, request
from utils.predictor import predict_sentiment
import pandas as pd
import os

import matplotlib
matplotlib.use("Agg")   # non-GUI backend

import matplotlib.pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
CHART_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Single Review Analysis
@app.route("/analyze", methods=["POST"])
def analyze():

    review = request.form["review"]

    if review.strip() == "":
        return render_template(
            "index.html",
            error="Please enter review text."
        )

    result = predict_sentiment(review)

    return render_template(
        "result.html",
        review=review,
        sentiment=result["sentiment"],
        confidence=result["confidence"]
    )

# CSV Upload Analysis
@app.route("/upload", methods=["POST"])
def upload():

    file = request.files["csvfile"]

    if not file:
        return render_template(
            "index.html",
            error="Please upload a CSV file."
        )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        if "review" not in df.columns:
            return render_template(
                "index.html",
                error="CSV must contain column name: review"
            )

    except Exception:
        return render_template(
            "index.html",
            error="Invalid CSV file."
        )

    positive = 0
    negative = 0

    for text in df["review"]:

        result = predict_sentiment(
            str(text)
        )

        if result["sentiment"] == "Positive":
            positive += 1
        else:
            negative += 1

    total = len(df)

    # percentages
    positive_percent = round(
        (positive / total) * 100,2
    )

    negative_percent = round(
        (negative / total) * 100,2
    )

    # insight
    if positive_percent >= 70:
        insight = "Customer sentiment is very healthy."

    elif positive_percent >= 50:
        insight = "Overall sentiment is moderately positive."

    else:
        insight = "High negative sentiment detected. Improvement needed."

    # Generate Pie Chart
    labels = ["Positive", "Negative"]
    sizes = [positive, negative]

    plt.figure(figsize=(6, 6))

    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%"
    )

    plt.title("Sentiment Distribution")

    chart_path = os.path.join(
        CHART_FOLDER,
        "chart.png"
    )

    plt.savefig(chart_path)
    plt.close()

    return render_template(
        "dashboard.html",
        total=total,
        positive=positive,
        negative=negative,
        positive_percent=positive_percent,
        negative_percent=negative_percent,
        insight=insight
    )

# Run App
if __name__ == "__main__":
    app.run()