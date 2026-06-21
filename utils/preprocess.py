import re
import nltk
from nltk.corpus import stopwords

# Load stopwords only once
try:
    stop_words = set(stopwords.words("english"))

except LookupError:

    nltk.download("stopwords", quiet=True)

    stop_words = set(stopwords.words("english"))

# Text Cleaning Function
def clean_text(text):

    text = str(text).lower()

    # remove html tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # keep only letters
    text = re.sub(
        r"[^a-zA-Z]",
        " ",
        text
    )

    words = text.split()

    # remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)