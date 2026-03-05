# /app/preprocessing/text_cleaner.py
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))

def clean_text(text):
    return " ".join(
        [w for w in str(text).split() if w.lower() not in STOP_WORDS]
    )