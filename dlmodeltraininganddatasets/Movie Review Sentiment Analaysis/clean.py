import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
df = pd.read_csv(r'')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)      # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)   # remove punctuation
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)

df["review"] = df["review"].apply(clean_text)
df.to_csv('Cleaned data.csv' , index=False)