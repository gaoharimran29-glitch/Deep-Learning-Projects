import pickle
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

model = load_model(r"Movie Review Sentiment Analaysis\sentiment_lstm.keras")
STOPWORDS = set(stopwords.words('english'))

tokenizer = pickle.load(open(r'Movie Review Sentiment Analaysis\tokenizer.pkl' , 'rb'))

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=150, padding="post" , truncating='post')
    score = model.predict(padded)[0][0]

    if score >= 0.5:
        return "Positive", score
    else:
        return "Negative", score

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)      # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)   # remove punctuation
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)

review = "This movie was boring and too long"
review = clean_text(review)
sentiment, confidence = predict_sentiment(review)
print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")