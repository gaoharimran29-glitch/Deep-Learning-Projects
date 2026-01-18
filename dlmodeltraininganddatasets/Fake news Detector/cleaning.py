import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt_tab')
STOPWORDS = set(stopwords.words('english'))

df = pd.read_csv(r"/kaggle/input/fake-news-classification/WELFake_Dataset.csv")
df.drop(columns='Unnamed: 0' , inplace=True)
df.dropna(inplace=True)
df["content"] = df["title"] + " " + df["text"]
df.drop(columns=['title' , 'text'] , inplace=True)
print(df.shape)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)      # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)   # remove punctuation
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in STOPWORDS]
    return " ".join(tokens)

df['content'] = df["content"].apply(clean_text)
df.to_csv(r'Cleaned fake news dataset.csv' , index=False)
print('File saved')