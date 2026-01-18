import pandas as pd
df = pd.read_csv(r'spam.csv' , encoding='latin-1')
df.drop(columns=['Unnamed: 2' , "Unnamed: 3" , "Unnamed: 4"] , inplace=True)
print(df.head())
print(df.isnull().sum())
print(df.info())

df['label'] = df['v1'].map({"ham":0 , "spam":1})
df.drop(columns="v1" , inplace=True)
print(df.head())

import re
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df['cleaned_text'] = df['v2'].apply(clean_text)

df.drop(columns='v2' , inplace=True)
print(df.head())
df.dropna(subset=['cleaned_text'] , inplace=True)
df.to_csv('Cleaned Email Dataset.csv' , index=False)
print("File saved")