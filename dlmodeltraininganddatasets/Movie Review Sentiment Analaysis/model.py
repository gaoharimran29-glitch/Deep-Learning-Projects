from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pickle

df = pd.read_csv(r'Movie Review Sentiment Analaysis\Cleaned data.csv')

X = df["review"].values
y = df["sentiment"].map({"negative": 0, "positive": 1}).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train_seq, maxlen=150, padding="post", truncating="post")
X_test = pad_sequences(X_test_seq, maxlen=150, padding="post", truncating="post")

early_stop = EarlyStopping(monitor="val_loss", patience=4 , restore_best_weights=True)

model = Sequential([
        Embedding(input_dim=15000, output_dim=128, input_length=200),
        Bidirectional(LSTM(64, return_sequences=False , dropout=0.3)),
        Dropout(0.5),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

model.summary()
model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[early_stop]
)

model.save("sentiment_lstm.keras")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model training complete")