import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# --- 1. Load Data ---
df = pd.read_csv(r'Fake news Detector\Cleaned fake news dataset.csv')

X = df['content'].astype(str)
y = df['label']

# --- 2. Train-test split FIRST (no leakage) ---
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --- 3. Tokenization ---
VOCAB_SIZE = 10000
MAX_LEN = 450

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# --- 4. Model ---
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=100, input_length=MAX_LEN),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    Bidirectional(LSTM(64, return_sequences=False, dropout=0.5 , recurrent_dropout=0.2)),
    Dropout(0.5) ,
    Dense(64, activation='relu' , activity_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# --- 5. Compile ---
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC()]
)

model.build(input_shape=(None, MAX_LEN))
model.summary()

# --- 6. Train ---
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='max')

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# --- 7. Evaluate ---
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
model.save("Fake_news_model2.keras")
print("Model saved")

import pickle
with open("tokenizer2.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Model training complete")