import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv(r"Email Classifier\Cleaned Email Dataset.csv")

df.dropna(subset=["cleaned_text"], inplace=True)

X = df["cleaned_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

best_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=7500,
        ngram_range=(1, 2),
        stop_words="english"
    )),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=10,
        solver="liblinear",
    ))
])

best_pipeline.fit(X_train, y_train)

y_pred = best_pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(best_pipeline , 'Email classifier.pkl')

print("Model saved successfully")