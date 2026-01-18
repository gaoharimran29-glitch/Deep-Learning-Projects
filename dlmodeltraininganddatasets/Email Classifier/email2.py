import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# Load the dataset
df = pd.read_csv(r"Email Classifier\Cleaned Email Dataset.csv")

# Drop rows with missing 'cleaned_text'
df.dropna(subset=["cleaned_text"], inplace=True)

X = df["cleaned_text"] 
y = df["label"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Define the models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42 # Added random state for reproducibility
    ),
    "MultinomialNB": MultinomialNB()
}

# Training and Evaluation Loop
for name, model in models.items():
    print(f"\n========== {name} ==========")

    # Define the Pipeline
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english" 
            # The "english" string tells TfidfVectorizer to use a predefined list of English stop words
        )),
        ("model", model)
    ])
    
    # Train the model
    pipe.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipe.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

#logistic regression is winner
""" ========== Logistic Regression ==========
Accuracy: 0.9775784753363229
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       965
           1       0.92      0.91      0.92       150

    accuracy                           0.98      1115
   macro avg       0.95      0.95      0.95      1115
weighted avg       0.98      0.98      0.98      1115


========== MultinomialNB ==========
Accuracy: 0.9650224215246637
              precision    recall  f1-score   support

           0       0.96      1.00      0.98       965
           1       1.00      0.74      0.85       150

    accuracy                           0.97      1115
   macro avg       0.98      0.87      0.92      1115
weighted avg       0.97      0.97      0.96      1115

PS C:\Users\hp\Downloads\dlmodeltraininganddatasets>  """