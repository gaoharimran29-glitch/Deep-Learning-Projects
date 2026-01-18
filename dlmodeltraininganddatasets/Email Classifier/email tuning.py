import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import numpy as np 
# Note: I'm focusing only on the Logistic Regression part now for tuning.


# --- 1. Load and Split Data (Same as before) ---
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

# --- 2. Define the Pipeline ---
# We use the same pipeline structure, but name the components for Grid Search
pipeline = Pipeline([
    # Step 1: TF-IDF Vectorizer
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english" 
    )),
    # Step 2: Logistic Regression Classifier
    ('lr', LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

# --- 3. Define the Hyperparameter Grid ---
# The parameter names MUST start with the step name from the pipeline ('lr' or 'tfidf') 
# followed by a double underscore and the parameter name (e.g., 'lr__C').
param_grid = {
    # Tuning the Regularization Strength (C)
    'lr__C': [0.1, 1.0, 10.0],  # Test weak (10.0), default (1.0), and strong (0.1) regularization
    
    # Tuning the Solver (Try different optimization algorithms)
    'lr__solver': ['liblinear', 'lbfgs'], 
    
    # Optionally, tune a TF-IDF parameter (e.g., test a different max_features)
    'tfidf__max_features': [5000, 7500] 
}

# --- 4. Perform Grid Search Cross-Validation ---
# We optimize for 'f1_macro' because it's a balanced metric that considers both classes well.
print("Starting Grid Search...")
grid_search = GridSearchCV(
    pipeline,          # The pipeline to tune
    param_grid,        # The dictionary of parameters to test
    cv=5,              # 5-fold cross-validation
    scoring='f1_macro',# The metric to optimize (important for imbalanced data)
    n_jobs=-1,         # Use all available cores
    verbose=2          # Show progress
)

grid_search.fit(X_train, y_train)

# --- 5. Evaluate Best Model ---
print("\n========== Grid Search Results ==========")
print("Best parameters found on training set:")
print(grid_search.best_params_)

# Evaluate the best estimator on the separate test set
best_pipeline = grid_search.best_estimator_
y_pred = best_pipeline.predict(X_test)

print("\nAccuracy of Best Model on Test Set:", accuracy_score(y_test, y_pred))
print("\nClassification Report of Best Model:")
print(classification_report(y_test, y_pred))

""" ========== Grid Search Results ==========
Best parameters found on training set:
{'lr__C': 10.0, 'lr__solver': 'liblinear', 'tfidf__max_features': 7500}

Accuracy of Best Model on Test Set: 0.9820627802690582

Classification Report of Best Model:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99       965
           1       0.95      0.92      0.93       150

    accuracy                           0.98      1115
   macro avg       0.97      0.96      0.96      1115
weighted avg       0.98      0.98      0.98      1115

PS C:\Users\hp\Downloads\dlmodeltraininganddatasets>  """