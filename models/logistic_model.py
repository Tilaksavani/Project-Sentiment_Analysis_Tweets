import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

def run_logistic_regression(df):
    """
    Train and evaluate a Logistic Regression model using TF-IDF features.
    """
    print("[INFO] Running Logistic Regression...")

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['clean_text'])
    y = df['target']

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Predict and Evaluate
    preds = clf.predict(X_test)
    print("[INFO] Classification Report:")
    print(classification_report(y_test, preds))

    # ✅ Save model and vectorizer
    os.makedirs("models/saved_models", exist_ok=True)
    joblib.dump(clf, "models/saved_models/logistic_model.pkl")
    joblib.dump(tfidf, "models/saved_models/tfidf_vectorizer.pkl")
    print("[INFO] Logistic Regression model and TF-IDF vectorizer saved.")

    return clf
