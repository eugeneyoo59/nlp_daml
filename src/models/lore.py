# src/models/lore.py

from utils.feature_engineering import TFIDFFeatureEngineer as Feature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load dataset
df = pd.read_csv("/Users/akarenin/daml/nlp_daml/data/WELFake_Dataset.csv")

# 1. Split your dataframe
X_train, X_test, y_train, y_test = Feature.split_data(
    df, test_size=0.2, random_state=42, stratify=True
)

# 2. TF-IDF feature engineering
fe = Feature(max_features=5000)  # you can tune this number

# X_train/X_test are pandas Series, convert to plain lists/arrays
X_train_tfidf = fe.fit_transform(X_train.tolist())
X_test_tfidf = fe.transform(X_test.tolist())

# 3. Set up and train Logistic Regression
log_reg = LogisticRegression(
    solver="liblinear",      # good for smaller datasets / binary classification
    max_iter=1000,
    class_weight="balanced"  # optional but helpful if classes are imbalanced
)

log_reg.fit(X_train_tfidf, y_train)

# 4. Evaluate on test set
y_pred = log_reg.predict(X_test_tfidf)
y_proba = log_reg.predict_proba(X_test_tfidf)[:, 1]  # probability of positive class

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))

