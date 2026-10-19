
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

print("Loading dataset...")

df = pd.read_csv("data/paysim.csv")

# Create features
df["origin_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

df["destination_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

df["origin_balance_error"] = (
    df["amount"] - df["origin_balance_change"]
).abs()

df["destination_balance_error"] = (
    df["amount"] - df["destination_balance_change"]
).abs()

# Select input features and target
features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "origin_balance_change",
    "destination_balance_change",
    "origin_balance_error",
    "destination_balance_error"
]

X = df[features]
y = df["isFraud"]

# Split data before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Preprocess numeric and categorical features
numeric_features = [
    col for col in features if col != "type"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        ("type", OneHotEncoder(handle_unknown="ignore"), ["type"])
    ]
)

# Build model pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

print("Training Logistic Regression...")

model.fit(X_train, y_train)

print("\nEvaluating model...")

y_pred = model.predict(X_test)
y_scores = model.predict_proba(X_test)[:, 1]

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print(
    "Average precision (PR-AUC summary):",
    round(average_precision_score(y_test, y_scores), 4)
)

print("\nModel training and evaluation completed!")
