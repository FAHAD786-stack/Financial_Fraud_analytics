
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score
)

print("Loading dataset...")

df = pd.read_csv("data/paysim.csv")

# Feature engineering
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

features = [
    "step", "type", "amount",
    "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest",
    "origin_balance_change",
    "destination_balance_change",
    "origin_balance_error",
    "destination_balance_error"
]

X = df[features]
y = df["isFraud"]

# Create a stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Combine training features and labels
train_data = X_train.copy()
train_data["target"] = y_train.to_numpy()

# Sample up to 500,000 training rows while preserving both classes
sample_size = min(500_000, len(train_data))

train_data = train_data.groupby(
    "target", group_keys=False
).sample(
    frac=sample_size / len(train_data),
    random_state=42
)

X_train_small = train_data[features]
y_train_small = train_data["target"]

print("Training rows:", len(X_train_small))
print("Fraud cases in training sample:", int(y_train_small.sum()))

# Preprocessing
preprocessor = ColumnTransformer([
    (
        "type",
        OneHotEncoder(handle_unknown="ignore"),
        ["type"]
    ),
    (
        "numeric",
        "passthrough",
        [col for col in features if col != "type"]
    )
])

# Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    ))
])

print("\nTraining Random Forest...")
model.fit(X_train_small, y_train_small)

print("\nEvaluating on the held-out test set...")

predictions = model.predict(X_test)
scores = model.predict_proba(X_test)[:, 1]

print("\n===== RANDOM FOREST REPORT =====")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))

print(
    "Average precision:",
    round(average_precision_score(y_test, scores), 4)
)

print("\nRandom Forest evaluation completed!")
