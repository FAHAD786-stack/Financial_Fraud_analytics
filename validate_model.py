
import pandas as pd

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

# Chronological split: earlier steps for training,
# later steps for testing
cutoff = df["step"].quantile(0.80)

train_df = df[df["step"] <= cutoff].copy()
test_df = df[df["step"] > cutoff].copy()

print("Training cutoff step:", cutoff)
print("Training rows:", len(train_df))
print("Test rows:", len(test_df))
print("Fraud cases in test:", int(test_df["isFraud"].sum()))

# Sample training rows to limit memory usage
if len(train_df) > 500_000:
    train_df = train_df.groupby(
        "isFraud", group_keys=False
    ).sample(
        frac=500_000 / len(train_df),
        random_state=42
    )

X_train = train_df[features]
y_train = train_df["isFraud"]

X_test = test_df[features]
y_test = test_df["isFraud"]

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

print("\nTraining on earlier transactions...")
model.fit(X_train, y_train)

print("\nTesting on later transactions...")
predictions = model.predict(X_test)
scores = model.predict_proba(X_test)[:, 1]

print("\n===== CHRONOLOGICAL TEST RESULTS =====")
print(classification_report(
    y_test, predictions, zero_division=0
))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print(
    "Average precision:",
    round(average_precision_score(y_test, scores), 4)
)
