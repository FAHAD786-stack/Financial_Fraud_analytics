
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

print("Loading dataset...")
df = pd.read_csv("data/paysim.csv")

# Features intended to be available before the transaction
features = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest"
]

# Chronological 60% / 20% / 20% split
train_end = df["step"].quantile(0.60)
validation_end = df["step"].quantile(0.80)

train_df = df[df["step"] <= train_end].copy()
val_df = df[
    (df["step"] > train_end) &
    (df["step"] <= validation_end)
].copy()
test_df = df[df["step"] > validation_end].copy()

print("\n===== DATA SPLIT =====")
print("Training end step:", train_end)
print("Validation end step:", validation_end)
print("Training rows:", len(train_df))
print("Validation rows:", len(val_df))
print("Test rows:", len(test_df))
print("Training fraud cases:", int(train_df["isFraud"].sum()))
print("Validation fraud cases:", int(val_df["isFraud"].sum()))
print("Test fraud cases:", int(test_df["isFraud"].sum()))

if (
    train_df["isFraud"].nunique() < 2
    or val_df["isFraud"].nunique() < 2
    or test_df["isFraud"].nunique() < 2
):
    raise ValueError(
        "A data split contains only one class. "
        "Check the step boundaries before continuing."
    )

# Limit training size to manage memory and runtime.
# Keep the original class proportions.
if len(train_df) > 500_000:
    train_df = train_df.groupby(
        "isFraud", group_keys=False
    ).sample(
        frac=500_000 / len(train_df),
        random_state=42
    )

X_train = train_df[features]
y_train = train_df["isFraud"]

X_val = val_df[features]
y_val = val_df["isFraud"]

X_test = test_df[features]
y_test = test_df["isFraud"]

# Preprocessing and model
preprocessor = ColumnTransformer([
    (
        "type",
        OneHotEncoder(handle_unknown="ignore"),
        ["type"]
    ),
    (
        "numeric",
        "passthrough",
        ["step", "amount", "oldbalanceOrg", "oldbalanceDest"]
    )
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
    )
])

print("\nTraining model...")
model.fit(X_train, y_train)

# Select the threshold using validation data only
print("\nScoring validation data...")
val_scores = model.predict_proba(X_val)[:, 1]

target_recall = 0.80
thresholds = np.arange(0.01, 0.501, 0.01)
results = []

for threshold in thresholds:
    val_predictions = (val_scores >= threshold).astype(int)

    precision = precision_score(
        y_val, val_predictions, zero_division=0
    )
    recall = recall_score(
        y_val, val_predictions, zero_division=0
    )
    f1 = f1_score(
        y_val, val_predictions, zero_division=0
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

results_df = pd.DataFrame(results)

eligible = results_df[
    results_df["recall"] >= target_recall
]

if not eligible.empty:
    # Highest threshold that still meets target recall
    selected = eligible.sort_values(
        "threshold", ascending=False
    ).iloc[0]

    print(
        f"\nSelected threshold: {selected['threshold']:.2f}"
    )
    print(
        f"Reason: validation recall is at least "
        f"{target_recall:.0%}."
    )
else:
    # Fallback if no tested threshold reaches target recall
    selected = results_df.sort_values(
        ["recall", "precision"],
        ascending=[False, False]
    ).iloc[0]

    print("\nWARNING: Target recall was not reached.")
    print("Using the threshold with the highest validation recall.")

best_threshold = float(selected["threshold"])

print("\n===== VALIDATION THRESHOLD COMPARISON =====")
print(
    results_df[
        results_df["threshold"].isin(
            [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
        )
    ].round(4).to_string(index=False)
)

print("\nAverage precision on validation:",
      round(average_precision_score(y_val, val_scores), 4))

# Evaluate once on the untouched final test period
print("\nEvaluating on final test data...")
test_scores = model.predict_proba(X_test)[:, 1]
test_predictions = (test_scores >= best_threshold).astype(int)

print("\n===== FINAL TEST RESULTS =====")
print("Selected threshold:", round(best_threshold, 2))
print(
    "Average precision:",
    round(average_precision_score(y_test, test_scores), 4)
)
print(
    "Fraud precision:",
    round(precision_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "Fraud recall:",
    round(recall_score(y_test, test_predictions, zero_division=0), 4)
)
print(
    "Fraud F1-score:",
    round(f1_score(y_test, test_predictions, zero_division=0), 4)
)
print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nThreshold tuning complete.")

import joblib
import json

# Save the trained pipeline
joblib.dump(model, "fraud_model.joblib")

# Save the threshold selected using validation data
with open("fraud_threshold.json", "w") as file:
    json.dump(
        {
            "threshold": best_threshold,
            "target_recall": target_recall,
            "features": features
        },
        file,
        indent=4
    )

print("\nModel saved as fraud_model.joblib")
print("Threshold saved as fraud_threshold.json")
