
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

print("Loading model and threshold...")
model = joblib.load("fraud_model.joblib")

with open("fraud_threshold.json", "r") as file:
    settings = json.load(file)

threshold = float(settings["threshold"])
features = settings["features"]

print("Loading test data...")
df = pd.read_csv("data/paysim.csv")

# Match the chronological test period used during validation
test_df = df[df["step"] > 355].copy()

X_test = test_df[features]
y_test = test_df["isFraud"]

print("Scoring test transactions...")
scores = model.predict_proba(X_test)[:, 1]
predictions = (scores >= threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, predictions, labels=[0, 1]
).ravel()

print("\n===== MODEL EVALUATION =====")
print("Threshold:", threshold)
print("Test transactions:", len(test_df))
print("Actual fraud cases:", int(y_test.sum()))

print("\nFraud precision:", round(
    precision_score(y_test, predictions, zero_division=0), 4
))
print("Fraud recall:", round(
    recall_score(y_test, predictions, zero_division=0), 4
))
print("Fraud F1-score:", round(
    f1_score(y_test, predictions, zero_division=0), 4
))
print("Average precision:", round(
    average_precision_score(y_test, scores), 4
))

print("\nConfusion matrix:")
print([[int(tn), int(fp)], [int(fn), int(tp)]])

print("\nTrue negatives:", int(tn))
print("False positives:", int(fp))
print("False negatives:", int(fn))
print("True positives:", int(tp))
