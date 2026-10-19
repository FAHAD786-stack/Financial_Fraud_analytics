
import json
import joblib
import pandas as pd

print("Loading saved model...")
model = joblib.load("fraud_model.joblib")

with open("fraud_threshold.json", "r") as file:
    settings = json.load(file)

features = settings["features"]

print("Loading PaySim dataset...")
df = pd.read_csv("data/paysim.csv")

# Use later transactions, as in the final test period
test_df = df[df["step"] > 355].copy()

# Select actual fraud and legitimate transactions
fraud_rows = test_df[test_df["isFraud"] == 1].head(10)
legit_rows = test_df[test_df["isFraud"] == 0].sample(
    n=10, random_state=42
)

examples = pd.concat([fraud_rows, legit_rows])

X = examples[features]
probabilities = model.predict_proba(X)[:, 1]

results = pd.DataFrame({
    "actual_label": examples["isFraud"].values,
    "transaction_type": examples["type"].values,
    "amount": examples["amount"].values,
    "oldbalanceOrg": examples["oldbalanceOrg"].values,
    "oldbalanceDest": examples["oldbalanceDest"].values,
    "fraud_probability": probabilities
})

results["predicted_label"] = (
    results["fraud_probability"] >= settings["threshold"]
).astype(int)

print("\n===== REAL TRANSACTION CHECK =====")
print(results.to_string(index=False))

print("\nActual fraud cases:", int(results["actual_label"].sum()))
print("Fraud cases flagged:", int(results["predicted_label"].sum()))
print("Threshold:", settings["threshold"])
