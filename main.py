
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# STEP 1: Load dataset
df = pd.read_csv("data/paysim.csv")

print("Dataset loaded:", df.shape)

# STEP 2: Create graphs folder
os.makedirs("graphs", exist_ok=True)

sns.set_theme(style="whitegrid")

# GRAPH 1: Fraud vs Legitimate Transactions
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="isFraud")

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("0 = Legitimate, 1 = Fraud")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig("graphs/fraud_distribution.png")
plt.close()

# GRAPH 2: Fraud rate by transaction type
fraud_rate = df.groupby("type")["isFraud"].mean() * 100
fraud_rate = fraud_rate.sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=fraud_rate.index, y=fraud_rate.values)

plt.title("Fraud Rate by Transaction Type")
plt.xlabel("Transaction Type")
plt.ylabel("Fraud Rate (%)")
plt.tight_layout()
plt.savefig("graphs/fraud_by_type.png")
plt.close()

# GRAPH 3: Transaction amount by fraud status
sample_df = pd.concat([
    df[df["isFraud"] == 0].sample(n=5000, random_state=42),
    df[df["isFraud"] == 1].sample(n=5000, random_state=42)
])

plt.figure(figsize=(9, 5))
sns.boxplot(data=sample_df, x="isFraud", y="amount")

plt.title("Transaction Amount by Fraud Status")
plt.xlabel("0 = Legitimate, 1 = Fraud")
plt.ylabel("Transaction Amount")
plt.tight_layout()
plt.savefig("graphs/amount_by_fraud.png")
plt.close()

print("\nSUCCESS! All 3 graphs have been saved.")
print("Check the graphs folder.")



# STEP 7A: Statistical Analysis of Balance Changes

print("\n===== BALANCE CHANGE ANALYSIS =====")

# Calculate changes in the origin account balance
df["origin_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# Calculate changes in the destination account balance
df["destination_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

# Compare average values for legitimate and fraudulent transactions
balance_analysis = df.groupby("isFraud").agg(
    transaction_count=("isFraud", "size"),
    average_amount=("amount", "mean"),
    average_origin_balance_change=("origin_balance_change", "mean"),
    average_destination_balance_change=(
        "destination_balance_change", "mean"
    )
)

print(balance_analysis.round(2))

# Save results
balance_analysis.to_csv("balance_analysis.csv")

print("\nBalance analysis saved to balance_analysis.csv")




# STEP 7B: Feature Engineering

print("\n===== FEATURE ENGINEERING =====")

# 1. Sender balance change
df["origin_balance_change"] = (
    df["oldbalanceOrg"] - df["newbalanceOrig"]
)

# 2. Receiver balance change
df["destination_balance_change"] = (
    df["newbalanceDest"] - df["oldbalanceDest"]
)

# 3. Difference between amount sent and sender balance change
df["origin_balance_error"] = (
    df["amount"] - df["origin_balance_change"]
).abs()

# 4. Difference between amount sent and receiver balance change
df["destination_balance_error"] = (
    df["amount"] - df["destination_balance_change"]
).abs()

# 5. Is this a large transaction?
df["large_transaction"] = (
    df["amount"] >= 200000
).astype(int)

# 6. Is the transaction a transfer or cash-out?
df["high_risk_type"] = (
    df["type"].isin(["TRANSFER", "CASH_OUT"])
).astype(int)

# Display new features
new_features = [
    "origin_balance_change",
    "destination_balance_change",
    "origin_balance_error",
    "destination_balance_error",
    "large_transaction",
    "high_risk_type"
]

print(df[new_features].head())

# Save the engineered dataset
df[new_features + ["isFraud"]].to_csv(
    "engineered_features.csv",
    index=False
)

print("\nFeature engineering completed!")
print("Saved to engineered_features.csv")
