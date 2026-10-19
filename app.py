
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Financial Fraud Intelligence",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# LOAD MODEL, SETTINGS, AND DATA
# ============================================================
@st.cache_resource
def load_model():
    model = joblib.load("fraud_model.joblib")

    with open("fraud_threshold.json", "r") as file:
        settings = json.load(file)

    return model, settings


@st.cache_data
def load_data():
    return pd.read_csv("data/paysim.csv")


model, settings = load_model()
threshold = float(settings["threshold"])
features = settings["features"]


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("🔍 Fraud Intelligence")
st.sidebar.caption("Financial Fraud Detection System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Analyze Transaction",
        "Transaction Testing",
        "Model Evaluation"
    ]
)

st.sidebar.divider()
st.sidebar.write(f"**Decision threshold:** {threshold:.0%}")
st.sidebar.caption("Academic project using simulated PaySim data")


# ============================================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================================
if page == "Dashboard Overview":

    st.title("Financial Fraud Detection & Anomaly Intelligence")
    st.caption("PaySim transaction analytics and machine-learning insights")

    with st.spinner("Loading dataset analytics..."):
        df = load_data()

    total_transactions = len(df)
    fraud_count = int(df["isFraud"].sum())
    legitimate_count = total_transactions - fraud_count
    fraud_rate = fraud_count / total_transactions * 100

    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Transactions", f"{total_transactions:,}")
    c2.metric("Fraud Cases", f"{fraud_count:,}")
    c3.metric("Legitimate Cases", f"{legitimate_count:,}")
    c4.metric("Fraud Rate", f"{fraud_rate:.3f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Fraud vs Legitimate")

        distribution = pd.DataFrame({
            "Transaction Status": ["Legitimate", "Fraud"],
            "Count": [legitimate_count, fraud_count]
        })

        fig = px.pie(
            distribution,
            names="Transaction Status",
            values="Count",
            hole=0.45,
            title="Transaction Class Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Transaction Volume by Type")

        type_summary = (
            df.groupby("type")
            .agg(
                total_transactions=("isFraud", "size"),
                fraud_cases=("isFraud", "sum")
            )
            .reset_index()
        )

        fig = px.bar(
            type_summary,
            x="type",
            y="total_transactions",
            labels={
                "type": "Transaction Type",
                "total_transactions": "Transaction Count"
            },
            title="Transaction Volume by Type"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Cases by Transaction Type")

    fig = px.bar(
        type_summary,
        x="type",
        y="fraud_cases",
        color="type",
        labels={
            "type": "Transaction Type",
            "fraud_cases": "Fraud Cases"
        },
        title="Fraud Count by Transaction Type"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud Rate by Transaction Type")

    type_summary["fraud_rate_percent"] = (
        type_summary["fraud_cases"]
        / type_summary["total_transactions"] * 100
    )

    fig = px.bar(
        type_summary,
        x="type",
        y="fraud_rate_percent",
        labels={
            "type": "Transaction Type",
            "fraud_rate_percent": "Fraud Rate (%)"
        },
        title="Percentage of Transactions That Are Fraud"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Statistics describe the supplied simulated PaySim dataset, "
        "not live financial transactions."
    )


# ============================================================
# PAGE 2: ANALYZE TRANSACTION
# ============================================================
elif page == "Analyze Transaction":

    st.title("Analyze a Transaction")
    st.caption("Machine-learning-based transaction risk assessment")

    st.info(
        f"Decision threshold: {threshold:.2f}. "
        "Scores at or above this value are flagged for review."
    )

    with st.form("transaction_form"):

        col1, col2 = st.columns(2)

        with col1:
            step = st.number_input(
                "Simulation time step",
                min_value=1,
                value=300
            )

            transaction_type = st.selectbox(
                "Transaction type",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
            )

            amount = st.number_input(
                "Transaction amount",
                min_value=0.0,
                value=1000.0,
                step=100.0
            )

        with col2:
            oldbalance_org = st.number_input(
                "Origin balance before transaction",
                min_value=0.0,
                value=5000.0,
                step=100.0
            )

            oldbalance_dest = st.number_input(
                "Destination balance before transaction",
                min_value=0.0,
                value=2000.0,
                step=100.0
            )

        submitted = st.form_submit_button("Analyze Transaction")

    if submitted:

        transaction = pd.DataFrame([{
            "step": step,
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalance_org,
            "oldbalanceDest": oldbalance_dest
        }])

        probabilities = model.predict_proba(transaction)[0]

        legitimate_probability = float(probabilities[0])
        fraud_probability = float(probabilities[1])

        flagged = fraud_probability >= threshold

        st.subheader("Analysis Result")

        c1, c2 = st.columns(2)

        c1.metric(
            "Estimated Fraud Score",
            f"{fraud_probability:.2%}"
        )

        c2.metric(
            "Decision Threshold",
            f"{threshold:.0%}"
        )

        if flagged:
            st.error(
                "HIGH RISK: Flag this transaction for further review."
            )
        else:
            st.success(
                "LOWER MODEL-ESTIMATED RISK: "
                "Not flagged at this threshold."
            )

        st.progress(
            min(max(fraud_probability, 0.0), 1.0)
        )

        with st.expander("View model diagnostic details"):

            st.write("Input sent to the model:")
            st.dataframe(transaction, use_container_width=True)

            st.json({
                "legitimate_class_score": legitimate_probability,
                "fraud_class_score": fraud_probability,
                "decision_threshold": threshold
            })

        st.warning(
            "This is an experimental model score, not a verified "
            "probability of fraud or a real banking decision."
        )


# ============================================================
# PAGE 3: TRANSACTION TESTING
# ============================================================
elif page == "Transaction Testing":

    st.title("Transaction Testing & Model Verification")

    st.caption(
        "Compare model predictions with actual labels in PaySim."
    )

    st.warning(
        "These are historical, labeled simulated transactions, "
        "not live banking data."
    )

    @st.cache_data
    def load_test_sample():
        data = pd.read_csv("data/paysim.csv")

        # Later period used as the held-out test period
        data = data[data["step"] > 355].copy()

        fraud_data = data[data["isFraud"] == 1]
        legitimate_data = data[data["isFraud"] == 0]

        fraud_sample = fraud_data.sample(
            n=min(100, len(fraud_data)),
            random_state=42
        )

        legitimate_sample = legitimate_data.sample(
            n=min(900, len(legitimate_data)),
            random_state=42
        )

        sample = pd.concat([
            fraud_sample,
            legitimate_sample
        ])

        return sample.sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

    with st.spinner("Preparing transaction sample..."):
        sample_df = load_test_sample()

    st.write(
        f"Browsing {len(sample_df):,} sampled transactions "
        "from the later test period."
    )

    selected_index = st.selectbox(
        "Select a transaction",
        options=list(range(len(sample_df))),
        format_func=lambda i: (
            f"Record {i + 1} | "
            f"{sample_df.iloc[i]['type']} | "
            f"Amount: {sample_df.iloc[i]['amount']:,.2f}"
        )
    )

    record = sample_df.iloc[selected_index]

    transaction = pd.DataFrame([{
        feature: record[feature]
        for feature in features
    }])

    probabilities = model.predict_proba(transaction)[0]
    fraud_probability = float(probabilities[1])

    predicted_label = int(
        fraud_probability >= threshold
    )

    actual_label = int(record["isFraud"])

    st.subheader("Transaction Details")

    details = pd.DataFrame([{
        "Transaction Type": record["type"],
        "Amount": record["amount"],
        "Origin Balance Before": record["oldbalanceOrg"],
        "Destination Balance Before": record["oldbalanceDest"],
        "Simulation Step": record["step"]
    }])

    st.dataframe(details, use_container_width=True)

    st.subheader("Prediction Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Estimated Fraud Score",
        f"{fraud_probability:.2%}"
    )

    c2.metric(
        "Decision Threshold",
        f"{threshold:.0%}"
    )

    c3.metric(
        "Predicted Label",
        "Fraud" if predicted_label else "Legitimate"
    )

    st.write(
        "**Actual dataset label:**",
        "Fraud" if actual_label else "Legitimate"
    )

    if predicted_label == actual_label:
        st.success("Correct prediction for this sampled record.")
    else:
        st.error("Incorrect prediction for this sampled record.")

    with st.expander("View model input"):
        st.dataframe(transaction, use_container_width=True)

    st.caption(
        "A single correct prediction does not establish overall model accuracy."
    )


# ============================================================
# PAGE 4: MODEL EVALUATION
# ============================================================
elif page == "Model Evaluation":

    st.title("Model Evaluation & Performance")
    st.caption("Chronological held-out test evaluation")

    st.info(
        "The evaluation uses transactions after simulation step 355 "
        "and the threshold selected on validation data."
    )

    @st.cache_data
    def get_evaluation_results():

        evaluation_model = joblib.load("fraud_model.joblib")

        with open("fraud_threshold.json", "r") as file:
            config = json.load(file)

        data = pd.read_csv("data/paysim.csv")
        test_data = data[data["step"] > 355].copy()

        X = test_data[config["features"]]
        y = test_data["isFraud"]

        scores = evaluation_model.predict_proba(X)[:, 1]

        predictions = (
            scores >= float(config["threshold"])
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions,
            labels=[0, 1]
        ).ravel()

        metrics = {
            "Test transactions": len(test_data),
            "Actual fraud cases": int(y.sum()),
            "Threshold": float(config["threshold"]),

            "Precision": precision_score(
                y, predictions, zero_division=0
            ),

            "Recall": recall_score(
                y, predictions, zero_division=0
            ),

            "F1-score": f1_score(
                y, predictions, zero_division=0
            ),

            "Average precision": average_precision_score(
                y, scores
            ),

            "True negatives": int(tn),
            "False positives": int(fp),
            "False negatives": int(fn),
            "True positives": int(tp)
        }

        return metrics

    with st.spinner("Evaluating held-out transactions..."):
        metrics = get_evaluation_results()

    st.subheader("Performance Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Fraud Precision",
        f"{metrics['Precision']:.2%}"
    )

    c2.metric(
        "Fraud Recall",
        f"{metrics['Recall']:.2%}"
    )

    c3.metric(
        "F1-score",
        f"{metrics['F1-score']:.2%}"
    )

    c4.metric(
        "Average Precision",
        f"{metrics['Average precision']:.4f}"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Test Transactions",
        f"{metrics['Test transactions']:,}"
    )

    c2.metric(
        "Actual Fraud Cases",
        f"{metrics['Actual fraud cases']:,}"
    )

    c3.metric(
        "Decision Threshold",
        f"{metrics['Threshold']:.2f}"
    )

    st.subheader("Confusion Matrix")

    matrix = pd.DataFrame(
        [
            [
                metrics["True negatives"],
                metrics["False positives"]
            ],
            [
                metrics["False negatives"],
                metrics["True positives"]
            ]
        ],
        index=[
            "Actual Legitimate",
            "Actual Fraud"
        ],
        columns=[
            "Predicted Legitimate",
            "Predicted Fraud"
        ]
    )

    st.dataframe(matrix, use_container_width=True)

    # Visual confusion matrix
    heatmap_data = pd.DataFrame(
        [
            [
                metrics["True negatives"],
                metrics["False positives"]
            ],
            [
                metrics["False negatives"],
                metrics["True positives"]
            ]
        ],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    fig = px.imshow(
        heatmap_data,
        text_auto=",",
        aspect="auto",
        labels={
            "x": "Predicted Class",
            "y": "Actual Class",
            "color": "Transaction Count"
        },
        title="Confusion Matrix Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Detection Summary")

    left, right = st.columns(2)

    left.metric(
        "Fraud Cases Detected",
        f"{metrics['True positives']:,}"
    )

    left.metric(
        "Fraud Cases Missed",
        f"{metrics['False negatives']:,}"
    )

    right.metric(
        "Legitimate Cases Correctly Identified",
        f"{metrics['True negatives']:,}"
    )

    right.metric(
        "Legitimate Cases Flagged",
        f"{metrics['False positives']:,}"
    )

    st.warning(
        "These are experimental results on simulated PaySim data. "
        "They do not establish production banking performance."
    )

# ============================================================
# END OF APPLICATION
# ============================================================
