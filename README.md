# Financial Fraud Detection & Anomaly Intelligence System

## 1. Project Overview

This project develops a machine-learning-based financial fraud detection system using data analytics and a Random Forest classifier. It analyzes transaction characteristics, estimates fraud risk, and provides an interactive dashboard for exploring transactions and evaluating model performance.

## 2. Objectives

- Analyze financial transaction patterns using exploratory data analysis.
- Identify patterns associated with fraudulent transactions.
- Train and evaluate a Random Forest classification model.
- Investigate the effect of classification thresholds on fraud detection.
- Develop an interactive dashboard for transaction analysis and model evaluation.

## 3. Dataset

**Dataset:** PaySim — Synthetic Financial Dataset for Mobile Money Transactions.

The dataset contains transaction attributes such as transaction type, amount, account balances, simulation time step, and fraud labels.

The project uses simulated data. Results should not be interpreted as proof of performance on real banking transactions.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib
- Matplotlib

## 5. System Features

### Dashboard Overview

Displays transaction statistics, fraud distribution, transaction volume by type, and fraud rates.

### Analyze Transaction

Accepts transaction details and returns a model-generated fraud score with a threshold-based classification.

### Transaction Testing

Allows sampled PaySim transactions to be inspected and compares model predictions with their known dataset labels.

### Model Evaluation

Reports fraud precision, recall, F1-score, average precision, and confusion-matrix counts on a chronological held-out test period.

## 6. Model Evaluation

The current experiment uses a classification threshold of 0.19, selected using validation data.

| Metric                                  | Result |
| --------------------------------------- | -----: |
| Fraud precision                         | 26.01% |
| Fraud recall                            | 80.92% |
| Fraud F1-score                          | 39.37% |
| Average precision                       | 0.7195 |
| Fraud cases detected                    |  3,439 |
| Fraud cases missed                      |    811 |
| Legitimate transactions falsely flagged |  9,782 |

These results are specific to the current model, feature set, threshold, and PaySim test period. The low fraud precision indicates that many flagged transactions are legitimate.

## 7. Project Structure

```text
Financial_Fraud_Analytics/
├── data/
│   └── paysim.csv
├── main.py
├── train_model.py
├── random_forest.py
├── validate_model.py
├── validate_model_realistic.py
├── threshold_tuning.py
├── model_evaluation.py
├── check_model.py
├── app.py
├── fraud_model.joblib
├── fraud_threshold.json
└── README.md
```

## 8. Installation

Install Python and open a terminal in the project directory.

Install the required packages:

```bash
python -m pip install pandas numpy scikit-learn streamlit plotly joblib matplotlib
```

Ensure the PaySim CSV file is available at `data/paysim.csv`.

## 9. Run the Application

Start the Streamlit dashboard:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

The saved model and threshold files must be present in the project directory.

## 10. Limitations

- PaySim is a simulated dataset, not a real banking transaction feed.
- Model scores are estimates and are not verified probabilities of fraud.
- Performance depends on the selected features and classification threshold.
- False positives can create substantial investigation workload.
- Further validation is required before any real-world deployment.

## 11. Future Scope

- Compare Random Forest with Logistic Regression and other classifiers.
- Add anomaly detection methods.
- Improve false-positive handling and threshold selection.
- Add transaction monitoring and alert logging.
- Investigate explainable AI techniques for understanding model predictions.
- Evaluate robustness using additional datasets and validation strategies.

## 12. Conclusion

The project demonstrates a complete experimental workflow for financial fraud analytics, including exploratory data analysis, feature selection, model training, chronological evaluation, threshold tuning, and interactive visualization.
