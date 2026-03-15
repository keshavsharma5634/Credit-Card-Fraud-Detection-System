# Credit Card Fraud Detection System

An end-to-end Machine Learning project that detects fraudulent credit card transactions
using XGBoost, Optuna hyperparameter tuning, and MLOps monitoring.

---

## Project Overview

Banks and fintech companies need systems that automatically detect fraud in milliseconds
without blocking genuine customers. This project builds a complete fraud detection
pipeline from raw data to a served ML model with drift monitoring.

- PR-AUC: 0.97+ on imbalanced data
- Handles class imbalance (only 4% fraud)
- Cost-sensitive threshold to minimize business loss
- MLOps monitoring for production readiness

---

## Tech Stack

- Machine Learning: scikit-learn, XGBoost, Optuna
- Data: pandas, numpy
- API Serving: FastAPI, uvicorn
- MLOps: PSI drift detection, latency monitoring

---

## Project Structure

```
fraud_project/
│
├── 01_Data_Generation_and_Ingestion.ipynb
├── 02_EDA_and_Feature_Engineering.ipynb
├── 03_Baseline_Model_Training.ipynb
├── 04_XGBoost_Tuning_and_Threshold.ipynb
├── 05_Evaluation_and_Explainability.ipynb
├── 06_MLOps_Monitoring.ipynb
│
├── data/
│   └── transactions.csv
│
└── models/
    └── fraud_xgb.joblib
```

---

## Notebooks Summary

| Notebook | What it does |
|----------|-------------|
| 01 - Data Generation | Creates 50,000 synthetic PII-safe transactions |
| 02 - EDA and Features | Finds fraud patterns, builds velocity features |
| 03 - Baseline Models | Trains Logistic Regression and Random Forest |
| 04 - XGBoost Tuning | Tunes XGBoost with Optuna, selects best threshold |
| 05 - Evaluation | PR/ROC curves, feature importance analysis |
| 06 - MLOps Monitoring | PSI drift detection, latency SLOs, PII audit |

---

## Key Features Engineered

- velocity_ratio: spending speed vs 24h average (top fraud signal)
- log_amount: log transform to reduce amount skewness
- avg_tx_amt_24h: normal spending behavior of card
- high_velocity_flag: more than 3 transactions in 1 hour

---

## How to Run

```bash
# Step 1 - Install dependencies
pip install jupyterlab pandas numpy scikit-learn xgboost optuna joblib matplotlib

# Step 2 - Launch JupyterLab
jupyter lab

# Step 3 - Run notebooks in order from 01 to 06
```

---

## Results

| Metric | Score |
|--------|-------|
| PR-AUC | 0.97+ |
| ROC-AUC | 0.99+ |
| Latency p95 | under 150ms |
| PII in model | None (clean) |

---

## Author

Keshav — BCA Student, AI and ML Specialization
