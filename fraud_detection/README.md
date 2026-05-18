# FraudShield — AI-Powered Financial Fraud Detection System V2

FraudShield is a full-stack AI-powered fraud detection platform designed to detect suspicious financial transactions in real time.

The system combines:

- XGBoost Machine Learning
- Isolation Forest Anomaly Detection
- Rule-Based Hybrid Risk Scoring
- FastAPI Backend
- Interactive Analytics Dashboard

The goal of the system is to analyze transaction behavior patterns and classify transactions into:

- LOW
- MEDIUM
- HIGH
- CRITICAL

risk categories.

---

# System Architecture

```text
Frontend Dashboard
        ↓
FastAPI Backend
        ↓
Input Preprocessing
        ↓
Feature Engineering
        ↓
XGBoost Fraud Model
        ↓
Isolation Forest
        ↓
Hybrid Risk Engine
        ↓
Fraud Prediction + Analytics
```

---

# Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | FastAPI |
| ML Framework | Scikit-learn, XGBoost |
| Anomaly Detection | Isolation Forest |
| Data Processing | Pandas, NumPy |
| Visualization | Chart.js |
| Model Storage | Joblib |
| API Server | Uvicorn |

---

# Project Structure

```text
fraud_detection/
│
├── backend/
│   ├── main.py
│   ├── train.py
│   ├── history.json
│   └── models/
│       ├── fraud_pipeline.pkl
│       ├── anomaly_model.pkl
│       ├── preprocessor.pkl
│       └── metrics.json
│
├── frontend/
│   ├── index.html
│   ├── assets/
│   │   ├── css/
│   │   │   └── style.css
│   │   └── js/
│   │       └── utils.js
│   │
│   └── pages/
│       ├── dashboard.html
│       ├── checker.html
│       ├── history.html
│       ├── analytics.html
│       └── model.html
│
├── requirements.txt
└── README.md
```

---

# Machine Learning Pipeline

## Supervised Learning

FraudShield uses:

```text
XGBoost Classifier
```

for fraud prediction.

Why XGBoost?

- handles tabular data efficiently
- better fraud recall
- handles imbalance better than RandomForest
- strong performance on structured financial datasets

---

## Handling Imbalanced Data

Fraud transactions were only around:

```text
3.5%
```

of the dataset.

To solve this imbalance problem, the project uses:

```text
SMOTE (Synthetic Minority Oversampling Technique)
```

which generates synthetic fraud samples to help the model learn fraud patterns more effectively.

---

## Anomaly Detection

The system also includes:

```text
Isolation Forest
```

to detect unusual transactions.

Why?

- XGBoost detects known fraud patterns
- Isolation Forest detects unseen anomalies

This creates a:

```text
Hybrid Fraud Detection System
```

combining supervised and unsupervised learning.

---

# Feature Engineering

Feature engineering was one of the most important parts of the project.

The following engineered fraud signals were created:

| Feature | Purpose |
|---|---|
| amount_log | Normalize large transaction amounts |
| is_night_transaction | Detect suspicious late-night activity |
| is_weekend | Detect unusual weekend behavior |
| high_velocity_flag | Detect rapid transaction bursts |
| high_geo_flag | Detect geographic anomalies |
| large_amount_flag | Detect high-value transfers |
| extreme_deviation_flag | Detect unusual spending behavior |
| behavior_risk_score | Combined behavioral risk indicator |
| amount_velocity_interaction | Large rapid transactions |
| geo_deviation_interaction | Combined geographic + behavioral anomaly |

---

# Risk Scoring Engine

The final fraud decision is NOT based only on ML probability.

FraudShield uses a:

```text
Hybrid Risk Engine
```

that combines:

| Component | Weight |
|---|---|
| ML Fraud Probability | 45% |
| Anomaly Score | 20% |
| Velocity Score | 15% |
| Geo Anomaly | 10% |
| Off-Hours Activity | 10% |

---

# Risk Thresholds

| Risk Level | Logic |
|---|---|
| CRITICAL | Amount > ₹1,00,000 AND suspicious behavior |
| HIGH | Composite score ≥ 0.65 |
| MEDIUM | Composite score ≥ 0.35 |
| LOW | Composite score < 0.35 |

---

# Why Accuracy Is Misleading

Fraud datasets are highly imbalanced.

Example:

If 96% of transactions are legitimate, a model predicting everything as legitimate would still achieve:

```text
96% accuracy
```

while detecting:

```text
0 frauds
```

Therefore the project focuses more on:

- Recall
- Precision
- F1 Score
- ROC-AUC

instead of accuracy alone.

---

# Dataset Columns Used

| Column | Usage |
|---|---|
| amount | Core numerical feature |
| transaction_type | One-hot encoded |
| merchant_category | One-hot encoded |
| device_used | One-hot encoded |
| payment_channel | One-hot encoded |
| velocity_score | Fraud behavior signal |
| geo_anomaly_score | Geographic anomaly detection |
| spending_deviation_score | Behavioral anomaly |
| time_since_last_transaction | Temporal pattern |
| timestamp | Converted to hour/day/month |
| is_fraud | Target label |

---

# Dropped Columns

The following columns were removed:

| Column | Reason |
|---|---|
| transaction_id | Identifier only |
| ip_address | High cardinality |
| device_hash | High cardinality |
| sender_account | Identifier |
| receiver_account | Identifier |
| fraud_type | Data leakage risk |

---

# Frontend Features

## Dashboard

- KPI Cards
- Risk Distribution
- Live Transaction Feed
- Recent Flagged Transactions

## Fraud Checker

- Real-time fraud analysis
- Risk probability
- Composite score
- Fraud flags explanation

## History

- Transaction history
- Filters
- CSV export

## Analytics

- Fraud charts
- Distribution graphs
- Scatter plots
- Hourly trends

## Model Info

- Feature importance
- Metrics
- Confusion matrix
- Risk logic explanation

---

# API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/predict` | Analyze transaction |
| GET | `/history` | Transaction history |
| DELETE | `/history` | Clear history |
| GET | `/stats` | Dashboard analytics |
| GET | `/metrics` | ML metrics |

---

# Running The Project

## Step 1 — Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost imbalanced-learn joblib python-multipart
```

---

## Step 2 — Train The Model

```bash
cd backend
python train.py
```

This generates:

```text
fraud_pipeline.pkl
anomaly_model.pkl
preprocessor.pkl
metrics.json
```

inside:

```text
backend/models/
```

---

## Step 3 — Start FastAPI Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

Swagger Docs:

```text
http://127.0.0.1:8000/docs
```

---

## Step 4 — Start Frontend

Open another terminal:

```bash
cd frontend
python -m http.server 5500
```

Frontend URL:

```text
http://localhost:5500
```

---

# Demo Credentials

| Username | Password | Role |
|---|---|---|
| admin | fraud2024 | Admin |
| analyst | analyze123 | Analyst |

---

# Sample Fraud Prediction Request

```json
{
  "amount": 150000,
  "transaction_type": "transfer",
  "merchant_category": "electronics",
  "device_used": "atm",
  "payment_channel": "wire_transfer",
  "time_since_last_transaction": 2,
  "spending_deviation_score": 4.5,
  "velocity_score": 20,
  "geo_anomaly_score": 0.95,
  "hour": 2,
  "day_of_week": 6
}
```

---

# Sample Response

```json
{
  "prediction": "FRAUD",
  "fraud_probability": 0.81,
  "risk_level": "HIGH",
  "composite_score": 0.74,
  "flags": [
    "Large transaction amount",
    "High transaction velocity",
    "Geographic anomaly detected",
    "Off-hours transaction"
  ]
}
```

---

# Major Learning Outcomes

This project demonstrates:

- End-to-end ML pipeline development
- Fraud detection system design
- Feature engineering
- Imbalanced classification handling
- API development with FastAPI
- Frontend-backend integration
- Real-time analytics dashboards
- Hybrid AI systems

---

# Future Improvements

Possible future upgrades:

- SHAP Explainability
- PostgreSQL integration
- JWT Authentication
- Cloud deployment
- Kafka real-time streaming
- Deep learning sequence models
- User behavior profiling

---

# Important Review Concepts

## Precision

Out of predicted frauds:

```text
How many were actually fraud?
```

## Recall

Out of actual frauds:

```text
How many frauds did the model catch?
```

Fraud systems usually prioritize:

```text
High Recall
```

because missing fraud is expensive.

---

# Final Summary

FraudShield is a hybrid AI-powered fraud detection platform combining:

- Machine Learning
- Anomaly Detection
- Risk Rules
- Real-Time APIs
- Interactive Analytics

to simulate a real-world financial fraud monitoring system.