# ===============================
# FRAUD DETECTION API (FINAL)
# ===============================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib

# -------------------------------
# 1. LOAD MODELS
# -------------------------------
print("Loading models...")

model = joblib.load("fraud_model.pkl")
anomaly_model = joblib.load("anomaly_model.pkl")
model_columns = joblib.load("model_columns.pkl")

print("Models loaded successfully!")

# -------------------------------
# 2. CREATE APP
# -------------------------------
app = FastAPI()

# -------------------------------
# 3. ENABLE CORS (FIX ERROR)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all for project
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 4. HOME ROUTE
# -------------------------------
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running 🚀"}

# -------------------------------
# 5. PREDICT ROUTE
# -------------------------------
@app.post("/predict")
def predict(data: dict):

    # Convert input to DataFrame
    df = pd.DataFrame([data])

    # ---------------------------
    # PREPROCESSING
    # ---------------------------

    # Handle timestamp if provided
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df["hour"] = df["timestamp"].dt.hour
        df["day"] = df["timestamp"].dt.dayofweek
        df = df.drop(columns=["timestamp"])

    # Fill missing values
    df = df.fillna(0)

    # One-hot encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    # ---------------------------
    # MODEL PREDICTIONS
    # ---------------------------

    fraud_prob = model.predict_proba(df)[0][1]
    anomaly_score = anomaly_model.decision_function(df)[0]

    # ---------------------------
    # HYBRID RISK LOGIC
    # ---------------------------

    amount = data.get("amount", 0)
    deviation = data.get("spending_deviation_score", 0)
    velocity = data.get("velocity_score", 0)
    geo = data.get("geo_anomaly_score", 0)

    # HIGH risk
    if (amount > 100000 and (deviation > 0.8 or velocity > 0.8 or geo > 0.8)):
        risk = "HIGH"

    # MEDIUM risk
    elif (deviation > 0.5 or velocity > 0.5 or geo > 0.5):
        risk = "MEDIUM"

    # fallback to ML model
    elif fraud_prob > 0.2:
        risk = "MEDIUM"

    # LOW risk
    else:
        risk = "LOW"

    # ---------------------------
    # RESPONSE
    # ---------------------------
    return {
        "fraud_probability": float(fraud_prob),
        "anomaly_score": float(anomaly_score),
        "risk_level": risk
    }