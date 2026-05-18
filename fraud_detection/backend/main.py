# ================================================
# FRAUD DETECTION - FastAPI BACKEND (FINAL VERSION)
# ================================================
# Run:
# uvicorn main:app --reload --port 8000
# ================================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from typing import Optional

import pandas as pd
import numpy as np
import joblib
import json
import os
import uuid

from datetime import datetime

# ------------------------------------------------
# BASE PATHS
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "history.json"
)

# ------------------------------------------------
# LOAD MODELS
# ------------------------------------------------
print("Loading models...")

try:

    model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "fraud_pipeline.pkl"
        )
    )

    preprocessor = joblib.load(
        os.path.join(
            MODELS_DIR,
            "preprocessor.pkl"
        )
    )

    anomaly_model = joblib.load(
        os.path.join(
            MODELS_DIR,
            "anomaly_model.pkl"
        )
    )

    print("✅ fraud_pipeline.pkl loaded")
    print("✅ preprocessor.pkl loaded")
    print("✅ anomaly_model.pkl loaded")

except Exception as e:

    print(f"❌ Error loading models: {e}")
    raise

# ------------------------------------------------
# LOAD METRICS
# ------------------------------------------------
try:

    with open(
        os.path.join(
            MODELS_DIR,
            "metrics.json"
        )
    ) as f:

        MODEL_METRICS = json.load(f)

    print("✅ metrics.json loaded")

except FileNotFoundError:

    MODEL_METRICS = {
        "note": "Run train.py first"
    }

# ------------------------------------------------
# HISTORY STORAGE
# ------------------------------------------------
def load_history():

    if os.path.exists(HISTORY_FILE):

        with open(HISTORY_FILE) as f:
            return json.load(f)

    return []


def save_history(history):

    with open(HISTORY_FILE, "w") as f:

        json.dump(
            history[-500:],
            f,
            indent=2
        )

# ------------------------------------------------
# FASTAPI APP
# ------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description="AI-powered fraud detection platform",
    version="5.0.0"
)

# ------------------------------------------------
# CORS
# ------------------------------------------------
app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# STATIC FRONTEND
# ------------------------------------------------
FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "..",
    "frontend"
)

if os.path.exists(FRONTEND_DIR):

    app.mount(
        "/static",
        StaticFiles(directory=FRONTEND_DIR),
        name="static"
    )

# ------------------------------------------------
# INPUT SCHEMA
# ------------------------------------------------
class TransactionInput(BaseModel):

    amount: float = Field(
        ...,
        gt=0,
        description="Transaction amount"
    )

    transaction_type: Optional[str] = "purchase"

    merchant_category: Optional[str] = "general"

    device_used: Optional[str] = "mobile"

    payment_channel: Optional[str] = "card"

    time_since_last_transaction: Optional[float] = Field(
        0,
        ge=0
    )

    spending_deviation_score: Optional[float] = Field(
        0.0,
        ge=-5.0,
        le=5.0
    )

    velocity_score: Optional[float] = Field(
        0,
        ge=0
    )

    geo_anomaly_score: Optional[float] = Field(
        0.0,
        ge=0.0,
        le=1.0
    )

    hour: Optional[int] = Field(
        None,
        ge=0,
        le=23
    )

    day_of_week: Optional[int] = Field(
        None,
        ge=0,
        le=6
    )

    @validator("amount")
    def validate_amount(cls, v):

        if v > 10_000_000:

            raise ValueError(
                "Amount exceeds maximum allowed value"
            )

        return v

# ------------------------------------------------
# PREPROCESS INPUT
# ------------------------------------------------
def preprocess_input(data: dict) -> pd.DataFrame:

    df = pd.DataFrame([data])

    now = datetime.now()

    # --------------------------------------------
    # AUTO FILL TIME
    # --------------------------------------------
    if df["hour"].isna().any():
        df["hour"] = now.hour

    if df["day_of_week"].isna().any():
        df["day_of_week"] = now.weekday()

    # --------------------------------------------
    # MONTH
    # --------------------------------------------
    df["month"] = now.month

    # --------------------------------------------
    # FEATURE ENGINEERING
    # --------------------------------------------

    # Log amount
    df["amount_log"] = np.log1p(df["amount"])

    # Night transaction
    df["is_night_transaction"] = (
        (df["hour"] < 6) |
        (df["hour"] > 22)
    ).astype(int)

    # Weekend
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Large amount
    df["large_amount_flag"] = (
        df["amount"] > 50000
    ).astype(int)

    # High velocity
    df["high_velocity_flag"] = (
        df["velocity_score"] > 10
    ).astype(int)

    # High geo anomaly
    df["high_geo_flag"] = (
        df["geo_anomaly_score"] > 0.7
    ).astype(int)

    # Extreme spending deviation
    df["extreme_deviation_flag"] = (
        abs(df["spending_deviation_score"]) > 2
    ).astype(int)

    # Behavioral risk score
    df["behavior_risk_score"] = (
        df["velocity_score"] * 0.4 +
        df["geo_anomaly_score"] * 0.3 +
        abs(df["spending_deviation_score"]) * 0.3
    )

    # Amount × velocity
    df["amount_velocity_interaction"] = (
        df["amount_log"] *
        df["velocity_score"]
    )

    # Geo × deviation
    df["geo_deviation_interaction"] = (
        df["geo_anomaly_score"] *
        abs(df["spending_deviation_score"])
    )

    return df

# ------------------------------------------------
# RISK ENGINE
# ------------------------------------------------
def compute_risk(
    fraud_prob: float,
    anomaly_score: float,
    data: dict
):

    amount = data.get("amount", 0)

    deviation = abs(
        data.get(
            "spending_deviation_score",
            0
        )
    )

    velocity = data.get(
        "velocity_score",
        0
    )

    geo = data.get(
        "geo_anomaly_score",
        0
    )

    hour = data.get(
        "hour",
        12
    )

    # --------------------------------------------
    # COMPONENT SCORES
    # --------------------------------------------
    ml_score = fraud_prob

    anomaly_norm = max(
        0,
        min(
            1,
            (-anomaly_score + 0.5)
        )
    )

    off_hours = (
        1.0
        if (hour < 6 or hour > 22)
        else 0.0
    )

    # --------------------------------------------
    # COMPOSITE SCORE
    # --------------------------------------------
    composite = (
        0.45 * ml_score +
        0.20 * anomaly_norm +
        0.15 * min(velocity / 20.0, 1.0) +
        0.10 * geo +
        0.10 * off_hours
    )

    # --------------------------------------------
    # RISK LEVELS
    # --------------------------------------------
    if (
        amount > 100_000 and
        (
            deviation > 2 or
            velocity > 15 or
            geo > 0.8
        )
    ):
        level = "CRITICAL"

    elif composite >= 0.65:
        level = "HIGH"

    elif composite >= 0.35:
        level = "MEDIUM"

    else:
        level = "LOW"

    # --------------------------------------------
    # FLAGS
    # --------------------------------------------
    flags = []

    if amount > 50_000:
        flags.append(
            "Large transaction amount"
        )

    if deviation > 2:
        flags.append(
            "Extreme spending deviation"
        )

    if velocity > 10:
        flags.append(
            "High transaction velocity"
        )

    if geo > 0.7:
        flags.append(
            "Geographic anomaly detected"
        )

    if off_hours:
        flags.append(
            "Off-hours transaction"
        )

    if anomaly_norm > 0.6:
        flags.append(
            "Anomaly detector flagged transaction"
        )

    return {
        "level": level,
        "composite_score": round(composite, 4),
        "flags": flags,
    }

# ------------------------------------------------
# ROOT
# ------------------------------------------------
@app.get("/")
def root():

    return {
        "status": "running",
        "api": "Fraud Detection API v5"
    }

# ------------------------------------------------
# HEALTH
# ------------------------------------------------
@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": True,
        "timestamp": datetime.now().isoformat()
    }

# ------------------------------------------------
# PREDICT
# ------------------------------------------------
@app.post("/predict")
def predict(tx: TransactionInput):

    data = tx.dict()

    try:

        # PREPROCESS INPUT
        df = preprocess_input(data)

        # MODEL PREDICTION
        fraud_prob = float(
            model.predict_proba(df)[0][1]
        )

        prediction = int(
            fraud_prob >= 0.22
        )

        # PREPROCESS FOR ANOMALY MODEL
        processed_df = preprocessor.transform(df)

        anomaly_score = float(
            anomaly_model.decision_function(
                processed_df
            )[0]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {e}"
        )

    # RISK ENGINE
    risk = compute_risk(
        fraud_prob,
        anomaly_score,
        data
    )

    # RESULT
    result = {

        "transaction_id":
            f"TXN-{uuid.uuid4().hex[:8].upper()}",

        "timestamp":
            datetime.now().isoformat(),

        "prediction":
            "FRAUD"
            if prediction == 1
            else "LEGITIMATE",

        "fraud_probability":
            round(fraud_prob, 4),

        "anomaly_score":
            round(anomaly_score, 4),

        "risk_level":
            risk["level"],

        "composite_score":
            risk["composite_score"],

        "flags":
            risk["flags"],

        "input":
            data
    }

    # SAVE HISTORY
    history = load_history()

    history.append(result)

    save_history(history)

    return result

# ------------------------------------------------
# HISTORY
# ------------------------------------------------
@app.get("/history")
def get_history(limit: int = 50):

    history = load_history()

    return {
        "total": len(history),
        "records": history[-limit:][::-1]
    }

# ------------------------------------------------
# CLEAR HISTORY
# ------------------------------------------------
@app.delete("/history")
def clear_history():

    save_history([])

    return {
        "message": "History cleared"
    }

# ------------------------------------------------
# STATS
# ------------------------------------------------
@app.get("/stats")
def get_stats():

    history = load_history()

    if not history:

        return {
            "message": "No transactions yet",
            "total": 0
        }

    levels = [
        r["risk_level"]
        for r in history
    ]

    amounts = [
        r["input"]["amount"]
        for r in history
    ]

    probs = [
        r["fraud_probability"]
        for r in history
    ]

    return {

        "total_transactions":
            len(history),

        "risk_distribution": {

            "LOW":
                levels.count("LOW"),

            "MEDIUM":
                levels.count("MEDIUM"),

            "HIGH":
                levels.count("HIGH"),

            "CRITICAL":
                levels.count("CRITICAL"),
        },

        "avg_fraud_probability":
            round(np.mean(probs), 4),

        "avg_amount":
            round(np.mean(amounts), 2),

        "max_amount":
            round(max(amounts), 2),

        "high_risk_count":
            levels.count("HIGH") +
            levels.count("CRITICAL"),

        "recent_flagged": [

            r for r in history[-10:]

            if r["risk_level"] in (
                "HIGH",
                "CRITICAL"
            )

        ][-5:]
    }

# ------------------------------------------------
# METRICS
# ------------------------------------------------
@app.get("/metrics")
def get_metrics():

    return MODEL_METRICS