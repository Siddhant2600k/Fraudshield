# ===============================
# FRAUD DETECTION MODEL TRAINING (FIXED)
# ===============================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report

# -------------------------------
# 1. LOAD DATA
# -------------------------------
print("Loading dataset...")

df = pd.read_csv("financial_fraud_detection_dataset.csv")

# Use sample for faster training (IMPORTANT)
df = df.sample(n=200000, random_state=42)

print("Data shape:", df.shape)

# -------------------------------
# 2. BASIC CLEANING (FIXED)
# -------------------------------
print("Cleaning data...")

# Drop high-cardinality + unnecessary columns
drop_cols = [
    "transaction_id",
    "ip_address",
    "device_hash",
    "sender_account",      # 🔥 prevent memory explosion
    "receiver_account",    # 🔥 prevent memory explosion
    "location"             # 🔥 prevent memory explosion
]

df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# Convert timestamp → useful features
df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.dayofweek

df = df.drop(columns=["timestamp"])

# Fill missing values
df = df.fillna(0)

# -------------------------------
# 3. ENCODE CATEGORICAL FEATURES
# -------------------------------
print("Encoding categorical variables...")

df = pd.get_dummies(df, drop_first=True)

print("Total features after encoding:", df.shape[1])

# -------------------------------
# 4. SPLIT DATA
# -------------------------------
print("Splitting data...")

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# -------------------------------
# 5. TRAIN FRAUD MODEL
# -------------------------------
print("Training RandomForest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

model.fit(X_train, y_train)

# -------------------------------
# 6. EVALUATE MODEL
# -------------------------------
print("\nEvaluating model...\n")

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# -------------------------------
# 7. TRAIN ANOMALY MODEL (OPTIONAL)
# -------------------------------
print("\nTraining anomaly detection model...")

anomaly_model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

anomaly_model.fit(X_train)

# -------------------------------
# 8. SAVE MODELS
# -------------------------------
print("\nSaving models...")

joblib.dump(model, "fraud_model.pkl")
joblib.dump(anomaly_model, "anomaly_model.pkl")

# Save feature columns (VERY IMPORTANT for API later)
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("\n✅ Training complete!")