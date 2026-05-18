# ================================================
# FRAUD DETECTION - FINAL PRODUCTION TRAINER
# ================================================

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV

from sklearn.ensemble import IsolationForest

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

# ------------------------------------------------
# PATHS
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "financial_fraud_detection_dataset.csv"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

# Sample for speed
df = df.sample(
    n=min(200_000, len(df)),
    random_state=42
)

print(f"Loaded {len(df):,} rows")
print(f"Columns: {len(df.columns)}")

# ------------------------------------------------
# CLEAN DATA
# ------------------------------------------------
print("Cleaning dataset...")

DROP_COLS = [
    "transaction_id",
    "ip_address",
    "device_hash",
    "sender_account",
    "receiver_account",
    "location",
    "fraud_type"
]

df = df.drop(
    columns=[c for c in DROP_COLS if c in df.columns]
)

# ------------------------------------------------
# FEATURE ENGINEERING
# ------------------------------------------------
print("Engineering features...")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce"
)

# Time Features
df["hour"] = df["timestamp"].dt.hour
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["month"] = df["timestamp"].dt.month

# --------------------------------------------
# ENGINEERED FRAUD FEATURES
# --------------------------------------------

# Log amount
df["amount_log"] = np.log1p(df["amount"])

# Night transactions
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

# Combined behavioral score
df["behavior_risk_score"] = (
    df["velocity_score"] * 0.4 +
    df["geo_anomaly_score"] * 0.3 +
    abs(df["spending_deviation_score"]) * 0.3
)

# Amount × velocity interaction
df["amount_velocity_interaction"] = (
    df["amount_log"] *
    df["velocity_score"]
)

# Geo × deviation interaction
df["geo_deviation_interaction"] = (
    df["geo_anomaly_score"] *
    abs(df["spending_deviation_score"])
)

# Remove timestamp
df = df.drop(columns=["timestamp"])

# ------------------------------------------------
# TARGET
# ------------------------------------------------
TARGET = "is_fraud"

X = df.drop(TARGET, axis=1)
y = df[TARGET].astype(int)

# ------------------------------------------------
# FEATURE TYPES
# ------------------------------------------------
numeric_features = X.select_dtypes(
    include=[np.number]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print(f"Numeric columns: {len(numeric_features)}")
print(f"Categorical columns: {len(categorical_features)}")

# ------------------------------------------------
# PREPROCESSOR
# ------------------------------------------------
numeric_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

categorical_transformer = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="constant",
            fill_value="unknown"
        )
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore"
        )
    )
])

preprocessor = ColumnTransformer([
    (
        "num",
        numeric_transformer,
        numeric_features
    ),

    (
        "cat",
        categorical_transformer,
        categorical_features
    )
])

# ------------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------------
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Fraud rate: {y_train.mean():.2%}")

# ------------------------------------------------
# PREPROCESS
# ------------------------------------------------
print("\nPreprocessing data...")

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)

# ------------------------------------------------
# APPLY SMOTE
# ------------------------------------------------
print("\nApplying SMOTE balancing...")

smote = SMOTE(
    sampling_strategy=0.5,
    random_state=42
)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nBalanced class distribution:")
print(
    pd.Series(y_train_balanced).value_counts()
)

# ------------------------------------------------
# BASE XGBOOST MODEL
# ------------------------------------------------
print("\nTraining XGBoost...")

base_model = XGBClassifier(

    n_estimators=500,

    max_depth=10,

    learning_rate=0.03,

    subsample=0.8,

    colsample_bytree=0.8,

    scale_pos_weight=10,

    gamma=1,

    min_child_weight=3,

    eval_metric="logloss",

    random_state=42,

    n_jobs=-1
)

# ------------------------------------------------
# CALIBRATION
# ------------------------------------------------
model = CalibratedClassifierCV(
    base_model,
    method="sigmoid",
    cv=3
)

model.fit(
    X_train_balanced,
    y_train_balanced
)

# ------------------------------------------------
# EVALUATION
# ------------------------------------------------
print("\nEvaluating model...")

# Probability predictions
y_proba = model.predict_proba(
    X_test_processed
)[:, 1]

# Optimized threshold
THRESHOLD = 0.22

y_pred = (
    y_proba >= THRESHOLD
).astype(int)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

cm = confusion_matrix(
    y_test,
    y_pred
).tolist()

auc = roc_auc_score(
    y_test,
    y_proba
)

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {auc:.4f}")

# ------------------------------------------------
# FEATURE IMPORTANCE
# ------------------------------------------------
print("\nExtracting feature importance...")

# Use underlying XGBoost estimator
xgb_model = model.calibrated_classifiers_[0].estimator

ohe = preprocessor.named_transformers_["cat"] \
    .named_steps["encoder"]

encoded_cat_features = ohe.get_feature_names_out(
    categorical_features
)

all_features = (
    numeric_features +
    list(encoded_cat_features)
)

feature_importance = pd.Series(
    xgb_model.feature_importances_,
    index=all_features
)

top_features = feature_importance \
    .sort_values(ascending=False) \
    .head(15)

print("\nTop Features:")
print(top_features)

# ------------------------------------------------
# CREATE PIPELINE
# ------------------------------------------------
fraud_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# ------------------------------------------------
# TRAIN ANOMALY MODEL
# ------------------------------------------------
print("\nTraining Isolation Forest...")

normal_transactions = X_train[
    y_train == 0
]

X_normal_processed = preprocessor.transform(
    normal_transactions
)

anomaly_model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)

anomaly_model.fit(
    X_normal_processed
)

# ------------------------------------------------
# SAVE MODELS
# ------------------------------------------------
print("\nSaving models...")

joblib.dump(
    fraud_pipeline,
    os.path.join(
        MODELS_DIR,
        "fraud_pipeline.pkl"
    )
)

joblib.dump(
    anomaly_model,
    os.path.join(
        MODELS_DIR,
        "anomaly_model.pkl"
    )
)

joblib.dump(
    preprocessor,
    os.path.join(
        MODELS_DIR,
        "preprocessor.pkl"
    )
)

# ------------------------------------------------
# SAVE METRICS
# ------------------------------------------------
metrics = {

    "accuracy":
        round(report["accuracy"], 4),

    "precision":
        round(
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            4
        ),

    "recall":
        round(
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            4
        ),

    "f1_score":
        round(
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            ),
            4
        ),

    "roc_auc":
        round(auc, 4),

    "threshold":
        THRESHOLD,

    "confusion_matrix":
        cm,

    "top_features": {
        k: round(float(v), 5)
        for k, v in top_features.items()
    },

    "train_size":
        len(X_train),

    "test_size":
        len(X_test),

    "fraud_rate":
        round(float(y_train.mean()), 4)
}

with open(
    os.path.join(
        MODELS_DIR,
        "metrics.json"
    ),
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

# ------------------------------------------------
# DONE
# ------------------------------------------------
print("\n✅ TRAINING COMPLETE")

print("\nSaved files:")

for fname in [
    "fraud_pipeline.pkl",
    "anomaly_model.pkl",
    "preprocessor.pkl",
    "metrics.json"
]:

    path = os.path.join(
        MODELS_DIR,
        fname
    )

    size = os.path.getsize(path) / 1024

    print(f"{fname} ({size:.1f} KB)")