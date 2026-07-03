"""
British Airways — Booking Completion Prediction Model
======================================================
Trains XGBoost classifier to predict whether a customer
will complete their flight booking.

Business context:
    Only 15% of bookings are completed (85% drop off).
    Identifying likely completers helps BA:
    - Target retention campaigns at high-risk drop-offs
    - Optimize pricing for likely completers
    - Prioritize customer service interventions

Optimization target: RECALL for completed bookings
    A missed completer = lost revenue opportunity
    A false positive = wasted marketing spend (acceptable)

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import psycopg2
import mlflow
import mlflow.xgboost
import logging
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/train_model.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ba_intelligence_db",
    "user":     "postgres",
    "password": "postgres"
}

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "ba_booking_prediction"
DECISION_THRESHOLD  = 0.35
RANDOM_STATE        = 42

# ── Load from PostgreSQL ──────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    logger.info("Loading booking data from PostgreSQL")
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT num_passengers, sales_channel, trip_type,
               purchase_lead, length_of_stay, flight_hour,
               flight_day_num, wants_extra_baggage, wants_preferred_seat,
               wants_in_flight_meals, flight_duration, booking_complete,
               booking_window_category, route_popularity,
               is_weekend_booking, total_extras, duration_category
        FROM ba_bookings
    """, conn)
    conn.close()
    logger.info(f"Loaded {len(df)} rows")
    return df

# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame):
    logger.info("Engineering features")
    df = df.copy()

    # Encode categorical columns
    cat_cols = ["sales_channel", "trip_type",
                "booking_window_category", "duration_category"]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Interaction features
    df["lead_x_extras"]    = df["purchase_lead"] * df["total_extras"]
    df["popularity_x_lead"] = df["route_popularity"] * df["purchase_lead"]
    df["extras_x_duration"] = df["total_extras"] * df["flight_duration"]

    feature_cols = [c for c in df.columns if c != "booking_complete"]
    X = df[feature_cols]
    y = df["booking_complete"]

    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Class distribution: {dict(y.value_counts())}")
    logger.info(f"Class imbalance ratio: {y.value_counts()[0]/y.value_counts()[1]:.2f}:1")

    return X, y, encoders

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold=DECISION_THRESHOLD):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = (y_proba >= threshold).astype(int)

    metrics = {
        "accuracy":           round(accuracy_score(y_test, y_pred), 4),
        "precision":          round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":             round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":           round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":            round(roc_auc_score(y_test, y_proba), 4),
        "decision_threshold": threshold
    }

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    logger.info(f"\n{'='*50}")
    logger.info("XGBoost Booking Completion Model Performance")
    logger.info(f"{'='*50}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"  True Negatives  (Correct drop-offs):     {tn}")
    logger.info(f"  False Positives (Predicted complete, didn't): {fp}")
    logger.info(f"  False Negatives (MISSED completers):     {fn} ← minimize")
    logger.info(f"  True Positives  (Correct completers):    {tp}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Drop-off', 'Complete'])}")

    return metrics, y_proba

# ── Train ─────────────────────────────────────────────────────────────────────
def train(X, y):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    class_ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

    params = {
        "n_estimators":      300,
        "max_depth":         5,
        "learning_rate":     0.05,
        "subsample":         0.8,
        "colsample_bytree":  0.8,
        "scale_pos_weight":  class_ratio,
        "eval_metric":       "logloss",
        "random_state":      RANDOM_STATE
    }

    with mlflow.start_run(run_name="xgboost_booking_v1"):
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)

        metrics, y_proba = evaluate(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_params({
            "decision_threshold": DECISION_THRESHOLD,
            "train_size":         len(X_train),
            "test_size":          len(X_test),
            "class_ratio":        round(class_ratio, 2)
        })
        mlflow.log_metrics(metrics)
        mlflow.xgboost.log_model(model, artifact_path="model")

        # Feature importance
        importance_df = pd.DataFrame({
            "feature":    X.columns,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        importance_df.to_csv("data/processed/feature_importance.csv", index=False)
        mlflow.log_artifact("data/processed/feature_importance.csv")

        logger.info("\nTop 10 Feature Importance:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"\nMLflow run ID: {run_id}")

    return model, metrics, run_id, X_test, y_test

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== BA Booking Prediction Training Start ===")
    try:
        df              = load_data()
        X, y, encoders = engineer_features(df)
        model, metrics, run_id, X_test, y_test = train(X, y)

        logger.info("\n=== Training Complete ===")
        logger.info(f"  ROC-AUC:  {metrics['roc_auc']}")
        logger.info(f"  Recall:   {metrics['recall']}")
        logger.info(f"  F1:       {metrics['f1_score']}")
        logger.info(f"  Run ID:   {run_id}")
        logger.info(f"\nView: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
