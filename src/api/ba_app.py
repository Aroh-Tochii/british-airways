"""
British Airways — Airline Customer Intelligence API
====================================================
Two prediction endpoints:

    POST /analyze/review-sentiment
        Input:  Raw review text
        Output: Sentiment score, label, subjectivity

    POST /predict/booking-completion
        Input:  Booking features
        Output: Completion probability, risk level, recommendation

    GET  /insights/topic-summary  — Top topics from database
    GET  /insights/sentiment-summary — Sentiment overview from database
    GET  /health — Service health check

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import psycopg2
import logging
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/api.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME     = "ba_booking_prediction"
DECISION_THRESHOLD  = 0.35

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ba_intelligence_db",
    "user":     "postgres",
    "password": "postgres"
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Airline Customer Intelligence API",
    description="""
## British Airways Customer Intelligence Platform

Two AI-powered modules:

### Module A — Review Sentiment Analysis
`POST /analyze/review-sentiment`
Analyzes any customer review text using dual sentiment models (VADER + TextBlob).
Returns sentiment label, scores, and business interpretation.

### Module B — Booking Completion Prediction
`POST /predict/booking-completion`
Predicts whether a customer will complete their flight booking.
Helps BA target retention campaigns at high drop-off risk customers.

### Business Insights
`GET /insights/topic-summary` — What customers complain about most
`GET /insights/sentiment-summary` — Overall sentiment health
    """,
    version="1.0.0"
)

# ── Global model store ────────────────────────────────────────────────────────
models = {}
vader_analyzer = SentimentIntensityAnalyzer()

# ── Load booking model at startup ─────────────────────────────────────────────
@app.on_event("startup")
def load_models():
    global models
    import joblib
    import os
    model_path = "models/booking_model.pkl"
    if os.path.exists(model_path):
        models["booking"] = joblib.load(model_path)
        logger.info(f"Booking model loaded from {model_path}")
    else:
        logger.error(f"Model not found at {model_path}")
    logger.info(f"Models loaded: {list(models.keys())}")

# ══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════════════════════════════════════

class ReviewInput(BaseModel):
    review_text: str = Field(..., min_length=10,
        description="Customer review text to analyze")

    class Config:
        json_schema_extra = {
            "example": {
                "review_text": "The flight was delayed by 3 hours and the staff were unhelpful. Food was terrible."
            }
        }

class SentimentResponse(BaseModel):
    sentiment:          str
    vader_compound:     float
    vader_positive:     float
    vader_negative:     float
    textblob_polarity:  float
    textblob_subjectivity: float
    confidence:         str
    business_interpretation: str
    analyzed_at:        str

class BookingInput(BaseModel):
    num_passengers:         int   = Field(..., ge=1, le=9)
    sales_channel:          str   = Field(..., description="Internet or Mobile")
    trip_type:              str   = Field(..., description="RoundTrip, OneWay, CircleTrip")
    purchase_lead:          int   = Field(..., ge=0, description="Days before flight")
    length_of_stay:         int   = Field(..., ge=0)
    flight_hour:            int   = Field(..., ge=0, le=23)
    flight_day_num:         int   = Field(..., ge=1, le=7, description="1=Mon, 7=Sun")
    wants_extra_baggage:    int   = Field(..., ge=0, le=1)
    wants_preferred_seat:   int   = Field(..., ge=0, le=1)
    wants_in_flight_meals:  int   = Field(..., ge=0, le=1)
    flight_duration:        float = Field(..., ge=0.5)
    route_popularity:       int   = Field(100, ge=1)
    is_weekend_booking:     int   = Field(0, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "num_passengers": 1,
                "sales_channel": "Internet",
                "trip_type": "RoundTrip",
                "purchase_lead": 30,
                "length_of_stay": 7,
                "flight_hour": 10,
                "flight_day_num": 3,
                "wants_extra_baggage": 1,
                "wants_preferred_seat": 0,
                "wants_in_flight_meals": 1,
                "flight_duration": 5.5,
                "route_popularity": 150,
                "is_weekend_booking": 0
            }
        }

class BookingResponse(BaseModel):
    completion_prediction:  int
    completion_probability: float
    drop_off_risk:          str
    recommendation:         str
    key_factors:            list
    decision_threshold:     float
    predicted_at:           str

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def get_business_interpretation(sentiment: str, compound: float) -> str:
    if sentiment == "Positive" and compound > 0.5:
        return "Strong brand advocate — high likelihood of repeat booking"
    elif sentiment == "Positive":
        return "Satisfied customer — likely to consider BA for future travel"
    elif sentiment == "Negative" and compound < -0.5:
        return "High churn risk — immediate service recovery recommended"
    elif sentiment == "Negative":
        return "Dissatisfied customer — follow-up recommended to prevent churn"
    else:
        return "Neutral experience — opportunity to improve and convert to advocate"

def engineer_booking_features(data: dict) -> pd.DataFrame:
    """Apply same feature engineering used during training."""
    # Booking window category
    lead = data["purchase_lead"]
    if lead <= 7:
        window = "Last Minute"
    elif lead <= 30:
        window = "Short Term"
    elif lead <= 90:
        window = "Medium Term"
    else:
        window = "Long Term"

    # Duration category
    dur = data["flight_duration"]
    if dur <= 3:
        dur_cat = "Short"
    elif dur <= 6:
        dur_cat = "Medium"
    elif dur <= 12:
        dur_cat = "Long"
    else:
        dur_cat = "Ultra Long"

    total_extras = (data["wants_extra_baggage"] +
                    data["wants_preferred_seat"] +
                    data["wants_in_flight_meals"])

    # Encode categoricals same way as training
    sales_map    = {"Internet": 0, "Mobile": 1}
    trip_map     = {"CircleTrip": 0, "OneWay": 1, "RoundTrip": 2}
    window_map   = {"Last Minute": 0, "Long Term": 1, "Medium Term": 2, "Short Term": 3}
    dur_map      = {"Long": 0, "Medium": 1, "Short": 2, "Ultra Long": 3}

    features = {
        "num_passengers":         data["num_passengers"],
        "sales_channel":          sales_map.get(data["sales_channel"], 0),
        "trip_type":              trip_map.get(data["trip_type"], 2),
        "purchase_lead":          data["purchase_lead"],
        "length_of_stay":         data["length_of_stay"],
        "flight_hour":            data["flight_hour"],
        "flight_day_num":         data["flight_day_num"],
        "wants_extra_baggage":    data["wants_extra_baggage"],
        "wants_preferred_seat":   data["wants_preferred_seat"],
        "wants_in_flight_meals":  data["wants_in_flight_meals"],
        "flight_duration":        data["flight_duration"],
        "booking_window_category":window_map.get(window, 2),
        "route_popularity":       data["route_popularity"],
        "is_weekend_booking":     data["is_weekend_booking"],
        "total_extras":           total_extras,
        "duration_category":      dur_map.get(dur_cat, 1),
        "lead_x_extras":          data["purchase_lead"] * total_extras,
        "popularity_x_lead":      data["route_popularity"] * data["purchase_lead"],
        "extras_x_duration":      total_extras * data["flight_duration"]
    }

    return pd.DataFrame([features])

def get_booking_factors(data: dict, proba: float) -> list:
    factors = []
    if data["purchase_lead"] > 90:
        factors.append("High drop-off risk: booked 90+ days in advance")
    if data["wants_extra_baggage"] == 0 and data["wants_in_flight_meals"] == 0:
        factors.append("Low engagement: no extras selected")
    if data["sales_channel"] == "Mobile":
        factors.append("Mobile bookings have lower completion rates")
    if data["flight_duration"] > 10:
        factors.append("Long-haul flight: higher consideration period")
    if data["wants_extra_baggage"] == 1:
        factors.append("Extra baggage selected: positive commitment signal")
    if data["purchase_lead"] <= 7:
        factors.append("Last-minute booking: higher completion likelihood")
    return factors if factors else ["No significant risk factors detected"]

def get_booking_recommendation(risk: str) -> str:
    recs = {
        "Low Drop-off Risk":    "Customer likely to complete. Standard follow-up email sufficient.",
        "Medium Drop-off Risk": "Send personalized reminder with incentive within 24 hours.",
        "High Drop-off Risk":   "Immediate intervention needed. Offer discount or flexible rebooking."
    }
    return recs.get(risk, "Monitor customer journey.")

# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "service": "Airline Customer Intelligence API",
        "version": "1.0.0",
        "modules": {
            "nlp":     "Review sentiment analysis",
            "ml":      "Booking completion prediction",
            "insights":"Database-driven business insights"
        }
    }

@app.get("/health")
def health():
    return {
        "status":         "healthy",
        "booking_model":  "booking" in models,
        "sentiment_model": "vader_ready"
    }

@app.post("/analyze/review-sentiment", response_model=SentimentResponse)
def analyze_sentiment(review: ReviewInput):
    """Analyze sentiment of a customer review using VADER + TextBlob."""
    try:
        text = review.review_text

        # VADER
        v = vader_analyzer.polarity_scores(text)
        compound = v["compound"]
        if compound >= 0.05:
            vader_sent = "Positive"
        elif compound <= -0.05:
            vader_sent = "Negative"
        else:
            vader_sent = "Neutral"

        # TextBlob
        blob = TextBlob(text)
        tb_polarity = round(blob.sentiment.polarity, 4)
        tb_subj     = round(blob.sentiment.subjectivity, 4)

        # Agreement
        confidence = "High" if (
            (vader_sent == "Positive" and tb_polarity > 0) or
            (vader_sent == "Negative" and tb_polarity < 0)
        ) else "Medium"

        interpretation = get_business_interpretation(vader_sent, compound)

        logger.info(f"Sentiment analyzed: {vader_sent} (compound={compound:.3f})")

        return SentimentResponse(
            sentiment             = vader_sent,
            vader_compound        = round(compound, 4),
            vader_positive        = round(v["pos"], 4),
            vader_negative        = round(v["neg"], 4),
            textblob_polarity     = tb_polarity,
            textblob_subjectivity = tb_subj,
            confidence            = confidence,
            business_interpretation = interpretation,
            analyzed_at           = datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/booking-completion", response_model=BookingResponse)
def predict_booking(booking: BookingInput):
    """Predict whether a customer will complete their flight booking."""
    if "booking" not in models:
        raise HTTPException(status_code=503, detail="Booking model not loaded")

    try:
        input_dict = booking.dict()
        df = engineer_booking_features(input_dict)

        model = models["booking"]
        proba = model.predict_proba(df)[0][1]
        pred  = int(proba >= DECISION_THRESHOLD)

        if proba < 0.35:
            risk = "High Drop-off Risk"
        elif proba < 0.60:
            risk = "Medium Drop-off Risk"
        else:
            risk = "Low Drop-off Risk"

        factors        = get_booking_factors(input_dict, proba)
        recommendation = get_booking_recommendation(risk)

        logger.info(f"Booking prediction: prob={proba:.4f}, risk={risk}")

        return BookingResponse(
            completion_prediction  = pred,
            completion_probability = round(float(proba), 4),
            drop_off_risk          = risk,
            recommendation         = recommendation,
            key_factors            = factors,
            decision_threshold     = DECISION_THRESHOLD,
            predicted_at           = datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/insights/topic-summary")
def topic_summary():
    """What are customers talking about most — and what sentiment?"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("""
            SELECT topic_label,
                   COUNT(*) AS total_reviews,
                   ROUND(AVG(vader_compound)::NUMERIC, 3) AS avg_sentiment,
                   SUM(CASE WHEN combined_sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative_count,
                   ROUND(
                       SUM(CASE WHEN combined_sentiment = 'Negative' THEN 1 ELSE 0 END)::NUMERIC
                       / COUNT(*) * 100, 1
                   ) AS negative_pct
            FROM ba_reviews
            WHERE topic_label IS NOT NULL
            GROUP BY topic_label
            ORDER BY negative_pct DESC
        """)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return {"topics": [
            {
                "topic":         r[0],
                "total_reviews": r[1],
                "avg_sentiment": float(r[2]) if r[2] else 0,
                "negative_count":r[3],
                "negative_pct":  float(r[4]) if r[4] else 0
            }
            for r in rows
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insights/sentiment-summary")
def sentiment_summary():
    """Overall sentiment health of British Airways reviews."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur  = conn.cursor()
        cur.execute("""
            SELECT
                combined_sentiment,
                COUNT(*) AS count,
                ROUND(AVG(vader_compound)::NUMERIC, 3) AS avg_score,
                ROUND(COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER() * 100, 1) AS percentage
            FROM ba_reviews
            WHERE combined_sentiment IS NOT NULL
            GROUP BY combined_sentiment
            ORDER BY count DESC
        """)
        rows = cur.fetchall()
        cur.execute("SELECT COUNT(*) FROM ba_reviews")
        total = cur.fetchone()[0]
        cur.close()
        conn.close()

        return {
            "total_reviews": total,
            "sentiment_breakdown": [
                {
                    "sentiment":  r[0],
                    "count":      r[1],
                    "avg_score":  float(r[2]) if r[2] else 0,
                    "percentage": float(r[3]) if r[3] else 0
                }
                for r in rows
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ba_app:app", host="0.0.0.0", port=8003, reload=True)
