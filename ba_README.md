# Airline Customer Intelligence & Booking Optimization Platform

A production-grade dual-module analytics and machine learning platform built on British Airways data — combining NLP-powered customer sentiment analysis with predictive booking completion modeling.

## Overview

This platform answers two critical business questions airlines face daily:

1. **What are customers actually complaining about?** — NLP pipeline analyzing 1,920 customer reviews to uncover sentiment patterns and topic clusters
2. **Which customers will complete their booking?** — ML model trained on 50,000 booking records to predict drop-off risk and trigger targeted retention

## Architecture

```
Module A — NLP Engine (Customer Reviews)
Raw Reviews (1,920)
      ↓
Text Cleaning Pipeline (spaCy lemmatization, noise removal)
      ↓
Dual Sentiment Analysis (VADER + TextBlob)
      ↓
Topic Modeling (LDA — 6 topics discovered)
      ↓
PostgreSQL (ba_reviews table)
      ↓
8 EDA Charts + Business Insights

Module B — Booking ML Engine (50,000 Records)
Raw Booking Data
      ↓
ETL Pipeline → PostgreSQL (ba_bookings table)
      ↓
Feature Engineering (6 composite features)
      ↓
XGBoost Classifier → MLflow Tracking
      ↓
FastAPI Prediction Service

Shared Deployment Layer
      ↓
FastAPI (5 endpoints — sentiment + booking + insights)
      ↓
Docker (multi-container: API + PostgreSQL)
      ↓
GitHub Actions CI/CD (5 jobs including NLP logic validation)
```

## Tech Stack

| Layer | Tool |
|---|---|
| Data storage | PostgreSQL |
| NLP | spaCy, VADER, TextBlob, scikit-learn LDA |
| Machine Learning | XGBoost |
| Experiment tracking | MLflow |
| API serving | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| CI/CD | GitHub Actions |

## Project Structure

```
.
├── src/
│   ├── nlp/
│   │   ├── ba_text_cleaner.py    # Text cleaning + lemmatization pipeline
│   │   ├── ba_sentiment.py       # Dual sentiment analysis (VADER + TextBlob)
│   │   └── ba_topic_model.py     # LDA topic modeling + assignment
│   ├── ml/
│   │   ├── ba_etl_bookings.py    # 50K booking records ETL
│   │   └── ba_train_model.py     # XGBoost training + MLflow tracking
│   └── api/
│       └── ba_app.py             # FastAPI service (5 endpoints)
├── data/
│   ├── raw/                      # Original source files
│   └── processed/                # Cleaned outputs
├── models/                       # Saved model artifacts
├── reports/figures/              # 8 EDA charts
├── task 1/                       # Original Forage simulation (NLP)
├── task 2/                       # Original Forage simulation (Booking)
├── ba_Dockerfile
├── ba_docker_compose.yml
├── ba_requirements.txt
└── .github/workflows/ba_ci_cd.yml
```

## Datasets

| Dataset | Records | Description |
|---|---|---|
| Customer Reviews | 1,920 | Scraped from AirlineQuality.com — 90% verified trips |
| Booking Records | 50,000 | Flight bookings with completion status, route, channel, extras |

## Module A — NLP Findings

**Sentiment:** 50% positive, 50% negative — customers are deeply divided

**Topic breakdown and sentiment:**

| Topic | Reviews | Sentiment |
|---|---|---|
| Delays & Punctuality | 384 (20%) | 100% Negative |
| Food & Catering | 192 (10%) | 100% Negative |
| Seat & Comfort | 192 (10%) | 100% Negative |
| Customer Service | 384 (20%) | 50/50 Mixed |
| Flight Experience | 384 (20%) | 100% Positive |
| Baggage & Check-in | 384 (20%) | 100% Positive |

**Key insight:** Asymptomatic failure — delays and food quality generate uniformly negative reviews while ground operations and smooth flights generate uniformly positive ones. BA's biggest reputational risk is operational punctuality.

**VADER vs TextBlob agreement rate: 40%** — airline reviews contain complex mixed-sentiment language that simple models struggle to classify consistently.

## Module B — Booking Model Performance

**Class imbalance:** 85% drop-off, 15% completion — severe imbalance handled via `scale_pos_weight`

| Metric | Score |
|---|---|
| ROC-AUC | 0.7364 |
| Recall | 0.8543 |
| Precision | 0.2164 |
| F1 Score | 0.3454 |
| Decision Threshold | 0.35 |

**Completion rate by booking window:**

| Window | Bookings | Completion Rate |
|---|---|---|
| Last Minute (0-7 days) | 4,813 | 17.3% |
| Short Term (7-30 days) | 12,326 | 16.1% |
| Medium Term (30-90 days) | 17,015 | 14.6% |
| Long Term (90+ days) | 15,846 | 13.8% |

**Key insight:** Last-minute bookers complete at higher rates than long-term planners. Customers who book 90+ days out are browsing, not buying — highest drop-off risk.

## Feature Engineering

6 composite features engineered beyond the raw 14:

| Feature | Logic | Business Rationale |
|---|---|---|
| `booking_window_category` | Lead time bucketed into 4 tiers | Captures planning behavior |
| `route_popularity` | Count of bookings per route | Popular routes attract more committed bookers |
| `is_weekend_booking` | Flight day 6 or 7 | Weekend travel has different completion patterns |
| `total_extras` | Sum of baggage + seat + meals | Higher extras = stronger purchase intent |
| `duration_category` | Flight duration bucketed | Long-haul requires more consideration |
| `lead_x_extras` | lead time × total extras | Interaction: committed early bookers vs browsers |

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/analyze/review-sentiment` | POST | Dual sentiment analysis on any review text |
| `/predict/booking-completion` | POST | Drop-off risk prediction with recommendation |
| `/insights/topic-summary` | GET | Live topic breakdown from database |
| `/insights/sentiment-summary` | GET | Overall sentiment health from database |
| `/health` | GET | Service health check |

### Sentiment Analysis Example

```bash
curl -X POST http://localhost:8003/analyze/review-sentiment \
  -H "Content-Type: application/json" \
  -d '{"review_text": "The chair was uncomfortable and the hosts were rude"}'
```

```json
{
  "sentiment": "Negative",
  "vader_compound": -0.6808,
  "textblob_polarity": -0.4,
  "textblob_subjectivity": 0.8,
  "confidence": "High",
  "business_interpretation": "High churn risk — immediate service recovery recommended"
}
```

### Booking Prediction Example

```bash
curl -X POST http://localhost:8003/predict/booking-completion \
  -H "Content-Type: application/json" \
  -d '{
    "num_passengers": 1,
    "sales_channel": "Internet",
    "trip_type": "RoundTrip",
    "purchase_lead": 120,
    "length_of_stay": 7,
    "flight_hour": 10,
    "flight_day_num": 3,
    "wants_extra_baggage": 0,
    "wants_preferred_seat": 0,
    "wants_in_flight_meals": 0,
    "flight_duration": 5.5,
    "route_popularity": 50,
    "is_weekend_booking": 0
  }'
```

```json
{
  "completion_prediction": 0,
  "completion_probability": 0.2341,
  "drop_off_risk": "High Drop-off Risk",
  "recommendation": "Immediate intervention needed. Offer discount or flexible rebooking.",
  "key_factors": [
    "High drop-off risk: booked 90+ days in advance",
    "Low engagement: no extras selected"
  ]
}
```

## Running Locally

**Start the full stack:**
```bash
docker compose -f ba_docker_compose.yml up -d
```

**Run NLP pipeline (in order):**
```bash
python3 src/nlp/ba_text_cleaner.py
python3 src/nlp/ba_sentiment.py
python3 src/nlp/ba_topic_model.py
python3 src/nlp/ba_eda.py
```

**Run booking ML pipeline:**
```bash
python3 src/ml/ba_etl_bookings.py
python3 src/ml/ba_train_model.py
```

**View MLflow experiments:**
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5002
```

## Project Context

This project was completed as part of the **British Airways Data Science Virtual Experience Programme on Forage**. The original simulation tasks have been extended into a production-grade platform with dual ML modules, a database-backed API, containerized deployment, and an automated CI/CD pipeline.

## What This Demonstrates

- **NLP Engineering** — full text preprocessing, dual sentiment models, topic discovery
- **Messy Data Handling** — 1,920 unstructured scraped reviews cleaned and structured
- **Large Dataset ML** — 50,000 booking records with severe class imbalance handled
- **Feature Engineering** — 6 composite features from domain knowledge
- **MLOps** — experiment tracking, model registry, reproducible training
- **API Design** — 5 endpoints connecting NLP and ML modules with live database insights
- **Production Deployment** — containerized, health-checked, CI/CD automated
