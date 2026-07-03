"""
British Airways — Sentiment Analysis Pipeline
==============================================
Two-layer sentiment analysis on cleaned reviews:
    Layer 1: VADER — rule-based, fast, handles negations well
    Layer 2: TextBlob — statistical polarity + subjectivity

Adds sentiment columns to ba_reviews table in PostgreSQL.
Also generates a sentiment summary table for dashboard use.

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import logging
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/sentiment.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ba_intelligence_db",
    "user":     "postgres",
    "password": "postgres"
}

# ── Add sentiment columns to existing table ───────────────────────────────────
def add_sentiment_columns():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    columns = [
        ("vader_positive",   "FLOAT"),
        ("vader_negative",   "FLOAT"),
        ("vader_neutral",    "FLOAT"),
        ("vader_compound",   "FLOAT"),
        ("vader_sentiment",  "VARCHAR(10)"),
        ("textblob_polarity","FLOAT"),
        ("textblob_subjectivity", "FLOAT"),
        ("textblob_sentiment","VARCHAR(10)"),
        ("combined_sentiment","VARCHAR(10)"),
        ("sentiment_confidence", "FLOAT"),
    ]

    for col_name, col_type in columns:
        try:
            cur.execute(f"ALTER TABLE ba_reviews ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
        except Exception:
            pass

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Sentiment columns added to ba_reviews")

# ── Load cleaned reviews ──────────────────────────────────────────────────────
def load_reviews() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT id, clean_review, verified, word_count
        FROM ba_reviews
        ORDER BY id
    """, conn)
    conn.close()
    logger.info(f"Loaded {len(df)} reviews for sentiment analysis")
    return df

# ── VADER sentiment ───────────────────────────────────────────────────────────
def analyze_vader(text: str) -> dict:
    """VADER — good for short social-media style text, handles negations."""
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(str(text))
    compound = scores["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "vader_positive":  round(scores["pos"], 4),
        "vader_negative":  round(scores["neg"], 4),
        "vader_neutral":   round(scores["neu"], 4),
        "vader_compound":  round(compound, 4),
        "vader_sentiment": sentiment
    }

# ── TextBlob sentiment ────────────────────────────────────────────────────────
def analyze_textblob(text: str) -> dict:
    """TextBlob — polarity (-1 to 1) and subjectivity (0 to 1)."""
    blob = TextBlob(str(text))
    polarity    = round(blob.sentiment.polarity, 4)
    subjectivity = round(blob.sentiment.subjectivity, 4)

    if polarity > 0.05:
        sentiment = "Positive"
    elif polarity < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "textblob_polarity":     polarity,
        "textblob_subjectivity": subjectivity,
        "textblob_sentiment":    sentiment
    }

# ── Combined sentiment ────────────────────────────────────────────────────────
def combine_sentiment(vader_sent: str, textblob_sent: str,
                      vader_compound: float, textblob_polarity: float) -> dict:
    """
    When both models agree — high confidence.
    When they disagree — use VADER as tiebreaker (better for reviews).
    """
    if vader_sent == textblob_sent:
        confidence = abs(vader_compound + textblob_polarity) / 2
        return {"combined_sentiment": vader_sent, "sentiment_confidence": round(confidence, 4)}
    else:
        # VADER wins on disagreement — better for longer review text
        confidence = abs(vader_compound) * 0.6
        return {"combined_sentiment": vader_sent, "sentiment_confidence": round(confidence, 4)}

# ── Run sentiment analysis ────────────────────────────────────────────────────
def analyze_all(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Running sentiment analysis...")

    vader_results    = []
    textblob_results = []
    combined_results = []

    for _, row in df.iterrows():
        text = row["clean_review"]

        v = analyze_vader(text)
        t = analyze_textblob(text)
        c = combine_sentiment(
            v["vader_sentiment"], t["textblob_sentiment"],
            v["vader_compound"],  t["textblob_polarity"]
        )

        vader_results.append(v)
        textblob_results.append(t)
        combined_results.append(c)

    vader_df    = pd.DataFrame(vader_results)
    textblob_df = pd.DataFrame(textblob_results)
    combined_df = pd.DataFrame(combined_results)

    df = pd.concat([df, vader_df, textblob_df, combined_df], axis=1)

    logger.info("Sentiment analysis complete")
    logger.info(f"\nSentiment distribution (combined):")
    dist = df["combined_sentiment"].value_counts()
    for sentiment, count in dist.items():
        logger.info(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")

    logger.info(f"\nVADER vs TextBlob agreement rate: "
                f"{(df['vader_sentiment'] == df['textblob_sentiment']).mean()*100:.1f}%")

    return df

# ── Save back to PostgreSQL ───────────────────────────────────────────────────
def save_sentiment(df: pd.DataFrame):
    logger.info("Saving sentiment scores to PostgreSQL")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    for _, row in df.iterrows():
        cur.execute("""
            UPDATE ba_reviews SET
                vader_positive       = %s,
                vader_negative       = %s,
                vader_neutral        = %s,
                vader_compound       = %s,
                vader_sentiment      = %s,
                textblob_polarity    = %s,
                textblob_subjectivity= %s,
                textblob_sentiment   = %s,
                combined_sentiment   = %s,
                sentiment_confidence = %s
            WHERE id = %s
        """, (
            row["vader_positive"], row["vader_negative"],
            row["vader_neutral"],  row["vader_compound"],
            row["vader_sentiment"], row["textblob_polarity"],
            row["textblob_subjectivity"], row["textblob_sentiment"],
            row["combined_sentiment"], row["sentiment_confidence"],
            int(row["id"])
        ))

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Saved sentiment scores for {len(df)} reviews")

    # Save processed file
    df.to_csv("data/processed/reviews_sentiment.csv", index=False)
    logger.info("Saved: data/processed/reviews_sentiment.csv")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== BA Sentiment Analysis Pipeline Start ===")

    try:
        add_sentiment_columns()
        df = load_reviews()
        df = analyze_all(df)
        save_sentiment(df)
        logger.info("=== Sentiment Pipeline Complete ===")

    except Exception as e:
        logger.error(f"Sentiment pipeline failed: {e}")
        raise
