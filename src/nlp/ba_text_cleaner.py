"""
British Airways — Text Cleaning Pipeline
=========================================
Cleans raw scraped reviews and loads them into PostgreSQL.

Steps:
    1. Extract verified/unverified flag
    2. Remove HTML, emojis, special characters
    3. Normalize whitespace
    4. Lemmatize with spaCy
    5. Remove stopwords
    6. Load clean data into PostgreSQL (ba_reviews table)

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import re
import string
import psycopg2
from psycopg2.extras import execute_values
import spacy
import logging
import os

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/text_cleaner.log"), logging.StreamHandler()]
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

RAW_DATA_PATH = "data/raw/reviews.csv"

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# ── Database setup ─────────────────────────────────────────────────────────────
def setup_database():
    """Create database and tables if they don't exist."""
    # Connect to default postgres DB first
    conn = psycopg2.connect(**{**DB_CONFIG, "database": "postgres"})
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'ba_intelligence_db'")
    if not cur.fetchone():
        cur.execute("CREATE DATABASE ba_intelligence_db")
        logger.info("Created database: ba_intelligence_db")
    cur.close()
    conn.close()

    # Now connect to the actual DB and create tables
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ba_reviews (
            id                  SERIAL PRIMARY KEY,
            raw_review          TEXT,
            verified            BOOLEAN,
            clean_review        TEXT,
            lemmatized_review   TEXT,
            word_count          INTEGER,
            ingested_at         TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Database tables ready")

# ── Extract ────────────────────────────────────────────────────────────────────
def extract(filepath: str) -> pd.DataFrame:
    logger.info(f"Loading reviews from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} reviews")
    return df

# ── Transform ──────────────────────────────────────────────────────────────────
def extract_verified_flag(text: str) -> bool:
    """Check if review is verified trip."""
    if pd.isna(text):
        return False
    return "Trip Verified" in text or "✅" in text

def extract_clean_text(text: str) -> str:
    """Extract the review content after the pipe separator."""
    if pd.isna(text):
        return ""
    # Split on | and take the second part if it exists
    parts = str(text).split("|")
    if len(parts) > 1:
        return parts[1].strip()
    return str(text).strip()

def remove_noise(text: str) -> str:
    """Remove HTML, emojis, special characters."""
    if not text:
        return ""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    # Remove emojis and special unicode
    text = text.encode("ascii", "ignore").decode("ascii")
    # Remove special characters but keep letters, numbers, spaces, basic punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

def lemmatize_text(text: str) -> str:
    """Lemmatize and remove stopwords using spaCy."""
    if not text:
        return ""
    doc = nlp(text)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and len(token.lemma_) > 2
        and token.is_alpha
    ]
    return " ".join(tokens)

def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting text transformation pipeline")

    # Step 1: Extract verified flag
    df["verified"] = df["reviews"].apply(extract_verified_flag)
    logger.info(f"Verified reviews: {df['verified'].sum()} / {len(df)}")

    # Step 2: Extract clean text (after pipe separator)
    df["clean_review"] = df["reviews"].apply(extract_clean_text)

    # Step 3: Remove noise
    df["clean_review"] = df["clean_review"].apply(remove_noise)

    # Step 4: Lemmatize
    logger.info("Lemmatizing reviews — this may take a minute...")
    df["lemmatized_review"] = df["clean_review"].apply(lemmatize_text)

    # Step 5: Word count
    df["word_count"] = df["clean_review"].apply(lambda x: len(x.split()) if x else 0)

    # Drop empty reviews
    before = len(df)
    df = df[df["clean_review"].str.len() > 20].reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} empty/too-short reviews")
    logger.info(f"Transform complete: {len(df)} reviews ready")

    return df

# ── Load ───────────────────────────────────────────────────────────────────────
def load(df: pd.DataFrame):
    logger.info("Loading cleaned reviews into PostgreSQL")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE ba_reviews RESTART IDENTITY;")

    records = [
        (
            str(row["reviews"]),
            bool(row["verified"]),
            str(row["clean_review"]),
            str(row["lemmatized_review"]),
            int(row["word_count"])
        )
        for _, row in df.iterrows()
    ]

    execute_values(cur, """
        INSERT INTO ba_reviews
        (raw_review, verified, clean_review, lemmatized_review, word_count)
        VALUES %s
    """, records)

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM ba_reviews")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    logger.info(f"Loaded {count} reviews into ba_reviews")
    return count

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    logger.info("=== BA Text Cleaning Pipeline Start ===")

    try:
        setup_database()
        df = extract(RAW_DATA_PATH)
        df = transform(df)
        count = load(df)

        # Save processed data locally too
        df.to_csv("data/processed/reviews_clean.csv", index=False)
        logger.info("Saved: data/processed/reviews_clean.csv")

        logger.info(f"=== Pipeline Complete — {count} reviews loaded ===")

        # Quick stats
        logger.info(f"\nQuick stats:")
        logger.info(f"  Verified reviews:   {df['verified'].sum()} ({df['verified'].mean()*100:.1f}%)")
        logger.info(f"  Avg word count:     {df['word_count'].mean():.0f} words")
        logger.info(f"  Longest review:     {df['word_count'].max()} words")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
