"""
British Airways — Topic Modeling Pipeline
==========================================
Uses LDA (Latent Dirichlet Allocation) to discover hidden topics
in customer reviews.

Topics expected (based on airline review patterns):
    - Flight delays and punctuality
    - Cabin crew and customer service
    - Seat comfort and legroom
    - Food and catering quality
    - Baggage handling
    - Check-in and boarding process
    - Value for money

Results stored in PostgreSQL ba_reviews table (topic columns).

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import logging
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/topic_model.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ba_intelligence_db",
    "user":     "postgres",
    "password": "postgres"
}

# Number of topics to discover
N_TOPICS = 6

# Topic labels — assigned based on top words after running LDA
TOPIC_LABELS = {
    0: "Flight Experience",
    1: "Customer Service",
    2: "Delays & Punctuality",
    3: "Seat & Comfort",
    4: "Food & Catering",
    5: "Baggage & Check-in"
}

# ── Add topic columns ─────────────────────────────────────────────────────────
def add_topic_columns():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    columns = [
        ("dominant_topic",    "INTEGER"),
        ("topic_label",       "VARCHAR(50)"),
        ("topic_confidence",  "FLOAT"),
    ]
    for col_name, col_type in columns:
        cur.execute(f"ALTER TABLE ba_reviews ADD COLUMN IF NOT EXISTS {col_name} {col_type};")
    conn.commit()
    cur.close()
    conn.close()
    logger.info("Topic columns added to ba_reviews")

# ── Load lemmatized reviews ───────────────────────────────────────────────────
def load_reviews() -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT id, lemmatized_review, combined_sentiment, verified, word_count
        FROM ba_reviews
        WHERE lemmatized_review IS NOT NULL
        AND LENGTH(lemmatized_review) > 10
    """, conn)
    conn.close()
    logger.info(f"Loaded {len(df)} reviews for topic modeling")
    return df

# ── Build LDA model ───────────────────────────────────────────────────────────
def build_lda_model(texts):
    logger.info(f"Building LDA model with {N_TOPICS} topics...")

    # Vectorize — convert text to word count matrix
    vectorizer = CountVectorizer(
        max_df=0.90,      # ignore words appearing in >90% of docs
        min_df=5,         # ignore words appearing in <5 docs
        max_features=1000,
        ngram_range=(1, 2) # unigrams and bigrams
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    logger.info(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

    # Train LDA
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        max_iter=20,
        learning_method="online"
    )
    lda.fit(doc_term_matrix)
    logger.info("LDA model trained")

    return lda, vectorizer, doc_term_matrix

# ── Print top words per topic ─────────────────────────────────────────────────
def print_top_words(lda, vectorizer, n_words=12):
    feature_names = vectorizer.get_feature_names_out()
    logger.info(f"\nTop {n_words} words per topic:")
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words-1:-1]]
        label = TOPIC_LABELS.get(topic_idx, f"Topic {topic_idx}")
        logger.info(f"  Topic {topic_idx} — {label}:")
        logger.info(f"    {', '.join(top_words)}")

# ── Assign dominant topic to each review ─────────────────────────────────────
def assign_topics(df, lda, vectorizer, doc_term_matrix) -> pd.DataFrame:
    topic_distributions = lda.transform(doc_term_matrix)

    df["dominant_topic"]   = topic_distributions.argmax(axis=1)
    df["topic_confidence"] = topic_distributions.max(axis=1).round(4)
    df["topic_label"]      = df["dominant_topic"].map(TOPIC_LABELS)

    logger.info("\nTopic distribution across reviews:")
    dist = df["topic_label"].value_counts()
    for topic, count in dist.items():
        logger.info(f"  {topic}: {count} ({count/len(df)*100:.1f}%)")

    return df

# ── Cross-analysis: sentiment by topic ───────────────────────────────────────
def sentiment_by_topic(df):
    logger.info("\nSentiment breakdown by topic:")
    cross = df.groupby(["topic_label", "combined_sentiment"]).size().unstack(fill_value=0)
    logger.info(f"\n{cross.to_string()}")

    # Which topic has the most negative sentiment?
    if "Negative" in cross.columns:
        most_negative = cross["Negative"].idxmax()
        logger.info(f"\nMost complained-about topic: {most_negative}")

# ── Save to PostgreSQL ────────────────────────────────────────────────────────
def save_topics(df: pd.DataFrame):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    for _, row in df.iterrows():
        cur.execute("""
            UPDATE ba_reviews SET
                dominant_topic   = %s,
                topic_label      = %s,
                topic_confidence = %s
            WHERE id = %s
        """, (
            int(row["dominant_topic"]),
            str(row["topic_label"]),
            float(row["topic_confidence"]),
            int(row["id"])
        ))

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Saved topic assignments for {len(df)} reviews")

    df.to_csv("data/processed/reviews_topics.csv", index=False)
    logger.info("Saved: data/processed/reviews_topics.csv")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== BA Topic Modeling Pipeline Start ===")

    try:
        add_topic_columns()
        df = load_reviews()

        texts = df["lemmatized_review"].fillna("").tolist()

        lda, vectorizer, doc_term_matrix = build_lda_model(texts)
        print_top_words(lda, vectorizer)

        df = assign_topics(df, lda, vectorizer, doc_term_matrix)
        sentiment_by_topic(df)
        save_topics(df)

        # Save model for reuse in API
        os.makedirs("models", exist_ok=True)
        with open("models/lda_model.pkl", "wb") as f:
            pickle.dump(lda, f)
        with open("models/vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info("Saved LDA model and vectorizer to models/")

        logger.info("=== Topic Modeling Complete ===")

    except Exception as e:
        logger.error(f"Topic modeling failed: {e}")
        raise
