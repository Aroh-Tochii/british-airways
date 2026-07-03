"""
British Airways — Booking Data ETL Pipeline
=============================================
Loads 50,000 booking records into PostgreSQL with:
    - Data validation and cleaning
    - Feature engineering (booking window, route popularity, etc.)
    - Load into ba_bookings table

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/bookings_etl.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ba_intelligence_db",
    "user":     "postgres",
    "password": "postgres"
}

RAW_PATH = "data/raw/customer_booking.csv"

# ── Setup table ───────────────────────────────────────────────────────────────
def setup_table():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ba_bookings (
            id                      SERIAL PRIMARY KEY,
            num_passengers          INTEGER,
            sales_channel           VARCHAR(20),
            trip_type               VARCHAR(20),
            purchase_lead           INTEGER,
            length_of_stay          INTEGER,
            flight_hour             INTEGER,
            flight_day              VARCHAR(5),
            flight_day_num          INTEGER,
            route                   VARCHAR(10),
            booking_origin          VARCHAR(50),
            wants_extra_baggage     INTEGER,
            wants_preferred_seat    INTEGER,
            wants_in_flight_meals   INTEGER,
            flight_duration         FLOAT,
            booking_complete        INTEGER,
            booking_window_category VARCHAR(20),
            route_popularity        INTEGER,
            is_weekend_booking      INTEGER,
            total_extras            INTEGER,
            duration_category       VARCHAR(20),
            ingested_at             TIMESTAMP DEFAULT NOW()
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    logger.info("ba_bookings table ready")

# ── Extract ───────────────────────────────────────────────────────────────────
def extract() -> pd.DataFrame:
    logger.info(f"Loading booking data from {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, encoding="ISO-8859-1")
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df

# ── Transform ─────────────────────────────────────────────────────────────────
def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Transforming booking data")
    df = df.copy()

    # 1. Map flight_day to numeric
    day_map = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4,
               "Fri": 5, "Sat": 6, "Sun": 7}
    df["flight_day_num"] = df["flight_day"].map(day_map)

    # 2. Booking window category (how far in advance did they book?)
    df["booking_window_category"] = pd.cut(
        df["purchase_lead"],
        bins=[-1, 7, 30, 90, 999],
        labels=["Last Minute", "Short Term", "Medium Term", "Long Term"]
    ).astype(str)

    # 3. Route popularity (how many bookings on each route?)
    route_counts = df["route"].value_counts()
    df["route_popularity"] = df["route"].map(route_counts)

    # 4. Weekend booking flag
    df["is_weekend_booking"] = df["flight_day_num"].isin([6, 7]).astype(int)

    # 5. Total extras requested
    df["total_extras"] = (df["wants_extra_baggage"] +
                          df["wants_preferred_seat"] +
                          df["wants_in_flight_meals"])

    # 6. Flight duration category
    df["duration_category"] = pd.cut(
        df["flight_duration"],
        bins=[0, 3, 6, 12, 99],
        labels=["Short", "Medium", "Long", "Ultra Long"]
    ).astype(str)

    # Validate
    invalid_passengers = df[df["num_passengers"] < 1]
    if len(invalid_passengers) > 0:
        logger.warning(f"Removing {len(invalid_passengers)} rows with invalid passenger count")
        df = df[df["num_passengers"] >= 1]

    logger.info(f"Transform complete: {len(df)} rows")
    logger.info(f"  Booking completion rate: {df['booking_complete'].mean()*100:.1f}%")
    logger.info(f"  Class imbalance: {(df['booking_complete']==0).sum()} incomplete vs {(df['booking_complete']==1).sum()} complete")
    logger.info(f"  Booking window distribution:\n{df['booking_window_category'].value_counts().to_string()}")

    return df

# ── Load ──────────────────────────────────────────────────────────────────────
def load(df: pd.DataFrame):
    logger.info("Loading booking data into PostgreSQL")
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE ba_bookings RESTART IDENTITY;")

    records = []
    for _, row in df.iterrows():
        records.append((
            int(row["num_passengers"]),
            str(row["sales_channel"]),
            str(row["trip_type"]),
            int(row["purchase_lead"]),
            int(row["length_of_stay"]),
            int(row["flight_hour"]),
            str(row["flight_day"]),
            int(row["flight_day_num"]),
            str(row["route"]),
            str(row["booking_origin"]),
            int(row["wants_extra_baggage"]),
            int(row["wants_preferred_seat"]),
            int(row["wants_in_flight_meals"]),
            float(row["flight_duration"]),
            int(row["booking_complete"]),
            str(row["booking_window_category"]),
            int(row["route_popularity"]),
            int(row["is_weekend_booking"]),
            int(row["total_extras"]),
            str(row["duration_category"])
        ))

    execute_values(cur, """
        INSERT INTO ba_bookings (
            num_passengers, sales_channel, trip_type, purchase_lead,
            length_of_stay, flight_hour, flight_day, flight_day_num,
            route, booking_origin, wants_extra_baggage, wants_preferred_seat,
            wants_in_flight_meals, flight_duration, booking_complete,
            booking_window_category, route_popularity, is_weekend_booking,
            total_extras, duration_category
        ) VALUES %s
    """, records)

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM ba_bookings")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    logger.info(f"Loaded {count} booking records into ba_bookings")

    df.to_csv("data/processed/bookings_clean.csv", index=False)
    logger.info("Saved: data/processed/bookings_clean.csv")
    return count

# ── Validate with SQL ─────────────────────────────────────────────────────────
def validate():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM ba_bookings")
    total = cur.fetchone()[0]

    cur.execute("""
        SELECT booking_window_category, COUNT(*),
               ROUND(AVG(booking_complete)*100, 1) AS completion_rate
        FROM ba_bookings
        GROUP BY booking_window_category
        ORDER BY completion_rate DESC
    """)
    rows = cur.fetchall()

    cur.close()
    conn.close()

    logger.info(f"\nValidation — {total} total records")
    logger.info("\nCompletion rate by booking window:")
    for row in rows:
        logger.info(f"  {row[0]}: {row[1]} bookings, {row[2]}% completion")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=== BA Bookings ETL Pipeline Start ===")
    try:
        setup_table()
        df = extract()
        df = transform(df)
        count = load(df)
        validate()
        logger.info(f"=== ETL Complete — {count} records loaded ===")
    except Exception as e:
        logger.error(f"ETL failed: {e}")
        raise
