"""
British Airways — EDA & Visualization
======================================
8 clinical-quality charts covering:
    1. Sentiment distribution (overview)
    2. Topic frequency (what customers talk about)
    3. Sentiment by topic (where complaints concentrate)
    4. Verified vs unverified sentiment gap
    5. Word count distribution by sentiment
    6. Top words per sentiment (positive vs negative)
    7. Topic confidence distribution
    8. Sentiment heatmap (topic × sentiment intensity)

All charts saved to reports/figures/

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import psycopg2
from wordcloud import WordCloud
import warnings
import os
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ba_intelligence_db",
    "user":     "postgres",
    "password": "postgres"
}

OUTPUT_DIR = "reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
POSITIVE_COLOR = "#1565C0"
NEGATIVE_COLOR = "#C62828"
NEUTRAL_COLOR  = "#607D8B"
NAVY           = "#0B2545"
TEAL           = "#028090"
OFFWHITE       = "#F7F9FA"

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT id, verified, clean_review, lemmatized_review,
               word_count, vader_compound, vader_sentiment,
               textblob_polarity, textblob_subjectivity,
               combined_sentiment, sentiment_confidence,
               dominant_topic, topic_label, topic_confidence
        FROM ba_reviews
        WHERE combined_sentiment IS NOT NULL
        AND topic_label IS NOT NULL
    """, conn)
    conn.close()
    print(f"Loaded {len(df)} reviews for visualization")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 — Sentiment Distribution Overview
# ══════════════════════════════════════════════════════════════════════════════
def plot_sentiment_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("British Airways Customer Sentiment Overview",
                 fontsize=16, fontweight="bold", y=1.01)

    counts = df["combined_sentiment"].value_counts()
    colors = {"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR,
              "Neutral": NEUTRAL_COLOR}
    bar_colors = [colors.get(s, NEUTRAL_COLOR) for s in counts.index]

    # Bar chart
    bars = axes[0].bar(counts.index, counts.values, color=bar_colors,
                       width=0.5, edgecolor="white")
    for bar, count in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(count), ha="center", fontweight="bold", fontsize=12)
    axes[0].set_title("Count by Sentiment", fontsize=13)
    axes[0].set_ylabel("Number of Reviews")
    axes[0].spines[["top", "right"]].set_visible(False)

    # Pie chart
    pie_colors = [colors.get(s, NEUTRAL_COLOR) for s in counts.index]
    axes[1].pie(counts.values, labels=counts.index, colors=pie_colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 2},
                textprops={"fontsize": 12})
    axes[1].set_title("Proportion by Sentiment", fontsize=13)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/01_sentiment_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 — Topic Frequency
# ══════════════════════════════════════════════════════════════════════════════
def plot_topic_frequency(df):
    topic_counts = df["topic_label"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [NEGATIVE_COLOR if "Delay" in t or "Food" in t or "Seat" in t
              else POSITIVE_COLOR for t in topic_counts.index]
    bars = ax.barh(topic_counts.index, topic_counts.values,
                   color=colors, edgecolor="white", height=0.6)
    for bar, count in zip(bars, topic_counts.values):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
               f"{count} ({count/len(df)*100:.1f}%)",
               va="center", fontsize=10)

    ax.set_title("What Customers Talk About Most",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Number of Reviews")
    ax.set_xlim(0, topic_counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    pos_patch = mpatches.Patch(color=POSITIVE_COLOR, label="Generally positive topic")
    neg_patch = mpatches.Patch(color=NEGATIVE_COLOR, label="Generally negative topic")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/02_topic_frequency.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 — Sentiment by Topic (Stacked Bar)
# ══════════════════════════════════════════════════════════════════════════════
def plot_sentiment_by_topic(df):
    cross = df.groupby(["topic_label", "combined_sentiment"]).size().unstack(fill_value=0)
    cross_pct = cross.div(cross.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(cross_pct))

    sentiment_colors = {"Positive": POSITIVE_COLOR, "Negative": NEGATIVE_COLOR,
                        "Neutral": NEUTRAL_COLOR}

    for sentiment in ["Positive", "Neutral", "Negative"]:
        if sentiment in cross_pct.columns:
            values = cross_pct[sentiment].values
            bars = ax.bar(cross_pct.index, values, bottom=bottom,
                         label=sentiment, color=sentiment_colors[sentiment],
                         edgecolor="white", width=0.6)
            for bar, val in zip(bars, values):
                if val > 5:
                    ax.text(bar.get_x() + bar.get_width()/2,
                           bar.get_y() + bar.get_height()/2,
                           f"{val:.0f}%", ha="center", va="center",
                           fontsize=9, color="white", fontweight="bold")
            bottom += values

    ax.set_title("Sentiment Breakdown by Topic\n(% of reviews in each topic)",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Percentage of Reviews (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = f"{OUTPUT_DIR}/03_sentiment_by_topic.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 — Verified vs Unverified Sentiment Gap
# ══════════════════════════════════════════════════════════════════════════════
def plot_verified_sentiment_gap(df):
    verified   = df[df["verified"] == True]
    unverified = df[df["verified"] == False]

    def sentiment_rates(subset):
        counts = subset["combined_sentiment"].value_counts(normalize=True) * 100
        return counts.get("Positive", 0), counts.get("Negative", 0)

    v_pos, v_neg = sentiment_rates(verified)
    u_pos, u_neg = sentiment_rates(unverified)

    categories = ["Positive Rate", "Negative Rate"]
    verified_vals   = [v_pos, v_neg]
    unverified_vals = [u_pos, u_neg]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width/2, verified_vals,   width, label=f"Verified (n={len(verified)})",
                color=TEAL, alpha=0.85, edgecolor="white")
    b2 = ax.bar(x + width/2, unverified_vals, width, label=f"Unverified (n={len(unverified)})",
                color=NEUTRAL_COLOR, alpha=0.85, edgecolor="white")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f"{bar.get_height():.1f}%", ha="center", fontsize=10)

    ax.set_title("Verified vs Unverified Reviews — Sentiment Gap",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylabel("Percentage of Reviews (%)")
    ax.set_ylim(0, 80)
    ax.legend(fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/04_verified_sentiment_gap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 — VADER Compound Score Distribution
# ══════════════════════════════════════════════════════════════════════════════
def plot_vader_distribution(df):
    fig, ax = plt.subplots(figsize=(12, 5))

    positive = df[df["combined_sentiment"] == "Positive"]["vader_compound"]
    negative = df[df["combined_sentiment"] == "Negative"]["vader_compound"]

    ax.hist(positive, bins=30, alpha=0.7, color=POSITIVE_COLOR,
            label=f"Positive reviews (n={len(positive)})")
    ax.hist(negative, bins=30, alpha=0.7, color=NEGATIVE_COLOR,
            label=f"Negative reviews (n={len(negative)})")

    ax.axvline(x=0.05,  color="green",  linestyle="--", linewidth=1.5,
               label="Positive threshold (0.05)")
    ax.axvline(x=-0.05, color="red",    linestyle="--", linewidth=1.5,
               label="Negative threshold (-0.05)")
    ax.axvline(x=positive.mean(), color=POSITIVE_COLOR, linestyle="-",
               linewidth=2, label=f"Positive avg: {positive.mean():.2f}")
    ax.axvline(x=negative.mean(), color=NEGATIVE_COLOR, linestyle="-",
               linewidth=2, label=f"Negative avg: {negative.mean():.2f}")

    ax.set_title("VADER Sentiment Score Distribution\n(-1 = Most Negative, +1 = Most Positive)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("VADER Compound Score")
    ax.set_ylabel("Number of Reviews")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/05_vader_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 — Word Clouds (Positive vs Negative)
# ══════════════════════════════════════════════════════════════════════════════
def plot_word_clouds(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Most Common Words: Positive vs Negative Reviews",
                 fontsize=15, fontweight="bold")

    for ax, sentiment, color, title in [
        (axes[0], "Positive", POSITIVE_COLOR, "Positive Reviews"),
        (axes[1], "Negative", NEGATIVE_COLOR, "Negative Reviews")
    ]:
        text = " ".join(
            df[df["combined_sentiment"] == sentiment]["lemmatized_review"].dropna()
        )
        if text.strip():
            wc = WordCloud(
                width=700, height=400,
                background_color="white",
                colormap="Blues" if sentiment == "Positive" else "Reds",
                max_words=80,
                collocations=False
            ).generate(text)
            ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", color=color)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/06_word_clouds.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 — TextBlob Polarity vs Subjectivity
# ══════════════════════════════════════════════════════════════════════════════
def plot_polarity_subjectivity(df):
    fig, ax = plt.subplots(figsize=(11, 7))

    colors = df["combined_sentiment"].map({
        "Positive": POSITIVE_COLOR,
        "Negative": NEGATIVE_COLOR,
        "Neutral":  NEUTRAL_COLOR
    })

    ax.scatter(df["textblob_polarity"], df["textblob_subjectivity"],
               c=colors, alpha=0.4, s=20)

    ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1)

    ax.text(-0.95, 0.95, "Negative\n& Subjective", fontsize=10,
            color=NEGATIVE_COLOR, fontweight="bold")
    ax.text(0.6, 0.95, "Positive\n& Subjective", fontsize=10,
            color=POSITIVE_COLOR, fontweight="bold")
    ax.text(-0.95, 0.02, "Negative\n& Objective", fontsize=10,
            color=NEGATIVE_COLOR)
    ax.text(0.6, 0.02, "Positive\n& Objective", fontsize=10,
            color=POSITIVE_COLOR)

    pos_patch = mpatches.Patch(color=POSITIVE_COLOR, label="Positive")
    neg_patch = mpatches.Patch(color=NEGATIVE_COLOR, label="Negative")
    neu_patch = mpatches.Patch(color=NEUTRAL_COLOR,  label="Neutral")
    ax.legend(handles=[pos_patch, neg_patch, neu_patch], fontsize=10)

    ax.set_title("Review Polarity vs Subjectivity\n(TextBlob Analysis)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Polarity (-1 = Very Negative, +1 = Very Positive)", fontsize=11)
    ax.set_ylabel("Subjectivity (0 = Factual, 1 = Opinion-based)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/07_polarity_subjectivity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# CHART 8 — Average VADER Score by Topic (Business Insight Chart)
# ══════════════════════════════════════════════════════════════════════════════
def plot_vader_by_topic(df):
    topic_sentiment = df.groupby("topic_label")["vader_compound"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [NEGATIVE_COLOR if v < 0 else POSITIVE_COLOR
              for v in topic_sentiment.values]
    bars = ax.barh(topic_sentiment.index, topic_sentiment.values,
                   color=colors, edgecolor="white", height=0.6)

    for bar, val in zip(bars, topic_sentiment.values):
        ax.text(val + (0.01 if val >= 0 else -0.01),
               bar.get_y() + bar.get_height()/2,
               f"{val:.3f}", va="center", fontsize=10,
               ha="left" if val >= 0 else "right")

    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_title("Average Sentiment Score by Topic\n(Most Negative → Most Positive)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Average VADER Compound Score")
    ax.spines[["top", "right"]].set_visible(False)

    pos_patch = mpatches.Patch(color=POSITIVE_COLOR, label="Net Positive")
    neg_patch = mpatches.Patch(color=NEGATIVE_COLOR, label="Net Negative")
    ax.legend(handles=[pos_patch, neg_patch], fontsize=10)

    plt.tight_layout()
    path = f"{OUTPUT_DIR}/08_vader_by_topic.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("British Airways EDA — Generating Charts")
    print("=" * 60)

    df = load_data()

    print("\nGenerating 8 charts...")
    plot_sentiment_distribution(df)
    plot_topic_frequency(df)
    plot_sentiment_by_topic(df)
    plot_verified_sentiment_gap(df)
    plot_vader_distribution(df)
    plot_word_clouds(df)
    plot_polarity_subjectivity(df)
    plot_vader_by_topic(df)

    print("\n" + "=" * 60)
    print(f"EDA Complete — 8 charts saved to {OUTPUT_DIR}/")
    print("=" * 60)
