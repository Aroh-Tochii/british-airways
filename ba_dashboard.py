"""
British Airways — KPI Dashboard
=================================
Interactive Plotly Dash dashboard with three tabs:
    Tab 1: Customer Voice — sentiment & topic analysis
    Tab 2: Booking Intelligence — completion rates & drop-off risk
    Tab 3: Live KPIs — real-time metrics from PostgreSQL

Run: python3 ba_dashboard.py
Open: http://localhost:8050

Author: Alfred (Aroh-Tochii)
"""

import pandas as pd
import numpy as np
import psycopg2
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, html, dcc, Input, Output, callback
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "localhost",
    "port":     5432,
    "database": "ba_intelligence_db",
    "user":     "postgres",
    "password": "postgres"
}

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY    = "#0B2545"
TEAL    = "#028090"
MINT    = "#02C39A"
RED     = "#C62828"
AMBER   = "#F57F17"
WHITE   = "#FFFFFF"
OFFWHITE= "#F7F9FA"
SLATE   = "#475569"
LIGHT   = "#E8F4F8"

# ── Load data ─────────────────────────────────────────────────────────────────
def load_reviews():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT verified, word_count, vader_compound, vader_sentiment,
               textblob_polarity, textblob_subjectivity,
               combined_sentiment, sentiment_confidence,
               dominant_topic, topic_label, topic_confidence
        FROM ba_reviews
        WHERE combined_sentiment IS NOT NULL
    """, conn)
    conn.close()
    return df

def load_bookings():
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT num_passengers, sales_channel, trip_type,
               purchase_lead, length_of_stay, flight_hour,
               wants_extra_baggage, wants_preferred_seat,
               wants_in_flight_meals, flight_duration,
               booking_complete, booking_window_category,
               route_popularity, is_weekend_booking, total_extras,
               duration_category
        FROM ba_bookings
    """, conn)
    conn.close()
    return df

# Load data once at startup
print("Loading data from PostgreSQL...")
reviews_df  = load_reviews()
bookings_df = load_bookings()
print(f"Loaded {len(reviews_df)} reviews and {len(bookings_df)} bookings")

# ── KPI Calculations ──────────────────────────────────────────────────────────
total_reviews     = len(reviews_df)
positive_pct      = round(reviews_df["combined_sentiment"].eq("Positive").mean() * 100, 1)
negative_pct      = round(reviews_df["combined_sentiment"].eq("Negative").mean() * 100, 1)
avg_sentiment     = round(reviews_df["vader_compound"].mean(), 3)
verified_pct      = round(reviews_df["verified"].mean() * 100, 1)
most_complained   = reviews_df["topic_label"].value_counts().index[0] if len(reviews_df) > 0 else "N/A"

total_bookings    = len(bookings_df)
completion_rate   = round(bookings_df["booking_complete"].mean() * 100, 1)
drop_off_rate     = round(100 - completion_rate, 1)
avg_lead_time     = round(bookings_df["purchase_lead"].mean(), 0)
high_extras_pct   = round(bookings_df["total_extras"].ge(2).mean() * 100, 1)
est_revenue_risk  = f"${int(bookings_df['booking_complete'].eq(0).sum() * 450):,}"

# ── App Layout ────────────────────────────────────────────────────────────────
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "BA Customer Intelligence Dashboard"

# KPI Card helper
def kpi_card(value, label, sublabel="", color=TEAL, width="23%"):
    return html.Div([
        html.H2(value, style={"color": color, "margin": "0", "fontSize": "2.2rem", "fontWeight": "800"}),
        html.P(label, style={"margin": "4px 0 2px", "fontWeight": "600", "color": NAVY, "fontSize": "0.9rem"}),
        html.P(sublabel, style={"margin": "0", "color": SLATE, "fontSize": "0.75rem"}) if sublabel else html.Span()
    ], style={
        "background": WHITE, "borderRadius": "10px", "padding": "18px 20px",
        "width": width, "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
        "borderTop": f"4px solid {color}", "minWidth": "150px"
    })

app.layout = html.Div([
    # ── Header ────────────────────────────────────────────────────────────────
    html.Div([
        html.Div([
            html.H1("✈ Airline Customer Intelligence", style={
                "color": WHITE, "margin": "0", "fontSize": "1.8rem", "fontWeight": "800"}),
            html.P("British Airways · 1,920 Reviews · 50,000 Bookings · Live PostgreSQL Data",
                   style={"color": "#9FC9D3", "margin": "4px 0 0", "fontSize": "0.85rem"})
        ]),
        html.Div([
            html.Span("● LIVE", style={"color": MINT, "fontWeight": "700", "fontSize": "0.9rem"}),
            html.Span(" Data connected", style={"color": "#9FC9D3", "fontSize": "0.85rem"})
        ])
    ], style={
        "background": NAVY, "padding": "20px 32px", "display": "flex",
        "justifyContent": "space-between", "alignItems": "center"
    }),

    # ── Top KPI Bar ───────────────────────────────────────────────────────────
    html.Div([
        kpi_card(f"{positive_pct}%",    "Positive Reviews",    f"of {total_reviews} total", TEAL),
        kpi_card(f"{negative_pct}%",    "Negative Reviews",    "Delays #1 driver", RED),
        kpi_card(f"{completion_rate}%", "Booking Completion",  f"of {total_bookings:,} bookings", MINT),
        kpi_card(f"{drop_off_rate}%",   "Drop-off Rate",       "Revenue at risk", AMBER),
        kpi_card(est_revenue_risk,      "Est. Revenue at Risk","@ $450 avg ticket", RED),
    ], style={
        "display": "flex", "gap": "16px", "padding": "20px 32px",
        "background": OFFWHITE, "flexWrap": "wrap"
    }),

    # ── Tabs ──────────────────────────────────────────────────────────────────
    dcc.Tabs(id="tabs", value="tab-voice", children=[
        dcc.Tab(label="📣 Customer Voice", value="tab-voice",
                style={"fontWeight": "600"}, selected_style={"fontWeight": "700", "color": TEAL}),
        dcc.Tab(label="🎫 Booking Intelligence", value="tab-booking",
                style={"fontWeight": "600"}, selected_style={"fontWeight": "700", "color": TEAL}),
        dcc.Tab(label="📊 KPI Summary", value="tab-kpi",
                style={"fontWeight": "600"}, selected_style={"fontWeight": "700", "color": TEAL}),
    ], style={"padding": "0 32px", "background": WHITE}),

    html.Div(id="tab-content", style={"padding": "24px 32px", "background": OFFWHITE, "minHeight": "500px"})

], style={"fontFamily": "Calibri, Arial, sans-serif", "background": OFFWHITE, "minHeight": "100vh"})

# ── Tab 1: Customer Voice ─────────────────────────────────────────────────────
def render_voice_tab():
    # Chart 1 — Sentiment Donut
    sent_counts = reviews_df["combined_sentiment"].value_counts()
    fig_sent = go.Figure(go.Pie(
        labels=sent_counts.index, values=sent_counts.values,
        hole=0.55, marker_colors=[TEAL if l=="Positive" else RED if l=="Negative" else SLATE
                                   for l in sent_counts.index],
        textinfo="label+percent"
    ))
    fig_sent.update_layout(title="Overall Sentiment Split", showlegend=False,
        margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        annotations=[dict(text=f"{positive_pct}%<br>Positive", x=0.5, y=0.5,
                         font_size=16, showarrow=False, font_color=TEAL)])

    # Chart 2 — Topic Sentiment Heatmap
    topic_sent = reviews_df.groupby(["topic_label", "combined_sentiment"]).size().unstack(fill_value=0)
    topic_pct  = topic_sent.div(topic_sent.sum(axis=1), axis=0) * 100

    neg_vals = topic_pct.get("Negative", pd.Series(0, index=topic_pct.index))
    fig_topic = go.Figure(go.Bar(
        x=neg_vals.sort_values(ascending=True).values,
        y=neg_vals.sort_values(ascending=True).index,
        orientation="h",
        marker_color=[RED if v > 50 else AMBER if v > 20 else MINT
                      for v in neg_vals.sort_values(ascending=True).values],
        text=[f"{v:.0f}%" for v in neg_vals.sort_values(ascending=True).values],
        textposition="outside"
    ))
    fig_topic.update_layout(
        title="% Negative Reviews by Topic", xaxis_title="Negative Review Rate (%)",
        margin=dict(t=40, b=10, l=10, r=60), plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        xaxis=dict(range=[0, 120])
    )

    # Chart 3 — Verified vs Unverified
    ver_sent = reviews_df.groupby(["verified", "combined_sentiment"]).size().unstack(fill_value=0)
    ver_pct  = ver_sent.div(ver_sent.sum(axis=1), axis=0) * 100
    fig_ver  = go.Figure()
    for sent, color in [("Positive", TEAL), ("Negative", RED)]:
        if sent in ver_pct.columns:
            fig_ver.add_trace(go.Bar(
                name=sent,
                x=["Verified", "Unverified"],
                y=[ver_pct.loc[True, sent] if True in ver_pct.index else 0,
                   ver_pct.loc[False, sent] if False in ver_pct.index else 0],
                marker_color=color
            ))
    fig_ver.update_layout(
        title="Verified vs Unverified Sentiment", barmode="group",
        margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor=WHITE, paper_bgcolor=WHITE
    )

    # Chart 4 — VADER Score Distribution
    fig_vader = go.Figure()
    for sent, color in [("Positive", TEAL), ("Negative", RED)]:
        subset = reviews_df[reviews_df["combined_sentiment"] == sent]["vader_compound"]
        fig_vader.add_trace(go.Histogram(x=subset, name=sent, marker_color=color,
                                          opacity=0.7, nbinsx=30))
    fig_vader.update_layout(
        title="Sentiment Score Distribution (VADER)", barmode="overlay",
        xaxis_title="Compound Score (-1 to +1)",
        margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor=WHITE, paper_bgcolor=WHITE
    )

    return html.Div([
        html.Div([
            html.Div([
                html.H4("Key Insight", style={"color": NAVY, "margin": "0 0 8px"}),
                html.P(f"Delays & Punctuality generates 100% negative reviews — the single biggest reputational risk. "
                       f"Food and seat comfort follow. Ground operations (baggage, check-in) are strengths.",
                       style={"color": SLATE, "margin": "0", "fontSize": "0.95rem"})
            ], style={"background": LIGHT, "borderLeft": f"4px solid {TEAL}",
                      "padding": "16px", "borderRadius": "8px", "marginBottom": "20px"})
        ]),
        html.Div([
            html.Div([dcc.Graph(figure=fig_sent)],  style={"width": "48%"}),
            html.Div([dcc.Graph(figure=fig_topic)], style={"width": "48%"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_ver)],   style={"width": "48%"}),
            html.Div([dcc.Graph(figure=fig_vader)], style={"width": "48%"}),
        ], style={"display": "flex", "gap": "16px"}),
    ])

# ── Tab 2: Booking Intelligence ───────────────────────────────────────────────
def render_booking_tab():
    # Chart 1 — Completion rate by booking window
    window_stats = bookings_df.groupby("booking_window_category").agg(
        total=("booking_complete", "count"),
        completed=("booking_complete", "sum")
    ).reset_index()
    window_stats["rate"] = (window_stats["completed"] / window_stats["total"] * 100).round(1)
    window_order = ["Last Minute", "Short Term", "Medium Term", "Long Term"]
    window_stats["booking_window_category"] = pd.Categorical(
        window_stats["booking_window_category"], categories=window_order, ordered=True)
    window_stats = window_stats.sort_values("booking_window_category")

    fig_window = go.Figure(go.Bar(
        x=window_stats["booking_window_category"],
        y=window_stats["rate"],
        marker_color=[MINT if r > 16 else TEAL if r > 15 else AMBER for r in window_stats["rate"]],
        text=[f"{r}%" for r in window_stats["rate"]], textposition="outside"
    ))
    fig_window.update_layout(
        title="Booking Completion Rate by Lead Time",
        yaxis_title="Completion Rate (%)", yaxis=dict(range=[0, 25]),
        margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor=WHITE, paper_bgcolor=WHITE
    )

    # Chart 2 — Completion by sales channel
    channel_stats = bookings_df.groupby("sales_channel")["booking_complete"].agg(["mean", "count"]).reset_index()
    channel_stats["rate"] = (channel_stats["mean"] * 100).round(1)
    fig_channel = go.Figure(go.Bar(
        x=channel_stats["sales_channel"], y=channel_stats["rate"],
        marker_color=[TEAL, NAVY], text=[f"{r}%" for r in channel_stats["rate"]],
        textposition="outside"
    ))
    fig_channel.update_layout(
        title="Completion Rate by Sales Channel",
        yaxis_title="Completion Rate (%)", yaxis=dict(range=[0, 25]),
        margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor=WHITE, paper_bgcolor=WHITE
    )

    # Chart 3 — Extras vs completion
    extras_stats = bookings_df.groupby("total_extras")["booking_complete"].agg(["mean", "count"]).reset_index()
    extras_stats["rate"] = (extras_stats["mean"] * 100).round(1)
    fig_extras = go.Figure(go.Bar(
        x=extras_stats["total_extras"].astype(str),
        y=extras_stats["rate"],
        marker_color=[MINT, TEAL, NAVY, AMBER],
        text=[f"{r}%" for r in extras_stats["rate"]], textposition="outside"
    ))
    fig_extras.update_layout(
        title="Completion Rate by Number of Extras Selected",
        xaxis_title="Total Extras (baggage + seat + meals)",
        yaxis_title="Completion Rate (%)", yaxis=dict(range=[0, 30]),
        margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor=WHITE, paper_bgcolor=WHITE
    )

    # Chart 4 — Flight duration vs completion
    dur_stats = bookings_df.groupby("duration_category")["booking_complete"].agg(["mean", "count"]).reset_index()
    dur_stats["rate"] = (dur_stats["mean"] * 100).round(1)
    fig_dur = go.Figure(go.Bar(
        x=dur_stats["duration_category"], y=dur_stats["rate"],
        marker_color=TEAL,
        text=[f"{r}%" for r in dur_stats["rate"]], textposition="outside"
    ))
    fig_dur.update_layout(
        title="Completion Rate by Flight Duration",
        yaxis_title="Completion Rate (%)", yaxis=dict(range=[0, 25]),
        margin=dict(t=40, b=10, l=10, r=10), plot_bgcolor=WHITE, paper_bgcolor=WHITE
    )

    return html.Div([
        html.Div([
            html.H4("Key Insight", style={"color": NAVY, "margin": "0 0 8px"}),
            html.P("Last-minute bookers complete at 17.3% vs 13.8% for long-term planners. "
                   "Customers selecting more extras show higher intent — total_extras is the strongest "
                   "commitment signal. Target long-term planners with no extras for intervention campaigns.",
                   style={"color": SLATE, "margin": "0", "fontSize": "0.95rem"})
        ], style={"background": LIGHT, "borderLeft": f"4px solid {AMBER}",
                  "padding": "16px", "borderRadius": "8px", "marginBottom": "20px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_window)],  style={"width": "48%"}),
            html.Div([dcc.Graph(figure=fig_channel)], style={"width": "48%"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
        html.Div([
            html.Div([dcc.Graph(figure=fig_extras)], style={"width": "48%"}),
            html.Div([dcc.Graph(figure=fig_dur)],    style={"width": "48%"}),
        ], style={"display": "flex", "gap": "16px"}),
    ])

# ── Tab 3: KPI Summary ────────────────────────────────────────────────────────
def render_kpi_tab():
    # Gauge — Sentiment Health Score (0-100)
    health_score = round((positive_pct / 100) * 100)
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=health_score,
        title={"text": "Sentiment Health Score", "font": {"size": 16}},
        delta={"reference": 70, "increasing": {"color": MINT}, "decreasing": {"color": RED}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": TEAL},
            "steps": [
                {"range": [0, 40],  "color": "#FFEBEE"},
                {"range": [40, 70], "color": "#FFF8E1"},
                {"range": [70, 100],"color": "#E8F5E9"},
            ],
            "threshold": {"line": {"color": NAVY, "width": 3}, "value": 70}
        }
    ))
    fig_gauge.update_layout(margin=dict(t=30, b=10, l=30, r=30),
                             paper_bgcolor=WHITE, height=280)

    # Gauge — Booking Conversion
    fig_conv = go.Figure(go.Indicator(
        mode="gauge+number",
        value=completion_rate,
        title={"text": "Booking Conversion Rate (%)", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 30]},
            "bar":  {"color": MINT},
            "steps": [
                {"range": [0, 10],  "color": "#FFEBEE"},
                {"range": [10, 20], "color": "#FFF8E1"},
                {"range": [20, 30], "color": "#E8F5E9"},
            ],
            "threshold": {"line": {"color": NAVY, "width": 3}, "value": 20}
        }
    ))
    fig_conv.update_layout(margin=dict(t=30, b=10, l=30, r=30),
                            paper_bgcolor=WHITE, height=280)

    kpis = [
        ("Total Reviews Analyzed",   f"{total_reviews:,}",    "NLP pipeline processed"),
        ("Positive Sentiment",        f"{positive_pct}%",      "of all reviews"),
        ("Verified Review Rate",      f"{verified_pct}%",      "high credibility data"),
        ("Avg Sentiment Score",       f"{avg_sentiment:+.3f}", "VADER compound (-1 to +1)"),
        ("Total Bookings",            f"{total_bookings:,}",   "booking records analyzed"),
        ("Completion Rate",           f"{completion_rate}%",   "industry avg ~15-20%"),
        ("Avg Booking Lead Time",     f"{int(avg_lead_time)}d","days before flight"),
        ("High-Intent Customers",     f"{high_extras_pct}%",   "selected 2+ extras"),
        ("Est. Revenue at Risk",      est_revenue_risk,        "from drop-offs @ $450/ticket"),
    ]

    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_gauge)], style={"width": "48%", "background": WHITE,
                "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}),
            html.Div([dcc.Graph(figure=fig_conv)],  style={"width": "48%", "background": WHITE,
                "borderRadius": "10px", "boxShadow": "0 2px 8px rgba(0,0,0,0.08)"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "20px"}),

        html.H4("Full KPI Table", style={"color": NAVY, "marginBottom": "12px"}),
        html.Div([
            html.Div([
                html.Div(str(kpi[1]), style={"fontSize": "1.6rem", "fontWeight": "800", "color": TEAL}),
                html.Div(kpi[0], style={"fontWeight": "600", "color": NAVY, "fontSize": "0.85rem"}),
                html.Div(kpi[2], style={"color": SLATE, "fontSize": "0.75rem"})
            ], style={
                "background": WHITE, "borderRadius": "8px", "padding": "14px 16px",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.06)", "minWidth": "150px"
            })
            for kpi in kpis
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"})
    ])

# ── Callback ──────────────────────────────────────────────────────────────────
@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-voice":
        return render_voice_tab()
    elif tab == "tab-booking":
        return render_booking_tab()
    elif tab == "tab-kpi":
        return render_kpi_tab()

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
