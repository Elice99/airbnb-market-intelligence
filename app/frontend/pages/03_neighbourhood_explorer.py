# pages/03_neighbourhood_explorer.py
# Compare any two neighbourhoods or boroughs side by side.
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from app.frontend.utils import set_page_config, load_data, PRIMARY, SECONDARY

set_page_config("Neighbourhood Explorer")

st.title("🏘️ Neighbourhood Explorer")
st.caption("Compare any two areas side by side.")

df = load_data()

# ── Borough Selector ───────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Area A")
    borough_a = st.selectbox(
        "Borough A",
        df["neighbourhood_group"].unique(),
        index=0,
        key="borough_a"
    )
    neighbourhoods_a = sorted(
        df[df["neighbourhood_group"] == borough_a]["neighbourhood"]
        .unique().tolist()
    )
    neighbourhood_a = st.selectbox(
        "Neighbourhood A",
        neighbourhoods_a,
        key="neigh_a"
    )

with col2:
    st.subheader("Area B")
    borough_b = st.selectbox(
        "Borough B",
        df["neighbourhood_group"].unique(),
        index=1,
        key="borough_b"
    )
    neighbourhoods_b = sorted(
        df[df["neighbourhood_group"] == borough_b]["neighbourhood"]
        .unique().tolist()
    )
    neighbourhood_b = st.selectbox(
        "Neighbourhood B",
        neighbourhoods_b,
        key="neigh_b"
    )

st.markdown("---")

# ── Filter data for each area ──────────────────────────────
df_a = df[df["neighbourhood"] == neighbourhood_a]
df_b = df[df["neighbourhood"] == neighbourhood_b]

# ── Side by Side KPIs ──────────────────────────────────────
st.subheader("📊 Key Metrics Comparison")

metrics = {
    "Total Listings"   : (len(df_a), len(df_b)),
    "Avg Price"        : (
        f"${df_a['price'].mean():.0f}",
        f"${df_b['price'].mean():.0f}"
    ),
    "Median Price"     : (
        f"${df_a['price'].median():.0f}",
        f"${df_b['price'].median():.0f}"
    ),
    "Avg Availability" : (
        f"{df_a['availability_365'].mean():.0f} days",
        f"{df_b['availability_365'].mean():.0f} days"
    ),
    "Avg Reviews"      : (
        f"{df_a['number_of_reviews'].mean():.1f}",
        f"{df_b['number_of_reviews'].mean():.1f}"
    ),
}

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    st.markdown("**Metric**")
with col2:
    st.markdown(f"**{neighbourhood_a}**")
with col3:
    st.markdown(f"**{neighbourhood_b}**")

st.markdown("---")

for metric, (val_a, val_b) in metrics.items():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(metric)
    with col2:
        st.write(val_a)
    with col3:
        st.write(val_b)

# ── Price Distribution Comparison ─────────────────────────
st.markdown("---")
st.subheader("Price Distribution Comparison")

fig = go.Figure()

fig.add_trace(go.Histogram(
    x=df_a["price"],
    name=neighbourhood_a,
    opacity=0.7,
    marker_color=PRIMARY,
    nbinsx=40
))

fig.add_trace(go.Histogram(
    x=df_b["price"],
    name=neighbourhood_b,
    opacity=0.7,
    marker_color=SECONDARY,
    nbinsx=40
))

fig.update_layout(
    barmode="overlay",
    xaxis_title="Price ($)",
    yaxis_title="Number of Listings",
    plot_bgcolor="white",
    legend=dict(orientation="h", y=1.1),
    margin=dict(t=30, b=20)
)

st.plotly_chart(fig, width="stretch")

# ── Room Type Mix ──────────────────────────────────────────
st.subheader("Room Type Mix")

col1, col2 = st.columns(2)

for col, area_df, name in [
    (col1, df_a, neighbourhood_a),
    (col2, df_b, neighbourhood_b)
]:
    with col:
        room_counts = area_df["room_type"].value_counts().reset_index()
        room_counts.columns = ["Room Type", "Count"]
        fig = px.pie(
            room_counts,
            names="Room Type",
            values="Count",
            title=name,
            color_discrete_sequence=[PRIMARY, SECONDARY, "#FFB400"],
            hole=0.4
        )
        fig.update_layout(margin=dict(t=40, b=10))
        st.plotly_chart(fig, width="stretch")