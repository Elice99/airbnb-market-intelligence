# pages/01_market_analytics.py
# Interactive market analytics — price charts, distributions,
# room type breakdowns, top neighbourhoods.
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.frontend.utils import set_page_config, load_data, PRIMARY, SECONDARY

set_page_config("Market Analytics")

st.title("📈 Market Analytics")
st.caption("Explore pricing patterns across NYC's Airbnb market.")

df = load_data()

# ── Sidebar Filters ────────────────────────────────────────
st.sidebar.header("🔍 Filters")

boroughs = st.sidebar.multiselect(
    "Borough",
    options=df["neighbourhood_group"].unique().tolist(),
    default=df["neighbourhood_group"].unique().tolist()
)

room_types = st.sidebar.multiselect(
    "Room Type",
    options=df["room_type"].unique().tolist(),
    default=df["room_type"].unique().tolist()
)

price_range = st.sidebar.slider(
    "Price Range ($)",
    min_value=int(df["price"].min()),
    max_value=int(df["price"].max()),
    value=(10, 400)
)

# Apply filters
mask = (
    df["neighbourhood_group"].isin(boroughs) &
    df["room_type"].isin(room_types) &
    df["price"].between(price_range[0], price_range[1])
)
filtered = df[mask]

st.markdown(f"Showing **{len(filtered):,}** listings after filters.")
st.markdown("---")

# ── Row 1: Price Distribution + Room Type ─────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Price Distribution")
    fig = px.histogram(
        filtered, x="price", nbins=60,
        color_discrete_sequence=[PRIMARY],
        labels={"price": "Price ($)", "count": "Listings"},
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=20, b=20),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, width="stretch")

with col2:
    st.subheader("Listings by Room Type")
    room_counts = filtered["room_type"].value_counts().reset_index()
    room_counts.columns = ["Room Type", "Count"]
    fig = px.pie(
        room_counts, names="Room Type", values="Count",
        color_discrete_sequence=[PRIMARY, SECONDARY, "#FFB400"],
        hole=0.4
    )
    fig.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

# ── Row 2: Avg Price by Borough ────────────────────────────
st.subheader("Average Price by Borough")

borough_price = (
    filtered.groupby("neighbourhood_group")["price"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
borough_price.columns = ["Borough", "Avg Price"]

fig = px.bar(
    borough_price, x="Borough", y="Avg Price",
    color="Avg Price",
    color_continuous_scale=[[0, SECONDARY], [1, PRIMARY]],
    text=borough_price["Avg Price"].apply(lambda x: f"${x:.0f}"),
    labels={"Avg Price": "Average Price ($)"}
)
fig.update_traces(textposition="outside")
fig.update_layout(
    showlegend=False,
    plot_bgcolor="white",
    margin=dict(t=20, b=20),
    coloraxis_showscale=False
)
st.plotly_chart(fig, width="stretch")

# ── Row 3: Top 15 Neighbourhoods ───────────────────────────
st.subheader("Top 15 Neighbourhoods by Average Price")
st.caption("Minimum 30 listings required to appear.")

counts = filtered["neighbourhood"].value_counts()
valid  = counts[counts >= 30].index

top15 = (
    filtered[filtered["neighbourhood"].isin(valid)]
    .groupby("neighbourhood")["price"]
    .mean()
    .sort_values(ascending=False)
    .head(15)
    .reset_index()
)
top15.columns = ["Neighbourhood", "Avg Price"]

fig = px.bar(
    top15, x="Avg Price", y="Neighbourhood",
    orientation="h",
    color="Avg Price",
    color_continuous_scale=[[0, SECONDARY], [1, PRIMARY]],
    text=top15["Avg Price"].apply(lambda x: f"${x:.0f}"),
)
fig.update_traces(textposition="outside")
fig.update_layout(
    yaxis=dict(autorange="reversed"),
    plot_bgcolor="white",
    margin=dict(t=20, b=20),
    coloraxis_showscale=False
)
st.plotly_chart(fig, width="stretch")

# ── Row 4: Price by Room Type per Borough ─────────────────
st.subheader("Average Price by Room Type and Borough")

pivot = (
    filtered.groupby(["neighbourhood_group", "room_type"])["price"]
    .mean()
    .reset_index()
)

fig = px.bar(
    pivot,
    x="neighbourhood_group",
    y="price",
    color="room_type",
    barmode="group",
    color_discrete_sequence=[PRIMARY, SECONDARY, "#FFB400"],
    labels={
        "neighbourhood_group": "Borough",
        "price": "Avg Price ($)",
        "room_type": "Room Type"
    }
)
fig.update_layout(plot_bgcolor="white", margin=dict(t=20, b=20))
st.plotly_chart(fig, width="stretch")