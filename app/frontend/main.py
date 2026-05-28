# app/frontend/main.py
# Home page — project overview, key stats, navigation guide.
# Run with: streamlit run app/frontend/main.py

import streamlit as st
import pandas as pd
from app.frontend.utils import set_page_config, load_data, check_api_health
from app.frontend.utils import PRIMARY, SECONDARY

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

set_page_config("Home")

# ── Header ─────────────────────────────────────────────────
st.markdown(
    f"""
    <div style='background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
                padding: 2.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
        <h1 style='color: white; margin: 0; font-size: 2.2rem;'>
            🏠 Airbnb Market Intelligence
        </h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0;
                  font-size: 1.1rem;'>
            NYC 2019 · Price Prediction · Market Analytics · Host Insights
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ── API Status ─────────────────────────────────────────────
api_ok = check_api_health()
if api_ok:
    st.success("✅ Prediction API is online and ready.")
else:
    st.warning(
        "⚠️ Prediction API is offline. "
        "Start it with: uvicorn app.backend.main:app --port 8000"
    )

st.markdown("---")

# ── Dataset KPIs ───────────────────────────────────────────
df = load_data()

st.subheader("📊 Dataset at a Glance")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Listings", f"{len(df):,}")
with col2:
    st.metric("Unique Hosts", f"{df['host_id'].nunique():,}")
with col3:
    st.metric("Neighbourhoods", f"{df['neighbourhood'].nunique()}")
with col4:
    st.metric("Avg Price/Night", f"${df['price'].mean():.0f}")
with col5:
    st.metric("Median Price/Night", f"${df['price'].median():.0f}")

st.markdown("---")

# ── Borough Summary Table ──────────────────────────────────
st.subheader("🗺️ Market Summary by Borough")

borough_summary = (
    df.groupby("neighbourhood_group")
    .agg(
        Listings=("id", "count"),
        Avg_Price=("price", "mean"),
        Median_Price=("price", "median"),
        Avg_Availability=("availability_365", "mean"),
    )
    .round(2)
    .sort_values("Avg_Price", ascending=False)
    .reset_index()
    .rename(columns={"neighbourhood_group": "Borough"})
)

st.dataframe(
    borough_summary,
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ── Navigation Guide ───────────────────────────────────────
st.subheader("🧭 What Can You Do Here?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📈 Market Analytics**
    Explore price distributions, room type breakdowns,
    availability trends, and top neighbourhoods.

    **🏘️ Neighbourhood Explorer**
    Compare any two boroughs or neighbourhoods
    side by side on price, reviews, and availability.
    """)

with col2:
    st.markdown("""
    **💰 Price Prediction**
    Enter your listing details and get an instant
    AI-powered price recommendation with a confidence range.

    **🤖 Model Performance**
    See how the XGBoost model was built, what features
    it uses, and how accurate it is.
    """)

st.markdown("---")
st.caption(
    "Built by Elisha (Elice99) · "
    "Airbnb NYC 2019 Dataset · "
    "XGBoost · FastAPI · Streamlit"
)