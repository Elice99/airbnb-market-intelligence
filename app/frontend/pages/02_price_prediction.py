# pages/02_price_prediction.py
# The main prediction interface — host enters listing details,
# gets a price recommendation back from the FastAPI backend.
import sys
import os

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
)

import streamlit as st
import requests
import pandas as pd

from app.frontend.utils import (
    set_page_config, load_data, check_api_health,
    PREDICT_URL, BOROUGHS, ROOM_TYPES, PRIMARY, SECONDARY
)

set_page_config("Price Prediction")

st.title("💰 Price Prediction")
st.caption(
    "Enter your listing details below to get an AI-powered "
    "price recommendation."
)

# ── API Status Check ───────────────────────────────────────
if not check_api_health():
    st.error(
        "❌ The prediction API is offline. "
        "Please start it with: "
        "`uvicorn app.backend.main:app --port 8000`"
    )
    st.stop()

df = load_data()

# ── Input Form ─────────────────────────────────────────────
st.subheader("🏠 Your Listing Details")

col1, col2 = st.columns(2)

with col1:
    borough = st.selectbox("Borough", BOROUGHS)

    # Filter neighbourhoods to selected borough
    neighbourhoods = sorted(
        df[df["neighbourhood_group"] == borough]["neighbourhood"]
        .unique()
        .tolist()
    )
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)

    room_type = st.selectbox("Room Type", ROOM_TYPES)

    minimum_nights = st.slider(
        "Minimum Nights Required",
        min_value=1, max_value=45, value=2
    )

with col2:
    number_of_reviews = st.number_input(
        "Number of Reviews",
        min_value=0, max_value=629, value=10, step=1
    )

    reviews_per_month = st.number_input(
        "Reviews per Month",
        min_value=0.0, max_value=58.5, value=1.0, step=0.1
    )

    host_listings = st.number_input(
        "Your Total Listings Count",
        min_value=1, max_value=327, value=1, step=1
    )

    availability = st.slider(
        "Days Available per Year",
        min_value=0, max_value=365, value=180
    )

listing_name_length = st.slider(
    "Listing Name Length (characters)",
    min_value=5, max_value=150, value=45,
    help="Longer, descriptive names may signal more effort."
)

st.markdown("---")

# ── Predict Button ─────────────────────────────────────────
if st.button("🔮 Get Price Recommendation", type="primary"):

    payload = {
        "neighbourhood_group"             : borough,
        "neighbourhood"                   : neighbourhood,
        "room_type"                       : room_type,
        "minimum_nights"                  : minimum_nights,
        "number_of_reviews"               : number_of_reviews,
        "reviews_per_month"               : reviews_per_month,
        "calculated_host_listings_count"  : host_listings,
        "availability_365"                : availability,
        "name_length"                     : listing_name_length,
    }

    with st.spinner("Getting prediction..."):
        try:
            response = requests.post(PREDICT_URL, json=payload, timeout=10)
            result   = response.json()

            if response.status_code == 200:

                pred  = result["predicted_price"]
                low   = result["price_range_low"]
                high  = result["price_range_high"]

                # ── Result Display ─────────────────────────
                st.markdown("---")
                st.subheader("💡 Your Price Recommendation")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Recommended Price",
                        f"${pred:.2f}",
                        help="XGBoost model prediction"
                    )
                with col2:
                    st.metric(
                        "Lower Estimate",
                        f"${low:.2f}",
                        help="Predicted price minus model MAE"
                    )
                with col3:
                    st.metric(
                        "Upper Estimate",
                        f"${high:.2f}",
                        help="Predicted price plus model MAE"
                    )

                # ── Neighbourhood Benchmark ────────────────
                st.markdown("---")
                st.subheader("📊 How You Compare to Your Neighbourhood")

                neigh_df = df[
                    (df["neighbourhood"] == neighbourhood) &
                    (df["room_type"] == room_type)
                ]["price"]

                if len(neigh_df) >= 5:
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Neighbourhood Median",
                            f"${neigh_df.median():.0f}"
                        )
                    with col2:
                        st.metric(
                            "Neighbourhood Avg",
                            f"${neigh_df.mean():.0f}"
                        )
                    with col3:
                        st.metric("Cheapest", f"${neigh_df.min():.0f}")
                    with col4:
                        st.metric("Most Expensive", f"${neigh_df.max():.0f}")

                    # Gauge: where does prediction sit?
                    pct = (neigh_df < pred).mean() * 100
                    st.info(
                        f"Your predicted price of **${pred:.0f}** is higher "
                        f"than **{pct:.0f}%** of {room_type} listings "
                        f"in {neighbourhood}."
                    )

                else:
                    st.info(
                        f"Not enough listings in {neighbourhood} "
                        f"for {room_type} to benchmark against."
                    )

            else:
                st.error(f"API Error: {result.get('detail', 'Unknown error')}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Cannot connect to the prediction API. "
                "Make sure it's running on port 8000."
            )
        except Exception as e:
            st.error(f"Something went wrong: {e}")