# pages/04_model_performance.py
# Shows model metrics, feature importance, and evaluation charts.

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
from app.frontend.utils import set_page_config, PRIMARY, SECONDARY


set_page_config("Model Performance")

st.title("🤖 Model Performance")
st.caption("How the XGBoost price prediction model was built and evaluated.")

# ── Model Metrics ──────────────────────────────────────────
st.subheader("📊 Model Comparison")

results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "XGBoost ✓"
    ],
    "RMSE ($)": [90.58, 83.83, 83.10],
    "MAE ($)" : [53.19, 47.43, 47.20],
    "R²"      : [0.3529, 0.4457, 0.4553],
})

st.dataframe(results, width="stretch", hide_index=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Best Model RMSE", "$83.10",
              help="Average prediction error in dollars")
with col2:
    st.metric("Best Model MAE", "$47.20",
              help="Mean absolute error in dollars")
with col3:
    st.metric("Best Model R²", "0.4553",
              help="45.5% of price variance explained")

st.markdown("---")

# ── Cross Validation ───────────────────────────────────────
st.subheader("🔁 Cross-Validation Results (5-Fold)")
st.caption(
    "5-fold CV splits training data 5 ways to verify the model "
    "isn't just getting lucky on one split."
)

cv_data = pd.DataFrame({
    "Fold"     : ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
    "RMSE ($)" : [82.44, 80.56, 78.98, 79.48, 81.78],
})

fig = px.bar(
    cv_data, x="Fold", y="RMSE ($)",
    color_discrete_sequence=[PRIMARY],
    text=cv_data["RMSE ($)"].apply(lambda x: f"${x:.2f}"),
)
fig.add_hline(
    y=80.65,
    line_dash="dash",
    line_color=SECONDARY,
    annotation_text="Mean RMSE: $80.65",
    annotation_position="top right"
)
fig.update_traces(textposition="outside")
fig.update_layout(plot_bgcolor="white", margin=dict(t=30, b=20))
st.plotly_chart(fig, width="stretch")

st.info("Mean CV RMSE: **$80.65** ±$1.31 — low variance means the model is consistent across different data splits.")

st.markdown("---")

# ── Feature Importance ─────────────────────────────────────
st.subheader("🎯 Feature Importance")
st.caption("Which features drive price predictions the most.")

features = pd.DataFrame({
    "Feature": [
        "room_type",
        "neighbourhood_group_median_price",
        "neighbourhood_median_price",
        "is_long_term",
        "host_is_superhost_proxy",
        "availability_365",
        "availability_ratio",
        "minimum_nights",
        "number_of_reviews",
        "name_length",
        "reviews_per_month",
        "calculated_host_listings_count",
        "is_reviewed",
        "neighbourhood_group",
        "neighbourhood",
        "days_since_last_review",
    ],
    "Importance": [
        0.4318, 0.1290, 0.1071, 0.0532, 0.0490,
        0.0463, 0.0380, 0.0194, 0.0150, 0.0140,
        0.0130, 0.0120, 0.0100, 0.0080, 0.0071,
        0.0070,
    ]
}).sort_values("Importance", ascending=True)

fig = px.bar(
    features, x="Importance", y="Feature",
    orientation="h",
    color="Importance",
    color_continuous_scale=[[0, SECONDARY], [1, PRIMARY]],
    text=features["Importance"].apply(lambda x: f"{x:.3f}"),
)
fig.update_traces(textposition="outside")
fig.update_layout(
    coloraxis_showscale=False,
    plot_bgcolor="white",
    margin=dict(t=20, b=20),
    height=500
)
st.plotly_chart(fig, width="stretch")

st.markdown("---")

# ── Model Notes ────────────────────────────────────────────
st.subheader("📝 Model Notes")

st.markdown("""
**Why R² of 0.45?**

This dataset contains no bedroom count, amenities, photos,
or bathroom information — all major price drivers on Airbnb.
The model predicts from borough, room type, host history,
and availability alone.

With metadata-only features, 45% explained variance is the
realistic ceiling for this dataset. Room type alone accounts
for 43% of all feature importance — because it's the only
real size signal available.

**Cross-validation confirms reliability.**
Mean RMSE of $80.65 ±$1.31 across 5 folds means
the model performs consistently, not just on one lucky split.

**Further improvement would require:**
- Number of bedrooms and bathrooms
- Amenities list (pool, gym, kitchen, WiFi)
- Official superhost status
- Proximity to subway and landmarks
- Seasonal pricing data
""")