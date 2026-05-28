# app/frontend/utils.py
import sys
import os

# Add project root to Python path so 'app', 'pipelines' etc are importable
# This is needed because Streamlit runs files directly, not as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pandas as pd
import streamlit as st

# ── API config ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")
PREDICT_URL  = f"{API_BASE_URL}/predict"
HEALTH_URL   = f"{API_BASE_URL}/health"

# ── Data path ──────────────────────────────────────────────
FEATURED_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "airbnb_featured.csv"
)

# ── Brand colors ───────────────────────────────────────────
PRIMARY   = "#FF5A5F"
SECONDARY = "#00A699"
DARK      = "#484848"
LIGHT     = "#F7F7F7"

# ── Options ────────────────────────────────────────────────
BOROUGHS   = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room"]


def set_page_config(title: str) -> None:
    st.set_page_config(
        page_title=f"{title} | Airbnb Intelligence",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(FEATURED_PATH, low_memory=False)
    return df


def metric_card(label: str, value: str, delta: str = "") -> None:
    st.metric(label=label, value=value, delta=delta)


def check_api_health() -> bool:
    import requests
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.status_code == 200
    except Exception:
        return False