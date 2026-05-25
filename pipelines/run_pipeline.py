# pipelines/run_pipeline.py
# Master pipeline runner — now includes feature engineering as Step 3.

import os
import pandas as pd
from pipelines.data_loader import load_raw_data
from pipelines.data_cleaner import run_cleaning_pipeline, validate_cleaned_data
from pipelines.feature_engineer import run_feature_engineering, print_feature_summary
from pipelines.db_loader import load_to_postgres, verify_table_load
from pipelines.logger import logger


PROCESSED_DATA_PATH = os.path.join("data", "processed", "airbnb_cleaned.csv")
FEATURED_DATA_PATH  = os.path.join("data", "processed", "airbnb_featured.csv")


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Saves a DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info(f"Saved to: {path}")
    print(f"\n Saved: {path}")


def run_full_pipeline() -> pd.DataFrame:

    print("\n" + "=" * 55)
    print("   AIRBNB FULL PIPELINE — STARTING")
    print("=" * 55 + "\n")

    # ── 1. Load ────────────────────────────────────────────
    logger.info("STEP 1: Loading raw data...")
    df_raw = load_raw_data()
    print(f"\n Raw data loaded: {len(df_raw):,} rows")

    # ── 2. Clean ───────────────────────────────────────────
    logger.info("STEP 2: Cleaning...")
    df_clean = run_cleaning_pipeline(df_raw)
    validate_cleaned_data(df_clean)
    save_dataframe(df_clean, PROCESSED_DATA_PATH)

    # ── 3. Feature Engineering ─────────────────────────────
    logger.info("STEP 3: Feature engineering...")
    df_featured = run_feature_engineering(df_clean)
    print_feature_summary(df_featured)
    save_dataframe(df_featured, FEATURED_DATA_PATH)

    # ── 4. Load to PostgreSQL ──────────────────────────────
    logger.info("STEP 4: Loading to PostgreSQL...")
    load_to_postgres(df_clean,    "cleaned_listings",  if_exists="replace")
    load_to_postgres(df_featured, "featured_listings", if_exists="replace")
    verify_table_load("featured_listings")

    print("\n" + "=" * 55)
    print("   PIPELINE COMPLETE")
    print("=" * 55 + "\n")

    return df_featured


if __name__ == "__main__":
    df = run_full_pipeline()