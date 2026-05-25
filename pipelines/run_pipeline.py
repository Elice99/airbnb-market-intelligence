# pipelines/run_pipeline.py
# This is the master script that runs the full pipeline:
# Load raw data → Clean → Save to CSV → Save to PostgreSQL

import os
import pandas as pd
from pipelines.data_loader import load_raw_data
from pipelines.data_cleaner import run_cleaning_pipeline, validate_cleaned_data
from pipelines.db_loader import load_to_postgres, verify_table_load
from pipelines.logger import logger


# Where to save the cleaned CSV
PROCESSED_DATA_PATH = os.path.join("data", "processed", "airbnb_cleaned.csv")


def save_cleaned_csv(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    """
    Saves the cleaned DataFrame to a CSV file.

    index=False means we don't write the row numbers as a column.
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    df.to_csv(path, index=False)

    logger.info(f"Cleaned data saved to: {path}")
    print(f"\n Cleaned CSV saved to: {path}")


def run_full_pipeline() -> pd.DataFrame:
    """
    Master function — runs the complete pipeline end to end.
    """

    print("\n" + "=" * 55)
    print("   AIRBNB CLEANING PIPELINE — STARTING")
    print("=" * 55 + "\n")

    # ── 1. Load raw data ──────────────────────────────────
    logger.info("STEP 1: Loading raw data...")
    df_raw = load_raw_data()
    print(f"\n Raw data loaded: {len(df_raw):,} rows")

    # ── 2. Run cleaning ───────────────────────────────────
    logger.info("STEP 2: Running cleaning pipeline...")
    df_clean = run_cleaning_pipeline(df_raw)
    print(f" Cleaned data  : {len(df_clean):,} rows")
    print(f" Rows removed  : {len(df_raw) - len(df_clean):,}")

    # ── 3. Validate ───────────────────────────────────────
    logger.info("STEP 3: Validating cleaned data...")
    validate_cleaned_data(df_clean)

    # ── 4. Save cleaned CSV ───────────────────────────────
    logger.info("STEP 4: Saving cleaned CSV...")
    save_cleaned_csv(df_clean)

    # ── 5. Load into PostgreSQL ───────────────────────────
    logger.info("STEP 5: Loading into PostgreSQL...")
    load_to_postgres(df_clean, table_name="cleaned_listings", if_exists="replace")
    verify_table_load("cleaned_listings")

    print("\n" + "=" * 55)
    print("   PIPELINE COMPLETE")
    print("=" * 55 + "\n")

    return df_clean


# Run directly: python pipelines/run_pipeline.py
if __name__ == "__main__":
    df = run_full_pipeline()