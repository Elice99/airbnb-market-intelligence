# pipelines/data_cleaner.py
# Contains every cleaning function for the Airbnb dataset.
# Each function does ONE job and can be tested independently.
# The main function run_cleaning_pipeline() chains them all together.

import pandas as pd
from pipelines.logger import logger


# ─────────────────────────────────────────
# 1. DROP ZERO-PRICE LISTINGS
# ─────────────────────────────────────────

def drop_zero_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes listings where price is 0.
    A $0 listing is invalid, it breaks ML models and skews averages.

    The inspection found 11 of these rows.
    """

    before = len(df)

    # Keep only rows where price is greater than 0
    df = df[df["price"] > 0].copy()

    dropped = before - len(df)
    logger.info(f"drop_zero_prices: Dropped {dropped} rows with price = 0.")

    return df


# ─────────────────────────────────────────
# 2. CAP PRICE OUTLIERS
# ─────────────────────────────────────────

def cap_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Caps extreme price values at the 99th percentile.

    Why cap instead of drop?
    - Dropping removes real data (some listings ARE expensive).
    - Capping keeps the row but limits its influence on the model.
    - The 99th percentile is a standard industry threshold.

    The inspection showed max price = $10,000 which is clearly an outlier.
    The mean was $152 and median $106 — so $10,000 distorts everything.
    """

    # Calculate the 99th percentile value
    cap_value = df["price"].quantile(0.99)

    logger.info(
        f"cap_price_outliers: Capping price at 99th percentile "
        f"= ${cap_value:,.2f}"
    )

    # Replace any price above the cap with the cap value
    df["price"] = df["price"].clip(upper=cap_value)

    return df


# ─────────────────────────────────────────
# 3. CAP MINIMUM NIGHTS OUTLIERS
# ─────────────────────────────────────────

def cap_minimum_nights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Caps extreme minimum_nights values at the 99th percentile.

    The inspection showed a max of 1,250 nights — that's 3.4 years.
    This is either a data entry error or an unusual edge case.
    Either way it would badly distort our ML model.
    """

    cap_value = df["minimum_nights"].quantile(0.99)

    logger.info(
        f"cap_minimum_nights: Capping minimum_nights at 99th percentile "
        f"= {cap_value}"
    )

    df["minimum_nights"] = df["minimum_nights"].clip(upper=cap_value)

    return df


# ─────────────────────────────────────────
# 4. FILL MISSING TEXT COLUMNS
# ─────────────────────────────────────────

def fill_missing_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in text columns with a placeholder.

    - name        : 16 missing  → fill with 'Unknown'
    - host_name   : 21 missing  → fill with 'Unknown'

    We don't drop these rows — the listing data is still valid
    even if the name is missing.
    """

    # fillna() replaces NaN with the value you provide
    df["name"] = df["name"].fillna("Unknown")
    df["host_name"] = df["host_name"].fillna("Unknown")

    logger.info("fill_missing_text: Filled missing name and host_name.")

    return df


# ─────────────────────────────────────────
# 5. FIX REVIEWS COLUMNS
# ─────────────────────────────────────────

def fix_reviews_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles the 10,052 rows missing reviews_per_month and last_review.

    Why are they missing?
    These are listings that have NEVER been reviewed.
    So the correct fill is:
    - reviews_per_month → 0  (zero reviews per month, not unknown)
    - last_review       → 'No Review'  (placeholder before date conversion)
    """

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df["last_review"] = df["last_review"].fillna("No Review")

    logger.info(
        "fix_reviews_columns: Filled 10,052 missing review values."
    )

    return df


# ─────────────────────────────────────────
# 6. CONVERT LAST_REVIEW TO DATETIME
# ─────────────────────────────────────────

def convert_last_review_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the last_review column from a string to a proper datetime.

    The inspection showed it was stored as dtype 'str' with values
    like '2019-06-23'. We need it as a real date for time-based analysis.

    Rows with 'No Review' will become NaT (Not a Time) — that's fine,
    it correctly represents "no date available".

    errors='coerce' means: if a value can't be parsed as a date,
    set it to NaT instead of crashing.
    """

    df["last_review"] = pd.to_datetime(
        df["last_review"],
        errors="coerce"
    )

    # Count how many converted successfully vs became NaT
    valid_dates = df["last_review"].notna().sum()
    nat_count = df["last_review"].isna().sum()

    logger.info(
        f"convert_last_review_date: "
        f"{valid_dates:,} valid dates | {nat_count:,} NaT (no review)."
    )

    return df


# ─────────────────────────────────────────
# 7. STRIP WHITESPACE FROM TEXT COLUMNS
# ─────────────────────────────────────────

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes leading and trailing whitespace from all text columns.

    Example: '  Manhattan  ' becomes 'Manhattan'

    This prevents issues like groupby treating
    'Manhattan' and ' Manhattan' as different categories.
    """

    # Get all string/object columns
    str_cols = df.select_dtypes(include=["object", "str"]).columns

    for col in str_cols:
        # .str.strip() removes whitespace from both ends of each string
        df[col] = df[col].str.strip()

    logger.info(
        f"strip_whitespace: Stripped whitespace from "
        f"{len(str_cols)} text columns."
    )

    return df


# ─────────────────────────────────────────
# 8. STANDARDIZE TEXT CASE
# ─────────────────────────────────────────

def standardize_text_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the case of key categorical columns.

    Why this matters:
    'entire home/apt' and 'Entire home/apt' would be treated
    as two different categories by our ML model.

    We use title case for neighbourhood columns and
    keep room_type as-is (it's already consistent).
    """

    # Title case: 'bedford-stuyvesant' → 'Bedford-Stuyvesant'
    df["neighbourhood"] = df["neighbourhood"].str.title()
    df["neighbourhood_group"] = df["neighbourhood_group"].str.title()

    logger.info("standardize_text_case: Standardized neighbourhood columns.")

    return df


# ─────────────────────────────────────────
# 9. RESET INDEX
# ─────────────────────────────────────────

def reset_dataframe_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resets the DataFrame index after dropping rows.

    When we drop rows (like zero-price listings), the index
    has gaps: 0, 1, 3, 4... (missing 2).
    Resetting gives us a clean 0, 1, 2, 3... index.

    drop=True means we don't add the old index as a column.
    """

    df = df.reset_index(drop=True)
    logger.info("reset_dataframe_index: Index reset.")

    return df


# ─────────────────────────────────────────
# 10. FINAL VALIDATION
# ─────────────────────────────────────────

def validate_cleaned_data(df: pd.DataFrame) -> None:
    """
    Runs a final check on the cleaned dataset.
    Prints a summary so we can confirm everything looks right
    before saving.
    """

    logger.info("=== VALIDATION REPORT ===")

    print("\n" + "=" * 50)
    print("   CLEANED DATASET — VALIDATION REPORT")
    print("=" * 50)

    print(f"\n Rows             : {len(df):,}")
    print(f" Columns          : {df.shape[1]}")

    # Check remaining missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print(" Missing values   : None ✓")
    else:
        print(f" Missing values   : {len(missing)} columns still have nulls")
        print(missing)

    # Check price range
    print(f"\n Price min        : ${df['price'].min():,.2f}")
    print(f" Price max        : ${df['price'].max():,.2f}")
    print(f" Price mean       : ${df['price'].mean():,.2f}")

    # Check minimum_nights range
    print(f"\n Min nights max   : {df['minimum_nights'].max()}")

    # Check room_type values
    print(f"\n Room types       : {df['room_type'].unique().tolist()}")

    # Check neighbourhood_group values
    print(
        f"\n Neighbourhoods   : "
        f"{df['neighbourhood_group'].unique().tolist()}"
    )

    print("\n" + "=" * 50 + "\n")


# ─────────────────────────────────────────
# MAIN PIPELINE — CHAINS ALL STEPS
# ─────────────────────────────────────────

def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all cleaning steps in the correct order.

    Order matters:
    1. Drop invalid rows first (zero prices)
    2. Cap outliers
    3. Fill missing values
    4. Fix data types
    5. Standardize text
    6. Reset index last (after all drops)
    """

    logger.info("Starting cleaning pipeline...")

    # Step 1 — Remove invalid listings
    df = drop_zero_prices(df)

    # Step 2 — Cap outliers
    df = cap_price_outliers(df)
    df = cap_minimum_nights(df)

    # Step 3 — Fill missing values
    df = fill_missing_text(df)
    df = fix_reviews_columns(df)

    # Step 4 — Fix data types
    df = convert_last_review_date(df)

    # Step 5 — Standardize text
    df = strip_whitespace(df)
    df = standardize_text_case(df)

    # Step 6 — Clean up index
    df = reset_dataframe_index(df)

    logger.info(
        f"Cleaning pipeline complete. "
        f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns."
    )

    return df