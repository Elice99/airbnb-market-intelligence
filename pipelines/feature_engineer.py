# pipelines/feature_engineer.py
# Builds all engineered features on top of the cleaned dataset.
# Every feature has a clear business reason and ML justification.
# Run AFTER data_cleaner.py — this expects a clean DataFrame.

import pandas as pd
import numpy as np
from pipelines.logger import logger


# Reference date for calculating "days since last review"
# We use the latest date in the dataset (July 2019) as our anchor
REFERENCE_DATE = pd.Timestamp("2019-07-09")


# ─────────────────────────────────────────
# 1. PRICE PER REVIEW
# ─────────────────────────────────────────

def add_price_per_review(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: price_per_review = price / number_of_reviews

    Business meaning:
    A $200 listing with 100 reviews is a better value signal
    than a $200 listing with 0 reviews.

    Edge case: listings with 0 reviews would cause division by zero.
    We replace those with 0 using np.where.
    """

    df["price_per_review"] = np.where(
        df["number_of_reviews"] > 0,                    # condition
        df["price"] / df["number_of_reviews"],          # if true
        0                                               # if false (no reviews)
    )

    # Round to 2 decimal places
    df["price_per_review"] = df["price_per_review"].round(2)

    logger.info("add_price_per_review: Created price_per_review column.")

    return df


# ─────────────────────────────────────────
# 2. IS REVIEWED FLAG
# ─────────────────────────────────────────

def add_is_reviewed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: is_reviewed = 1 if the listing has at least 1 review, else 0

    ML meaning:
    Whether a listing has been reviewed at all is a strong signal.
    A listing with zero reviews behaves very differently from one
    with even a single review — it may be new, inactive, or overpriced.
    """

    # astype(int) converts True/False to 1/0
    df["is_reviewed"] = (df["number_of_reviews"] > 0).astype(int)

    reviewed_count = df["is_reviewed"].sum()
    logger.info(
        f"add_is_reviewed: {reviewed_count:,} listings have reviews "
        f"({reviewed_count / len(df) * 100:.1f}%)."
    )

    return df


# ─────────────────────────────────────────
# 3. HOST IS SUPERHOST PROXY
# ─────────────────────────────────────────

def add_host_superhost_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: host_is_superhost_proxy = 1 if host manages 5+ listings

    We don't have an official superhost flag in this dataset.
    But hosts with many listings are likely professional operators
    (property management companies, serial hosts) — they price
    and manage differently from a person renting one spare room.

    Threshold of 5 is a common industry benchmark.
    """

    df["host_is_superhost_proxy"] = (
        df["calculated_host_listings_count"] >= 5
    ).astype(int)

    superhost_count = df["host_is_superhost_proxy"].sum()
    logger.info(
        f"add_host_superhost_proxy: {superhost_count:,} proxy superhosts "
        f"({superhost_count / len(df) * 100:.1f}%)."
    )

    return df


# ─────────────────────────────────────────
# 4. AVAILABILITY RATIO
# ─────────────────────────────────────────

def add_availability_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: availability_ratio = availability_365 / 365

    Converts raw days available into a 0–1 ratio.
    0.0 = fully booked all year
    1.0 = available every single day

    Ratios are easier for ML models to work with than raw day counts
    because they're already normalized between 0 and 1.
    """

    df["availability_ratio"] = (df["availability_365"] / 365).round(4)

    logger.info("add_availability_ratio: Created availability_ratio column.")

    return df


# ─────────────────────────────────────────
# 5. AVAILABILITY CATEGORY
# ─────────────────────────────────────────

def add_availability_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: availability_category — bucketed label for availability

    Buckets:
    - 'Rarely Available'  : 0–30 days/year
    - 'Occasionally'      : 31–90 days/year
    - 'Moderately'        : 91–180 days/year
    - 'Highly Available'  : 181–365 days/year

    Useful for dashboard filtering and as a categorical ML feature.
    pd.cut() splits a continuous column into labeled buckets.
    """

    bins = [0, 30, 90, 180, 365]
    labels = ["Rarely Available", "Occasionally", "Moderately", "Highly Available"]

    df["availability_category"] = pd.cut(
        df["availability_365"],
        bins=bins,
        labels=labels,
        include_lowest=True    # include 0 in the first bucket
    )

    logger.info("add_availability_category: Created availability_category column.")

    return df


# ─────────────────────────────────────────
# 6. REVIEW SCORE CATEGORY
# ─────────────────────────────────────────

def add_review_score_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: review_score_category — label for review activity level

    Based on reviews_per_month:
    - 'No Reviews'   : 0
    - 'Low'          : 0.01 – 1.0
    - 'Medium'       : 1.01 – 3.0
    - 'High'         : 3.01+

    This captures how active and popular a listing is.
    """

    def categorize_reviews(rpm):
        if rpm == 0:
            return "No Reviews"
        elif rpm <= 1.0:
            return "Low"
        elif rpm <= 3.0:
            return "Medium"
        else:
            return "High"

    df["review_score_category"] = df["reviews_per_month"].apply(
        categorize_reviews
    )

    # Log the distribution
    dist = df["review_score_category"].value_counts()
    logger.info(f"add_review_score_category: Distribution:\n{dist}")

    return df


# ─────────────────────────────────────────
# 7. PRICE CATEGORY
# ─────────────────────────────────────────

def add_price_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: price_category — market segment label for each listing

    Buckets (NYC market context):
    - 'Budget'    : $0   – $75
    - 'Mid'       : $76  – $150
    - 'Premium'   : $151 – $300
    - 'Luxury'    : $301+

    Used in the Power BI dashboard for segment analysis
    and as a feature for classification tasks.
    """

    bins = [0, 75, 150, 300, float("inf")]
    labels = ["Budget", "Mid", "Premium", "Luxury"]

    df["price_category"] = pd.cut(
        df["price"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    dist = df["price_category"].value_counts()
    logger.info(f"add_price_category: Distribution:\n{dist}")

    return df


# ─────────────────────────────────────────
# 8. IS LONG TERM LISTING
# ─────────────────────────────────────────

def add_is_long_term(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: is_long_term = 1 if minimum_nights >= 30

    A 30-night minimum signals a fundamentally different listing type —
    monthly rentals rather than short-stay tourism.
    These price very differently and should be flagged for the model.
    """

    df["is_long_term"] = (df["minimum_nights"] >= 30).astype(int)

    long_term_count = df["is_long_term"].sum()
    logger.info(
        f"add_is_long_term: {long_term_count:,} long-term listings "
        f"({long_term_count / len(df) * 100:.1f}%)."
    )

    return df


# ─────────────────────────────────────────
# 9. NAME LENGTH
# ─────────────────────────────────────────

def add_name_length(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates: name_length = number of characters in the listing name

    Hypothesis: hosts who put effort into their listing title
    (longer, descriptive names) may also price higher or attract
    more bookings. This is a soft signal worth including.

    fillna('') handles any remaining null names safely.
    """

    df["name_length"] = df["name"].fillna("").str.len()

    logger.info(
        f"add_name_length: Avg name length = "
        f"{df['name_length'].mean():.1f} chars."
    )

    return df


# ─────────────────────────────────────────
# 10. DATE-BASED FEATURES
# ─────────────────────────────────────────

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts year, month, and recency from last_review.

    Creates:
    - last_review_year  : e.g. 2019
    - last_review_month : e.g. 6 (June)
    - days_since_last_review : days between last review and REFERENCE_DATE

    Listings with a recent review are likely more active and
    command different prices than stale or unreviewed listings.

    NaT rows (no review) will produce NaN for these columns — that's fine.
    We'll handle them in the ML preprocessing pipeline.
    """

    df["last_review_year"] = df["last_review"].dt.year
    df["last_review_month"] = df["last_review"].dt.month

    # Calculate days since last review
    # (REFERENCE_DATE - last_review).dt.days gives a whole number of days
    df["days_since_last_review"] = (
        REFERENCE_DATE - df["last_review"]
    ).dt.days

    valid = df["days_since_last_review"].notna().sum()
    logger.info(
        f"add_date_features: Date features created. "
        f"{valid:,} listings have a valid last_review date."
    )

    return df

# ─────────────────────────────────────────
# 11. NEIGHBOURHOOD MEDIAN PRICE ENCODING
# ─────────────────────────────────────────

def add_neighbourhood_price_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces raw neighbourhood names with their median price.

    Why this works better than ordinal encoding:
    OrdinalEncoder assigns arbitrary numbers — 'Tribeca'=187,
    'Bronx'=23. The model has no idea 187 means expensive.

    Median price encoding gives the model real signal:
    'Tribeca' → $295, 'Fordham' → $55.
    Now the model knows the actual price context of each area.

    This is called Target Encoding — encoding a categorical
    variable using the target variable's statistics.

    We calculate medians on the FULL dataset (before split)
    because neighbourhoods are stable geographic facts,
    not leakage-prone individual rows.

    We create NEW columns so originals are preserved:
    - neighbourhood_median_price
    - neighbourhood_group_median_price
    """

    # Calculate median price per neighbourhood
    neigh_median = (
        df.groupby("neighbourhood")["price"]
        .median()
        .rename("neighbourhood_median_price")
    )

    # Calculate median price per borough
    group_median = (
        df.groupby("neighbourhood_group")["price"]
        .median()
        .rename("neighbourhood_group_median_price")
    )

    # Map the medians back to each row
    df["neighbourhood_median_price"] = (
        df["neighbourhood"].map(neigh_median)
    )
    df["neighbourhood_group_median_price"] = (
        df["neighbourhood_group"].map(group_median)
    )

    logger.info(
        f"add_neighbourhood_price_encoding: "
        f"neighbourhood_median_price range: "
        f"${df['neighbourhood_median_price'].min():.0f} – "
        f"${df['neighbourhood_median_price'].max():.0f}"
    )

    return df

# ─────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs all feature engineering steps in order.
    Returns the enriched DataFrame with all new columns added.
    """

    logger.info("Starting feature engineering pipeline...")

    original_cols = df.shape[1]

    df = add_price_per_review(df)
    df = add_is_reviewed(df)
    df = add_host_superhost_proxy(df)
    df = add_availability_ratio(df)
    df = add_availability_category(df)
    df = add_review_score_category(df)
    df = add_price_category(df)
    df = add_is_long_term(df)
    df = add_name_length(df)
    df = add_date_features(df)
    df = add_neighbourhood_price_encoding(df)

    new_cols = df.shape[1] - original_cols

    logger.info(
        f"Feature engineering complete. "
        f"Added {new_cols} new columns. "
        f"Total columns: {df.shape[1]}"
    )

    return df


def print_feature_summary(df: pd.DataFrame) -> None:
    """
    Prints a summary of all new features and their distributions.
    """

    new_features = [
        "price_per_review", "is_reviewed", "host_is_superhost_proxy",
        "availability_ratio", "availability_category",
        "review_score_category", "price_category", "is_long_term",
        "name_length", "last_review_year", "last_review_month",
        "days_since_last_review", "neighbourhood_price_encoding"
    ]

    print("\n" + "=" * 55)
    print("   FEATURE ENGINEERING — SUMMARY")
    print("=" * 55)

    for feat in new_features:
        if feat not in df.columns:
            continue

        print(f"\n {feat}")
        print(f" dtype : {df[feat].dtype}")

        # For numeric features show basic stats
        if pd.api.types.is_numeric_dtype(df[feat]):
            print(
                f" mean  : {df[feat].mean():.2f} | "
                f"min : {df[feat].min()} | "
                f"max : {df[feat].max()}"
            )
        else:
            # For categorical features show value counts
            counts = df[feat].value_counts()
            for val, count in counts.items():
                print(f"   {str(val):<25} → {count:>7,}")

    print("\n" + "=" * 55 + "\n")