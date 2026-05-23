# pipelines/schema_inspector.py
# Inspects the raw dataset and produces a full schema report.
# Run this BEFORE any cleaning. Understand the data first.

import pandas as pd
import numpy as np
import os
from pipelines.data_loader import load_raw_data
from pipelines.logger import logger


def inspect_shape(df: pd.DataFrame) -> None:
    """Prints the dimensions of the dataset."""

    logger.info("=== SHAPE ===")
    print(f"\n Rows    : {df.shape[0]:,}")
    print(f" Columns : {df.shape[1]}")


def inspect_columns(df: pd.DataFrame) -> None:
    """Prints all column names and their data types."""

    logger.info("=== COLUMNS & DATA TYPES ===")
    print("\n Column Name                          | dtype")
    print("-" * 50)

    for col in df.columns:
        print(f" {col:<40} | {df[col].dtype}")


def inspect_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates missing values per column.
    Returns a DataFrame sorted by missing % descending.
    """

    logger.info("=== MISSING VALUES ===")

    # Count nulls and calculate percentage
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)

    missing_df = pd.DataFrame({
        "column": missing_count.index,
        "missing_count": missing_count.values,
        "missing_pct": missing_pct.values
    })

    # Only show columns that actually have missing values
    missing_df = missing_df[missing_df["missing_count"] > 0]
    missing_df = missing_df.sort_values("missing_pct", ascending=False)

    if missing_df.empty:
        print("\n No missing values found.")
    else:
        print(f"\n {len(missing_df)} columns have missing values:\n")
        print(missing_df.to_string(index=False))

    return missing_df


def inspect_duplicates(df: pd.DataFrame) -> None:
    """Checks for fully duplicated rows."""

    logger.info("=== DUPLICATES ===")

    duplicate_count = df.duplicated().sum()
    print(f"\n Duplicate rows: {duplicate_count:,}")

    if duplicate_count > 0:
        print(" ACTION NEEDED: Duplicates found — will be removed in cleaning.")
    else:
        print(" No duplicates found.")


def inspect_numeric_summary(df: pd.DataFrame) -> None:
    """
    Descriptive statistics for all numeric columns.
    Shows count, mean, std, min, max and key percentiles.
    """

    logger.info("=== NUMERIC COLUMN SUMMARY ===")

    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])

    if numeric_cols.empty:
        print("\n No numeric columns found.")
        return

    # describe() gives us count, mean, std, min, 25%, 50%, 75%, max
    summary = numeric_cols.describe().T
    summary["null_count"] = df[numeric_cols.columns].isnull().sum().values

    print("\n")
    print(summary.to_string())


def inspect_categorical_summary(df: pd.DataFrame, top_n: int = 5) -> None:
    """
    For each categorical (text) column, shows:
    - number of unique values
    - top N most frequent values
    """

    logger.info("=== CATEGORICAL COLUMN SUMMARY ===")

    # Select object (string) columns
    cat_cols = df.select_dtypes(include=["object"])

    if cat_cols.empty:
        print("\n No categorical columns found.")
        return

    for col in cat_cols.columns:
        unique_count = df[col].nunique()
        top_values = df[col].value_counts().head(top_n)

        print(f"\n Column   : {col}")
        print(f" Unique   : {unique_count:,}")
        print(f" Top {top_n}   :")

        for val, count in top_values.items():
            pct = count / len(df) * 100
            print(f"   {str(val):<35} → {count:>6,} ({pct:.1f}%)")


def inspect_price_column(df: pd.DataFrame) -> None:
    """
    Special inspection for the price column specifically.
    This is our ML target variable — we need to understand it well.
    """

    logger.info("=== PRICE COLUMN DEEP DIVE ===")

    # Try to find price column — handle different naming conventions
    price_col = None
    for candidate in ["price", "Price", "PRICE"]:
        if candidate in df.columns:
            price_col = candidate
            break

    if price_col is None:
        print("\n WARNING: No 'price' column found. Check column names above.")
        return

    price_series = df[price_col]

    # Price may be stored as a string like "$120.00" — detect this
    sample = price_series.dropna().iloc[0]
    print(f"\n Price column   : '{price_col}'")
    print(f" Sample value   : {sample}")
    print(f" Stored as      : {price_series.dtype}")

    # If it's numeric already, show distribution
    if pd.api.types.is_numeric_dtype(price_series):
        print(f" Min            : ${price_series.min():,.2f}")
        print(f" Max            : ${price_series.max():,.2f}")
        print(f" Mean           : ${price_series.mean():,.2f}")
        print(f" Median         : ${price_series.median():,.2f}")
        print(f" Zero values    : {(price_series == 0).sum():,}")
        print(f" Negative values: {(price_series < 0).sum():,}")
    else:
        print(
            " NOTE: Price is stored as a string. "
            "Will need cleaning (remove $ and commas)."
        )


def save_schema_report(df: pd.DataFrame) -> None:
    """
    Saves a simple schema report to reports/schema_report.txt
    so you have a record of the raw dataset's state.
    """

    logger.info("=== SAVING SCHEMA REPORT ===")

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/schema_report.txt"

    with open(report_path, "w") as f:
        f.write("AIRBNB 2019 — RAW DATASET SCHEMA REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n")
        f.write("Column Name | dtype | Missing Count | Missing %\n")
        f.write("-" * 60 + "\n")

        for col in df.columns:
            missing = df[col].isnull().sum()
            pct = missing / len(df) * 100
            f.write(f"{col} | {df[col].dtype} | {missing} | {pct:.2f}%\n")

    logger.info(f"Schema report saved to: {report_path}")
    print(f"\n Schema report saved to: {report_path}")


def run_full_inspection(df: pd.DataFrame) -> None:
    """
    Runs ALL inspection functions in sequence.
    This is the main entry point for Day 2.
    """

    print("\n" + "=" * 60)
    print("   AIRBNB 2019 — FULL SCHEMA INSPECTION")
    print("=" * 60)

    inspect_shape(df)
    print()
    inspect_columns(df)
    print()
    inspect_missing_values(df)
    print()
    inspect_duplicates(df)
    print()
    inspect_numeric_summary(df)
    print()
    inspect_categorical_summary(df)
    print()
    inspect_price_column(df)
    print()
    save_schema_report(df)

    print("\n" + "=" * 60)
    print("   INSPECTION COMPLETE")
    print("=" * 60 + "\n")


# Run directly: python pipelines/schema_inspector.py
if __name__ == "__main__":
    df = load_raw_data()
    run_full_inspection(df)