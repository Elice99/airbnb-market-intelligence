# pipelines/db_loader.py
# Loads a DataFrame into PostgreSQL.
# Keeps raw and cleaned data in separate tables.

import pandas as pd
from sqlalchemy import text
from pipelines.db_connection import get_db_engine
from pipelines.logger import logger


def load_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    if_exists: str = "replace"
) -> None:
    """
    Loads a pandas DataFrame into a PostgreSQL table.

    Args:
        df         : The DataFrame to load
        table_name : Name of the target table in PostgreSQL
        if_exists  : What to do if table exists:
                     'replace' — drop and recreate (good for dev)
                     'append'  — add rows to existing table
                     'fail'    — raise error if table exists
    """

    logger.info(
        f"Loading {len(df):,} rows into PostgreSQL table: '{table_name}'"
    )

    engine = get_db_engine()

    try:
        # to_sql handles table creation and data insertion automatically
        # index=False means we don't write the pandas row index as a column
        # chunksize=500 means it inserts 500 rows at a time
        # (more reliable than trying to insert everything at once)
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            index=False,
            chunksize=500,
            method="multi"     # Uses multi-row INSERT for better performance
        )

        logger.info(
            f"Successfully loaded {len(df):,} rows into '{table_name}'."
        )
        print(f"\n Loaded {len(df):,} rows into table: '{table_name}'")

    except Exception as e:
        logger.error(f"Failed to load data into '{table_name}': {e}")
        raise


def verify_table_load(table_name: str) -> None:
    """
    Verifies the table was loaded correctly by
    checking the row count in PostgreSQL.
    """

    engine = get_db_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM {table_name}")
            )
            count = result.scalar()
            print(f" Row count in '{table_name}': {count:,}")
            logger.info(f"Table '{table_name}' verified: {count:,} rows.")

    except Exception as e:
        logger.error(f"Failed to verify table '{table_name}': {e}")
        raise


# Run directly: python pipelines/db_loader.py
if __name__ == "__main__":
    from pipelines.data_loader import load_raw_data

    # Load raw CSV
    df = load_raw_data()

    # Push to PostgreSQL as raw table (we'll push clean data later)
    load_to_postgres(df, table_name="raw_listings", if_exists="replace")

    # Confirm it loaded correctly
    verify_table_load("raw_listings")