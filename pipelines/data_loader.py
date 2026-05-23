# pipelines/data_loader.py
# Responsible for loading raw data into the pipeline.
# This file does ONE thing: load the CSV. Nothing else.

import pandas as pd
import os
from pipelines.logger import logger


# Path to the raw dataset — uses os.path so it works on any OS
RAW_DATA_PATH = os.path.join( "data", "raw", "AB_NYC_2019.csv" )


def load_raw_data(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads the raw Airbnb CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file. Defaults to data/raw/AB_NYC_2019.csv

    Returns:
        A pandas DataFrame with the raw data loaded.
    """

    logger.info(f"Loading raw data from: {filepath}")

    # Check the file actually exists before trying to load it
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. "
            f"Please place your CSV in data/raw/"
        )

    try:
        # low_memory=False prevents pandas from guessing column types
        # incorrectly when columns have mixed data halfway through the file
        df = pd.read_csv(filepath, low_memory=False)

        logger.info(
            f"Data loaded successfully. "
            f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}"
        )

        return df

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


# Run this file directly to test: python pipelines/data_loader.py
if __name__ == "__main__":
    df = load_raw_data()
    print(df.head())