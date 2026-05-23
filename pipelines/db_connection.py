# pipelines/db_connection.py
# Handles all PostgreSQL connection logic in one place

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from pipelines.logger import logger

# Load variables from .env
load_dotenv()

def get_db_engine():
    """
    Creates and returns a SQLAlchemy database engine.
    All connection details come from environment variables.
    """
    
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    name = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    # Build the connection string
    connection_url = (
        f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{name}"
    )

    try:
        engine = create_engine(connection_url)
        logger.info("Database engine created successfully.")
        return engine

    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        raise


def test_connection():
    """
    Tests the database connection by running a simple query.
    Run this to verify your PostgreSQL setup is working.
    """
    engine = get_db_engine()

    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            logger.info("Database connection test PASSED.")
            return True

    except Exception as e:
        logger.error(f"Database connection test FAILED: {e}")
        return False


# Run this file directly to test: python pipelines/db_connection.py
if __name__ == "__main__":
    test_connection()