# pipelines/logger.py
# This sets up a reusable logger for the entire project

from loguru import logger
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read log level from environment (default to INFO)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Remove default logger and configure a clean one
logger.remove()
logger.add(
    "reports/app.log",          # Save logs to a file
    level=LOG_LEVEL,            # Use level from .env
    rotation="10 MB",           # Create new file after 10MB
    retention="7 days",         # Keep logs for 7 days
    format="{time} | {level} | {message}"
)

# Also show logs in the terminal
logger.add(
    sink=lambda msg: print(msg, end=""),
    level=LOG_LEVEL,
    colorize=True
)