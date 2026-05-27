# app/backend/model_loader.py
# Handles loading the saved XGBoost model pipeline.
# Loads once at startup — not on every request.
# This is critical for performance. Loading a model takes
# ~0.5 seconds. If we loaded on every request, a busy API
# would be extremely slow.

import os
import joblib
from pipelines.logger import logger

# Path to the saved model
MODEL_PATH = os.path.join("models", "best_model_xgboost.pkl")
MODEL_INFO_PATH = os.path.join("models", "model_info.txt")

# Global variable that holds the loaded model
# Populated once when the app starts via load_model()
_model = None


def load_model():
    """
    Loads the XGBoost pipeline from disk into memory.
    Called once at application startup.
    """

    global _model

    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found: {MODEL_PATH}")
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Run pipelines/ml_pipeline.py first."
        )

    _model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from: {MODEL_PATH}")


def get_model():
    """
    Returns the loaded model.
    Called by the prediction endpoint on every request.
    Raises an error if model hasn't been loaded yet.
    """

    if _model is None:
        raise RuntimeError(
            "Model is not loaded. "
            "The application may not have started correctly."
        )

    return _model


def get_model_info() -> dict:
    """
    Reads and returns model metadata from model_info.txt.
    Used by the /model-info endpoint.
    """

    if not os.path.exists(MODEL_INFO_PATH):
        return {"error": "model_info.txt not found"}

    info = {}
    with open(MODEL_INFO_PATH, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                info[key.strip()] = val.strip()

    return info