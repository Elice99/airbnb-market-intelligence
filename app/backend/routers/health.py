# app/backend/routers/health.py
# Simple health check endpoint.
# Used by Docker, cloud platforms, and monitoring tools
# to verify the API is alive and the model is loaded.

from fastapi import APIRouter
from app.backend.schemas import HealthResponse
from app.backend.model_loader import get_model
from pipelines.logger import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns the health status of the API.
    Checks whether the ML model is loaded and ready.
    """

    try:
        get_model()
        model_loaded = True
        message = "API is healthy. Model is loaded and ready."
    except RuntimeError:
        model_loaded = False
        message = "API is running but model is not loaded."

    logger.info(f"Health check: model_loaded={model_loaded}")

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        message=message
    )