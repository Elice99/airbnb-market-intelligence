# app/backend/main.py
# Entry point for the FastAPI application.
# Registers all routers, configures startup/shutdown events,
# and sets up CORS so the Streamlit frontend can talk to it.

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.backend.routers import predict, health
from app.backend.model_loader import load_model
from pipelines.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    We load the model here once at startup so every
    subsequent request has instant access to it.
    """
    logger.info("Starting Airbnb Price Prediction API...")
    load_model()
    logger.info("API startup complete.")
    yield
    logger.info("API shutting down.")


# Create the FastAPI app
app = FastAPI(
    title="Airbnb Market Intelligence API",
    description=(
        "Production-grade price prediction API for Airbnb listings. "
        "Built on XGBoost trained on NYC 2019 Airbnb data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allows the Streamlit frontend (running on a different port)
# to make requests to this API without being blocked by the browser
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers — each router handles a group of endpoints
app.include_router(health.router,  tags=["Health"])
app.include_router(predict.router, tags=["Predictions"])


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "message" : "Airbnb Market Intelligence API",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "health"  : "/health",
        "predict" : "/predict",
    }