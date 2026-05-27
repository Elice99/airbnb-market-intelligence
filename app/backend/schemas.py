# app/backend/schemas.py
# Defines the shape of data coming IN and going OUT of the API.
# Pydantic validates every request automatically.
# If a user sends wrong data types or missing fields,
# FastAPI returns a clear error before the model even runs.

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class PredictionRequest(BaseModel):
    """
    Input schema for the /predict endpoint.
    Every field maps directly to a feature the ML model expects.

    Field() lets us add:
    - description : shown in Swagger docs
    - ge / le     : greater-than-or-equal / less-than-or-equal bounds
    - example     : shown in Swagger docs
    """

    neighbourhood_group: Literal[
        "Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"
    ] = Field(
        description="NYC borough the listing is in",
        examples=["Manhattan"]
    )

    neighbourhood: str = Field(
        description="Specific neighbourhood name",
        examples=["Williamsburg"],
        min_length=2,
        max_length=100
    )

    room_type: Literal[
        "Entire home/apt", "Private room", "Shared room"
    ] = Field(
        description="Type of room being listed",
        examples=["Entire home/apt"]
    )

    minimum_nights: int = Field(
        ge=1, le=365,
        description="Minimum number of nights required",
        examples=[2]
    )

    number_of_reviews: int = Field(
        ge=0, le=629,
        description="Total number of reviews the listing has",
        examples=[45]
    )

    reviews_per_month: float = Field(
        ge=0.0, le=58.5,
        description="Average reviews per month",
        examples=[1.5]
    )

    calculated_host_listings_count: int = Field(
        ge=1, le=327,
        description="Total number of listings this host has",
        examples=[1]
    )

    availability_365: int = Field(
        ge=0, le=365,
        description="Number of days available in the year",
        examples=[200]
    )

    name_length: int = Field(
        ge=0, le=200,
        description="Character length of the listing name",
        examples=[40]
    )

    @field_validator("neighbourhood_group")
    @classmethod
    def title_case_borough(cls, v: str) -> str:
        """Ensures borough is properly title-cased."""
        return v.strip().title()

    @field_validator("neighbourhood")
    @classmethod
    def title_case_neighbourhood(cls, v: str) -> str:
        """Ensures neighbourhood is properly title-cased."""
        return v.strip().title()


class PredictionResponse(BaseModel):
    """
    Output schema for the /predict endpoint.
    What the API sends back after making a prediction.
    """

    predicted_price: float = Field(
        description="Predicted nightly price in USD"
    )
    price_range_low: float = Field(
        description="Lower bound estimate (predicted - MAE)"
    )
    price_range_high: float = Field(
        description="Upper bound estimate (predicted + MAE)"
    )
    currency: str = Field(default="USD")
    model_version: str = Field(default="xgboost_v1")
    input_summary: dict = Field(
        description="Echo of key input fields for confirmation"
    )


class HealthResponse(BaseModel):
    """Output schema for the /health endpoint."""
    status: str
    model_loaded: bool
    message: str