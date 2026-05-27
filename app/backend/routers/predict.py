# app/backend/routers/predict.py
# Handles the /predict and /model-info endpoints.

import pandas as pd
from fastapi import APIRouter, HTTPException
from app.backend.schemas import PredictionRequest, PredictionResponse
from app.backend.model_loader import get_model, get_model_info
from pipelines.logger import logger

# MODEL_MAE: the average error of our best model in dollars
# Used to calculate a price range around the prediction
# Update this if you retrain the model
MODEL_MAE = 47.20

router = APIRouter()


def build_input_dataframe(request: PredictionRequest) -> pd.DataFrame:
    """
    Converts a PredictionRequest into a pandas DataFrame
    that matches exactly the feature set the model was trained on.

    Features used during training (16 total):
    - neighbourhood_group
    - neighbourhood
    - room_type
    - neighbourhood_median_price
    - neighbourhood_group_median_price
    - availability_365
    - is_reviewed
    - host_is_superhost_proxy
    - availability_ratio
    - is_long_term
    - name_length
    - minimum_nights        ← raw, not log
    - number_of_reviews     ← raw, not log
    - reviews_per_month     ← raw, not log
    - calculated_host_listings_count  ← raw, not log
    - days_since_last_review ← raw, not log
    """

    NEIGHBOURHOOD_MEDIANS = {
        "Tribeca": 295.0, "Flatiron District": 225.0,
        "Noho": 250.0, "Midtown": 210.0, "Soho": 199.0,
        "West Village": 200.0, "Theater District": 190.0,
        "Chelsea": 199.0, "Battery Park City": 195.0,
        "Greenwich Village": 197.5, "Financial District": 200.0,
        "Murray Hill": 190.0, "Gramercy": 165.0,
        "Nolita": 179.0, "Brooklyn Heights": 150.0,
        "Williamsburg": 150.0, "Bushwick": 80.0,
        "Bedford-Stuyvesant": 95.0, "Harlem": 90.0,
        "Upper West Side": 150.0, "Upper East Side": 150.0,
        "Hell'S Kitchen": 150.0, "East Village": 150.0,
        "Lower East Side": 130.0, "Crown Heights": 85.0,
        "Astoria": 80.0, "Long Island City": 100.0,
    }

    BOROUGH_MEDIANS = {
        "Manhattan": 150.0,
        "Brooklyn": 90.0,
        "Queens": 75.0,
        "Bronx": 65.0,
        "Staten Island": 75.0,
    }

    # Derived features — same logic as feature_engineer.py
    is_reviewed  = 1 if request.number_of_reviews > 0 else 0
    is_long_term = 1 if request.minimum_nights >= 30 else 0
    host_proxy   = 1 if request.calculated_host_listings_count >= 5 else 0
    avail_ratio  = round(request.availability_365 / 365, 4)

    # Neighbourhood price encoding
    neigh_median = NEIGHBOURHOOD_MEDIANS.get(
        request.neighbourhood,
        BOROUGH_MEDIANS.get(request.neighbourhood_group, 106.0)
    )
    group_median = BOROUGH_MEDIANS.get(request.neighbourhood_group, 106.0)

    # days_since_last_review — use training mean for new listings
    days_since = 277.95

    # Exactly 16 features — raw values, no log columns
    data = {
        "neighbourhood_group"               : [request.neighbourhood_group],
        "neighbourhood"                     : [request.neighbourhood],
        "room_type"                         : [request.room_type],
        "neighbourhood_median_price"        : [neigh_median],
        "neighbourhood_group_median_price"  : [group_median],
        "availability_365"                  : [request.availability_365],
        "is_reviewed"                       : [is_reviewed],
        "host_is_superhost_proxy"           : [host_proxy],
        "availability_ratio"                : [avail_ratio],
        "is_long_term"                      : [is_long_term],
        "name_length"                       : [request.name_length],
        "minimum_nights"                    : [request.minimum_nights],
        "number_of_reviews"                 : [request.number_of_reviews],
        "reviews_per_month"                 : [request.reviews_per_month],
        "calculated_host_listings_count"    : [request.calculated_host_listings_count],
        "days_since_last_review"            : [days_since],
    }

    return pd.DataFrame(data)


@router.post("/predict", response_model=PredictionResponse)
async def predict_price(request: PredictionRequest):
    """
    Predicts the nightly price for an Airbnb listing.

    Accepts listing details, runs them through the XGBoost model,
    and returns a predicted price with a confidence range.
    """

    logger.info(
        f"Prediction request: {request.neighbourhood_group} | "
        f"{request.room_type} | nights={request.minimum_nights}"
    )

    try:
        # Get the loaded model
        model = get_model()

        # Build input DataFrame
        input_df = build_input_dataframe(request)

        # Run prediction
        predicted_price = float(model.predict(input_df)[0])

        # Clip to realistic range — model can sometimes predict
        # negative or extremely high values for edge cases
        predicted_price = max(10.0, min(predicted_price, 799.0))
        predicted_price = round(predicted_price, 2)

        # Build price range using MAE as uncertainty estimate
        price_low  = round(max(10.0, predicted_price - MODEL_MAE), 2)
        price_high = round(min(799.0, predicted_price + MODEL_MAE), 2)

        logger.info(f"Prediction result: ${predicted_price}")

        return PredictionResponse(
            predicted_price=predicted_price,
            price_range_low=price_low,
            price_range_high=price_high,
            currency="USD",
            model_version="xgboost_v1",
            input_summary={
                "borough"     : request.neighbourhood_group,
                "neighbourhood": request.neighbourhood,
                "room_type"   : request.room_type,
                "min_nights"  : request.minimum_nights,
            }
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get("/model-info")
async def model_info():
    """
    Returns metadata about the currently loaded model.
    Useful for checking which model version is running.
    """

    info = get_model_info()
    return {
        "model_info"   : info,
        "model_version": "xgboost_v1",
        "features_used": 16,
        "target"       : "price (USD per night)",
        "dataset"      : "Airbnb NYC 2019",
    }