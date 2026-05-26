# pipelines/ml_pipeline.py
# Full machine learning pipeline for Airbnb price prediction.
# Trains Linear Regression, Random Forest, and XGBoost.
# Evaluates all three and saves the best model to models/

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from xgboost import XGBRegressor

from pipelines.logger import logger

# ── Paths ─────────────────────────────────────────────────
FEATURED_PATH = os.path.join("data", "processed", "airbnb_featured.csv")
MODELS_DIR    = "models"
CHARTS_DIR    = os.path.join("reports", "charts")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# 1. LOAD & SELECT FEATURES
# ─────────────────────────────────────────

def load_ml_data() -> pd.DataFrame:
    """
    Loads the featured dataset and selects only the columns
    we'll use for ML training.

    Why these features?
    - neighbourhood_group : borough is a strong price signal
    - neighbourhood       : even stronger — Tribeca vs Bronx
    - room_type           : entire home vs private vs shared
    - minimum_nights      : long-term vs short-term signal
    - number_of_reviews   : popularity/demand proxy
    - reviews_per_month   : recent activity signal
    - calculated_host_listings_count : host scale signal
    - availability_365    : supply signal
    - is_reviewed         : binary — has any review
    - host_is_superhost_proxy : professional host flag
    - availability_ratio  : normalized availability
    - is_long_term        : 30+ night minimum flag
    - name_length         : effort signal
    - days_since_last_review : recency signal

    We exclude: id, name, host_id, host_name, last_review,
    last_review_year/month (redundant with days_since),
    price_per_review (derived from target — data leakage risk),
    price_category (derived from target — data leakage risk),
    availability_category, review_score_category (string categories
    we already have as numeric features)
    """

    df = pd.read_csv(FEATURED_PATH, low_memory=False)

    FEATURES = [
        # Categorical
        "neighbourhood_group",
        "neighbourhood",
        "room_type",
         # Numeric — price-encoded geography (NEW)
        "neighbourhood_median_price",
        "neighbourhood_group_median_price",
        # Numeric — original
        "minimum_nights",
        "number_of_reviews",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
        # Numeric — engineered
        "is_reviewed",
        "host_is_superhost_proxy",
        "availability_ratio",
        "is_long_term",
        "name_length",
        "days_since_last_review",
    ]

    TARGET = "price"

    # Drop rows where any feature or target is null
    cols_needed = FEATURES + [TARGET]
    df_ml = df[cols_needed].dropna()

    logger.info(
        f"ML data ready: {len(df_ml):,} rows, "
        f"{len(FEATURES)} features → target: price"
    )

    return df_ml, FEATURES, TARGET


# ─────────────────────────────────────────
# 2. TRAIN / TEST SPLIT
# ─────────────────────────────────────────

def split_data(df: pd.DataFrame, features: list, target: str):
    """
    Splits data into train (80%) and test (20%) sets.

    random_state=42 ensures reproducibility —
    every time you run this you get the same split.

    Returns X_train, X_test, y_train, y_test
    """

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    logger.info(
        f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows"
    )

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────
# 3. BUILD PREPROCESSING PIPELINE
# ─────────────────────────────────────────

def build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a preprocessing pipeline that handles:
    - Categorical columns → OrdinalEncoder
      (converts text like 'Manhattan' to a number)
    - Numeric columns → StandardScaler
      (scales values to mean=0, std=1 so no single feature dominates)

    Why OrdinalEncoder instead of OneHotEncoder?
    With 221 unique neighbourhoods, OneHotEncoding would create
    221 new columns — very wide and slow. OrdinalEncoder keeps
    it compact. XGBoost handles ordinal encoding well.

    ColumnTransformer applies different transformations to
    different column groups simultaneously.
    """

    # Identify categorical vs numeric columns automatically
    cat_cols = X_train.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    logger.info(f"Categorical features: {cat_cols}")
    logger.info(f"Numeric features    : {num_cols}")

    # Categorical pipeline: encode text to numbers
    cat_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1   # unseen categories get -1
        ))
    ])

    # Numeric pipeline: scale to zero mean, unit variance
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ("cat", cat_pipeline, cat_cols),
        ("num", num_pipeline, num_cols),
    ])

    return preprocessor


# ─────────────────────────────────────────
# 4. BUILD MODELS
# ─────────────────────────────────────────

def build_models(preprocessor: ColumnTransformer) -> dict:
    """
    Builds three model pipelines.

    XGBoost now has two versions:
    - XGBoost          : same as before (baseline)
    - XGBoost_Tuned    : better hyperparameters after analysis

    Tuning decisions explained:
    - n_estimators=500    : more trees = more learning rounds
    - learning_rate=0.02  : slower learning generalizes better
    - max_depth=5         : shallower trees reduce overfitting
    - min_child_weight=5  : leaf needs 5+ samples (prevents noise fitting)
    - subsample=0.75      : 75% row sampling adds randomness
    - colsample_bytree=0.75 : 75% feature sampling per tree
    - reg_alpha=0.1       : L1 regularization penalizes complexity
    - reg_lambda=1.5      : L2 regularization (stronger penalty)
    - gamma=0.1           : minimum loss reduction to make a split

    These prevent the model from memorizing training data
    and help it generalize to unseen listings.
    """

    models = {

        "LinearRegression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LinearRegression())
        ]),

        "RandomForest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_leaf=5,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            ))
        ]),

        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ))
        ]),

        "XGBoost_Tuned": Pipeline([
            ("preprocessor", preprocessor),
            ("model", XGBRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=5,
                min_child_weight=5,
                subsample=0.75,
                colsample_bytree=0.75,
                reg_alpha=0.1,
                reg_lambda=1.5,
                gamma=0.1,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ))
        ]),

    }

    return models

# ─────────────────────────────────────────
# 5. TRAIN & EVALUATE
# ─────────────────────────────────────────

def evaluate_model(
    name: str,
    model,
    X_train, X_test,
    y_train, y_test
) -> dict:
    """
    Trains a model and evaluates it on the test set.

    Metrics:
    - RMSE (Root Mean Squared Error):
      Average prediction error in dollars.
      Lower is better. RMSE of $40 means predictions
      are off by ~$40 on average.

    - MAE (Mean Absolute Error):
      Average absolute error — less sensitive to outliers
      than RMSE. Also in dollars.

    - R² (R-squared):
      How much variance the model explains.
      1.0 = perfect, 0.0 = no better than guessing the mean.
      A good model is typically R² > 0.60 for real-world data.
    """

    logger.info(f"Training {name}...")
    print(f"\n Training {name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Generate predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    results = {
        "name"  : name,
        "model" : model,
        "rmse"  : round(rmse, 2),
        "mae"   : round(mae, 2),
        "r2"    : round(r2, 4),
        "y_pred": y_pred,
        "y_test": y_test,
    }

    print(f"   RMSE : ${rmse:,.2f}")
    print(f"   MAE  : ${mae:,.2f}")
    print(f"   R²   : {r2:.4f}")

    logger.info(
        f"{name} → RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f} | R²: {r2:.4f}"
    )

    return results


# ─────────────────────────────────────────
# 6. COMPARE ALL MODELS
# ─────────────────────────────────────────

def compare_models(all_results: list) -> dict:
    """
    Compares all model results and returns the best one.
    Best = lowest RMSE (most accurate predictions in dollars).
    """

    print("\n" + "=" * 55)
    print("   MODEL COMPARISON")
    print("=" * 55)
    print(f"\n {'Model':<22} {'RMSE':>10} {'MAE':>10} {'R²':>8}")
    print("-" * 55)

    for r in all_results:
        print(
            f" {r['name']:<22} "
            f"${r['rmse']:>8,.2f} "
            f"${r['mae']:>8,.2f} "
            f"{r['r2']:>8.4f}"
        )

    # Pick model with lowest RMSE
    best = min(all_results, key=lambda x: x["rmse"])

    print(f"\n Best model: {best['name']} (RMSE = ${best['rmse']:,.2f})")
    print("=" * 55 + "\n")

    logger.info(f"Best model: {best['name']} — RMSE ${best['rmse']:,.2f}")

    return best

def cross_validate_best(best: dict, X_train, y_train) -> None:
    """
    Runs 5-fold cross-validation on the best model.

    Why cross-validate?
    A single train/test split can get lucky or unlucky depending
    on which rows land in each set. Cross-validation splits the
    training data into 5 different train/validation combinations
    and averages the scores — giving a much more reliable estimate
    of real-world performance.

    neg_root_mean_squared_error returns negative RMSE (sklearn
    convention — higher is better for all scorers). We negate it
    back to get positive RMSE values.
    """

    print(f"\n Running 5-fold cross-validation on {best['name']}...")
    logger.info(f"Cross-validating {best['name']}...")

    cv_scores = cross_val_score(
        best["model"],
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    # Negate back to positive RMSE values
    rmse_scores = -cv_scores

    print(f"\n Cross-Validation Results ({best['name']}):")
    print(f"   Fold scores : {[f'${s:.2f}' for s in rmse_scores]}")
    print(f"   Mean RMSE   : ${rmse_scores.mean():.2f}")
    print(f"   Std RMSE    : ±${rmse_scores.std():.2f}")
    print(
        f"\n   Interpretation: The model's average error across "
        f"5 different data splits is ${rmse_scores.mean():.2f} "
        f"±${rmse_scores.std():.2f}"
    )

    logger.info(
        f"CV Results — Mean RMSE: ${rmse_scores.mean():.2f} "
        f"±${rmse_scores.std():.2f}"
    )

# ─────────────────────────────────────────
# 7. SAVE BEST MODEL
# ─────────────────────────────────────────

def save_best_model(best: dict) -> None:
    """
    Saves the best model pipeline to disk using joblib.

    We save the entire sklearn Pipeline (preprocessor + model),
    not just the model weights. This means when FastAPI loads
    the model, it can accept raw input directly — no need to
    preprocess manually before predicting.
    """

    model_name = best["name"].lower().replace(" ", "_")
    model_path = os.path.join(MODELS_DIR, f"best_model_{model_name}.pkl")

    joblib.dump(best["model"], model_path)

    logger.info(f"Best model saved to: {model_path}")
    print(f" Model saved: {model_path}")

    # Also save model metadata as a simple text file
    meta_path = os.path.join(MODELS_DIR, "model_info.txt")
    with open(meta_path, "w") as f:
        f.write(f"Best Model   : {best['name']}\n")
        f.write(f"RMSE         : ${best['rmse']:,.2f}\n")
        f.write(f"MAE          : ${best['mae']:,.2f}\n")
        f.write(f"R²           : {best['r2']:.4f}\n")

    print(f" Metadata saved: {meta_path}")



# ─────────────────────────────────────────
# 8. PLOT ACTUAL VS PREDICTED
# ─────────────────────────────────────────

def plot_actual_vs_predicted(best: dict) -> None:
    """
    Scatter plot of actual prices vs predicted prices.
    A perfect model would show all dots on the diagonal line.
    Spread around the line shows where the model struggles.
    """

    y_test = best["y_test"]
    y_pred = best["y_pred"]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(y_test, y_pred, alpha=0.3, s=10, color="#4C72B0")

    # Perfect prediction line
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.5, label="Perfect prediction")

    ax.set_title(
        f"Actual vs Predicted Price — {best['name']}\n"
        f"RMSE: ${best['rmse']:,.2f} | R²: {best['r2']:.4f}",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.legend()

    path = os.path.join(CHARTS_DIR, "13_actual_vs_predicted.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    logger.info("Chart saved: 13_actual_vs_predicted.png")
    print("   Saved: 13_actual_vs_predicted.png")


# ─────────────────────────────────────────
# 9. PLOT FEATURE IMPORTANCE
# ─────────────────────────────────────────

def plot_feature_importance(best: dict, feature_names: list) -> None:
    """
    Bar chart of feature importances from the best model.
    Only works for tree-based models (RandomForest, XGBoost).
    Shows which features drive price predictions the most.
    """

    model_step = best["model"].named_steps["model"]

    # Check if model supports feature_importances_
    if not hasattr(model_step, "feature_importances_"):
        logger.info("Feature importance not available for this model type.")
        return

    importances = model_step.feature_importances_
    feat_df = pd.DataFrame({
        "feature"   : feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.barplot(
        data=feat_df,
        y="feature",
        x="importance",
        hue="feature",
        palette="Blues_d",
        legend=False,
        ax=ax
    )

    ax.set_title(
        f"Feature Importance — {best['name']}",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")

    path = os.path.join(CHARTS_DIR, "14_feature_importance.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    logger.info("Chart saved: 14_feature_importance.png")
    print("   Saved: 14_feature_importance.png")

    # Print top 5 features
    print("\n Top 5 most important features:")
    for _, row in feat_df.head(5).iterrows():
        print(f"   {row['feature']:<35} {row['importance']:.4f}")


# ─────────────────────────────────────────
# 10. PLOT RESIDUALS
# ─────────────────────────────────────────

def plot_residuals(best: dict) -> None:
    """
    Residuals = actual - predicted.
    A good model has residuals centered around 0 with no pattern.
    If residuals fan out at high prices, the model struggles with
    luxury listings (common with Airbnb data).
    """

    y_test  = best["y_test"]
    y_pred  = best["y_pred"]
    residuals = y_test.values - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Residual Analysis — {best['name']}",
        fontsize=13, fontweight="bold"
    )

    # Left: residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.3, s=10, color="#DD8452")
    axes[0].axhline(0, color="red", linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Price ($)")
    axes[0].set_ylabel("Residual ($)")
    axes[0].set_title("Residuals vs Predicted")

    # Right: distribution of residuals
    axes[1].hist(residuals, bins=60, color="#8172B2", edgecolor="white")
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution")

    plt.tight_layout()

    path = os.path.join(CHARTS_DIR, "15_residuals.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    logger.info("Chart saved: 15_residuals.png")
    print("   Saved: 15_residuals.png")


# ─────────────────────────────────────────
# MAIN — RUN FULL ML PIPELINE
# ─────────────────────────────────────────

def run_ml_pipeline() -> None:

    print("\n" + "=" * 55)
    print("   AIRBNB ML PIPELINE — STARTING")
    print("=" * 55)

    df_ml, features, target = load_ml_data()
    X_train, X_test, y_train, y_test = split_data(df_ml, features, target)
    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    all_results = []
    for name, model in models.items():
        result = evaluate_model(
            name, model,
            X_train, X_test,
            y_train, y_test
        )
        all_results.append(result)

    best = compare_models(all_results)

    # Cross-validate the best model
    cross_validate_best(best, X_train, y_train)

    save_best_model(best)

    print("\n Generating ML charts...")
    plot_actual_vs_predicted(best)
    plot_feature_importance(best, features)
    plot_residuals(best)

    print("\n" + "=" * 55)
    print("   ML PIPELINE COMPLETE")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_ml_pipeline()