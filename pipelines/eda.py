# pipelines/eda.py
# Performs full exploratory data analysis on the featured dataset.
# Generates and saves charts to reports/charts/
# Run after feature_engineer.py — expects airbnb_featured.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pipelines.logger import logger

# ── Chart output folder ───────────────────────────────────
CHARTS_DIR = os.path.join("reports", "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────
# A clean, professional look for all charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.family"] = "sans-serif"

FEATURED_PATH = os.path.join("data", "processed", "airbnb_featured.csv")


def load_featured_data() -> pd.DataFrame:
    """Loads the featured dataset and restores categorical dtypes."""

    df = pd.read_csv(FEATURED_PATH, low_memory=False)

    # Restore datetime column lost during CSV save
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    logger.info(f"EDA: Loaded {len(df):,} rows from featured dataset.")
    return df


def save_chart(filename: str) -> None:
    """Saves the current matplotlib figure and closes it."""
    path = os.path.join(CHARTS_DIR, filename)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    logger.info(f"Chart saved: {filename}")
    print(f"   Saved: {filename}")


# ─────────────────────────────────────────
# 1. PRICE DISTRIBUTION
# ─────────────────────────────────────────

def plot_price_distribution(df: pd.DataFrame) -> None:
    """
    Shows the overall distribution of listing prices.
    Helps us understand if prices are skewed (they will be).
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Price Distribution", fontsize=14, fontweight="bold")

    # Left: full distribution
    axes[0].hist(df["price"], bins=60, color="#4C72B0", edgecolor="white")
    axes[0].set_title("All Prices (after capping at $799)")
    axes[0].set_xlabel("Price ($)")
    axes[0].set_ylabel("Count")

    # Right: zoom into $0–$400 where most listings live
    filtered = df[df["price"] <= 400]
    axes[1].hist(filtered["price"], bins=60, color="#55A868", edgecolor="white")
    axes[1].set_title("Prices $0–$400 (zoomed)")
    axes[1].set_xlabel("Price ($)")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    save_chart("01_price_distribution.png")


# ─────────────────────────────────────────
# 2. PRICE BY ROOM TYPE
# ─────────────────────────────────────────

def plot_price_by_room_type(df: pd.DataFrame) -> None:
    """
    Compares price distributions across the 3 room types.
    Boxplot shows median, spread, and outliers clearly.
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    # Define a consistent room type order
    order = ["Entire home/apt", "Private room", "Shared room"]

    sns.boxplot(
        data=df,
        x="room_type",
        y="price",
        order=order,
        palette="muted",
        ax=ax
    )

    ax.set_title("Price Distribution by Room Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Room Type")
    ax.set_ylabel("Price ($)")

    # Add median labels on top of each box
    medians = df.groupby("room_type")["price"].median()
    for i, room in enumerate(order):
        ax.text(
            i, medians[room] + 5,
            f"${medians[room]:.0f}",
            ha="center", fontsize=10, fontweight="bold", color="black"
        )

    plt.tight_layout()
    save_chart("02_price_by_room_type.png")


# ─────────────────────────────────────────
# 3. PRICE BY NEIGHBOURHOOD GROUP
# ─────────────────────────────────────────

def plot_price_by_neighbourhood_group(df: pd.DataFrame) -> None:
    """
    Average price per borough.
    Manhattan and Brooklyn will dominate — this confirms data quality.
    """

    avg_price = (
        df.groupby("neighbourhood_group")["price"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        avg_price["neighbourhood_group"],
        avg_price["price"],
        color=sns.color_palette("muted", len(avg_price))
    )

    # Add price labels on top of each bar
    for bar, val in zip(bars, avg_price["price"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"${val:.0f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title(
        "Average Listing Price by Borough", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Borough")
    ax.set_ylabel("Average Price ($)")
    ax.set_ylim(0, avg_price["price"].max() * 1.15)

    plt.tight_layout()
    save_chart("03_price_by_borough.png")


# ─────────────────────────────────────────
# 4. TOP 15 NEIGHBOURHOODS BY AVERAGE PRICE
# ─────────────────────────────────────────

def plot_top_neighbourhoods_by_price(df: pd.DataFrame) -> None:
    """
    Top 15 neighbourhoods by average price.
    Only includes neighbourhoods with 30+ listings to avoid
    small-sample distortion (e.g. one $800 listing in a tiny area).
    """

    # Filter to neighbourhoods with enough listings
    counts = df["neighbourhood"].value_counts()
    valid_neighbourhoods = counts[counts >= 30].index

    top15 = (
        df[df["neighbourhood"].isin(valid_neighbourhoods)]
        .groupby("neighbourhood")["price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 7))

    sns.barplot(
        data=top15,
        y="neighbourhood",
        x="price",
        palette="Blues_d",
        ax=ax
    )

    ax.set_title(
        "Top 15 Neighbourhoods by Average Price (min 30 listings)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Average Price ($)")
    ax.set_ylabel("Neighbourhood")

    # Add value labels
    for i, (_, row) in enumerate(top15.iterrows()):
        ax.text(
            row["price"] + 1, i,
            f"${row['price']:.0f}",
            va="center", fontsize=9
        )

    plt.tight_layout()
    save_chart("04_top_neighbourhoods_by_price.png")


# ─────────────────────────────────────────
# 5. LISTING COUNT BY ROOM TYPE
# ─────────────────────────────────────────

def plot_room_type_distribution(df: pd.DataFrame) -> None:
    """
    Shows how the market splits across room types.
    """

    counts = df["room_type"].value_counts().reset_index()
    counts.columns = ["room_type", "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(
        counts["room_type"],
        counts["count"],
        color=sns.color_palette("muted", 3)
    )

    for bar, row in zip(bars, counts.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 100,
            f"{row.count:,}\n({row.pct}%)",
            ha="center", fontsize=10
        )

    ax.set_title("Listing Count by Room Type", fontsize=13, fontweight="bold")
    ax.set_xlabel("Room Type")
    ax.set_ylabel("Number of Listings")

    plt.tight_layout()
    save_chart("05_room_type_distribution.png")


# ─────────────────────────────────────────
# 6. AVAILABILITY DISTRIBUTION
# ─────────────────────────────────────────

def plot_availability_distribution(df: pd.DataFrame) -> None:
    """
    Shows how available listings are throughout the year.
    The spike at 0 days is interesting — fully booked listings.
    """

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.hist(
        df["availability_365"],
        bins=50,
        color="#DD8452",
        edgecolor="white"
    )

    ax.set_title(
        "Distribution of Availability (days/year)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Days Available per Year")
    ax.set_ylabel("Number of Listings")

    plt.tight_layout()
    save_chart("06_availability_distribution.png")


# ─────────────────────────────────────────
# 7. REVIEWS PER MONTH DISTRIBUTION
# ─────────────────────────────────────────

def plot_reviews_distribution(df: pd.DataFrame) -> None:
    """
    Distribution of review activity.
    Most listings get under 2 reviews/month — good to know for ML.
    """

    # Filter to reviewed listings only for a cleaner chart
    reviewed = df[df["reviews_per_month"] > 0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        reviewed["reviews_per_month"],
        bins=50,
        color="#8172B2",
        edgecolor="white"
    )

    ax.set_title(
        "Reviews Per Month (reviewed listings only)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Reviews per Month")
    ax.set_ylabel("Number of Listings")
    ax.set_xlim(0, 15)

    plt.tight_layout()
    save_chart("07_reviews_per_month.png")


# ─────────────────────────────────────────
# 8. CORRELATION HEATMAP
# ─────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Correlation matrix of all numeric features.
    Shows which features are related to price and to each other.
    High correlation between features = multicollinearity (ML concern).
    """

    numeric_cols = [
        "price", "minimum_nights", "number_of_reviews",
        "reviews_per_month", "calculated_host_listings_count",
        "availability_365", "price_per_review", "is_reviewed",
        "host_is_superhost_proxy", "availability_ratio",
        "is_long_term", "name_length", "days_since_last_review"
    ]

    # Only keep columns that exist in this DataFrame
    cols = [c for c in numeric_cols if c in df.columns]

    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(13, 10))

    sns.heatmap(
        corr,
        annot=True,           # show correlation numbers
        fmt=".2f",            # 2 decimal places
        cmap="coolwarm",      # red = positive, blue = negative
        center=0,             # white at 0
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(
        "Feature Correlation Heatmap", fontsize=13, fontweight="bold"
    )

    plt.tight_layout()
    save_chart("08_correlation_heatmap.png")


# ─────────────────────────────────────────
# 9. PRICE CATEGORY BREAKDOWN BY BOROUGH
# ─────────────────────────────────────────

def plot_price_category_by_borough(df: pd.DataFrame) -> None:
    """
    Stacked bar chart showing price category mix per borough.
    Reveals where luxury listings concentrate.
    """

    # Count listings per borough + price category
    grouped = (
        df.groupby(["neighbourhood_group", "price_category"])
        .size()
        .unstack(fill_value=0)
    )

    # Reorder columns from cheapest to most expensive
    col_order = ["Budget", "Mid", "Premium", "Luxury"]
    col_order = [c for c in col_order if c in grouped.columns]
    grouped = grouped[col_order]

    fig, ax = plt.subplots(figsize=(12, 6))

    grouped.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        colormap="tab10",
        edgecolor="white"
    )

    ax.set_title(
        "Price Category Mix by Borough",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Borough")
    ax.set_ylabel("Number of Listings")
    ax.legend(title="Price Category", bbox_to_anchor=(1.01, 1))
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    save_chart("09_price_category_by_borough.png")


# ─────────────────────────────────────────
# 10. TOP 10 HOSTS BY LISTING COUNT
# ─────────────────────────────────────────

def plot_top_hosts(df: pd.DataFrame) -> None:
    """
    The 10 hosts with the most listings.
    Reveals property management companies dominating supply.
    """

    top_hosts = (
        df.groupby("host_name")["id"]
        .count()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    top_hosts.columns = ["host_name", "listing_count"]

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        data=top_hosts,
        y="host_name",
        x="listing_count",
        palette="rocket_r",
        ax=ax
    )

    ax.set_title(
        "Top 10 Hosts by Number of Listings",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Number of Listings")
    ax.set_ylabel("Host Name")

    for i, row in enumerate(top_hosts.itertuples()):
        ax.text(
            row.listing_count + 0.5, i,
            str(row.listing_count),
            va="center", fontsize=10
        )

    plt.tight_layout()
    save_chart("10_top_hosts.png")


# ─────────────────────────────────────────
# 11. MINIMUM NIGHTS DISTRIBUTION
# ─────────────────────────────────────────

def plot_minimum_nights(df: pd.DataFrame) -> None:
    """
    Distribution of minimum night requirements.
    Shows how many listings require 30+ nights (long-term).
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(
        df["minimum_nights"],
        bins=45,
        color="#C44E52",
        edgecolor="white"
    )

    ax.set_title(
        "Distribution of Minimum Nights Required",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Minimum Nights")
    ax.set_ylabel("Number of Listings")

    plt.tight_layout()
    save_chart("11_minimum_nights.png")


# ─────────────────────────────────────────
# 12. PRICE VS NUMBER OF REVIEWS (SCATTER)
# ─────────────────────────────────────────

def plot_price_vs_reviews(df: pd.DataFrame) -> None:
    """
    Scatter plot of price vs number of reviews, coloured by room type.
    Shows whether cheaper or more expensive listings get more reviews.
    """

    # Sample 5,000 points so the chart doesn't get overloaded
    sample = df.sample(n=min(5000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(11, 6))

    room_types = sample["room_type"].unique()
    colors = sns.color_palette("muted", len(room_types))

    for room, color in zip(room_types, colors):
        subset = sample[sample["room_type"] == room]
        ax.scatter(
            subset["number_of_reviews"],
            subset["price"],
            alpha=0.4,
            s=15,
            label=room,
            color=color
        )

    ax.set_title(
        "Price vs Number of Reviews (sample 5,000)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Price ($)")
    ax.legend(title="Room Type")
    ax.set_xlim(0, df["number_of_reviews"].quantile(0.98))

    plt.tight_layout()
    save_chart("12_price_vs_reviews.png")


# ─────────────────────────────────────────
# MAIN — RUN ALL EDA
# ─────────────────────────────────────────

def run_eda() -> None:
    """Runs all EDA charts in sequence."""

    print("\n" + "=" * 55)
    print("   AIRBNB EDA — GENERATING CHARTS")
    print("=" * 55 + "\n")

    df = load_featured_data()

    plot_price_distribution(df)
    plot_price_by_room_type(df)
    plot_price_by_neighbourhood_group(df)
    plot_top_neighbourhoods_by_price(df)
    plot_room_type_distribution(df)
    plot_availability_distribution(df)
    plot_reviews_distribution(df)
    plot_correlation_heatmap(df)
    plot_price_category_by_borough(df)
    plot_top_hosts(df)
    plot_minimum_nights(df)
    plot_price_vs_reviews(df)

    print(f"\n All 12 charts saved to: {CHARTS_DIR}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    run_eda()