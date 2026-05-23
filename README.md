# 🏠 Airbnb Market Intelligence & Price Prediction Platform

> A production-grade, end-to-end analytics and machine learning platform that helps hosts price smarter and helps platforms understand their market — built on real Airbnb listing data.

---

## 📌 Table of Contents

- [Business Problem](#-business-problem)
- [Solution](#-solution)
- [Key Outcomes](#-key-outcomes)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Features](#-features)
- [Dashboard Preview](#-dashboard-preview)
- [ML Model Performance](#-ml-model-performance)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Project Status](#-project-status)
- [Author](#-author)

---

## 🚨 Business Problem

The short-term rental market is highly competitive and pricing is one of the biggest levers hosts have — yet most hosts price their listings based on gut feeling, manual comparisons, or static templates.

This creates two real problems:

**For hosts:**
- Underpricing means leaving money on the table. A host charging $80/night when comparable listings average $130 is losing ~$1,800/month in potential revenue.
- Overpricing drives away bookings entirely, hurting occupancy and review volume.
- Hosts have no easy way to benchmark their listing against the neighborhood, room type, or seasonal patterns in their area.

**For the platform (Airbnb/property managers):**
- Without market-level analytics, there's no visibility into which neighborhoods are saturated, which hosts are underperforming, or where pricing anomalies exist.
- Data sits in CSVs with no queryable layer, no dashboards, and no prediction capability.
- Business decisions are made reactively rather than from insight.

In short: **the data exists, but no one is using it intelligently.**

---

## ✅ Solution

This platform transforms raw Airbnb listing data into a fully operational market intelligence system — from raw CSV to live predictions.

It answers questions like:

- *"What should I charge for a 2-bedroom apartment in this neighborhood?"*
- *"Which neighborhoods have the best price-to-review ratio?"*
- *"What factors drive Airbnb prices the most?"*
- *"How do pricing trends vary by room type and availability?"*

The system delivers this through three layers:

**1. Analytics Layer**
A cleaned, structured data pipeline feeds a PostgreSQL database and Power BI dashboard with KPIs, neighborhood benchmarks, host performance metrics, and pricing intelligence — all refreshable and filterable.

**2. Machine Learning Layer**
Three trained models (Linear Regression, Random Forest, XGBoost) predict listing prices based on location, room type, host history, availability, and review signals. The best model is exposed via a REST API.

**3. Application Layer**
A Streamlit frontend lets hosts input their listing details and receive an instant price recommendation, market comparison, and neighborhood context — no SQL or Python knowledge required.

---

## 📈 Key Outcomes

| Outcome | Detail |
|--------|--------|
| Cleaned dataset | Removed nulls, outliers, duplicates — ready for analysis |
| SQL analytics | 15+ business queries covering pricing, availability, hosts |
| Power BI dashboard | 6-page interactive dashboard with live KPIs |
| ML price prediction | XGBoost model with RMSE < $X and R² > 0.X |
| REST API | FastAPI `/predict` endpoint with full input validation |
| Streamlit app | 5-page frontend for live predictions and market exploration |
| Docker deployment | Full stack containerized and deployable in one command |

> Model performance metrics will be updated after training on final dataset.

---

## 🏗 System Architecture

```
Raw CSV Data
     │
     ▼
┌─────────────────┐
│  ETL Pipeline   │  ← pandas, custom cleaning functions
│  (pipelines/)   │
└────────┬────────┘
         │
    ┌────▼────┐        ┌──────────────┐
    │PostgreSQL│◄──────│  SQL Scripts │
    │  (DB)    │       │  (sql/)      │
    └────┬─────┘       └──────────────┘
         │
    ┌────▼──────────┐
    │  Power BI     │  ← Connected to PostgreSQL
    │  Dashboard    │
    └───────────────┘

         │ (processed data also feeds ML)
         ▼
┌─────────────────────┐
│   ML Pipeline       │  ← sklearn, XGBoost
│   (pipelines/ml/)   │
└─────────┬───────────┘
          │  saved model (.pkl)
          ▼
┌─────────────────────┐      ┌──────────────────┐
│   FastAPI Backend   │◄─────│ Streamlit Frontend│
│   (app/backend/)    │      │ (app/frontend/)   │
└─────────────────────┘      └──────────────────┘
          │
          ▼
   Docker Compose
   (backend + frontend + db)
          │
          ▼
   Render / Railway (Cloud)
```

---

## 🛠 Tech Stack

| Layer | Tools |
|-------|-------|
| Data Processing | Python, Pandas, NumPy |
| Database | PostgreSQL, SQLAlchemy |
| Analytics | SQL (CTEs, window functions, aggregations) |
| Visualization | Power BI, Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn, XGBoost, Joblib |
| Backend API | FastAPI, Uvicorn, Pydantic |
| Frontend App | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | Render / Railway / Streamlit Cloud |
| Version Control | Git, GitHub |
| Logging | Loguru |
| Environment | python-dotenv |
| Testing | Pytest |

---

## 📁 Project Structure

```
airbnb-market-intelligence/
│
├── data/
│   ├── raw/                  # Original, untouched dataset
│   ├── processed/            # Cleaned and feature-engineered data
│   └── external/             # Any external reference data
│
├── notebooks/                # Jupyter notebooks for exploration
│
├── sql/                      # SQL analytics scripts
│
├── app/
│   ├── backend/              # FastAPI application
│   └── frontend/             # Streamlit application
│
├── models/                   # Saved ML models (.pkl / .joblib)
│
├── pipelines/                # ETL, feature engineering, ML pipelines
│
├── dashboards/               # Power BI files (.pbix)
│
├── reports/                  # Generated charts, EDA outputs, logs
│
├── tests/                    # Unit and API tests (pytest)
│
├── docs/                     # Architecture diagrams, documentation
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env                      # Environment variables (never committed)
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- PostgreSQL 14+
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/airbnb-market-intelligence.git
cd airbnb-market-intelligence
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 5. Create the database

```bash
psql -U postgres -c "CREATE DATABASE airbnb_db;"
```

### 6. Test the database connection

```bash
python pipelines/db_connection.py
```

### 7. Run with Docker (optional — full stack)

```bash
docker-compose up --build
```

---

## ✨ Features

**Data Pipeline**
- Automated ETL with null handling, outlier removal, type casting
- Reusable, modular cleaning functions
- Processed data saved to both CSV and PostgreSQL

**SQL Analytics**
- Neighborhood pricing rankings
- Host performance leaderboards
- Availability and occupancy trends
- Room type profitability analysis
- Window functions and CTEs for advanced aggregations

**Power BI Dashboard**
- Executive Overview — high-level KPIs and trends
- Pricing Intelligence — price distributions, benchmarks
- Host Analytics — top hosts, listing counts, performance
- Customer Experience — review scores and patterns
- Geospatial Insights — neighborhood maps and clusters
- Prediction Insights — model outputs and feature importance

**Machine Learning**
- Baseline to advanced model comparison
- Full preprocessing pipeline (encoding, scaling, feature selection)
- Hyperparameter tuning with cross-validation
- Model evaluation: RMSE, MAE, R²
- Saved model artifacts for production use

**FastAPI Backend**
- `/health` — system health check
- `/predict` — real-time price prediction
- `/model-info` — model metadata and feature details
- Full Pydantic input validation and error handling
- Auto-generated Swagger documentation at `/docs`

**Streamlit Frontend**
- Home page with project overview
- Market Analytics — interactive charts and filters
- Price Prediction — input your listing, get a price estimate
- Neighborhood Explorer — compare areas by price, reviews, availability
- Model Performance — visualize model metrics and feature importance

---

## 📊 Dashboard Preview

> Screenshots will be added after Power BI dashboard is completed (Day 10).

---

## 🤖 ML Model Performance

> Results will be updated after model training is complete (Day 6).

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Linear Regression | — | — | — |
| Random Forest | — | — | — |
| XGBoost | — | — | — |

---

## 🔌 API Reference

### `POST /predict`

Predicts the recommended nightly price for an Airbnb listing.

**Request body:**
```json
{
  "neighbourhood": "Brooklyn",
  "room_type": "Entire home/apt",
  "minimum_nights": 2,
  "number_of_reviews": 45,
  "availability_365": 200,
  "calculated_host_listings_count": 1
}
```

**Response:**
```json
{
  "predicted_price": 127.50,
  "currency": "USD",
  "model_version": "xgboost_v1"
}
```

Full API docs available at: `http://localhost:8000/docs`

---

## ☁️ Deployment

| Service | Component |
|---------|-----------|
| Render | FastAPI backend |
| Railway | PostgreSQL database |
| Streamlit Cloud | Streamlit frontend |

Deployment guide available in `docs/deployment.md`.

---

## ✅ Project Status

- [x] Day 1: Project initialization, folder structure, config setup
- [ ] Day 2: Data ingestion and schema inspection
- [ ] Day 3: Data cleaning pipeline
- [ ] Day 4: Feature engineering
- [ ] Day 5: EDA + SQL analytics
- [ ] Day 6: Machine learning pipeline
- [ ] Day 7: Review and buffer
- [ ] Day 8: FastAPI backend
- [ ] Day 9: Streamlit frontend
- [ ] Day 10: Power BI dashboard
- [ ] Day 11: Docker containerization
- [ ] Day 12: Cloud deployment
- [ ] Day 13: Testing and documentation
- [ ] Day 14: Final polish and portfolio ready

---

## 👤 Author

**Elisha (Elice99)**
Data Analyst | Python & Power BI

[LinkedIn](https://www.linkedin.com/in/YOUR_PROFILE) · [GitHub](https://github.com/YOUR_USERNAME)

---

> Built as a portfolio project demonstrating end-to-end data engineering, analytics, and machine learning on real-world data.
