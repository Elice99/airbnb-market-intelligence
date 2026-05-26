-- sql/analytics.sql
-- Business analytics queries for the Airbnb Market Intelligence Platform
-- All queries run against the featured_listings table in PostgreSQL
-- Run in psql or any SQL client connected to airbnb_db


-- ═══════════════════════════════════════════════
-- SECTION 1: MARKET OVERVIEW
-- ═══════════════════════════════════════════════

-- 1.1 Overall market summary
SELECT
    COUNT(*)                            AS total_listings,
    COUNT(DISTINCT host_id)             AS total_hosts,
    COUNT(DISTINCT neighbourhood)       AS total_neighbourhoods,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price,
    PERCENTILE_CONT(0.5)
        WITHIN GROUP (ORDER BY price)   AS median_price,
    MIN(price)                          AS min_price,
    MAX(price)                          AS max_price,
    ROUND(AVG(availability_365)::NUMERIC, 1) AS avg_availability_days
FROM featured_listings;


-- 1.2 Listings and average price per borough
SELECT
    neighbourhood_group                 AS borough,
    COUNT(*)                            AS total_listings,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price,
    ROUND(MIN(price)::NUMERIC, 2)       AS min_price,
    ROUND(MAX(price)::NUMERIC, 2)       AS max_price,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        1
    )                                   AS market_share_pct
FROM featured_listings
GROUP BY neighbourhood_group
ORDER BY avg_price DESC;


-- 1.3 Room type breakdown
SELECT
    room_type,
    COUNT(*)                            AS listing_count,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price,
    ROUND(
        PERCENTILE_CONT(0.5)
            WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    )                                   AS median_price,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        1
    )                                   AS share_pct
FROM featured_listings
GROUP BY room_type
ORDER BY listing_count DESC;


-- ═══════════════════════════════════════════════
-- SECTION 2: PRICING INTELLIGENCE
-- ═══════════════════════════════════════════════

-- 2.1 Top 15 most expensive neighbourhoods (min 30 listings)
SELECT
    neighbourhood,
    neighbourhood_group                 AS borough,
    COUNT(*)                            AS listing_count,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price,
    ROUND(
        PERCENTILE_CONT(0.5)
            WITHIN GROUP (ORDER BY price)::NUMERIC,
        2
    )                                   AS median_price
FROM featured_listings
GROUP BY neighbourhood, neighbourhood_group
HAVING COUNT(*) >= 30
ORDER BY avg_price DESC
LIMIT 15;


-- 2.2 Price category distribution per borough
SELECT
    neighbourhood_group                 AS borough,
    price_category,
    COUNT(*)                            AS listing_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER
            (PARTITION BY neighbourhood_group),
        1
    )                                   AS pct_within_borough
FROM featured_listings
WHERE price_category IS NOT NULL
GROUP BY neighbourhood_group, price_category
ORDER BY neighbourhood_group, listing_count DESC;


-- 2.3 Average price by room type and borough (cross tab)
SELECT
    neighbourhood_group                 AS borough,
    ROUND(AVG(CASE WHEN room_type = 'Entire home/apt'
        THEN price END)::NUMERIC, 0)    AS avg_entire_home,
    ROUND(AVG(CASE WHEN room_type = 'Private room'
        THEN price END)::NUMERIC, 0)    AS avg_private_room,
    ROUND(AVG(CASE WHEN room_type = 'Shared room'
        THEN price END)::NUMERIC, 0)    AS avg_shared_room
FROM featured_listings
GROUP BY neighbourhood_group
ORDER BY avg_entire_home DESC;


-- ═══════════════════════════════════════════════
-- SECTION 3: HOST ANALYTICS
-- ═══════════════════════════════════════════════

-- 3.1 Top 10 hosts by number of listings
SELECT
    host_id,
    host_name,
    COUNT(*)                            AS listing_count,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price,
    ROUND(AVG(number_of_reviews)::NUMERIC, 1) AS avg_reviews,
    ROUND(AVG(availability_365)::NUMERIC, 1)  AS avg_availability
FROM featured_listings
GROUP BY host_id, host_name
ORDER BY listing_count DESC
LIMIT 10;


-- 3.2 Superhost proxy vs regular host comparison
SELECT
    CASE
        WHEN host_is_superhost_proxy = 1 THEN 'Proxy Superhost (5+ listings)'
        ELSE 'Regular Host (1–4 listings)'
    END                                 AS host_type,
    COUNT(DISTINCT host_id)             AS host_count,
    COUNT(*)                            AS total_listings,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price,
    ROUND(AVG(number_of_reviews)::NUMERIC, 1) AS avg_reviews,
    ROUND(AVG(availability_365)::NUMERIC, 1)  AS avg_availability
FROM featured_listings
GROUP BY host_is_superhost_proxy
ORDER BY host_type;


-- 3.3 Hosts with highest average review count
-- (a proxy for booking volume)
SELECT
    host_name,
    COUNT(*)                            AS listing_count,
    ROUND(AVG(number_of_reviews)::NUMERIC, 1) AS avg_reviews_per_listing,
    SUM(number_of_reviews)              AS total_reviews,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price
FROM featured_listings
GROUP BY host_id, host_name
HAVING COUNT(*) >= 3
ORDER BY avg_reviews_per_listing DESC
LIMIT 10;


-- ═══════════════════════════════════════════════
-- SECTION 4: AVAILABILITY & DEMAND SIGNALS
-- ═══════════════════════════════════════════════

-- 4.1 Availability category distribution
SELECT
    availability_category,
    COUNT(*)                            AS listing_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        1
    )                                   AS share_pct,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price
FROM featured_listings
WHERE availability_category IS NOT NULL
GROUP BY availability_category
ORDER BY listing_count DESC;


-- 4.2 Average availability by borough and room type
SELECT
    neighbourhood_group                 AS borough,
    room_type,
    ROUND(AVG(availability_365)::NUMERIC, 1) AS avg_days_available,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price,
    COUNT(*)                            AS listing_count
FROM featured_listings
GROUP BY neighbourhood_group, room_type
ORDER BY neighbourhood_group, avg_days_available;


-- ═══════════════════════════════════════════════
-- SECTION 5: REVIEW ANALYTICS
-- ═══════════════════════════════════════════════

-- 5.1 Review activity by borough
SELECT
    neighbourhood_group                 AS borough,
    COUNT(*)                            AS total_listings,
    SUM(number_of_reviews)              AS total_reviews,
    ROUND(AVG(number_of_reviews)::NUMERIC, 1) AS avg_reviews,
    ROUND(AVG(reviews_per_month)::NUMERIC, 2) AS avg_reviews_per_month,
    SUM(CASE WHEN is_reviewed = 1
        THEN 1 ELSE 0 END)              AS listings_with_reviews,
    ROUND(
        SUM(CASE WHEN is_reviewed = 1
            THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        1
    )                                   AS pct_reviewed
FROM featured_listings
GROUP BY neighbourhood_group
ORDER BY avg_reviews_per_month DESC;


-- 5.2 Review score category breakdown
SELECT
    review_score_category,
    COUNT(*)                            AS listing_count,
    ROUND(AVG(price)::NUMERIC, 2)       AS avg_price,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (),
        1
    )                                   AS share_pct
FROM featured_listings
GROUP BY review_score_category
ORDER BY listing_count DESC;


-- ═══════════════════════════════════════════════
-- SECTION 6: WINDOW FUNCTION ANALYTICS
-- ═══════════════════════════════════════════════

-- 6.1 Rank neighbourhoods by average price within each borough
-- Using RANK() window function
WITH neighbourhood_avg AS (
    SELECT
        neighbourhood,
        neighbourhood_group             AS borough,
        COUNT(*)                        AS listing_count,
        ROUND(AVG(price)::NUMERIC, 2)   AS avg_price
    FROM featured_listings
    GROUP BY neighbourhood, neighbourhood_group
    HAVING COUNT(*) >= 20
)
SELECT
    neighbourhood,
    borough,
    listing_count,
    avg_price,
    RANK() OVER (
        PARTITION BY borough
        ORDER BY avg_price DESC
    )                                   AS price_rank_within_borough
FROM neighbourhood_avg
ORDER BY borough, price_rank_within_borough
LIMIT 30;


-- 6.2 Running total of listings and cumulative revenue potential
-- by neighbourhood (ordered by average price)
WITH neighbourhood_stats AS (
    SELECT
        neighbourhood,
        neighbourhood_group             AS borough,
        COUNT(*)                        AS listing_count,
        ROUND(AVG(price)::NUMERIC, 2)   AS avg_price,
        ROUND(SUM(price)::NUMERIC, 2)   AS total_price_value
    FROM featured_listings
    GROUP BY neighbourhood, neighbourhood_group
    HAVING COUNT(*) >= 30
)
SELECT
    neighbourhood,
    borough,
    listing_count,
    avg_price,
    SUM(listing_count) OVER (
        ORDER BY avg_price DESC
    )                                   AS cumulative_listings,
    ROUND(
        AVG(avg_price) OVER (
            ORDER BY avg_price DESC
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        )::NUMERIC, 2
    )                                   AS rolling_3_avg_price
FROM neighbourhood_stats
ORDER BY avg_price DESC
LIMIT 20;


-- 6.3 Percentile rank of every listing within its borough
-- Useful for telling a host "your listing is in the top X%"
SELECT
    id,
    name,
    neighbourhood_group AS borough,
    room_type,
    price,
    ROUND(
        (
            PERCENT_RANK() OVER (
                PARTITION BY neighbourhood_group
                ORDER BY price
            ) * 100
        )::numeric,
        1
    ) AS price_percentile_in_borough
FROM featured_listings
ORDER BY neighbourhood_group, price_percentile_in_borough DESC
LIMIT 20;