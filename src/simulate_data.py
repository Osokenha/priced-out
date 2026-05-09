"""
Data simulation for the Priced Out project.

This module generates a synthetic restaurant-level dataset calibrated to
real published statistics. The simulation is NOT a substitute for real
scraped menu data; it is a methodological scaffold that allows the
downstream pipeline (preprocessing, CPI-C construction, XGBoost regression,
Cox survival analysis, Random Forest classification) to be developed and
validated end-to-end.

Calibration sources (used to set distribution parameters):
- U.S. Bureau of Labor Statistics, CPI-U Food Away From Home, 2019-2024:
  cumulative ~26% increase nationally over the window.
  https://www.bls.gov/cpi/
- Yelp Local Economic Impact Report (2020-2022): pandemic-era closure
  rates of roughly 17-22% for independent restaurants in major metros.
- Bureau of Labor Statistics regional CPI tables (NYC, LA, Chicago):
  used to set city-level inflation offsets.

All "cuisine-specific" inflation offsets in this script are HYPOTHESES
drawn from the original project proposal (Mexican, Chinese, Vietnamese
expected to inflate more aggressively due to thinner margins). They are
NOT measured values and the paper is explicit about this.

Reproducibility: every random draw is keyed off SEED=42.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_RESTAURANTS = 6000

CITIES = ["NYC", "LA", "Chicago"]
CITY_WEIGHTS = [0.45, 0.35, 0.20]

# City-level baseline inflation 2019-2024, calibrated to BLS regional CPI
# food-away-from-home indices. These are the *city-wide* baselines; cuisine
# offsets are added on top.
CITY_BASE_INFLATION = {
    "NYC": 0.28,      # high rent-driven cost structure
    "LA": 0.26,       # minimum wage increases during window
    "Chicago": 0.23,  # lower rent baseline
}

CUISINES = [
    "Mexican", "Chinese", "Italian", "Japanese", "American",
    "Vietnamese", "Indian", "Thai", "Mediterranean", "FastFood",
]
CUISINE_WEIGHTS = [0.14, 0.13, 0.12, 0.10, 0.15,
                   0.06, 0.08, 0.07, 0.06, 0.09]

# Cuisine-level inflation offsets relative to the city baseline. Hypotheses
# from the project proposal: thin-margin cuisines (Mexican, Chinese,
# Vietnamese) raise prices more aggressively; higher-margin cuisines
# (Italian, Japanese) raise less.
CUISINE_INFLATION_OFFSET = {
    "Mexican":       0.06,
    "Chinese":       0.04,
    "Italian":      -0.03,
    "Japanese":     -0.01,
    "American":      0.00,
    "Vietnamese":    0.05,
    "Indian":        0.02,
    "Thai":          0.03,
    "Mediterranean": 0.01,
    "FastFood":      0.08,  # franchise pricing power
}

# 2019 baseline entree prices ($) by cuisine, drawn from publicly reported
# pre-pandemic averages. These are population means; per-restaurant prices
# vary around them.
CUISINE_BASELINE_PRICE_2019 = {
    "Mexican":       12.50,
    "Chinese":       13.00,
    "Italian":       18.00,
    "Japanese":      22.00,
    "American":      16.00,
    "Vietnamese":    11.50,
    "Indian":        14.50,
    "Thai":          14.00,
    "Mediterranean": 17.00,
    "FastFood":       8.50,
}

# City-level rent index multiplier (proxy for neighborhood rent burden)
CITY_RENT_MULTIPLIER = {"NYC": 1.45, "LA": 1.20, "Chicago": 1.00}

# Baseline 4-year closure probability (Yelp 2022 report ~ 17-22%)
BASE_CLOSURE_PROB = 0.22


def simulate_restaurants(n: int = N_RESTAURANTS, seed: int = SEED) -> pd.DataFrame:
    """Generate the restaurant-level dataset.

    Each row is one restaurant with: city, cuisine, 2019 baseline price,
    2024 price, years open before 2019, review trajectory, neighborhood
    rent index, closure event indicator, and time-to-event (months from
    Jan 2019 to closure or to study end Dec 2024).
    """
    rng = np.random.default_rng(seed)

    cities = rng.choice(CITIES, size=n, p=CITY_WEIGHTS)
    cuisines = rng.choice(CUISINES, size=n, p=CUISINE_WEIGHTS)

    # 2019 baseline price: mean by cuisine, log-normal noise per restaurant
    baseline_means = np.array([CUISINE_BASELINE_PRICE_2019[c] for c in cuisines])
    price_2019 = baseline_means * rng.lognormal(mean=0.0, sigma=0.18, size=n)

    # Inflation rate: city base + cuisine offset + per-restaurant noise
    city_infl = np.array([CITY_BASE_INFLATION[c] for c in cities])
    cuisine_infl = np.array([CUISINE_INFLATION_OFFSET[c] for c in cuisines])
    noise = rng.normal(0.0, 0.05, size=n)
    inflation_rate = city_infl + cuisine_infl + noise
    price_2024 = price_2019 * (1.0 + inflation_rate)

    # Years open before 2019 (Pareto-ish: most restaurants are young)
    years_open_pre_2019 = np.clip(rng.exponential(scale=6.0, size=n), 0.5, 40.0)

    # Pre-2019 review trajectory: standardized score in roughly [-2, 2]
    review_trajectory = rng.normal(0.0, 1.0, size=n)

    # Neighborhood rent index (city multiplier * local variation)
    rent_mult = np.array([CITY_RENT_MULTIPLIER[c] for c in cities])
    rent_index = rent_mult * rng.lognormal(mean=0.0, sigma=0.12, size=n)

    # 2019 price tier (Yelp-style $ / $$ / $$$ / $$$$)
    tiers = np.where(price_2019 < 12, "$",
             np.where(price_2019 < 18, "$$",
             np.where(price_2019 < 25, "$$$", "$$$$")))

    # Closure model: hazard increases with inflation, decreases with
    # established history and positive review trajectory, increases with
    # high rent. We construct a hazard score, convert to closure prob,
    # and then assign a uniform time-to-event in [1, 72] months for
    # closures; survivors are censored at month 72.
    hazard = (
        BASE_CLOSURE_PROB
        + 0.95 * (inflation_rate - inflation_rate.mean())
        - 0.010 * years_open_pre_2019
        - 0.05 * review_trajectory
        + 0.10 * (rent_index - rent_index.mean())
    )
    closure_prob = np.clip(hazard, 0.03, 0.85)
    closed = rng.uniform(0, 1, size=n) < closure_prob

    # Time to event: closures get a time drawn weighted toward later
    # months for low-hazard, earlier months for high-hazard.
    time_to_event_months = np.where(
        closed,
        np.clip(rng.normal(36 - 30 * (closure_prob - 0.18), 12, size=n), 1, 71),
        72,  # censored at study end
    ).astype(int)

    df = pd.DataFrame({
        "restaurant_id": [f"R{i:05d}" for i in range(n)],
        "city": cities,
        "cuisine": cuisines,
        "price_tier_2019": tiers,
        "price_2019": np.round(price_2019, 2),
        "price_2024": np.round(price_2024, 2),
        "inflation_rate": np.round(inflation_rate, 4),
        "years_open_pre_2019": np.round(years_open_pre_2019, 1),
        "review_trajectory": np.round(review_trajectory, 3),
        "rent_index": np.round(rent_index, 3),
        "closed": closed.astype(int),
        "time_to_event_months": time_to_event_months,
    })
    return df


def main() -> None:
    df = simulate_restaurants()
    out_path = "data/restaurants.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} restaurants to {out_path}")
    print(f"Closure rate: {df['closed'].mean():.1%}")
    print(f"Mean inflation: {df['inflation_rate'].mean():.1%}")
    print("\nClosure rate by cuisine:")
    print(df.groupby("cuisine")["closed"].mean().sort_values(ascending=False).to_string())
    print("\nMean inflation by city:")
    print(df.groupby("city")["inflation_rate"].mean().to_string())


if __name__ == "__main__":
    main()
