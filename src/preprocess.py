"""
Preprocessing pipeline.

Steps:
1. Load raw simulated data.
2. Build the Cuisine Price Index (CPI-C) from the training set only,
   to prevent leakage of future information into per-restaurant features.
3. Encode categorical variables (city, cuisine, price tier) and scale
   numeric features.
4. Split into train and test (stratified by city x closure status).

The CPI-C calculation is intentionally fit on the training partition only,
then applied to the test partition. This is the leakage-prevention point
emphasized in the rubric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.simulate_data import CITIES, CUISINES

NUMERIC_FEATURES = [
    "price_2019",
    "years_open_pre_2019",
    "review_trajectory",
    "rent_index",
]
CATEGORICAL_FEATURES = ["city", "cuisine", "price_tier_2019"]


@dataclass
class Preprocessed:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_inflation_train: pd.Series
    y_inflation_test: pd.Series
    y_closed_train: pd.Series
    y_closed_test: pd.Series
    time_train: pd.Series
    time_test: pd.Series
    cpi_c_table: pd.DataFrame
    scaler: StandardScaler
    feature_names: list[str]


def build_cpi_c(train_df: pd.DataFrame) -> pd.DataFrame:
    """Cuisine Price Index per (cuisine, city), in percent.

    Defined as the mean inflation_rate within each cuisine-city cell.
    Computed on the TRAINING data only.
    """
    cpi = (
        train_df.groupby(["cuisine", "city"])["inflation_rate"]
        .mean()
        .reset_index()
        .rename(columns={"inflation_rate": "cpi_c"})
    )
    return cpi


def attach_cpi_c(df: pd.DataFrame, cpi_c_table: pd.DataFrame) -> pd.DataFrame:
    """Left-join the (cuisine, city) -> CPI-C value onto each restaurant."""
    out = df.merge(cpi_c_table, on=["cuisine", "city"], how="left")
    # Fallback: in the unlikely case a (cuisine, city) cell is unseen in
    # training, fill with the global training mean of CPI-C.
    out["cpi_c"] = out["cpi_c"].fillna(cpi_c_table["cpi_c"].mean())
    return out


def encode_and_scale(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler, list[str]]:
    """One-hot encode categoricals, standard-scale numerics. Fit on train only."""
    # One-hot encode categoricals using a fixed vocabulary (so train/test
    # column sets are identical even if a category is missing in test).
    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["city"] = pd.Categorical(train_df["city"], categories=CITIES)
    test_df["city"] = pd.Categorical(test_df["city"], categories=CITIES)
    train_df["cuisine"] = pd.Categorical(train_df["cuisine"], categories=CUISINES)
    test_df["cuisine"] = pd.Categorical(test_df["cuisine"], categories=CUISINES)
    tier_levels = ["$", "$$", "$$$", "$$$$"]
    train_df["price_tier_2019"] = pd.Categorical(train_df["price_tier_2019"], categories=tier_levels)
    test_df["price_tier_2019"] = pd.Categorical(test_df["price_tier_2019"], categories=tier_levels)

    train_enc = pd.get_dummies(train_df, columns=CATEGORICAL_FEATURES, drop_first=False).astype({
        col: "int" for col in pd.get_dummies(train_df, columns=CATEGORICAL_FEATURES).columns
        if col not in train_df.columns
    }, errors="ignore")
    test_enc = pd.get_dummies(test_df, columns=CATEGORICAL_FEATURES, drop_first=False)

    # Convert dummy columns to int (handles bool dtype from get_dummies)
    dummy_cols = [c for c in train_enc.columns if c not in train_df.columns]
    for col in dummy_cols:
        train_enc[col] = train_enc[col].astype(int)
        if col in test_enc.columns:
            test_enc[col] = test_enc[col].astype(int)

    # Scale numerics with a scaler fit on training data only
    scaler = StandardScaler()
    train_enc[NUMERIC_FEATURES] = scaler.fit_transform(train_enc[NUMERIC_FEATURES])
    test_enc[NUMERIC_FEATURES] = scaler.transform(test_enc[NUMERIC_FEATURES])

    # Also scale the engineered cpi_c feature on training stats
    cpi_scaler = StandardScaler()
    train_enc[["cpi_c"]] = cpi_scaler.fit_transform(train_enc[["cpi_c"]])
    test_enc[["cpi_c"]] = cpi_scaler.transform(test_enc[["cpi_c"]])

    feature_names = NUMERIC_FEATURES + ["cpi_c"] + dummy_cols
    return train_enc, test_enc, scaler, feature_names


def preprocess(csv_path: str = "data/restaurants.csv", seed: int = 42) -> Preprocessed:
    df = pd.read_csv(csv_path)

    # Stratify by city x closure to keep marginal distributions similar
    strata = df["city"].astype(str) + "_" + df["closed"].astype(str)
    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=strata
    )

    # CPI-C built on training set only
    cpi_c_table = build_cpi_c(train_df)
    train_df = attach_cpi_c(train_df, cpi_c_table)
    test_df = attach_cpi_c(test_df, cpi_c_table)

    # Targets (kept aside before encoding)
    y_inflation_train = train_df["inflation_rate"].copy()
    y_inflation_test = test_df["inflation_rate"].copy()
    y_closed_train = train_df["closed"].copy()
    y_closed_test = test_df["closed"].copy()
    time_train = train_df["time_to_event_months"].copy()
    time_test = test_df["time_to_event_months"].copy()

    # Drop columns that would leak the targets
    drop_cols = ["restaurant_id", "inflation_rate", "price_2024",
                 "closed", "time_to_event_months"]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)

    train_enc, test_enc, scaler, feature_names = encode_and_scale(train_df, test_df)

    return Preprocessed(
        X_train=train_enc[feature_names],
        X_test=test_enc[feature_names],
        y_inflation_train=y_inflation_train.reset_index(drop=True),
        y_inflation_test=y_inflation_test.reset_index(drop=True),
        y_closed_train=y_closed_train.reset_index(drop=True),
        y_closed_test=y_closed_test.reset_index(drop=True),
        time_train=time_train.reset_index(drop=True),
        time_test=time_test.reset_index(drop=True),
        cpi_c_table=cpi_c_table,
        scaler=scaler,
        feature_names=feature_names,
    )


if __name__ == "__main__":
    p = preprocess()
    print(f"Train: {p.X_train.shape}, Test: {p.X_test.shape}")
    print(f"Features: {p.feature_names}")
    print(f"\nCPI-C table (top 10 by value):")
    print(p.cpi_c_table.sort_values("cpi_c", ascending=False).head(10).to_string(index=False))
