"""
Modeling pipeline.

Three models, all trained on the same preprocessed data:

1. XGBoost regression -> predicts inflation_rate (the per-restaurant
   fractional price change 2019->2024). Evaluated with RMSE and R^2.

2. Cox Proportional Hazards survival model -> models time-to-closure
   given covariates. Evaluated with Harrell's C-index.

3. Random Forest classifier -> binary closure prediction (sanity check
   alongside the survival model). Evaluated with ROC-AUC.

We additionally run a hybrid-vs-baseline ABLATION on the Random Forest:
  - Hybrid: includes the engineered cpi_c feature.
  - Baseline: drops cpi_c.
This isolates the marginal value of the unsupervised cuisine-city
clustering signal that the proposal hypothesized would help.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score

from src.preprocess import Preprocessed

SEED = 42


@dataclass
class ModelResults:
    inflation_rmse: float = 0.0
    inflation_r2: float = 0.0
    cox_c_index: float = 0.0
    rf_auc_hybrid: float = 0.0
    rf_auc_baseline: float = 0.0
    rf_feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    cox_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    xgb_model: object = None
    rf_hybrid_model: object = None
    cox_model: object = None
    inflation_preds: np.ndarray = field(default_factory=lambda: np.array([]))


def train_inflation_regressor(p: Preprocessed) -> tuple[xgb.XGBRegressor, float, float, np.ndarray]:
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED,
        n_jobs=4,
    )
    # The cpi_c feature would leak into inflation_rate prediction (it is
    # built FROM mean inflation in each cell). Drop it for this target.
    feats = [c for c in p.feature_names if c != "cpi_c"]
    model.fit(p.X_train[feats], p.y_inflation_train)
    preds = model.predict(p.X_test[feats])
    rmse = float(np.sqrt(mean_squared_error(p.y_inflation_test, preds)))
    r2 = float(r2_score(p.y_inflation_test, preds))
    return model, rmse, r2, preds


def train_cox_model(p: Preprocessed) -> tuple[CoxPHFitter, float, pd.DataFrame]:
    """Fit Cox PH on training data and evaluate C-index on test."""
    train = p.X_train.copy()
    train["duration"] = p.time_train.values
    train["event"] = p.y_closed_train.values

    test = p.X_test.copy()
    test["duration"] = p.time_test.values
    test["event"] = p.y_closed_test.values

    # Cox needs full rank; drop one one-hot per categorical to avoid
    # collinearity with the implicit baseline.
    drop_redundant = ["city_Chicago", "cuisine_American", "price_tier_2019_$$"]
    drop_redundant = [c for c in drop_redundant if c in train.columns]
    train_fit = train.drop(columns=drop_redundant)
    test_fit = test.drop(columns=drop_redundant)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(train_fit, duration_col="duration", event_col="event", show_progress=False)

    # C-index on held-out test
    risk_scores = cph.predict_partial_hazard(test_fit.drop(columns=["duration", "event"]))
    c_index = concordance_index(
        event_times=test_fit["duration"],
        predicted_scores=-risk_scores,  # higher hazard -> shorter survival
        event_observed=test_fit["event"],
    )
    return cph, float(c_index), cph.summary


def train_rf_classifier(p: Preprocessed) -> tuple[RandomForestClassifier, float, float, pd.DataFrame]:
    """Random Forest with ablation: hybrid (with cpi_c) vs baseline (without)."""
    rf_hybrid = RandomForestClassifier(
        n_estimators=400, max_depth=10, random_state=SEED, n_jobs=4,
        class_weight="balanced",
    )
    rf_hybrid.fit(p.X_train, p.y_closed_train)
    auc_hybrid = float(roc_auc_score(
        p.y_closed_test, rf_hybrid.predict_proba(p.X_test)[:, 1]
    ))

    feats_baseline = [c for c in p.feature_names if c != "cpi_c"]
    rf_baseline = RandomForestClassifier(
        n_estimators=400, max_depth=10, random_state=SEED, n_jobs=4,
        class_weight="balanced",
    )
    rf_baseline.fit(p.X_train[feats_baseline], p.y_closed_train)
    auc_baseline = float(roc_auc_score(
        p.y_closed_test, rf_baseline.predict_proba(p.X_test[feats_baseline])[:, 1]
    ))

    importance = pd.DataFrame({
        "feature": p.feature_names,
        "importance": rf_hybrid.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return rf_hybrid, auc_hybrid, auc_baseline, importance


def run_all(p: Preprocessed) -> ModelResults:
    print("Training XGBoost inflation regressor...")
    xgb_model, rmse, r2, preds = train_inflation_regressor(p)
    print(f"  RMSE = {rmse:.4f}, R^2 = {r2:.3f}")

    print("Training Cox proportional hazards model...")
    cox_model, c_index, cox_summary = train_cox_model(p)
    print(f"  C-index = {c_index:.3f}")

    print("Training Random Forest closure classifier (hybrid + ablation)...")
    rf_model, auc_h, auc_b, importance = train_rf_classifier(p)
    print(f"  AUC hybrid (with CPI-C) = {auc_h:.3f}")
    print(f"  AUC baseline (no CPI-C) = {auc_b:.3f}")
    print(f"  Hybrid uplift = {auc_h - auc_b:+.3f}")

    return ModelResults(
        inflation_rmse=rmse,
        inflation_r2=r2,
        cox_c_index=c_index,
        rf_auc_hybrid=auc_h,
        rf_auc_baseline=auc_b,
        rf_feature_importance=importance,
        cox_summary=cox_summary,
        xgb_model=xgb_model,
        rf_hybrid_model=rf_model,
        cox_model=cox_model,
        inflation_preds=preds,
    )


if __name__ == "__main__":
    from src.preprocess import preprocess
    p = preprocess()
    results = run_all(p)
    print("\nTop 10 features by RF importance:")
    print(results.rf_feature_importance.head(10).to_string(index=False))
