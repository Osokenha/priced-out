"""
Visualizations for the Priced Out report.

Produces (saved to results/figures/):
- price_trends_by_cuisine.png : 2019 vs 2024 mean prices, faceted by city
- inflation_heatmap.png       : cuisine x city CPI-C heatmap
- closure_vs_inflation.png    : scatter of cuisine-level closure rate vs CPI-C
- km_curves_cuisine.png       : Kaplan-Meier survival curves by cuisine
- pca_projection.png          : 2D PCA of feature space colored by closure
- rf_feature_importance.png   : top features from Random Forest
- shap_summary.png            : SHAP summary plot for the XGBoost regressor
- residuals.png               : XGBoost regression residuals
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA

from src.models import ModelResults
from src.preprocess import Preprocessed

FIG_DIR = "results/figures"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def _save(name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def plot_price_trends(raw: pd.DataFrame) -> str:
    means = raw.groupby(["cuisine", "city"])[["price_2019", "price_2024"]].mean().reset_index()
    long = means.melt(id_vars=["cuisine", "city"], var_name="year", value_name="price")
    long["year"] = long["year"].str.extract(r"(\d+)").astype(int)

    g = sns.FacetGrid(long, col="city", height=4.0, aspect=1.0, sharey=True)
    g.map_dataframe(sns.lineplot, x="year", y="price", hue="cuisine", marker="o")
    g.add_legend(title="Cuisine", bbox_to_anchor=(1.02, 0.5), loc="center left")
    g.set_axis_labels("Year", "Mean entree price ($)")
    g.set_titles("{col_name}")
    plt.subplots_adjust(top=0.88)
    g.figure.suptitle("Mean entree prices by cuisine, 2019 vs 2024")
    return _save("price_trends_by_cuisine.png")


def plot_inflation_heatmap(cpi_c_table: pd.DataFrame) -> str:
    pivot = cpi_c_table.pivot(index="cuisine", columns="city", values="cpi_c")
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)
    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="rocket_r",
                cbar_kws={"label": "CPI-C (%)"})
    plt.title("CPI-C by cuisine and city")
    plt.xlabel("City")
    plt.ylabel("Cuisine")
    return _save("inflation_heatmap.png")


def plot_closure_vs_inflation(raw: pd.DataFrame) -> str:
    agg = raw.groupby("cuisine").agg(
        closure_rate=("closed", "mean"),
        cpi_c=("inflation_rate", "mean"),
    ).reset_index()
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=agg, x="cpi_c", y="closure_rate", s=120, color="#444")
    for _, row in agg.iterrows():
        plt.annotate(row["cuisine"], (row["cpi_c"], row["closure_rate"]),
                     xytext=(5, 5), textcoords="offset points", fontsize=9)
    # Trend line
    x = agg["cpi_c"].values
    y = agg["closure_rate"].values
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    plt.plot(xs, np.polyval(coef, xs), "--", color="#c0392b", alpha=0.7,
             label=f"OLS fit: slope={coef[0]:.2f}")
    plt.xlabel("Mean inflation 2019-2024 (CPI-C)")
    plt.ylabel("4-year closure rate")
    plt.title("Cuisine-level closure rate vs price inflation")
    plt.legend()
    return _save("closure_vs_inflation.png")


def plot_km_curves(raw: pd.DataFrame) -> str:
    plt.figure(figsize=(7, 5))
    kmf = KaplanMeierFitter()
    cuisines_to_plot = (
        raw.groupby("cuisine")["closed"].mean().sort_values(ascending=False).index.tolist()
    )
    palette = sns.color_palette("tab10", n_colors=len(cuisines_to_plot))
    for cuisine, color in zip(cuisines_to_plot, palette):
        sub = raw[raw["cuisine"] == cuisine]
        kmf.fit(sub["time_to_event_months"], sub["closed"], label=cuisine)
        kmf.plot_survival_function(ci_show=False, color=color)
    plt.xlabel("Months since Jan 2019")
    plt.ylabel("Survival probability")
    plt.title("Kaplan-Meier survival curves by cuisine")
    plt.legend(loc="lower left", fontsize=8, ncol=2)
    plt.ylim(0.5, 1.02)
    return _save("km_curves_cuisine.png")


def plot_pca(p: Preprocessed) -> str:
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(p.X_train.values)
    df = pd.DataFrame({
        "PC1": coords[:, 0], "PC2": coords[:, 1],
        "Closed": p.y_closed_train.map({0: "Open", 1: "Closed"}).values,
    })
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Closed", alpha=0.45, s=14,
                    palette={"Open": "#3498db", "Closed": "#e74c3c"})
    plt.title(f"PCA projection of feature space "
              f"({pca.explained_variance_ratio_.sum():.0%} var explained)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    return _save("pca_projection.png")


def plot_rf_importance(results: ModelResults) -> str:
    top = results.rf_feature_importance.head(12).iloc[::-1].copy()
    # Escape $ for matplotlib (otherwise it tries to parse mathtext)
    top["feature"] = top["feature"].str.replace("$", r"\$", regex=False)
    plt.figure(figsize=(6.5, 5))
    plt.barh(top["feature"], top["importance"], color="#34495e")
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance (top 12)")
    return _save("rf_feature_importance.png")


def plot_shap(p: Preprocessed, results: ModelResults) -> str:
    feats = [c for c in p.feature_names if c != "cpi_c"]
    explainer = shap.TreeExplainer(results.xgb_model)
    sample = p.X_test[feats].sample(min(500, len(p.X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False, max_display=10)
    return _save("shap_summary.png")


def plot_residuals(p: Preprocessed, results: ModelResults) -> str:
    residuals = p.y_inflation_test.values - results.inflation_preds
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(results.inflation_preds, residuals, alpha=0.4, s=10, color="#2c3e50")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Predicted inflation rate")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title("Residuals vs fitted (XGBoost)")
    axes[1].hist(residuals, bins=40, color="#2c3e50", edgecolor="white")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    return _save("residuals.png")


def make_all(raw_csv: str = "data/restaurants.csv") -> dict[str, str]:
    from src.preprocess import preprocess
    from src.models import run_all

    raw = pd.read_csv(raw_csv)
    p = preprocess(raw_csv)
    results = run_all(p)

    paths = {
        "price_trends": plot_price_trends(raw),
        "heatmap": plot_inflation_heatmap(p.cpi_c_table),
        "closure_vs_inflation": plot_closure_vs_inflation(raw),
        "km_curves": plot_km_curves(raw),
        "pca": plot_pca(p),
        "rf_importance": plot_rf_importance(results),
        "shap": plot_shap(p, results),
        "residuals": plot_residuals(p, results),
    }
    return paths


if __name__ == "__main__":
    paths = make_all()
    for k, v in paths.items():
        print(f"  {k}: {v}")
