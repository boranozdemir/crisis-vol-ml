"""
Final forecast visualization for the 2008 crisis volatility project.

This script is intended to be placed directly under:

    src/forecast_visualizer.py

Purpose
-------
Create final forecast-comparison figures from the outputs of:

1. Strict pre-crisis ML experiment
2. Rolling ML experiment
3. Rolling GARCH-family forecasting experiment
4. Final forecast comparison script

Inputs
------
outputs/tables/final_comparison/final_forecast_performance_overall.csv
outputs/tables/final_comparison/final_forecast_performance_by_period.csv
outputs/tables/final_comparison/final_garch_vs_ml_rolling.csv

outputs/models/garch_forecasting/garch_rolling_predictions.csv
outputs/models/ml_rolling/ml_rolling_predictions.csv

Outputs
-------
outputs/figures/final_forecast/
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

def find_project_root() -> Path:
    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "outputs").exists() and (parent / "src").exists():
            return parent

    cwd = Path.cwd()
    if (cwd / "outputs").exists():
        return cwd

    raise FileNotFoundError(
        "Could not locate project root. Expected to find outputs/ "
        "above the current script or in the current working directory."
    )


PROJECT_ROOT = find_project_root()

FINAL_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "final_comparison"
GARCH_MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "garch_forecasting"
ML_ROLLING_MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "ml_rolling"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "final_forecast"

ASSETS = ["SPY", "XLF", "KBE"]


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def ensure_figure_dir() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}\n\n"
            "Run the previous pipeline scripts first."
        )

    return pd.read_csv(path)


def load_final_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    overall = read_csv_required(
        FINAL_TABLE_DIR / "final_forecast_performance_overall.csv"
    )
    by_period = read_csv_required(
        FINAL_TABLE_DIR / "final_forecast_performance_by_period.csv"
    )
    garch_vs_ml = read_csv_required(
        FINAL_TABLE_DIR / "final_garch_vs_ml_rolling.csv"
    )

    return overall, by_period, garch_vs_ml


def load_prediction_files() -> tuple[pd.DataFrame, pd.DataFrame]:
    garch_pred = read_csv_required(
        GARCH_MODEL_DIR / "garch_rolling_predictions.csv"
    )
    ml_pred = read_csv_required(
        ML_ROLLING_MODEL_DIR / "ml_rolling_predictions.csv"
    )

    garch_pred["experiment"] = "GARCH_Rolling"
    ml_pred["experiment"] = "ML_Rolling"

    if "target_date" in garch_pred.columns:
        garch_pred["plot_date"] = pd.to_datetime(garch_pred["target_date"])
    else:
        garch_pred["plot_date"] = pd.to_datetime(garch_pred["Date"])

    ml_pred["plot_date"] = pd.to_datetime(ml_pred["Date"])

    for df in [garch_pred, ml_pred]:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Asset"] = df["Asset"].astype(str)
        df["model"] = df["model"].astype(str)
        df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    return garch_pred, ml_pred


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def save_current_figure(filename: str) -> None:
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def short_label(row: pd.Series) -> str:
    exp = str(row["experiment"])
    model = str(row["model"])

    if exp == "GARCH_Rolling":
        return f"GARCH: {model}"
    if exp == "ML_Rolling":
        return f"Rolling ML: {model}"
    if exp == "ML_Strict_PreCrisis":
        return f"Strict ML: {model}"

    return f"{exp}: {model}"


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def plot_overall_qlike_by_asset(overall: pd.DataFrame) -> None:
    """
    Plot overall QLIKE comparison for all experiments/models by asset.
    """

    df = overall.copy()
    df["label"] = df.apply(short_label, axis=1)

    for asset in ASSETS:
        asset_df = df[df["Asset"] == asset].copy()

        if asset_df.empty:
            continue

        asset_df = asset_df.sort_values("qlike").head(12)

        plt.figure(figsize=(11, 6))
        plt.barh(asset_df["label"], asset_df["qlike"])
        plt.gca().invert_yaxis()
        plt.xlabel("QLIKE loss")
        plt.ylabel("Model")
        plt.title(f"Overall forecast performance by QLIKE — {asset}")
        plt.grid(axis="x", alpha=0.3)

        save_current_figure(f"overall_qlike_ranking_{asset}.png")


def plot_best_overall_models(overall: pd.DataFrame) -> None:
    """
    Plot the best overall model by QLIKE for each asset.
    """

    idx = overall.groupby("Asset")["qlike"].idxmin()
    best = overall.loc[idx].copy()
    best["label"] = best.apply(short_label, axis=1)

    best = best.sort_values("Asset")

    plt.figure(figsize=(9, 5))
    plt.bar(best["Asset"], best["qlike"])
    plt.xlabel("Asset")
    plt.ylabel("QLIKE loss")
    plt.title("Best overall forecast model by asset")
    plt.grid(axis="y", alpha=0.3)

    for i, (_, row) in enumerate(best.iterrows()):
        plt.text(
            i,
            row["qlike"],
            row["label"],
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=15,
        )

    save_current_figure("best_overall_model_by_asset.png")


def plot_best_by_period(by_period: pd.DataFrame) -> None:
    """
    Plot best QLIKE model by asset and test period.
    """

    idx = by_period.groupby(["Asset", "Period"])["qlike"].idxmin()
    best = by_period.loc[idx].copy()
    best["asset_period"] = best["Asset"] + " | " + best["Period"]
    best["label"] = best.apply(short_label, axis=1)

    best = best.sort_values(["Asset", "Period"])

    plt.figure(figsize=(11, 5))
    plt.bar(best["asset_period"], best["qlike"])
    plt.xlabel("Asset and period")
    plt.ylabel("QLIKE loss")
    plt.title("Best forecast model by asset and test period")
    plt.xticks(rotation=35, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for i, (_, row) in enumerate(best.iterrows()):
        plt.text(
            i,
            row["qlike"],
            row["model"],
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=20,
        )

    save_current_figure("best_model_by_asset_period.png")


def plot_rolling_garch_vs_ml(garch_vs_ml: pd.DataFrame) -> None:
    """
    Plot best rolling GARCH vs best rolling ML QLIKE values.
    """

    df = garch_vs_ml.copy()

    x = np.arange(len(df))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, df["best_garch_qlike"], width, label="Best rolling GARCH")
    plt.bar(x + width / 2, df["best_ml_qlike"], width, label="Best rolling ML")

    plt.xticks(x, df["Asset"])
    plt.xlabel("Asset")
    plt.ylabel("QLIKE loss")
    plt.title("Best rolling GARCH vs best rolling ML")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    save_current_figure("rolling_garch_vs_rolling_ml_qlike.png")


def plot_qlike_diff_ml_minus_garch(garch_vs_ml: pd.DataFrame) -> None:
    """
    Plot QLIKE difference: best ML minus best GARCH.
    Positive values mean GARCH has lower QLIKE.
    """

    df = garch_vs_ml.copy()

    plt.figure(figsize=(8, 5))
    plt.axhline(0.0, linewidth=1)
    plt.bar(df["Asset"], df["qlike_diff_ml_minus_garch"])
    plt.xlabel("Asset")
    plt.ylabel("QLIKE difference: ML - GARCH")
    plt.title("Rolling ML minus rolling GARCH QLIKE difference")
    plt.grid(axis="y", alpha=0.3)

    save_current_figure("qlike_diff_ml_minus_garch.png")


def build_forecast_path_frame(
    garch_pred: pd.DataFrame,
    ml_pred: pd.DataFrame,
    asset: str,
    period: str = "crisis",
) -> pd.DataFrame:
    """
    Build a merged forecast path frame for one asset and period.
    """

    garch_asset = garch_pred[
        (garch_pred["Asset"] == asset)
        & (garch_pred["Period"] == period)
        & (garch_pred["model"].isin(["Naive_Rolling22", "GJR_GARCH_11"]))
    ].copy()

    ml_asset = ml_pred[
        (ml_pred["Asset"] == asset)
        & (ml_pred["Period"] == period)
        & (ml_pred["model"].isin(["RandomForest"]))
    ].copy()

    frames = []

    if not garch_asset.empty:
        frames.append(garch_asset)

    if not ml_asset.empty:
        frames.append(ml_asset)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    combined["series"] = combined["experiment"] + " | " + combined["model"]

    return combined.sort_values("plot_date").reset_index(drop=True)


def plot_forecast_paths(
    garch_pred: pd.DataFrame,
    ml_pred: pd.DataFrame,
    period: str = "crisis",
) -> None:
    """
    Plot actual volatility proxy against selected rolling forecasts.

    The actual target is r_{t+1}^2. A 22-day rolling average of the actual
    target is also plotted to make the noisy volatility proxy easier to read.
    """

    for asset in ASSETS:
        df = build_forecast_path_frame(
            garch_pred=garch_pred,
            ml_pred=ml_pred,
            asset=asset,
            period=period,
        )

        if df.empty:
            continue

        actual_df = (
            df[["plot_date", "actual"]]
            .drop_duplicates("plot_date")
            .sort_values("plot_date")
            .copy()
        )
        actual_df["actual_22d_mean"] = actual_df["actual"].rolling(22).mean()

        plt.figure(figsize=(12, 6))

        plt.plot(
            actual_df["plot_date"],
            actual_df["actual_22d_mean"],
            linewidth=2,
            label="Actual r², 22-day mean",
        )

        for series_name, group in df.groupby("series"):
            group = group.sort_values("plot_date")
            plt.plot(
                group["plot_date"],
                group["prediction"],
                linewidth=1.5,
                label=series_name,
            )

        plt.xlabel("Date")
        plt.ylabel("Variance / squared return")
        plt.title(f"Forecast path comparison — {asset}, {period}")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)

        save_current_figure(f"forecast_path_{period}_{asset}.png")


def plot_prediction_mean_vs_actual_mean(overall: pd.DataFrame) -> None:
    """
    Plot actual mean versus prediction mean for final overall comparisons.
    """

    df = overall.copy()
    df = df[
        df["experiment"].isin(["GARCH_Rolling", "ML_Rolling"])
        & df["model"].isin(["GJR_GARCH_11", "RandomForest", "Naive_Rolling22"])
    ].copy()

    if df.empty:
        return

    df["label"] = df.apply(short_label, axis=1)

    for asset in ASSETS:
        asset_df = df[df["Asset"] == asset].copy()

        if asset_df.empty:
            continue

        x = np.arange(len(asset_df))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, asset_df["actual_mean"], width, label="Actual mean")
        plt.bar(x + width / 2, asset_df["prediction_mean"], width, label="Prediction mean")

        plt.xticks(x, asset_df["label"], rotation=25, ha="right")
        plt.ylabel("Mean squared return / variance forecast")
        plt.title(f"Actual mean vs prediction mean — {asset}")
        plt.legend()
        plt.grid(axis="y", alpha=0.3)

        save_current_figure(f"actual_vs_prediction_mean_{asset}.png")


# ---------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------

def write_visual_summary(
    overall: pd.DataFrame,
    by_period: pd.DataFrame,
    garch_vs_ml: pd.DataFrame,
) -> None:
    lines = []

    lines.append("Final forecast visualization summary")
    lines.append("=" * 44)
    lines.append("")
    lines.append("Figures created in:")
    lines.append(f"- {FIGURE_DIR}")
    lines.append("")
    lines.append("Main figures:")
    lines.append("- overall_qlike_ranking_[ASSET].png")
    lines.append("- best_overall_model_by_asset.png")
    lines.append("- best_model_by_asset_period.png")
    lines.append("- rolling_garch_vs_rolling_ml_qlike.png")
    lines.append("- qlike_diff_ml_minus_garch.png")
    lines.append("- forecast_path_crisis_[ASSET].png")
    lines.append("- actual_vs_prediction_mean_[ASSET].png")
    lines.append("")

    lines.append("Best overall QLIKE models:")
    best = overall.loc[overall.groupby("Asset")["qlike"].idxmin()].sort_values("Asset")
    for _, row in best.iterrows():
        lines.append(
            f"- {row['Asset']}: {row['experiment']} | {row['model']} "
            f"(QLIKE={row['qlike']:.6f})"
        )

    lines.append("")
    lines.append("Rolling GARCH vs rolling ML QLIKE difference:")
    for _, row in garch_vs_ml.sort_values("Asset").iterrows():
        lines.append(
            f"- {row['Asset']}: ML - GARCH = "
            f"{row['qlike_diff_ml_minus_garch']:.6f}; "
            f"winner={row['qlike_winner']}"
        )

    (FIGURE_DIR / "forecast_visual_summary.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ensure_figure_dir()

    overall, by_period, garch_vs_ml = load_final_tables()
    garch_pred, ml_pred = load_prediction_files()

    plot_overall_qlike_by_asset(overall)
    plot_best_overall_models(overall)
    plot_best_by_period(by_period)
    plot_rolling_garch_vs_ml(garch_vs_ml)
    plot_qlike_diff_ml_minus_garch(garch_vs_ml)
    plot_forecast_paths(garch_pred, ml_pred, period="crisis")
    plot_prediction_mean_vs_actual_mean(overall)

    write_visual_summary(
        overall=overall,
        by_period=by_period,
        garch_vs_ml=garch_vs_ml,
    )

    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Final forecast visualization completed.")
    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Summary saved to: {FIGURE_DIR / 'forecast_visual_summary.txt'}")


if __name__ == "__main__":
    main()
