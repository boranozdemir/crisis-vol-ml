"""
Final forecast comparison for the 2008 crisis volatility project.

This script is intended to be placed directly under:

    src/forecast_comparison.py

Purpose
-------
Combine forecast performance results from:

1. Strict pre-crisis ML experiment
2. Rolling-window ML experiment
3. Rolling GARCH-family forecasting experiment

and create final comparison tables and a compact summary.

Inputs
------
outputs/tables/ml/ml_model_performance_overall.csv
outputs/tables/ml/ml_model_performance_by_period.csv

outputs/tables/ml_rolling/ml_rolling_performance_overall.csv
outputs/tables/ml_rolling/ml_rolling_performance_by_period.csv

outputs/tables/garch_forecasting/garch_rolling_performance_overall.csv
outputs/tables/garch_forecasting/garch_rolling_performance_by_period.csv

Outputs
-------
outputs/tables/final_comparison/final_forecast_performance_overall.csv
outputs/tables/final_comparison/final_forecast_performance_by_period.csv
outputs/tables/final_comparison/final_model_ranking_overall.csv
outputs/tables/final_comparison/final_model_ranking_by_period.csv
outputs/tables/final_comparison/final_best_models_overall.csv
outputs/tables/final_comparison/final_best_models_by_period.csv
outputs/tables/final_comparison/final_best_by_experiment_overall.csv
outputs/tables/final_comparison/final_best_by_experiment_by_period.csv
outputs/tables/final_comparison/final_garch_vs_ml_rolling.csv
outputs/tables/final_comparison/final_forecast_summary.txt
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Find project root.

    Expected structure:

    crisis-vol-ml/
    ├── data/
    ├── outputs/
    └── src/
        └── forecast_comparison.py
    """

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

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "final_comparison"


INPUTS = {
    "ML_Strict_PreCrisis": {
        "overall": PROJECT_ROOT / "outputs" / "tables" / "ml" / "ml_model_performance_overall.csv",
        "by_period": PROJECT_ROOT / "outputs" / "tables" / "ml" / "ml_model_performance_by_period.csv",
        "description": "Strict pre-crisis train / crisis-post-crisis test ML experiment",
    },
    "ML_Rolling": {
        "overall": PROJECT_ROOT / "outputs" / "tables" / "ml_rolling" / "ml_rolling_performance_overall.csv",
        "by_period": PROJECT_ROOT / "outputs" / "tables" / "ml_rolling" / "ml_rolling_performance_by_period.csv",
        "description": "Adaptive rolling-window ML experiment",
    },
    "GARCH_Rolling": {
        "overall": PROJECT_ROOT / "outputs" / "tables" / "garch_forecasting" / "garch_rolling_performance_overall.csv",
        "by_period": PROJECT_ROOT / "outputs" / "tables" / "garch_forecasting" / "garch_rolling_performance_by_period.csv",
        "description": "Adaptive rolling-window GARCH-family experiment",
    },
}


METRIC_COLS = ["rmse", "mae", "qlike"]
REQUIRED_BASE_COLS = ["Asset", "model"]
OPTIONAL_COLS = ["scope", "trained_asset", "trained", "Period"]


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column variants across previous scripts.
    """

    df = df.copy()

    unnamed_cols = [col for col in df.columns if str(col).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    rename_map = {}

    for col in df.columns:
        clean = str(col).strip()
        lower = clean.lower()

        if lower in {"asset", "ticker", "symbol"}:
            rename_map[col] = "Asset"

        elif lower in {"period", "test_period", "target_period", "regime"}:
            rename_map[col] = "Period"

        elif lower in {"model", "model_name", "estimator"}:
            rename_map[col] = "model"

        elif lower in {"scope", "modeling_scope"}:
            rename_map[col] = "scope"

        elif lower in {"trained_asset", "trained", "trained_on", "train_asset"}:
            rename_map[col] = "trained_asset"

        elif lower in {"n_obs", "n", "observations"}:
            rename_map[col] = "n_obs"

        elif lower in {"rmse", "root_mean_squared_error"}:
            rename_map[col] = "rmse"

        elif lower in {"mae", "mean_absolute_error"}:
            rename_map[col] = "mae"

        elif lower in {"qlike", "quasi_likelihood", "quasi_likelihood_loss"}:
            rename_map[col] = "qlike"

        elif lower in {"actual_mean", "mean_actual"}:
            rename_map[col] = "actual_mean"

        elif lower in {"prediction_mean", "pred_mean", "mean_prediction"}:
            rename_map[col] = "prediction_mean"

    return df.rename(columns=rename_map)


def read_performance_file(
    path: Path,
    experiment: str,
    level: str,
) -> Optional[pd.DataFrame]:
    """
    Read one performance file and return normalized rows.
    """

    if not path.exists():
        return None

    df = pd.read_csv(path)
    df = normalize_column_names(df)

    missing = [col for col in REQUIRED_BASE_COLS if col not in df.columns]
    metric_missing = [col for col in METRIC_COLS if col not in df.columns]

    if missing or metric_missing:
        raise ValueError(
            f"Performance file has unexpected columns: {path}\n"
            f"Missing base columns: {missing}\n"
            f"Missing metric columns: {metric_missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    if "Period" not in df.columns:
        df["Period"] = "overall"

    if "scope" not in df.columns:
        if experiment == "GARCH_Rolling":
            df["scope"] = "asset_specific_rolling"
        elif experiment == "ML_Rolling":
            df["scope"] = "asset_specific_rolling"
        else:
            df["scope"] = "unknown"

    if "trained_asset" not in df.columns:
        if experiment in {"GARCH_Rolling", "ML_Rolling"}:
            df["trained_asset"] = df["Asset"]
        else:
            df["trained_asset"] = "unknown"

    if "n_obs" not in df.columns:
        df["n_obs"] = np.nan

    if "actual_mean" not in df.columns:
        df["actual_mean"] = np.nan

    if "prediction_mean" not in df.columns:
        df["prediction_mean"] = np.nan

    df["experiment"] = experiment
    df["level"] = level
    df["model_family"] = df.apply(assign_model_family, axis=1)

    keep_cols = [
        "experiment",
        "level",
        "model_family",
        "scope",
        "trained_asset",
        "Asset",
        "Period",
        "model",
        "n_obs",
        "rmse",
        "mae",
        "qlike",
        "actual_mean",
        "prediction_mean",
    ]

    df = df[keep_cols].copy()

    for col in ["rmse", "mae", "qlike", "actual_mean", "prediction_mean"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------
# Model labels
# ---------------------------------------------------------------------

def assign_model_family(row: pd.Series) -> str:
    model = str(row.get("model", "")).lower()
    experiment = str(row.get("experiment", ""))

    if "naive" in model or "rolling22" in model:
        return "Benchmark"

    if experiment == "GARCH_Rolling":
        return "GARCH"

    if experiment in {"ML_Strict_PreCrisis", "ML_Rolling"}:
        return "ML"

    if "garch" in model or "egarch" in model:
        return "GARCH"

    return "ML"


def add_rankings(
    df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Add metric-specific ranks within Asset or Asset-Period groups.
    Lower metric values are better.
    """

    out = df.copy()

    out["qlike_rank"] = (
        out.groupby(group_cols)["qlike"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    out["rmse_rank"] = (
        out.groupby(group_cols)["rmse"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    out["mae_rank"] = (
        out.groupby(group_cols)["mae"]
        .rank(method="min", ascending=True)
        .astype("Int64")
    )

    sort_cols = group_cols + ["qlike_rank", "rmse_rank", "mae_rank", "experiment", "model"]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


def best_by_qlike(
    df: pd.DataFrame,
    group_cols: List[str],
) -> pd.DataFrame:
    """
    Select best rows by QLIKE within each group.
    """

    valid = df.dropna(subset=["qlike"]).copy()

    if valid.empty:
        return pd.DataFrame(columns=df.columns)

    idx = valid.groupby(group_cols)["qlike"].idxmin()
    best = valid.loc[idx].sort_values(group_cols).reset_index(drop=True)

    return best


# ---------------------------------------------------------------------
# GARCH vs ML rolling comparison
# ---------------------------------------------------------------------

def build_garch_vs_ml_rolling(overall_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare best rolling GARCH-family model against best rolling ML model.

    The comparison excludes Naive_Rolling22 because the goal is to compare
    model families.
    """

    rows = []

    for asset, group in overall_df.groupby("Asset"):
        garch_group = group[
            (group["experiment"] == "GARCH_Rolling")
            & (group["model_family"] == "GARCH")
        ].copy()

        ml_group = group[
            (group["experiment"] == "ML_Rolling")
            & (group["model_family"] == "ML")
        ].copy()

        if garch_group.empty or ml_group.empty:
            continue

        best_garch = garch_group.loc[garch_group["qlike"].idxmin()]
        best_ml = ml_group.loc[ml_group["qlike"].idxmin()]

        qlike_diff = float(best_ml["qlike"] - best_garch["qlike"])
        rmse_diff = float(best_ml["rmse"] - best_garch["rmse"])
        mae_diff = float(best_ml["mae"] - best_garch["mae"])

        rows.append(
            {
                "Asset": asset,
                "best_garch_model": best_garch["model"],
                "best_garch_qlike": best_garch["qlike"],
                "best_garch_rmse": best_garch["rmse"],
                "best_garch_mae": best_garch["mae"],
                "best_ml_model": best_ml["model"],
                "best_ml_qlike": best_ml["qlike"],
                "best_ml_rmse": best_ml["rmse"],
                "best_ml_mae": best_ml["mae"],
                "qlike_diff_ml_minus_garch": qlike_diff,
                "rmse_diff_ml_minus_garch": rmse_diff,
                "mae_diff_ml_minus_garch": mae_diff,
                "qlike_winner": "GARCH_Rolling" if qlike_diff > 0 else "ML_Rolling",
                "rmse_winner": "GARCH_Rolling" if rmse_diff > 0 else "ML_Rolling",
                "mae_winner": "GARCH_Rolling" if mae_diff > 0 else "ML_Rolling",
            }
        )

    return pd.DataFrame(rows).sort_values("Asset").reset_index(drop=True)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def format_metric(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:.6f}"


def build_summary(
    found_inputs: Dict[str, Dict[str, bool]],
    overall_df: pd.DataFrame,
    by_period_df: pd.DataFrame,
    best_overall: pd.DataFrame,
    best_by_period: pd.DataFrame,
    best_by_experiment_overall: pd.DataFrame,
    best_by_experiment_by_period: pd.DataFrame,
    garch_vs_ml: pd.DataFrame,
) -> str:
    lines = []

    lines.append("Final forecast comparison summary")
    lines.append("=" * 41)
    lines.append("")
    lines.append("Purpose:")
    lines.append(
        "This file combines strict ML, rolling ML, and rolling GARCH-family "
        "forecast performance results into one comparison layer."
    )
    lines.append("")
    lines.append("Main reading rules:")
    lines.append("- Lower RMSE, MAE, and QLIKE indicate better forecast performance.")
    lines.append("- QLIKE is treated as the preferred volatility forecast loss.")
    lines.append("- Strict ML tests crisis-regime generalization from pre-crisis training.")
    lines.append("- Rolling ML and rolling GARCH test adaptive one-step-ahead forecasting.")
    lines.append("")

    lines.append("Input files found:")
    for experiment, levels in found_inputs.items():
        overall_flag = "yes" if levels.get("overall", False) else "no"
        period_flag = "yes" if levels.get("by_period", False) else "no"
        lines.append(
            f"- {experiment}: overall={overall_flag}, by_period={period_flag}"
        )

    lines.append("")
    lines.append("Combined rows:")
    lines.append(f"- Overall rows: {len(overall_df)}")
    lines.append(f"- By-period rows: {len(by_period_df)}")
    lines.append("")

    lines.append("Best models by QLIKE, overall:")
    for _, row in best_overall.iterrows():
        lines.append(
            f"- asset={row['Asset']}: "
            f"{row['experiment']} | {row['model']} "
            f"(family={row['model_family']}, "
            f"QLIKE={format_metric(row['qlike'])}, "
            f"RMSE={format_metric(row['rmse'])}, "
            f"MAE={format_metric(row['mae'])})"
        )

    lines.append("")
    lines.append("Best models by QLIKE, by test period:")
    for _, row in best_by_period.iterrows():
        lines.append(
            f"- asset={row['Asset']} | period={row['Period']}: "
            f"{row['experiment']} | {row['model']} "
            f"(family={row['model_family']}, "
            f"QLIKE={format_metric(row['qlike'])})"
        )

    lines.append("")
    lines.append("Best models by experiment, overall:")
    for _, row in best_by_experiment_overall.iterrows():
        lines.append(
            f"- {row['experiment']} | asset={row['Asset']}: "
            f"{row['model']} "
            f"(QLIKE={format_metric(row['qlike'])})"
        )

    if not garch_vs_ml.empty:
        lines.append("")
        lines.append("Rolling GARCH vs rolling ML, overall:")
        lines.append(
            "- qlike_diff_ml_minus_garch > 0 means the best rolling GARCH model "
            "has lower QLIKE than the best rolling ML model."
        )

        for _, row in garch_vs_ml.iterrows():
            lines.append(
                f"- asset={row['Asset']}: "
                f"best GARCH={row['best_garch_model']} "
                f"(QLIKE={format_metric(row['best_garch_qlike'])}), "
                f"best ML={row['best_ml_model']} "
                f"(QLIKE={format_metric(row['best_ml_qlike'])}), "
                f"winner={row['qlike_winner']}, "
                f"diff={format_metric(row['qlike_diff_ml_minus_garch'])}"
            )

    lines.append("")
    lines.append("Recommended interpretation:")
    lines.append(
        "- Use GARCH-family models, especially GJR-GARCH, for structural "
        "interpretation of persistence and downside asymmetry."
    )
    lines.append(
        "- Use the strict ML experiment as a regime-shift stress test."
    )
    lines.append(
        "- Use rolling ML as the adaptive machine-learning benchmark."
    )
    lines.append(
        "- The final forecast conclusion should primarily rely on rolling "
        "out-of-sample QLIKE comparisons."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ensure_output_dir()

    overall_frames = []
    by_period_frames = []
    found_inputs: Dict[str, Dict[str, bool]] = {}

    for experiment, paths in INPUTS.items():
        found_inputs[experiment] = {"overall": False, "by_period": False}

        overall_df = read_performance_file(
            path=paths["overall"],
            experiment=experiment,
            level="overall",
        )

        if overall_df is not None:
            found_inputs[experiment]["overall"] = True
            overall_frames.append(overall_df)

        by_period_df = read_performance_file(
            path=paths["by_period"],
            experiment=experiment,
            level="by_period",
        )

        if by_period_df is not None:
            found_inputs[experiment]["by_period"] = True
            by_period_frames.append(by_period_df)

    if not overall_frames:
        raise RuntimeError(
            "No overall performance files were found. "
            "Run ml_models.py, ml_rolling_models.py, and garch_forecasting.py first."
        )

    if not by_period_frames:
        raise RuntimeError(
            "No by-period performance files were found. "
            "Run ml_models.py, ml_rolling_models.py, and garch_forecasting.py first."
        )

    overall_df = pd.concat(overall_frames, ignore_index=True)
    by_period_df = pd.concat(by_period_frames, ignore_index=True)

    # Clean period labels for overall table
    overall_df["Period"] = "overall"

    # Rankings
    ranking_overall = add_rankings(overall_df, group_cols=["Asset"])
    ranking_by_period = add_rankings(by_period_df, group_cols=["Asset", "Period"])

    # Best rows
    best_overall = best_by_qlike(overall_df, group_cols=["Asset"])
    best_by_period = best_by_qlike(by_period_df, group_cols=["Asset", "Period"])

    best_by_experiment_overall = best_by_qlike(
        overall_df,
        group_cols=["experiment", "Asset"],
    )

    best_by_experiment_by_period = best_by_qlike(
        by_period_df,
        group_cols=["experiment", "Asset", "Period"],
    )

    garch_vs_ml = build_garch_vs_ml_rolling(overall_df)

    # Output paths
    overall_path = OUTPUT_DIR / "final_forecast_performance_overall.csv"
    by_period_path = OUTPUT_DIR / "final_forecast_performance_by_period.csv"
    ranking_overall_path = OUTPUT_DIR / "final_model_ranking_overall.csv"
    ranking_by_period_path = OUTPUT_DIR / "final_model_ranking_by_period.csv"
    best_overall_path = OUTPUT_DIR / "final_best_models_overall.csv"
    best_by_period_path = OUTPUT_DIR / "final_best_models_by_period.csv"
    best_by_experiment_overall_path = OUTPUT_DIR / "final_best_by_experiment_overall.csv"
    best_by_experiment_by_period_path = OUTPUT_DIR / "final_best_by_experiment_by_period.csv"
    garch_vs_ml_path = OUTPUT_DIR / "final_garch_vs_ml_rolling.csv"
    summary_path = OUTPUT_DIR / "final_forecast_summary.txt"

    # Save
    overall_df.to_csv(overall_path, index=False)
    by_period_df.to_csv(by_period_path, index=False)
    ranking_overall.to_csv(ranking_overall_path, index=False)
    ranking_by_period.to_csv(ranking_by_period_path, index=False)
    best_overall.to_csv(best_overall_path, index=False)
    best_by_period.to_csv(best_by_period_path, index=False)
    best_by_experiment_overall.to_csv(best_by_experiment_overall_path, index=False)
    best_by_experiment_by_period.to_csv(best_by_experiment_by_period_path, index=False)
    garch_vs_ml.to_csv(garch_vs_ml_path, index=False)

    summary = build_summary(
        found_inputs=found_inputs,
        overall_df=overall_df,
        by_period_df=by_period_df,
        best_overall=best_overall,
        best_by_period=best_by_period,
        best_by_experiment_overall=best_by_experiment_overall,
        best_by_experiment_by_period=best_by_experiment_by_period,
        garch_vs_ml=garch_vs_ml,
    )

    summary_path.write_text(summary, encoding="utf-8")

    print(f"Project root: {PROJECT_ROOT}")
    print("")
    print("Final forecast comparison completed.")
    print(f"Overall performance saved to: {overall_path}")
    print(f"By-period performance saved to: {by_period_path}")
    print(f"Overall ranking saved to: {ranking_overall_path}")
    print(f"By-period ranking saved to: {ranking_by_period_path}")
    print(f"Best overall models saved to: {best_overall_path}")
    print(f"Best by-period models saved to: {best_by_period_path}")
    print(f"Best by-experiment overall saved to: {best_by_experiment_overall_path}")
    print(f"Best by-experiment by-period saved to: {best_by_experiment_by_period_path}")
    print(f"GARCH vs ML rolling comparison saved to: {garch_vs_ml_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
