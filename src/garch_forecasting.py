"""
Rolling GARCH-family volatility forecasts for the 2008 crisis project.

This script is intended to be placed directly under:

    src/garch_forecasting.py

Purpose
-------
Estimate rolling one-step-ahead volatility forecasts from GARCH-family models
and compare them with a simple rolling-volatility benchmark.

Input
-----
data/processed/log_returns.csv

Output
------
outputs/models/garch_forecasting/garch_rolling_predictions.csv
outputs/tables/garch_forecasting/garch_rolling_performance_overall.csv
outputs/tables/garch_forecasting/garch_rolling_performance_by_period.csv
outputs/tables/garch_forecasting/garch_rolling_fit_errors.csv
outputs/tables/garch_forecasting/garch_rolling_summary.txt

Forecast design
---------------
For each asset and forecast origin date t:

1. Use only returns up to and including date t.
2. Fit the GARCH-family model on a rolling window.
3. Forecast one-step-ahead conditional variance for t+1.
4. Compare the forecast with realized squared return r_{t+1}^2.

The return series is assumed to be scaled by 100, consistent with the earlier
GARCH estimation pipeline.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Optional dependency check
# ---------------------------------------------------------------------

try:
    from arch import arch_model
except ImportError as exc:
    raise ImportError(
        "The 'arch' package is required for GARCH forecasting.\n"
        "Install it with:\n\n"
        "    pip install arch\n"
    ) from exc


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Find the project root.

    Expected structure:

    crisis-vol-ml/
    ├── data/
    ├── outputs/
    └── src/
        └── garch_forecasting.py
    """

    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "data" / "processed").exists() and (parent / "src").exists():
            return parent

    cwd = Path.cwd()
    if (cwd / "data" / "processed").exists():
        return cwd

    raise FileNotFoundError(
        "Could not locate project root. Expected to find data/processed/ "
        "above the current script or in the current working directory."
    )


PROJECT_ROOT = find_project_root()

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "log_returns.csv"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "garch_forecasting"
TABLE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "garch_forecasting"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ASSETS = ["SPY", "XLF", "KBE"]

TEST_START_DATE = "2007-07-01"

ROLLING_WINDOW = 500
MIN_TRAIN_SIZE = 250

# True one-step-ahead GARCH forecasting requires daily refitting because the
# conditional variance at t+1 depends on information through date t.
REFIT_EVERY = 1

NAIVE_WINDOW = 22

RANDOM_STATE = 42
EPS = 1e-8


MODEL_SPECS = {
    "GARCH_11": {
        "mean": "Constant",
        "vol": "GARCH",
        "p": 1,
        "o": 0,
        "q": 1,
        "dist": "t",
    },
    "GJR_GARCH_11": {
        "mean": "Constant",
        "vol": "GARCH",
        "p": 1,
        "o": 1,
        "q": 1,
        "dist": "t",
    },
    "EGARCH_11": {
        "mean": "Constant",
        "vol": "EGARCH",
        "p": 1,
        "o": 1,
        "q": 1,
        "dist": "t",
    },
}


@dataclass
class ForecastConfig:
    test_start_date: str = TEST_START_DATE
    rolling_window: int = ROLLING_WINDOW
    min_train_size: int = MIN_TRAIN_SIZE
    refit_every: int = REFIT_EVERY
    naive_window: int = NAIVE_WINDOW


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def ensure_output_dirs() -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    rename_map = {}

    for col in df.columns:
        clean = str(col).strip()
        lower = clean.lower()

        if lower in {"date", "datetime", "time", "timestamp"}:
            rename_map[col] = "Date"
        elif lower in {"asset", "ticker", "symbol"}:
            rename_map[col] = "Asset"
        elif lower in {"log_return", "log_returns", "return", "returns", "ret"}:
            rename_map[col] = "return"

    return df.rename(columns=rename_map)


def load_returns(path: Path = INPUT_PATH) -> pd.DataFrame:
    """
    Load log returns.

    Supported formats:
    1. Wide format:
       Date, SPY, XLF, KBE

    2. Long format:
       Date, Asset, return
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Return file not found: {path}\n"
            "Run src/data_fetcher.py before this script."
        )

    df = pd.read_csv(path)
    df = normalize_column_names(df)

    if "Date" not in df.columns:
        raise ValueError(
            "log_returns.csv must contain a Date column.\n"
            f"Available columns: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    # Long format support
    if {"Date", "Asset", "return"}.issubset(df.columns):
        wide = (
            df.pivot_table(index="Date", columns="Asset", values="return", aggfunc="first")
            .reset_index()
        )
        wide.columns.name = None
        df = wide

    available_assets = [asset for asset in ASSETS if asset in df.columns]

    if not available_assets:
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col != "Date"
        ]
        available_assets = numeric_cols

    if not available_assets:
        raise ValueError(
            "No asset return columns found in log_returns.csv.\n"
            f"Available columns: {list(df.columns)}"
        )

    keep_cols = ["Date"] + available_assets
    df = df[keep_cols].sort_values("Date").reset_index(drop=True)

    for asset in available_assets:
        df[asset] = pd.to_numeric(df[asset], errors="coerce")

    return df


# ---------------------------------------------------------------------
# Period labels
# ---------------------------------------------------------------------

def assign_period(date: pd.Timestamp) -> str:
    if date <= pd.Timestamp("2007-06-30"):
        return "pre_crisis"
    if date <= pd.Timestamp("2009-06-30"):
        return "crisis"
    return "post_crisis"


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.maximum(np.asarray(y_true, dtype=float), EPS)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), EPS)

    ratio = y_true / y_pred
    loss = ratio - np.log(ratio) - 1.0

    return float(np.mean(loss))


def evaluate_predictions(pred_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []

    for keys, group in pred_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_true = group["actual"].to_numpy(dtype=float)
        y_pred = group["prediction"].to_numpy(dtype=float)

        record = {col: value for col, value in zip(group_cols, keys)}
        record.update(
            {
                "n_obs": len(group),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "qlike": qlike(y_true, y_pred),
                "actual_mean": float(np.mean(y_true)),
                "prediction_mean": float(np.mean(y_pred)),
            }
        )

        rows.append(record)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# ---------------------------------------------------------------------
# Forecast helpers
# ---------------------------------------------------------------------

def fit_garch_forecast(
    train_returns: pd.Series,
    spec: Dict[str, object],
) -> Tuple[float, Dict[str, float]]:
    """
    Fit one GARCH-family model and return one-step-ahead variance forecast.

    Returns
    -------
    variance_forecast:
        Forecasted conditional variance for next period.

    info:
        Small dictionary with fit diagnostics.
    """

    y = train_returns.dropna().astype(float)

    am = arch_model(
        y,
        mean=spec["mean"],
        vol=spec["vol"],
        p=int(spec["p"]),
        o=int(spec["o"]),
        q=int(spec["q"]),
        dist=spec["dist"],
        rescale=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(
            disp="off",
            show_warning=False,
            options={"maxiter": 500},
        )

    forecast = res.forecast(horizon=1, reindex=False)
    variance_forecast = float(forecast.variance.iloc[-1, 0])
    variance_forecast = max(variance_forecast, EPS)

    info = {
        "aic": float(res.aic),
        "bic": float(res.bic),
        "loglikelihood": float(res.loglikelihood),
        "convergence_flag": float(getattr(res, "convergence_flag", np.nan)),
    }

    return variance_forecast, info


def rolling_forecast_asset(
    returns_df: pd.DataFrame,
    asset: str,
    config: ForecastConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run rolling GARCH-family forecasts for one asset.

    Forecast origin is date t.
    Actual target is squared return at t+1.
    """

    series_df = returns_df[["Date", asset]].dropna().copy()
    series_df = series_df.rename(columns={asset: "return"})
    series_df = series_df.sort_values("Date").reset_index(drop=True)

    test_start = pd.Timestamp(config.test_start_date)

    prediction_records = []
    error_records = []

    # Need pos + 1 because actual target is next-day squared return
    candidate_positions = series_df.index[
        (series_df["Date"] >= test_start)
        & (series_df.index < len(series_df) - 1)
    ].tolist()

    total_origins = len(candidate_positions)

    for count, pos in enumerate(candidate_positions, start=1):
        origin_date = pd.Timestamp(series_df.loc[pos, "Date"])
        target_date = pd.Timestamp(series_df.loc[pos + 1, "Date"])

        train_start = max(0, pos - config.rolling_window + 1)
        train_df = series_df.iloc[train_start : pos + 1].copy()
        train_returns = train_df["return"].dropna()

        if len(train_returns) < config.min_train_size:
            continue

        actual = float(series_df.loc[pos + 1, "return"] ** 2)
        period = assign_period(origin_date)
        target_period = assign_period(target_date)

        # Rolling-volatility benchmark
        naive_pred = float((train_returns.tail(config.naive_window) ** 2).mean())
        naive_pred = max(naive_pred, EPS)

        prediction_records.append(
            {
                "Date": origin_date,
                "target_date": target_date,
                "Asset": asset,
                "Period": period,
                "target_period": target_period,
                "model": "Naive_Rolling22",
                "actual": actual,
                "prediction": naive_pred,
                "train_start": train_df["Date"].min(),
                "train_end": train_df["Date"].max(),
                "n_train": len(train_returns),
                "aic": np.nan,
                "bic": np.nan,
                "loglikelihood": np.nan,
                "convergence_flag": np.nan,
            }
        )

        for model_name, spec in MODEL_SPECS.items():
            try:
                pred, info = fit_garch_forecast(train_returns, spec)

                prediction_records.append(
                    {
                        "Date": origin_date,
                        "target_date": target_date,
                        "Asset": asset,
                        "Period": period,
                        "target_period": target_period,
                        "model": model_name,
                        "actual": actual,
                        "prediction": pred,
                        "train_start": train_df["Date"].min(),
                        "train_end": train_df["Date"].max(),
                        "n_train": len(train_returns),
                        "aic": info["aic"],
                        "bic": info["bic"],
                        "loglikelihood": info["loglikelihood"],
                        "convergence_flag": info["convergence_flag"],
                    }
                )

            except Exception as exc:
                error_records.append(
                    {
                        "Date": origin_date,
                        "target_date": target_date,
                        "Asset": asset,
                        "Period": period,
                        "model": model_name,
                        "n_train": len(train_returns),
                        "error": repr(exc),
                    }
                )

        if count % 50 == 0 or count == total_origins:
            print(
                f"  {asset}: processed {count}/{total_origins} forecast origins "
                f"through {origin_date.date()}"
            )

    pred_df = pd.DataFrame(prediction_records)
    err_df = pd.DataFrame(error_records)

    return pred_df, err_df


def run_all_forecasts(
    returns_df: pd.DataFrame,
    config: ForecastConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    available_assets = [asset for asset in ASSETS if asset in returns_df.columns]

    if not available_assets:
        available_assets = [
            col for col in returns_df.columns
            if col != "Date" and pd.api.types.is_numeric_dtype(returns_df[col])
        ]

    all_predictions = []
    all_errors = []

    for asset in available_assets:
        print(f"Running rolling GARCH forecasts for {asset}...")
        pred_asset, err_asset = rolling_forecast_asset(
            returns_df=returns_df,
            asset=asset,
            config=config,
        )

        if not pred_asset.empty:
            all_predictions.append(pred_asset)

        if not err_asset.empty:
            all_errors.append(err_asset)

    if not all_predictions:
        raise RuntimeError("No GARCH rolling predictions were produced.")

    pred_df = pd.concat(all_predictions, ignore_index=True)
    pred_df["Date"] = pd.to_datetime(pred_df["Date"])
    pred_df["target_date"] = pd.to_datetime(pred_df["target_date"])

    if all_errors:
        err_df = pd.concat(all_errors, ignore_index=True)
    else:
        err_df = pd.DataFrame(
            columns=[
                "Date",
                "target_date",
                "Asset",
                "Period",
                "model",
                "n_train",
                "error",
            ]
        )

    return pred_df, err_df


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    pred_df: pd.DataFrame,
    err_df: pd.DataFrame,
    overall_perf: pd.DataFrame,
    period_perf: pd.DataFrame,
    config: ForecastConfig,
) -> str:
    lines = []

    lines.append("Rolling GARCH-family volatility forecast summary")
    lines.append("=" * 55)
    lines.append("")
    lines.append("Purpose:")
    lines.append(
        "This file summarizes rolling one-step-ahead forecasts from "
        "GARCH-family volatility models and a rolling-volatility benchmark."
    )
    lines.append("")
    lines.append("Forecast target:")
    lines.append("- next-day squared return, r_{t+1}^2")
    lines.append("")
    lines.append("Design:")
    lines.append(f"- Test origins start at: {config.test_start_date}")
    lines.append(f"- Rolling train window: {config.rolling_window} observations")
    lines.append(f"- Minimum train size: {config.min_train_size} observations")
    lines.append(f"- Naive benchmark window: {config.naive_window} observations")
    lines.append("- Forecast horizon: one trading day ahead")
    lines.append("- Split is chronological; no random shuffling is used.")
    lines.append("")
    lines.append("Models:")
    for model_name in ["Naive_Rolling22", *MODEL_SPECS.keys()]:
        lines.append(f"- {model_name}")
    lines.append("")
    lines.append("Successful prediction rows:")
    lines.append(f"- {len(pred_df)}")
    lines.append("")
    lines.append("Failed GARCH fits:")
    lines.append(f"- {len(err_df)}")
    lines.append("")
    lines.append("Main reading rules:")
    lines.append("- Lower RMSE, MAE, and QLIKE indicate better forecast performance.")
    lines.append("- QLIKE is the preferred volatility forecast loss.")
    lines.append("- GARCH forecasts are variance forecasts and are compared with r_{t+1}^2.")
    lines.append("- These are forecasting results, not parameter-interpretation results.")
    lines.append("")

    lines.append("Best models by QLIKE, overall:")
    best_overall = overall_perf.loc[
        overall_perf.groupby(["Asset"])["qlike"].idxmin()
    ].sort_values(["Asset"])

    for _, row in best_overall.iterrows():
        lines.append(
            f"- asset={row['Asset']}: {row['model']} "
            f"(QLIKE={row['qlike']:.6f}, RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f})"
        )

    lines.append("")
    lines.append("Best models by QLIKE, by test period:")
    best_period = period_perf.loc[
        period_perf.groupby(["Asset", "Period"])["qlike"].idxmin()
    ].sort_values(["Asset", "Period"])

    for _, row in best_period.iterrows():
        lines.append(
            f"- asset={row['Asset']} | period={row['Period']}: "
            f"{row['model']} (QLIKE={row['qlike']:.6f})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    config = ForecastConfig()

    returns_df = load_returns()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loaded returns: {returns_df.shape}")
    print(f"Assets available: {[col for col in returns_df.columns if col != 'Date']}")
    print(f"Rolling window: {config.rolling_window}")
    print("This may take a few minutes because GARCH models are refit daily.")

    pred_df, err_df = run_all_forecasts(
        returns_df=returns_df,
        config=config,
    )

    overall_perf = evaluate_predictions(
        pred_df,
        group_cols=["Asset", "model"],
    )

    period_perf = evaluate_predictions(
        pred_df,
        group_cols=["Asset", "Period", "model"],
    )

    pred_path = MODEL_OUTPUT_DIR / "garch_rolling_predictions.csv"
    overall_path = TABLE_OUTPUT_DIR / "garch_rolling_performance_overall.csv"
    period_path = TABLE_OUTPUT_DIR / "garch_rolling_performance_by_period.csv"
    errors_path = TABLE_OUTPUT_DIR / "garch_rolling_fit_errors.csv"
    summary_path = TABLE_OUTPUT_DIR / "garch_rolling_summary.txt"

    pred_df.to_csv(pred_path, index=False)
    overall_perf.to_csv(overall_path, index=False)
    period_perf.to_csv(period_path, index=False)
    err_df.to_csv(errors_path, index=False)

    summary = build_summary(
        pred_df=pred_df,
        err_df=err_df,
        overall_perf=overall_perf,
        period_perf=period_perf,
        config=config,
    )

    summary_path.write_text(summary, encoding="utf-8")

    print("")
    print("Rolling GARCH forecasting completed.")
    print(f"Predictions saved to: {pred_path}")
    print(f"Overall performance saved to: {overall_path}")
    print(f"Period performance saved to: {period_path}")
    print(f"Fit errors saved to: {errors_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
