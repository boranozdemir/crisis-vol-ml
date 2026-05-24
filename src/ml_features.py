"""
ML feature engineering for the 2008 mortgage crisis volatility project.

This module builds supervised-learning datasets for next-day volatility
forecasting. The target is the next-day squared return, and the predictors are
constructed only from information available up to time t.

Input:
    data/processed/log_returns.csv

Outputs:
    data/processed/ml_features_panel.csv
    data/processed/ml_features_wide.csv
    outputs/tables/ml/ml_feature_summary.csv
    outputs/tables/ml/ml_feature_dictionary.csv
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for volatility forecasting feature construction."""

    return_lags: tuple[int, ...] = (1, 2, 3, 5)
    squared_return_lags: tuple[int, ...] = (1, 2, 3, 5, 10)
    rolling_windows: tuple[int, ...] = (5, 22, 66)
    min_obs_by_asset: int = 250
    pre_crisis_end: str = "2007-06-30"
    crisis_start: str = "2007-07-01"
    crisis_end: str = "2009-06-30"
    post_crisis_start: str = "2009-07-01"
    target_horizon: int = 1
    assets: tuple[str, ...] | None = None
    output_feature_columns: tuple[str, ...] | None = None


def find_project_root() -> Path:
    """Find the project root by walking upward from this file."""
    current = Path(__file__).resolve()
    candidates = [current.parent, *current.parents]

    for parent in candidates:
        if (parent / "data").exists() and (parent / "src").exists():
            return parent

    # Fallback for interactive execution from the project root.
    return Path.cwd()


def ensure_directory(path: Path) -> None:
    """Create a directory if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def assign_period(date: pd.Timestamp, config: FeatureConfig) -> str:
    """Assign a date to pre-crisis, crisis, or post-crisis period."""
    date = pd.Timestamp(date)

    if date <= pd.Timestamp(config.pre_crisis_end):
        return "pre_crisis"
    if pd.Timestamp(config.crisis_start) <= date <= pd.Timestamp(config.crisis_end):
        return "crisis"
    if date >= pd.Timestamp(config.post_crisis_start):
        return "post_crisis"

    return "transition"


def read_returns(path: Path, assets: Iterable[str] | None = None) -> pd.DataFrame:
    """Read the processed log return file produced by data_fetcher.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run data_fetcher.py before ml_features.py."
        )

    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    returns.index.name = "Date"

    # Keep only numeric columns in case metadata columns are present.
    returns = returns.apply(pd.to_numeric, errors="coerce")

    if assets is not None:
        missing = sorted(set(assets) - set(returns.columns))
        if missing:
            raise ValueError(f"Requested assets are missing from returns file: {missing}")
        returns = returns[list(assets)]

    return returns.sort_index()


def build_asset_features(
    returns: pd.Series,
    asset: str,
    config: FeatureConfig,
) -> pd.DataFrame:
    """Build next-day volatility forecasting features for one asset."""
    r = returns.dropna().astype(float).copy()
    r.name = "return"

    df = pd.DataFrame(index=r.index)
    df.index.name = "Date"

    # Information available at time t.
    df["asset"] = asset
    df["return_t"] = r
    df["abs_return_t"] = r.abs()
    df["squared_return_t"] = r.pow(2)
    df["negative_return_dummy_t"] = (r < 0).astype(int)
    df["negative_squared_return_t"] = df["negative_return_dummy_t"] * df["squared_return_t"]

    # Lagged returns and lagged volatility proxies.
    for lag in config.return_lags:
        df[f"return_lag_{lag}"] = r.shift(lag)
        df[f"abs_return_lag_{lag}"] = r.abs().shift(lag)

    for lag in config.squared_return_lags:
        df[f"squared_return_lag_{lag}"] = r.pow(2).shift(lag)

    # Rolling volatility-style predictors based on information up to time t.
    squared = r.pow(2)
    abs_r = r.abs()

    for window in config.rolling_windows:
        df[f"rolling_mean_squared_return_{window}d"] = squared.rolling(window).mean()
        df[f"rolling_sum_squared_return_{window}d"] = squared.rolling(window).sum()
        df[f"rolling_volatility_{window}d"] = squared.rolling(window).mean().pow(0.5)
        df[f"rolling_abs_return_{window}d"] = abs_r.rolling(window).mean()

    # Simple downside-risk rolling measures.
    downside_squared = squared.where(r < 0, 0.0)
    for window in config.rolling_windows:
        df[f"rolling_downside_squared_return_{window}d"] = downside_squared.rolling(window).mean()
        df[f"rolling_negative_share_{window}d"] = (
            (r < 0).astype(float).rolling(window).mean()
        )

    # Calendar / regime information.
    df["period"] = [assign_period(date, config) for date in df.index]
    df["crisis_dummy"] = (df["period"] == "crisis").astype(int)
    df["post_crisis_dummy"] = (df["period"] == "post_crisis").astype(int)
    df["negative_x_crisis"] = df["negative_return_dummy_t"] * df["crisis_dummy"]
    df["negative_squared_x_crisis"] = df["negative_squared_return_t"] * df["crisis_dummy"]

    # Forecast target: next-day squared return.
    h = config.target_horizon
    df[f"target_squared_return_t_plus_{h}"] = squared.shift(-h)
    df[f"target_abs_return_t_plus_{h}"] = abs_r.shift(-h)
    df[f"target_return_t_plus_{h}"] = r.shift(-h)

    # Drop rows with unavailable lags/rolling features or target.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    if len(df) < config.min_obs_by_asset:
        raise ValueError(
            f"Too few usable observations for {asset}: {len(df)}. "
            f"Minimum required: {config.min_obs_by_asset}."
        )

    return df.reset_index()


def build_feature_panel(returns: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Build a long-format ML feature panel for all assets."""
    assets = config.assets if config.assets is not None else tuple(returns.columns)

    frames = []
    for asset in assets:
        frames.append(build_asset_features(returns[asset], asset, config))

    panel = pd.concat(frames, axis=0, ignore_index=True)
    panel = panel.sort_values(["asset", "Date"]).reset_index(drop=True)

    return panel


def build_wide_feature_data(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Create a wide version with asset dummies for pooled ML models."""
    wide = panel.copy()
    asset_dummies = pd.get_dummies(wide["asset"], prefix="asset", dtype=int)
    period_dummies = pd.get_dummies(wide["period"], prefix="period", dtype=int)
    wide = pd.concat([wide, asset_dummies, period_dummies], axis=1)
    return wide


def summarize_features(panel: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Summarize the generated feature panel by asset and period."""
    target_col = f"target_squared_return_t_plus_{config.target_horizon}"

    summary = (
        panel.groupby(["asset", "period"])
        .agg(
            n_obs=(target_col, "size"),
            start_date=("Date", "min"),
            end_date=("Date", "max"),
            mean_target_squared_return=(target_col, "mean"),
            median_target_squared_return=(target_col, "median"),
            std_target_squared_return=(target_col, "std"),
            mean_return_t=("return_t", "mean"),
            mean_squared_return_t=("squared_return_t", "mean"),
            negative_return_share=("negative_return_dummy_t", "mean"),
        )
        .reset_index()
    )

    return summary


def build_feature_dictionary(config: FeatureConfig) -> pd.DataFrame:
    """Create a compact dictionary explaining key feature groups."""
    rows = [
        ("return_t", "Current daily log return scaled by 100."),
        ("abs_return_t", "Absolute value of current daily return."),
        ("squared_return_t", "Current squared return; daily volatility proxy."),
        ("negative_return_dummy_t", "Equals 1 if current return is negative."),
        ("negative_squared_return_t", "Squared return interacted with negative-return dummy."),
        ("return_lag_k", "Lagged daily returns."),
        ("squared_return_lag_k", "Lagged squared returns."),
        ("rolling_mean_squared_return_wd", "Rolling average of squared returns over w trading days."),
        ("rolling_volatility_wd", "Square root of rolling mean squared returns."),
        ("rolling_downside_squared_return_wd", "Rolling average of squared returns on negative-return days."),
        ("rolling_negative_share_wd", "Share of negative-return days over the rolling window."),
        ("crisis_dummy", "Equals 1 during the 2007-07-01 to 2009-06-30 crisis window."),
        ("negative_x_crisis", "Negative-return dummy interacted with crisis dummy."),
        ("negative_squared_x_crisis", "Negative squared return interacted with crisis dummy."),
        (
            f"target_squared_return_t_plus_{config.target_horizon}",
            "Next-day squared return used as the volatility forecasting target.",
        ),
    ]

    return pd.DataFrame(rows, columns=["feature", "description"])


def main() -> None:
    config = FeatureConfig()

    project_root = find_project_root()
    processed_dir = project_root / "data" / "processed"
    table_dir = project_root / "outputs" / "tables" / "ml"

    ensure_directory(processed_dir)
    ensure_directory(table_dir)

    returns_path = processed_dir / "log_returns.csv"
    panel_output_path = processed_dir / "ml_features_panel.csv"
    wide_output_path = processed_dir / "ml_features_wide.csv"
    summary_output_path = table_dir / "ml_feature_summary.csv"
    dictionary_output_path = table_dir / "ml_feature_dictionary.csv"

    returns = read_returns(returns_path, assets=config.assets)
    panel = build_feature_panel(returns, config)
    wide = build_wide_feature_data(panel, config)
    summary = summarize_features(panel, config)
    feature_dictionary = build_feature_dictionary(config)

    panel.to_csv(panel_output_path, index=False)
    wide.to_csv(wide_output_path, index=False)
    summary.to_csv(summary_output_path, index=False)
    feature_dictionary.to_csv(dictionary_output_path, index=False)

    print("ML feature engineering completed.")
    print(f"Input returns: {returns_path}")
    print(f"Panel feature file: {panel_output_path}")
    print(f"Wide feature file: {wide_output_path}")
    print(f"Feature summary: {summary_output_path}")
    print(f"Feature dictionary: {dictionary_output_path}")
    print(f"Rows generated: {len(panel):,}")
    print(f"Assets: {', '.join(sorted(panel['asset'].unique()))}")


if __name__ == "__main__":
    main()
