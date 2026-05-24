"""
data_visualizer.py

Creates first-stage visual checks for the 2008 mortgage crisis.

Inputs expected from data_fetcher.py:
    data/raw/prices.csv
    data/processed/log_returns.csv
    data/processed/volatility_proxy.csv
    data/processed/panel_dataset.csv

Outputs:
    outputs/figures/data_overview/*.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CRISIS_WINDOWS: Dict[str, tuple[str, str]] = {
    "pre_crisis": ("2005-01-01", "2007-06-30"),
    "crisis": ("2007-07-01", "2009-06-30"),
    "post_crisis": ("2009-07-01", "2012-12-31"),
}


@dataclass(frozen=True)
class VisualizationConfig:
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs/figures/data_overview")
    rolling_window: int = 22
    figure_dpi: int = 150


def ensure_output_dir(config: VisualizationConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)


def load_time_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run data_fetcher.py first.")

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df = df.sort_index()
    return df


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run data_fetcher.py first.")

    panel = pd.read_csv(path, parse_dates=["Date"])
    panel = panel.sort_values(["Asset", "Date"])
    return panel


def add_crisis_windows(ax: plt.Axes) -> None:
    """Add sample-period background spans to a time-series plot."""
    for label, (start, end) in CRISIS_WINDOWS.items():
        alpha = 0.06 if label != "crisis" else 0.12
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=alpha)


def save_figure(fig: plt.Figure, output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_multiseries(
    df: pd.DataFrame,
    title: str,
    ylabel: str,
    output_path: Path,
    config: VisualizationConfig,
    add_windows: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    for column in df.columns:
        ax.plot(df.index, df[column], linewidth=1.1, label=column)

    if add_windows:
        add_crisis_windows(ax)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    save_figure(fig, output_path, config.figure_dpi)


def plot_each_asset(
    df: pd.DataFrame,
    title_prefix: str,
    ylabel: str,
    filename_prefix: str,
    config: VisualizationConfig,
    add_windows: bool = True,
) -> None:
    for asset in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df[asset], linewidth=1.0)

        if add_windows:
            add_crisis_windows(ax)

        ax.set_title(f"{title_prefix}: {asset}")
        ax.set_xlabel("Date")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        output_path = config.output_dir / f"{filename_prefix}_{asset}.png"
        save_figure(fig, output_path, config.figure_dpi)


def plot_rolling_volatility(
    returns: pd.DataFrame,
    window: int,
    config: VisualizationConfig,
) -> None:
    rolling_vol = returns.rolling(window=window).std().dropna(how="all")

    plot_multiseries(
        rolling_vol,
        title=f"Rolling Volatility ({window}-Day Window)",
        ylabel="Rolling standard deviation of daily log returns",
        output_path=config.output_dir / f"rolling_volatility_{window}d.png",
        config=config,
    )


def plot_return_distributions(returns: pd.DataFrame, config: VisualizationConfig) -> None:
    for asset in returns.columns:
        series = returns[asset].dropna()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(series, bins=60, density=True, alpha=0.8)
        ax.axvline(series.mean(), linestyle="--", linewidth=1.0, label="Mean")
        ax.set_title(f"Return Distribution: {asset}")
        ax.set_xlabel("Daily log return (%)")
        ax.set_ylabel("Density")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        output_path = config.output_dir / f"return_distribution_{asset}.png"
        save_figure(fig, output_path, config.figure_dpi)


def plot_period_counts(panel: pd.DataFrame, config: VisualizationConfig) -> None:
    counts = panel.groupby(["Asset", "Period"]).size().unstack(fill_value=0)
    ordered_cols = [col for col in ["pre_crisis", "crisis", "post_crisis"] if col in counts.columns]
    counts = counts[ordered_cols]

    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Observation Counts by Asset and Period")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Number of observations")
    ax.legend(title="Period", loc="best")
    ax.grid(True, axis="y", alpha=0.3)

    save_figure(fig, config.output_dir / "period_observation_counts.png", config.figure_dpi)


def plot_boxplot_by_period(
    panel: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    filename: str,
    config: VisualizationConfig,
) -> None:
    period_order = ["pre_crisis", "crisis", "post_crisis"]
    asset_order = sorted(panel["Asset"].unique())

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(asset_order),
        figsize=(5 * len(asset_order), 5),
        sharey=True,
    )

    if len(asset_order) == 1:
        axes = [axes]

    for ax, asset in zip(axes, asset_order):
        asset_panel = panel[panel["Asset"] == asset]
        values = [
            asset_panel.loc[asset_panel["Period"] == period, value_column].dropna().values
            for period in period_order
        ]
        ax.boxplot(values, labels=period_order, showfliers=False)
        ax.set_title(asset)
        ax.set_xlabel("Period")
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)

    save_figure(fig, config.output_dir / filename, config.figure_dpi)


def plot_correlation_matrix(returns: pd.DataFrame, config: VisualizationConfig) -> None:
    corr = returns.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1)
    ax.set_title("Return Correlation Matrix")
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.index)

    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, config.output_dir / "return_correlation_matrix.png", config.figure_dpi)


def write_visual_summary(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volatility_proxy: pd.DataFrame,
    panel: pd.DataFrame,
    config: VisualizationConfig,
) -> None:
    summary_path = config.output_dir / "visual_summary.txt"

    lines = [
        "Data visualization summary",
        "==========================",
        "",
        f"Price sample: {prices.index.min().date()} to {prices.index.max().date()}",
        f"Return sample: {returns.index.min().date()} to {returns.index.max().date()}",
        f"Assets: {', '.join(returns.columns)}",
        "",
        "Generated figures:",
    ]

    figure_files = sorted(config.output_dir.glob("*.png"))
    for file in figure_files:
        lines.append(f"- {file.name}")

    lines.extend(
        [
            "",
            "Period counts:",
            panel.groupby(["Asset", "Period"]).size().unstack(fill_value=0).to_string(),
            "",
            "Return summary:",
            returns.describe().T[["mean", "std", "min", "max"]].to_string(),
            "",
            "Volatility proxy summary:",
            volatility_proxy.describe().T[["mean", "std", "min", "max"]].to_string(),
        ]
    )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def create_all_data_visualizations(config: Optional[VisualizationConfig] = None) -> None:
    config = config or VisualizationConfig()
    ensure_output_dir(config)

    prices = load_time_series(config.raw_dir / "prices.csv")
    returns = load_time_series(config.processed_dir / "log_returns.csv")
    volatility_proxy = load_time_series(config.processed_dir / "volatility_proxy.csv")
    panel = load_panel(config.processed_dir / "panel_dataset.csv")

    plot_multiseries(
        prices,
        title="Adjusted Close Prices",
        ylabel="Adjusted close price",
        output_path=config.output_dir / "prices_all_assets.png",
        config=config,
    )
    plot_each_asset(
        prices,
        title_prefix="Adjusted Close Price",
        ylabel="Adjusted close price",
        filename_prefix="price",
        config=config,
    )

    plot_multiseries(
        returns,
        title="Daily Log Returns",
        ylabel="Daily log return (%)",
        output_path=config.output_dir / "returns_all_assets.png",
        config=config,
    )
    plot_each_asset(
        returns,
        title_prefix="Daily Log Return",
        ylabel="Daily log return (%)",
        filename_prefix="return",
        config=config,
    )

    plot_multiseries(
        volatility_proxy,
        title="Daily Squared Return Volatility Proxy",
        ylabel="Squared daily log return",
        output_path=config.output_dir / "volatility_proxy_all_assets.png",
        config=config,
    )
    plot_each_asset(
        volatility_proxy,
        title_prefix="Daily Squared Return Volatility Proxy",
        ylabel="Squared daily log return",
        filename_prefix="volatility_proxy",
        config=config,
    )

    plot_rolling_volatility(returns, window=config.rolling_window, config=config)
    plot_return_distributions(returns, config=config)
    plot_correlation_matrix(returns, config=config)
    plot_period_counts(panel, config=config)

    plot_boxplot_by_period(
        panel,
        value_column="Return",
        title="Daily Returns by Crisis Period",
        ylabel="Daily log return (%)",
        filename="return_boxplot_by_period.png",
        config=config,
    )
    plot_boxplot_by_period(
        panel,
        value_column="RV_proxy",
        title="Volatility Proxy by Crisis Period",
        ylabel="Squared daily log return",
        filename="rv_proxy_boxplot_by_period.png",
        config=config,
    )

    write_visual_summary(prices, returns, volatility_proxy, panel, config)

    print(f"Data overview figures saved to: {config.output_dir}")
    print(f"Summary file saved to: {config.output_dir / 'visual_summary.txt'}")


def main() -> None:
    create_all_data_visualizations()


if __name__ == "__main__":
    main()
