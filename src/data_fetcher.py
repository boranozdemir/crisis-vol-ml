"""
data_fetcher.py

Fetches daily ETF price data for the 2008 mortgage crisis volatility project.

Main assets:
    SPY : S&P 500 ETF
    XLF : Financial sector ETF
    KBE : Bank ETF

Outputs:
    data/raw/prices.csv
    data/processed/log_returns.csv
    data/processed/volatility_proxy.csv
    data/processed/panel_dataset.csv
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class DataConfig:
    start_date: str = "2005-01-01"
    end_date: str = "2012-12-31"
    interval: str = "1d"
    auto_adjust: bool = True
    return_scale: float = 100.0
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    tickers: Dict[str, str] = field(
        default_factory=lambda: {
            "SPY": "SPY",   # broad U.S. equity market
            "XLF": "XLF",   # financial sector
            "KBE": "KBE",   # banking sector
        }
    )


CRISIS_WINDOWS = {
    "pre_crisis": ("2005-01-01", "2007-06-30"),
    "crisis": ("2007-07-01", "2009-06-30"),
    "post_crisis": ("2009-07-01", "2012-12-31"),
}


def ensure_dirs(config: DataConfig) -> None:
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    config.processed_dir.mkdir(parents=True, exist_ok=True)


def fetch_prices(config: DataConfig) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance via yfinance."""
    symbols = list(config.tickers.values())

    data = yf.download(
        symbols,
        start=config.start_date,
        end=config.end_date,
        interval=config.interval,
        auto_adjust=config.auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )

    prices = extract_close_prices(data)

    ticker_to_name = {ticker: name for name, ticker in config.tickers.items()}
    prices = prices.rename(columns=ticker_to_name)
    prices = prices.sort_index()
    prices.index.name = "Date"

    missing = prices.isna().sum()
    if missing.any():
        print("Missing observations before dropna:")
        print(missing[missing > 0])

    prices = prices.dropna(how="any")

    if prices.empty:
        raise ValueError("Price data is empty after dropping missing values.")

    return prices


def extract_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Extract Close prices from yfinance output for single or multiple tickers."""
    if isinstance(data.columns, pd.MultiIndex):
        level_0 = data.columns.get_level_values(0)
        if "Close" in level_0:
            prices = data["Close"].copy()
        elif "Adj Close" in level_0:
            prices = data["Adj Close"].copy()
        else:
            raise KeyError("Could not find Close or Adj Close in downloaded data.")
    else:
        if "Close" in data.columns:
            prices = data[["Close"]].copy()
        elif "Adj Close" in data.columns:
            prices = data[["Adj Close"]].copy()
        else:
            raise KeyError("Could not find Close or Adj Close in downloaded data.")

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    return prices


def compute_log_returns(prices: pd.DataFrame, scale: float = 100.0) -> pd.DataFrame:
    """Compute scaled daily log returns."""
    returns = scale * np.log(prices / prices.shift(1))
    returns = returns.dropna(how="any")
    returns.index.name = "Date"
    return returns


def compute_volatility_proxy(returns: pd.DataFrame) -> pd.DataFrame:
    """Daily squared return proxy for volatility."""
    rv_proxy = returns.pow(2)
    rv_proxy.index.name = "Date"
    return rv_proxy


def assign_period(date: pd.Timestamp) -> str:
    """Assign pre-crisis, crisis, or post-crisis label."""
    for label, (start, end) in CRISIS_WINDOWS.items():
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return label
    return "outside_sample"


def build_dataset(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volatility_proxy: pd.DataFrame,
) -> pd.DataFrame:
    """Create a long-format panel dataset for diagnostics and ML feature building."""
    common_index = returns.index.intersection(volatility_proxy.index)
    prices = prices.loc[common_index]
    returns = returns.loc[common_index]
    volatility_proxy = volatility_proxy.loc[common_index]

    frames = []
    for asset in returns.columns:
        asset_df = pd.DataFrame(
            {
                "Date": common_index,
                "Asset": asset,
                "Price": prices[asset].values,
                "Return": returns[asset].values,
                "RV_proxy": volatility_proxy[asset].values,
            }
        )
        asset_df["Period"] = asset_df["Date"].apply(assign_period)
        frames.append(asset_df)

    panel = pd.concat(frames, ignore_index=True)
    return panel


def save_outputs(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    volatility_proxy: pd.DataFrame,
    panel: pd.DataFrame,
    config: DataConfig,
) -> None:
    prices.to_csv(config.raw_dir / "prices.csv")
    returns.to_csv(config.processed_dir / "log_returns.csv")
    volatility_proxy.to_csv(config.processed_dir / "volatility_proxy.csv")
    panel.to_csv(config.processed_dir / "panel_dataset.csv", index=False)


def print_summary(prices: pd.DataFrame, returns: pd.DataFrame, panel: pd.DataFrame) -> None:
    print("\nDownloaded price data")
    print("---------------------")
    print(f"Start date : {prices.index.min().date()}")
    print(f"End date   : {prices.index.max().date()}")
    print(f"Assets     : {', '.join(prices.columns)}")
    print(f"Rows       : {len(prices)}")

    print("\nReturn summary")
    print("--------------")
    print(returns.describe().T[["mean", "std", "min", "max"]])

    print("\nPeriod counts")
    print("-------------")
    print(panel.groupby(["Asset", "Period"]).size().unstack(fill_value=0))


def main():
    config = DataConfig()
    ensure_dirs(config)

    prices = fetch_prices(config)
    returns = compute_log_returns(prices, scale=config.return_scale)
    volatility_proxy = compute_volatility_proxy(returns)
    panel = build_dataset(prices, returns, volatility_proxy)

    save_outputs(prices, returns, volatility_proxy, panel, config)
    print_summary(prices, returns, panel)


if __name__ == "__main__":
    main()
