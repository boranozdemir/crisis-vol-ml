"""
diagnostics.py

Runs first-stage statistical diagnostics for the 2008 mortgage crisis volatility project.

Inputs expected from data_fetcher.py:
    data/processed/log_returns.csv
    data/processed/volatility_proxy.csv
    data/processed/panel_dataset.csv

Outputs:
    outputs/tables/diagnostics/descriptive_statistics_full_sample.csv
    outputs/tables/diagnostics/descriptive_statistics_by_period.csv
    outputs/tables/diagnostics/pre_model_diagnostics_full_sample.csv
    outputs/tables/diagnostics/pre_model_diagnostics_by_period.csv
    outputs/tables/diagnostics/diagnostics_summary.txt

Run from project root:
    python src/diagnostics.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import warnings

import numpy as np
import pandas as pd

try:
    from scipy.stats import jarque_bera, kurtosis, skew
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.stattools import adfuller, kpss
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "diagnostics.py requires scipy and statsmodels. "
        "Install them with: pip install scipy statsmodels"
    ) from exc


CRISIS_WINDOWS: Dict[str, tuple[str, str]] = {
    "pre_crisis": ("2005-01-01", "2007-06-30"),
    "crisis": ("2007-07-01", "2009-06-30"),
    "post_crisis": ("2009-07-01", "2012-12-31"),
}

PERIOD_ORDER = ["pre_crisis", "crisis", "post_crisis"]


@dataclass(frozen=True)
class DiagnosticsConfig:
    processed_dir: Path = Path("data/processed")
    output_dir: Path = Path("outputs/tables/diagnostics")
    ljung_box_lags: tuple[int, ...] = (10, 20)
    arch_lags: int = 10
    min_observations: int = 60
    annualization_factor: int = 252


def ensure_output_dir(config: DiagnosticsConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)


def load_time_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run data_fetcher.py first.")

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df.sort_index()


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run data_fetcher.py first.")

    panel = pd.read_csv(path, parse_dates=["Date"])
    return panel.sort_values(["Asset", "Date"])


def safe_series(values: pd.Series | np.ndarray) -> pd.Series:
    """Return a clean numeric series without NaN or infinite values."""
    series = pd.Series(values).astype(float)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    return series


def safe_pvalue(value: float | np.ndarray | tuple | list) -> float:
    """Convert test p-values to a clean float."""
    if isinstance(value, (tuple, list, np.ndarray)):
        value = value[0]
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def adf_test(series: pd.Series) -> dict[str, float | str]:
    """Augmented Dickey-Fuller test. Null hypothesis: unit root / non-stationary."""
    x = safe_series(series)

    if len(x) < 25 or x.nunique() <= 1:
        return {
            "adf_stat": np.nan,
            "adf_pvalue": np.nan,
            "adf_used_lag": np.nan,
            "adf_interpretation": "insufficient_data",
        }

    try:
        result = adfuller(x, autolag="AIC")
        pvalue = safe_pvalue(result[1])
        interpretation = "stationary" if pvalue < 0.05 else "non_stationary"
        return {
            "adf_stat": float(result[0]),
            "adf_pvalue": pvalue,
            "adf_used_lag": int(result[2]),
            "adf_interpretation": interpretation,
        }
    except Exception:
        return {
            "adf_stat": np.nan,
            "adf_pvalue": np.nan,
            "adf_used_lag": np.nan,
            "adf_interpretation": "failed",
        }


def kpss_test(series: pd.Series) -> dict[str, float | str]:
    """KPSS test. Null hypothesis: stationary around a constant."""
    x = safe_series(series)

    if len(x) < 25 or x.nunique() <= 1:
        return {
            "kpss_stat": np.nan,
            "kpss_pvalue": np.nan,
            "kpss_used_lag": np.nan,
            "kpss_interpretation": "insufficient_data",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = kpss(x, regression="c", nlags="auto")

        pvalue = safe_pvalue(result[1])
        interpretation = "stationary" if pvalue >= 0.05 else "non_stationary"
        return {
            "kpss_stat": float(result[0]),
            "kpss_pvalue": pvalue,
            "kpss_used_lag": int(result[2]),
            "kpss_interpretation": interpretation,
        }
    except Exception:
        return {
            "kpss_stat": np.nan,
            "kpss_pvalue": np.nan,
            "kpss_used_lag": np.nan,
            "kpss_interpretation": "failed",
        }


def ljung_box_pvalue(series: pd.Series, lag: int) -> float:
    """Ljung-Box p-value. Null hypothesis: no autocorrelation up to the selected lag."""
    x = safe_series(series)

    if len(x) <= lag + 5 or x.nunique() <= 1:
        return np.nan

    try:
        result = acorr_ljungbox(x, lags=[lag], return_df=True)
        return float(result["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan


def arch_lm_pvalue(series: pd.Series, lags: int) -> float:
    """ARCH-LM p-value. Null hypothesis: no remaining ARCH effect."""
    x = safe_series(series)

    if len(x) <= lags + 5 or x.nunique() <= 1:
        return np.nan

    try:
        result = het_arch(x, nlags=lags)
        return float(result[1])
    except Exception:
        return np.nan


def jarque_bera_test(series: pd.Series) -> dict[str, float | str]:
    """Jarque-Bera normality test. Null hypothesis: normal distribution."""
    x = safe_series(series)

    if len(x) < 8 or x.nunique() <= 1:
        return {
            "jb_stat": np.nan,
            "jb_pvalue": np.nan,
            "normality_interpretation": "insufficient_data",
        }

    try:
        result = jarque_bera(x)
        pvalue = safe_pvalue(result.pvalue)
        interpretation = "normal_not_rejected" if pvalue >= 0.05 else "non_normal"
        return {
            "jb_stat": float(result.statistic),
            "jb_pvalue": pvalue,
            "normality_interpretation": interpretation,
        }
    except Exception:
        return {
            "jb_stat": np.nan,
            "jb_pvalue": np.nan,
            "normality_interpretation": "failed",
        }


def descriptive_statistics(
    returns: pd.DataFrame,
    config: DiagnosticsConfig,
) -> pd.DataFrame:
    rows = []

    for asset in returns.columns:
        x = safe_series(returns[asset])
        if x.empty:
            continue

        rows.append(
            {
                "Asset": asset,
                "n_obs": len(x),
                "mean": x.mean(),
                "median": x.median(),
                "std": x.std(ddof=1),
                "annualized_vol": x.std(ddof=1) * np.sqrt(config.annualization_factor),
                "min": x.min(),
                "max": x.max(),
                "skewness": skew(x, bias=False),
                "kurtosis_excess": kurtosis(x, fisher=True, bias=False),
                "abs_return_mean": x.abs().mean(),
                "squared_return_mean": x.pow(2).mean(),
            }
        )

    return pd.DataFrame(rows)


def descriptive_statistics_by_period(
    panel: pd.DataFrame,
    config: DiagnosticsConfig,
) -> pd.DataFrame:
    rows = []

    for asset in sorted(panel["Asset"].unique()):
        for period in PERIOD_ORDER:
            subset = panel.loc[(panel["Asset"] == asset) & (panel["Period"] == period), "Return"]
            x = safe_series(subset)
            if len(x) < config.min_observations:
                continue

            rows.append(
                {
                    "Asset": asset,
                    "Period": period,
                    "n_obs": len(x),
                    "mean": x.mean(),
                    "median": x.median(),
                    "std": x.std(ddof=1),
                    "annualized_vol": x.std(ddof=1) * np.sqrt(config.annualization_factor),
                    "min": x.min(),
                    "max": x.max(),
                    "skewness": skew(x, bias=False),
                    "kurtosis_excess": kurtosis(x, fisher=True, bias=False),
                    "abs_return_mean": x.abs().mean(),
                    "squared_return_mean": x.pow(2).mean(),
                }
            )

    return pd.DataFrame(rows)


def pre_model_diagnostics_for_series(
    series: pd.Series,
    config: DiagnosticsConfig,
) -> dict[str, float | str]:
    x = safe_series(series)
    x_centered = x - x.mean()
    x_squared = x_centered.pow(2)

    row: dict[str, float | str] = {
        "n_obs": len(x),
    }

    row.update(adf_test(x))
    row.update(kpss_test(x))
    row.update(jarque_bera_test(x))

    for lag in config.ljung_box_lags:
        row[f"lb_return_pvalue_lag_{lag}"] = ljung_box_pvalue(x_centered, lag)
        row[f"lb_squared_return_pvalue_lag_{lag}"] = ljung_box_pvalue(x_squared, lag)

    row[f"arch_lm_pvalue_lag_{config.arch_lags}"] = arch_lm_pvalue(x_centered, config.arch_lags)

    row["volatility_clustering_flag"] = (
        "yes"
        if any(
            row.get(f"lb_squared_return_pvalue_lag_{lag}", np.nan) < 0.05
            for lag in config.ljung_box_lags
        )
        or row.get(f"arch_lm_pvalue_lag_{config.arch_lags}", np.nan) < 0.05
        else "no"
    )

    return row


def pre_model_diagnostics_full_sample(
    returns: pd.DataFrame,
    config: DiagnosticsConfig,
) -> pd.DataFrame:
    rows = []

    for asset in returns.columns:
        row = pre_model_diagnostics_for_series(returns[asset], config)
        row["Asset"] = asset
        rows.append(row)

    columns = ["Asset"] + [col for col in rows[0].keys() if col != "Asset"] if rows else []
    return pd.DataFrame(rows)[columns]


def pre_model_diagnostics_by_period(
    panel: pd.DataFrame,
    config: DiagnosticsConfig,
) -> pd.DataFrame:
    rows = []

    for asset in sorted(panel["Asset"].unique()):
        for period in PERIOD_ORDER:
            subset = panel.loc[(panel["Asset"] == asset) & (panel["Period"] == period), "Return"]
            x = safe_series(subset)

            if len(x) < config.min_observations:
                continue

            row = pre_model_diagnostics_for_series(x, config)
            row["Asset"] = asset
            row["Period"] = period
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    columns = ["Asset", "Period"] + [col for col in rows[0].keys() if col not in {"Asset", "Period"}]
    return pd.DataFrame(rows)[columns]


def save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_summary(
    desc_full: pd.DataFrame,
    desc_period: pd.DataFrame,
    diag_full: pd.DataFrame,
    diag_period: pd.DataFrame,
    config: DiagnosticsConfig,
) -> None:
    summary_path = config.output_dir / "diagnostics_summary.txt"

    lines = [
        "Diagnostics summary",
        "===================",
        "",
        "Purpose:",
        "These diagnostics check whether the return series display the empirical patterns that motivate volatility modeling.",
        "",
        "Key reading rules:",
        "- ADF p-value < 0.05 supports stationarity of returns.",
        "- KPSS p-value >= 0.05 supports stationarity of returns.",
        "- Jarque-Bera p-value < 0.05 suggests non-normal/heavy-tailed returns.",
        "- Ljung-Box on returns checks mean autocorrelation.",
        "- Ljung-Box on squared returns checks volatility clustering.",
        "- ARCH-LM p-value < 0.05 suggests ARCH effects and motivates GARCH models.",
        "",
        "Full-sample diagnostics:",
        diag_full.to_string(index=False),
        "",
        "Descriptive statistics by period:",
        desc_period.to_string(index=False),
        "",
    ]

    if not diag_period.empty:
        lines.extend(
            [
                "Pre-crisis / crisis / post-crisis diagnostics:",
                diag_period.to_string(index=False),
                "",
            ]
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def run_diagnostics(config: Optional[DiagnosticsConfig] = None) -> None:
    config = config or DiagnosticsConfig()
    ensure_output_dir(config)

    returns = load_time_series(config.processed_dir / "log_returns.csv")
    panel = load_panel(config.processed_dir / "panel_dataset.csv")

    desc_full = descriptive_statistics(returns, config)
    desc_period = descriptive_statistics_by_period(panel, config)
    diag_full = pre_model_diagnostics_full_sample(returns, config)
    diag_period = pre_model_diagnostics_by_period(panel, config)

    save_table(desc_full, config.output_dir / "descriptive_statistics_full_sample.csv")
    save_table(desc_period, config.output_dir / "descriptive_statistics_by_period.csv")
    save_table(diag_full, config.output_dir / "pre_model_diagnostics_full_sample.csv")
    save_table(diag_period, config.output_dir / "pre_model_diagnostics_by_period.csv")

    write_summary(desc_full, desc_period, diag_full, diag_period, config)

    print(f"Diagnostics tables saved to: {config.output_dir}")
    print(f"Summary file saved to: {config.output_dir / 'diagnostics_summary.txt'}")


def main() -> None:
    run_diagnostics()


if __name__ == "__main__":
    main()
