"""
garch_models.py

Estimate GARCH-family volatility models for the 2008 mortgage crisis project.

This script reads processed daily log returns, estimates GARCH(1,1),
GJR-GARCH(1,1), and EGARCH(1,1) models across assets and crisis subperiods,
and saves model-selection tables, parameter estimates, conditional volatility,
and standardized residuals.

Expected input:
    data/processed/log_returns.csv

Main outputs:
    outputs/tables/garch/garch_model_selection.csv
    outputs/tables/garch/garch_key_results.csv
    outputs/tables/garch/garch_parameter_estimates.csv
    outputs/tables/garch/garch_summary.txt
    outputs/models/garch/conditional_volatility.csv
    outputs/models/garch/standardized_residuals.csv

Run:
    python src/garch_models.py

Required package:
    pip install arch
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from arch import arch_model
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'arch' package is required for GARCH estimation. "
        "Install it with: pip install arch"
    ) from exc


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "log_returns.csv"

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "garch"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "garch"


# -----------------------------------------------------------------------------
# Empirical design
# -----------------------------------------------------------------------------

ASSETS = ["SPY", "XLF", "KBE"]

PERIODS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "full_sample": (None, None),
    "pre_crisis": ("2005-01-01", "2007-06-30"),
    "crisis": ("2007-07-01", "2009-06-30"),
    "post_crisis": ("2009-07-01", "2012-12-31"),
}

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

MIN_OBS = 250


@dataclass
class FitOutput:
    """Container for a fitted model or a failed fit."""

    asset: str
    period: str
    model_name: str
    result: Optional[object]
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def ensure_output_dirs() -> None:
    """Create output directories if they do not exist."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_returns(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load processed log returns.

    The expected format is a CSV with a date index and one column per asset.
    Returns should already be scaled as percentages, i.e.
    100 * log(P_t / P_{t-1}).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run data_fetcher.py before garch_models.py."
        )

    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()

    # Remove duplicated dates if any exist.
    returns = returns.loc[~returns.index.duplicated(keep="first")]

    # Keep numeric data only.
    returns = returns.apply(pd.to_numeric, errors="coerce")

    missing_assets = [asset for asset in ASSETS if asset not in returns.columns]
    if missing_assets:
        raise ValueError(
            f"Missing expected assets in {path}: {missing_assets}. "
            f"Available columns: {list(returns.columns)}"
        )

    return returns[ASSETS]


def select_period(series: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
    """Select a time period from a return series."""
    out = series.copy()
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out.dropna()


# -----------------------------------------------------------------------------
# Model fitting
# -----------------------------------------------------------------------------

def fit_single_model(
    series: pd.Series,
    asset: str,
    period: str,
    model_name: str,
    spec: Dict[str, object],
) -> FitOutput:
    """Fit a single GARCH-family model."""
    clean_series = series.dropna()

    if clean_series.shape[0] < MIN_OBS:
        return FitOutput(
            asset=asset,
            period=period,
            model_name=model_name,
            result=None,
            error=f"Not enough observations: {clean_series.shape[0]} < {MIN_OBS}",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                clean_series,
                mean=spec["mean"],
                vol=spec["vol"],
                p=int(spec["p"]),
                o=int(spec["o"]),
                q=int(spec["q"]),
                dist=str(spec["dist"]),
                rescale=False,
            )
            result = model.fit(
                disp="off",
                show_warning=False,
                options={"maxiter": 1000},
            )
        return FitOutput(asset, period, model_name, result, None)
    except Exception as exc:  # pragma: no cover - fit failures depend on environment/data
        return FitOutput(asset, period, model_name, None, str(exc))


def fit_all_models(returns: pd.DataFrame) -> Iterable[FitOutput]:
    """Fit all model specifications for all assets and periods."""
    for asset in ASSETS:
        for period_name, (start, end) in PERIODS.items():
            period_series = select_period(returns[asset], start, end)
            for model_name, spec in MODEL_SPECS.items():
                yield fit_single_model(
                    period_series,
                    asset=asset,
                    period=period_name,
                    model_name=model_name,
                    spec=spec,
                )


# -----------------------------------------------------------------------------
# Result extraction
# -----------------------------------------------------------------------------

def _safe_get(series: pd.Series, key: str) -> float:
    """Return a parameter value if present, otherwise NaN."""
    return float(series[key]) if key in series.index else np.nan


def extract_model_selection(fit: FitOutput) -> Dict[str, object]:
    """Extract model-level information for AIC/BIC comparison."""
    base = {
        "asset": fit.asset,
        "period": fit.period,
        "model": fit.model_name,
        "status": "failed" if fit.result is None else "success",
        "error": fit.error,
    }

    if fit.result is None:
        return {
            **base,
            "n_obs": np.nan,
            "loglikelihood": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "convergence_flag": np.nan,
        }

    res = fit.result
    return {
        **base,
        "n_obs": int(res.nobs),
        "loglikelihood": float(res.loglikelihood),
        "aic": float(res.aic),
        "bic": float(res.bic),
        "convergence_flag": getattr(res, "convergence_flag", np.nan),
    }


def extract_parameter_estimates(fit: FitOutput) -> pd.DataFrame:
    """Extract parameter estimates, standard errors, t-stats and p-values."""
    if fit.result is None:
        return pd.DataFrame()

    res = fit.result
    params = res.params
    std_err = res.std_err
    tvalues = res.tvalues
    pvalues = res.pvalues

    rows = []
    for param in params.index:
        rows.append(
            {
                "asset": fit.asset,
                "period": fit.period,
                "model": fit.model_name,
                "parameter": param,
                "estimate": float(params[param]),
                "std_error": float(std_err[param]) if param in std_err.index else np.nan,
                "t_stat": float(tvalues[param]) if param in tvalues.index else np.nan,
                "p_value": float(pvalues[param]) if param in pvalues.index else np.nan,
            }
        )
    return pd.DataFrame(rows)


def extract_key_results(fit: FitOutput) -> Dict[str, object]:
    """Create a compact row with economically important parameters."""
    base = {
        "asset": fit.asset,
        "period": fit.period,
        "model": fit.model_name,
        "status": "failed" if fit.result is None else "success",
        "error": fit.error,
    }

    if fit.result is None:
        return {
            **base,
            "mu": np.nan,
            "omega": np.nan,
            "alpha_1": np.nan,
            "gamma_1": np.nan,
            "beta_1": np.nan,
            "nu": np.nan,
            "alpha_plus_beta": np.nan,
            "gjr_approx_persistence": np.nan,
            "aic": np.nan,
            "bic": np.nan,
            "loglikelihood": np.nan,
            "n_obs": np.nan,
        }

    res = fit.result
    params = res.params

    alpha = _safe_get(params, "alpha[1]")
    beta = _safe_get(params, "beta[1]")
    gamma = _safe_get(params, "gamma[1]")

    # For GARCH(1,1), alpha + beta is the standard persistence measure.
    alpha_plus_beta = alpha + beta if np.isfinite(alpha) and np.isfinite(beta) else np.nan

    # For GJR-GARCH under roughly symmetric innovations, a common approximation is
    # alpha + beta + 0.5 * gamma. This is not used for EGARCH.
    if fit.model_name == "GJR_GARCH_11" and np.isfinite(alpha) and np.isfinite(beta) and np.isfinite(gamma):
        gjr_approx_persistence = alpha + beta + 0.5 * gamma
    else:
        gjr_approx_persistence = np.nan

    return {
        **base,
        "mu": _safe_get(params, "mu"),
        "omega": _safe_get(params, "omega"),
        "alpha_1": alpha,
        "gamma_1": gamma,
        "beta_1": beta,
        "nu": _safe_get(params, "nu"),
        "alpha_plus_beta": alpha_plus_beta,
        "gjr_approx_persistence": gjr_approx_persistence,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "loglikelihood": float(res.loglikelihood),
        "n_obs": int(res.nobs),
    }


def extract_conditional_volatility(fit: FitOutput) -> pd.DataFrame:
    """Extract conditional volatility and variance from a fitted model."""
    if fit.result is None:
        return pd.DataFrame()

    cond_vol = pd.Series(fit.result.conditional_volatility).dropna()
    out = pd.DataFrame(
        {
            "date": cond_vol.index,
            "asset": fit.asset,
            "period": fit.period,
            "model": fit.model_name,
            "conditional_volatility": cond_vol.values,
            "conditional_variance": np.square(cond_vol.values),
        }
    )
    return out


def extract_standardized_residuals(fit: FitOutput) -> pd.DataFrame:
    """Extract residuals and standardized residuals from a fitted model."""
    if fit.result is None:
        return pd.DataFrame()

    residuals = pd.Series(fit.result.resid).dropna()
    std_resid = pd.Series(fit.result.std_resid).dropna()

    common_index = residuals.index.intersection(std_resid.index)
    residuals = residuals.loc[common_index]
    std_resid = std_resid.loc[common_index]

    out = pd.DataFrame(
        {
            "date": common_index,
            "asset": fit.asset,
            "period": fit.period,
            "model": fit.model_name,
            "residual": residuals.values,
            "standardized_residual": std_resid.values,
            "squared_standardized_residual": np.square(std_resid.values),
        }
    )
    return out


# -----------------------------------------------------------------------------
# Summary generation
# -----------------------------------------------------------------------------

def create_text_summary(
    model_selection: pd.DataFrame,
    key_results: pd.DataFrame,
    path: Path,
) -> None:
    """Write a compact plain-text summary of GARCH estimation results."""
    lines = []
    lines.append("GARCH-family model summary")
    lines.append("===========================")
    lines.append("")
    lines.append("Purpose:")
    lines.append(
        "This file summarizes in-sample GARCH-family estimations for the 2008 mortgage crisis project."
    )
    lines.append("")
    lines.append("Models:")
    lines.append("- GARCH(1,1): symmetric volatility benchmark")
    lines.append("- GJR-GARCH(1,1): downside/asymmetric volatility model")
    lines.append("- EGARCH(1,1): robustness model for log-volatility asymmetry")
    lines.append("")
    lines.append("Main reading rules:")
    lines.append("- alpha + beta close to 1 indicates high volatility persistence.")
    lines.append("- GJR gamma > 0 suggests that negative shocks increase future volatility more than positive shocks.")
    lines.append("- Lower AIC/BIC values indicate better in-sample fit after penalizing model complexity.")
    lines.append("- AIC/BIC are not final forecast evaluation metrics; rolling forecast evaluation is handled later.")
    lines.append("")

    successful = model_selection[model_selection["status"] == "success"].copy()
    failed = model_selection[model_selection["status"] != "success"].copy()
    lines.append(f"Successful fits: {len(successful)}")
    lines.append(f"Failed fits: {len(failed)}")
    lines.append("")

    if not failed.empty:
        lines.append("Failed models:")
        for _, row in failed.iterrows():
            lines.append(
                f"- {row['asset']} | {row['period']} | {row['model']}: {row['error']}"
            )
        lines.append("")

    if not successful.empty:
        lines.append("Best model by AIC:")
        for (asset, period), group in successful.groupby(["asset", "period"]):
            best = group.loc[group["aic"].idxmin()]
            lines.append(
                f"- {asset} | {period}: {best['model']} "
                f"(AIC={best['aic']:.3f}, BIC={best['bic']:.3f})"
            )
        lines.append("")

        lines.append("Best model by BIC:")
        for (asset, period), group in successful.groupby(["asset", "period"]):
            best = group.loc[group["bic"].idxmin()]
            lines.append(
                f"- {asset} | {period}: {best['model']} "
                f"(AIC={best['aic']:.3f}, BIC={best['bic']:.3f})"
            )
        lines.append("")

    gjr = key_results[
        (key_results["model"] == "GJR_GARCH_11") & (key_results["status"] == "success")
    ].copy()
    if not gjr.empty:
        lines.append("GJR-GARCH downside asymmetry parameter gamma:")
        for _, row in gjr.sort_values(["asset", "period"]).iterrows():
            gamma = row["gamma_1"]
            persistence = row["gjr_approx_persistence"]
            lines.append(
                f"- {row['asset']} | {row['period']}: "
                f"gamma={gamma:.6f}, approx_persistence={persistence:.6f}"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def run() -> None:
    """Run the full GARCH estimation pipeline."""
    ensure_output_dirs()
    returns = load_returns()

    fit_outputs = list(fit_all_models(returns))

    model_selection_rows = []
    key_result_rows = []
    parameter_frames = []
    conditional_vol_frames = []
    residual_frames = []

    for fit in fit_outputs:
        model_selection_rows.append(extract_model_selection(fit))
        key_result_rows.append(extract_key_results(fit))

        params = extract_parameter_estimates(fit)
        if not params.empty:
            parameter_frames.append(params)

        cond_vol = extract_conditional_volatility(fit)
        if not cond_vol.empty:
            conditional_vol_frames.append(cond_vol)

        resid = extract_standardized_residuals(fit)
        if not resid.empty:
            residual_frames.append(resid)

    model_selection = pd.DataFrame(model_selection_rows)
    key_results = pd.DataFrame(key_result_rows)
    parameter_estimates = (
        pd.concat(parameter_frames, ignore_index=True)
        if parameter_frames
        else pd.DataFrame()
    )
    conditional_volatility = (
        pd.concat(conditional_vol_frames, ignore_index=True)
        if conditional_vol_frames
        else pd.DataFrame()
    )
    standardized_residuals = (
        pd.concat(residual_frames, ignore_index=True)
        if residual_frames
        else pd.DataFrame()
    )

    # Save tables.
    model_selection.to_csv(TABLE_DIR / "garch_model_selection.csv", index=False)
    key_results.to_csv(TABLE_DIR / "garch_key_results.csv", index=False)
    parameter_estimates.to_csv(TABLE_DIR / "garch_parameter_estimates.csv", index=False)

    # Save model-level time series.
    conditional_volatility.to_csv(MODEL_OUTPUT_DIR / "conditional_volatility.csv", index=False)
    standardized_residuals.to_csv(MODEL_OUTPUT_DIR / "standardized_residuals.csv", index=False)

    # Save text summary.
    create_text_summary(
        model_selection=model_selection,
        key_results=key_results,
        path=TABLE_DIR / "garch_summary.txt",
    )

    print("GARCH-family model estimation completed.")
    print(f"Model-selection table: {TABLE_DIR / 'garch_model_selection.csv'}")
    print(f"Key-results table:     {TABLE_DIR / 'garch_key_results.csv'}")
    print(f"Parameter table:       {TABLE_DIR / 'garch_parameter_estimates.csv'}")
    print(f"Conditional vol:       {MODEL_OUTPUT_DIR / 'conditional_volatility.csv'}")
    print(f"Standardized resid:    {MODEL_OUTPUT_DIR / 'standardized_residuals.csv'}")
    print(f"Summary:               {TABLE_DIR / 'garch_summary.txt'}")


if __name__ == "__main__":
    run()
