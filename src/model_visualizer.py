"""
model_visualizer.py

Create academic model-level figures for the 2008 mortgage crisis volatility
project.

This script does not estimate any model. It only reads outputs produced by
    - garch_models.py
    - model_diagnostics.py
and converts the main econometric findings into clean figures.

Expected inputs:
    outputs/tables/garch/garch_key_results.csv
    outputs/tables/garch/garch_model_selection.csv
    outputs/models/garch/conditional_volatility.csv
    outputs/tables/garch_diagnostics/post_model_diagnostics.csv
    data/processed/log_returns.csv

Main outputs:
    outputs/figures/model_overview/*.png
    outputs/figures/model_overview/model_visualization_summary.txt

Run:
    python src/model_visualizer.py

Required packages:
    pip install pandas numpy matplotlib
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GARCH_TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "garch"
GARCH_MODEL_DIR = PROJECT_ROOT / "outputs" / "models" / "garch"
GARCH_DIAG_DIR = PROJECT_ROOT / "outputs" / "tables" / "garch_diagnostics"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

KEY_RESULTS_PATH = GARCH_TABLE_DIR / "garch_key_results.csv"
MODEL_SELECTION_PATH = GARCH_TABLE_DIR / "garch_model_selection.csv"
CONDITIONAL_VOL_PATH = GARCH_MODEL_DIR / "conditional_volatility.csv"
POST_DIAGNOSTICS_PATH = GARCH_DIAG_DIR / "post_model_diagnostics.csv"
RETURNS_PATH = DATA_DIR / "log_returns.csv"

FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures" / "model_overview"


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

ASSET_ORDER = ["SPY", "XLF", "KBE"]
PERIOD_ORDER = ["pre_crisis", "crisis", "post_crisis", "full_sample"]
PERIOD_ORDER_NO_FULL = ["pre_crisis", "crisis", "post_crisis"]
MODEL_ORDER = ["GARCH_11", "GJR_GARCH_11", "EGARCH_11"]

CRISIS_START = pd.Timestamp("2007-07-01")
CRISIS_END = pd.Timestamp("2009-06-30")

DPI = 160


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------

def ensure_figure_dir() -> None:
    """Create the figure output directory if it does not exist."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. {hint}")


def load_key_results(path: Path = KEY_RESULTS_PATH) -> pd.DataFrame:
    """Load compact GARCH parameter results."""
    _require_file(path, "Run src/garch_models.py before src/model_visualizer.py.")
    df = pd.read_csv(path)
    required = {"asset", "period", "model", "alpha_1", "beta_1", "gamma_1", "gjr_approx_persistence", "aic", "bic"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def load_model_selection(path: Path = MODEL_SELECTION_PATH) -> pd.DataFrame:
    """Load AIC/BIC model-selection output."""
    _require_file(path, "Run src/garch_models.py before src/model_visualizer.py.")
    df = pd.read_csv(path)
    required = {"asset", "period", "model", "aic", "bic", "status"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def load_conditional_volatility(path: Path = CONDITIONAL_VOL_PATH) -> pd.DataFrame:
    """Load conditional volatility series from GARCH-family models."""
    _require_file(path, "Run src/garch_models.py before src/model_visualizer.py.")
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"date", "asset", "period", "model", "conditional_volatility", "conditional_variance"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    df = df.sort_values(["asset", "period", "model", "date"]).reset_index(drop=True)
    return df


def load_post_diagnostics(path: Path = POST_DIAGNOSTICS_PATH) -> pd.DataFrame:
    """Load post-GARCH diagnostic output."""
    _require_file(path, "Run src/model_diagnostics.py before src/model_visualizer.py.")
    df = pd.read_csv(path)
    required = {
        "asset",
        "period",
        "model",
        "diagnostic_pass_count",
        "diagnostic_total_count",
        "remaining_arch_effect",
        "remaining_squared_residual_autocorrelation",
        "remaining_simple_sign_bias",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return df


def load_returns(path: Path = RETURNS_PATH) -> pd.DataFrame:
    """Load processed scaled log returns."""
    _require_file(path, "Run src/data_fetcher.py before src/model_visualizer.py.")
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()
    return returns


# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def _save_figure(fig: plt.Figure, filename: str, generated_files: List[str]) -> None:
    """Save and close a matplotlib figure."""
    path = FIGURE_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    generated_files.append(str(path.relative_to(PROJECT_ROOT)))


def _ordered_periods(periods: Iterable[str], include_full: bool = False) -> List[str]:
    order = PERIOD_ORDER if include_full else PERIOD_ORDER_NO_FULL
    present = set(periods)
    ordered = [p for p in order if p in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _ordered_assets(assets: Iterable[str]) -> List[str]:
    present = set(assets)
    ordered = [a for a in ASSET_ORDER if a in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def _ordered_models(models: Iterable[str]) -> List[str]:
    present = set(models)
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------

def plot_conditional_volatility_crisis(
    cond_vol: pd.DataFrame,
    generated_files: List[str],
) -> None:
    """Plot crisis-period conditional volatility by model for each asset."""
    crisis = cond_vol[cond_vol["period"] == "crisis"].copy()
    if crisis.empty:
        return

    for asset in _ordered_assets(crisis["asset"].unique()):
        asset_df = crisis[crisis["asset"] == asset]
        if asset_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 5.5))
        for model in _ordered_models(asset_df["model"].unique()):
            subset = asset_df[asset_df["model"] == model]
            ax.plot(
                subset["date"],
                subset["conditional_volatility"],
                linewidth=1.4,
                label=model,
            )

        ax.set_title(f"Crisis-Period Conditional Volatility: {asset}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Conditional volatility (%)")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        _save_figure(fig, f"conditional_volatility_crisis_{asset}.png", generated_files)


def plot_gjr_volatility_vs_absolute_return_crisis(
    cond_vol: pd.DataFrame,
    returns: pd.DataFrame,
    generated_files: List[str],
) -> None:
    """Compare GJR conditional volatility with absolute returns during the crisis."""
    gjr = cond_vol[
        (cond_vol["period"] == "crisis") & (cond_vol["model"] == "GJR_GARCH_11")
    ].copy()
    if gjr.empty:
        return

    crisis_returns = returns.loc[(returns.index >= CRISIS_START) & (returns.index <= CRISIS_END)]

    for asset in _ordered_assets(gjr["asset"].unique()):
        if asset not in crisis_returns.columns:
            continue

        vol = gjr[gjr["asset"] == asset].set_index("date")["conditional_volatility"].sort_index()
        abs_ret = crisis_returns[asset].abs().rename("absolute_return")
        common_index = vol.index.intersection(abs_ret.index)
        if common_index.empty:
            continue

        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(common_index, abs_ret.loc[common_index], linewidth=0.9, alpha=0.65, label="Absolute return")
        ax.plot(common_index, vol.loc[common_index], linewidth=1.6, label="GJR conditional volatility")
        ax.set_title(f"GJR Conditional Volatility vs Absolute Returns: {asset}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Percent")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        _save_figure(fig, f"gjr_volatility_vs_absolute_return_crisis_{asset}.png", generated_files)


def plot_gjr_gamma_by_period(key_results: pd.DataFrame, generated_files: List[str]) -> None:
    """Plot the GJR downside asymmetry parameter by period and asset."""
    gjr = key_results[key_results["model"] == "GJR_GARCH_11"].copy()
    gjr = gjr[gjr["period"] != "full_sample"]
    if gjr.empty or "gamma_1" not in gjr.columns:
        return

    pivot = gjr.pivot(index="period", columns="asset", values="gamma_1")
    pivot = pivot.reindex(_ordered_periods(pivot.index)).reindex(columns=_ordered_assets(pivot.columns))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(0, linewidth=1.0)
    ax.set_title("GJR-GARCH Downside Asymmetry Parameter")
    ax.set_xlabel("Period")
    ax.set_ylabel("gamma")
    ax.legend(title="Asset", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, "gjr_gamma_by_period.png", generated_files)


def plot_gjr_persistence_by_period(key_results: pd.DataFrame, generated_files: List[str]) -> None:
    """Plot approximate GJR persistence by period and asset."""
    gjr = key_results[key_results["model"] == "GJR_GARCH_11"].copy()
    gjr = gjr[gjr["period"] != "full_sample"]
    if gjr.empty or "gjr_approx_persistence" not in gjr.columns:
        return

    pivot = gjr.pivot(index="period", columns="asset", values="gjr_approx_persistence")
    pivot = pivot.reindex(_ordered_periods(pivot.index)).reindex(columns=_ordered_assets(pivot.columns))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.axhline(1.0, linewidth=1.0, linestyle="--", label="Unit persistence")
    ax.set_title("Approximate Volatility Persistence in GJR-GARCH")
    ax.set_xlabel("Period")
    ax.set_ylabel("alpha + beta + 0.5 × gamma")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, "gjr_persistence_by_period.png", generated_files)


def plot_aic_delta_crisis(model_selection: pd.DataFrame, generated_files: List[str]) -> None:
    """Plot crisis-period AIC distance from the best model."""
    df = model_selection[(model_selection["period"] == "crisis") & (model_selection["status"] == "success")].copy()
    if df.empty:
        return

    df["delta_aic"] = df.groupby("asset")["aic"].transform(lambda x: x - x.min())
    pivot = df.pivot(index="asset", columns="model", values="delta_aic")
    pivot = pivot.reindex(_ordered_assets(pivot.index)).reindex(columns=_ordered_models(pivot.columns))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Crisis-Period AIC Distance from Best Model")
    ax.set_xlabel("Asset")
    ax.set_ylabel("AIC - minimum AIC")
    ax.legend(title="Model", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, "aic_delta_crisis.png", generated_files)


def plot_bic_delta_crisis(model_selection: pd.DataFrame, generated_files: List[str]) -> None:
    """Plot crisis-period BIC distance from the best model."""
    df = model_selection[(model_selection["period"] == "crisis") & (model_selection["status"] == "success")].copy()
    if df.empty:
        return

    df["delta_bic"] = df.groupby("asset")["bic"].transform(lambda x: x - x.min())
    pivot = df.pivot(index="asset", columns="model", values="delta_bic")
    pivot = pivot.reindex(_ordered_assets(pivot.index)).reindex(columns=_ordered_models(pivot.columns))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Crisis-Period BIC Distance from Best Model")
    ax.set_xlabel("Asset")
    ax.set_ylabel("BIC - minimum BIC")
    ax.legend(title="Model", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, "bic_delta_crisis.png", generated_files)


def plot_diagnostic_pass_count_crisis(diagnostics: pd.DataFrame, generated_files: List[str]) -> None:
    """Plot crisis-period diagnostic pass counts by model."""
    crisis = diagnostics[diagnostics["period"] == "crisis"].copy()
    if crisis.empty:
        return

    pivot = crisis.pivot(index="asset", columns="model", values="diagnostic_pass_count")
    pivot = pivot.reindex(_ordered_assets(pivot.index)).reindex(columns=_ordered_models(pivot.columns))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Crisis-Period Post-GARCH Diagnostic Pass Count")
    ax.set_xlabel("Asset")
    ax.set_ylabel("Passed checks out of 5")
    ax.set_ylim(0, 5.5)
    ax.legend(title="Model", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, "diagnostic_pass_count_crisis.png", generated_files)


def plot_remaining_problem_counts(diagnostics: pd.DataFrame, generated_files: List[str]) -> None:
    """Plot how often each model leaves residual problems across all cases."""
    if diagnostics.empty:
        return

    flags = {
        "ARCH effect": "remaining_arch_effect",
        "Squared residual autocorr.": "remaining_squared_residual_autocorrelation",
        "Sign bias": "remaining_simple_sign_bias",
    }

    rows = []
    for model in _ordered_models(diagnostics["model"].unique()):
        subset = diagnostics[diagnostics["model"] == model]
        for label, col in flags.items():
            rows.append({"model": model, "problem": label, "count": int((subset[col] == "yes").sum())})

    counts = pd.DataFrame(rows)
    pivot = counts.pivot(index="problem", columns="model", values="count")
    pivot = pivot.reindex(list(flags.keys())).reindex(columns=_ordered_models(pivot.columns))

    fig, ax = plt.subplots(figsize=(10, 5.2))
    pivot.plot(kind="bar", ax=ax, width=0.78)
    ax.set_title("Remaining Residual Problems Across Asset-Period Cases")
    ax.set_xlabel("Diagnostic issue")
    ax.set_ylabel("Number of cases")
    ax.legend(title="Model", frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    _save_figure(fig, "remaining_residual_problem_counts.png", generated_files)


def plot_best_model_heatmap_like(model_selection: pd.DataFrame, generated_files: List[str]) -> None:
    """Create a compact table-like figure showing best AIC model by asset and period."""
    df = model_selection[model_selection["status"] == "success"].copy()
    df = df[df["period"] != "full_sample"]
    if df.empty:
        return

    best_rows = []
    for (asset, period), group in df.groupby(["asset", "period"]):
        best = group.loc[group["aic"].idxmin()]
        best_rows.append({"asset": asset, "period": period, "best_model": best["model"]})
    best_df = pd.DataFrame(best_rows)

    assets = _ordered_assets(best_df["asset"].unique())
    periods = _ordered_periods(best_df["period"].unique())
    table = best_df.pivot(index="period", columns="asset", values="best_model").reindex(periods).reindex(columns=assets)

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.axis("off")
    cell_text = table.fillna("").values.tolist()
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=table.index.tolist(),
        colLabels=table.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.35)
    ax.set_title("Best In-Sample Model by AIC", pad=14)
    _save_figure(fig, "best_aic_model_table.png", generated_files)


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

def create_summary(generated_files: List[str], path: Path) -> None:
    """Write a short text summary of generated model figures."""
    lines = [
        "Model visualization summary",
        "===========================",
        "",
        "Purpose:",
        "These figures summarize the in-sample GARCH-family results and post-model diagnostics.",
        "They are designed for report-level interpretation before moving to rolling forecast evaluation and ML benchmarks.",
        "",
        "Main interpretation guide:",
        "- conditional_volatility_crisis_* shows how each GARCH-family model tracks crisis volatility.",
        "- gjr_gamma_by_period shows whether downside-risk asymmetry changes across periods.",
        "- gjr_persistence_by_period shows whether volatility shocks become more persistent during the crisis.",
        "- aic_delta_crisis and bic_delta_crisis show in-sample model-selection differences during the crisis.",
        "- diagnostic_pass_count_crisis and remaining_residual_problem_counts summarize post-GARCH diagnostic adequacy.",
        "",
        f"Generated figures: {len(generated_files)}",
    ]
    for file in generated_files:
        lines.append(f"- {file}")

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def run() -> None:
    """Run the model visualization pipeline."""
    ensure_figure_dir()

    key_results = load_key_results()
    model_selection = load_model_selection()
    cond_vol = load_conditional_volatility()
    diagnostics = load_post_diagnostics()
    returns = load_returns()

    generated_files: List[str] = []

    plot_conditional_volatility_crisis(cond_vol, generated_files)
    plot_gjr_volatility_vs_absolute_return_crisis(cond_vol, returns, generated_files)
    plot_gjr_gamma_by_period(key_results, generated_files)
    plot_gjr_persistence_by_period(key_results, generated_files)
    plot_aic_delta_crisis(model_selection, generated_files)
    plot_bic_delta_crisis(model_selection, generated_files)
    plot_diagnostic_pass_count_crisis(diagnostics, generated_files)
    plot_remaining_problem_counts(diagnostics, generated_files)
    plot_best_model_heatmap_like(model_selection, generated_files)

    summary_path = FIGURE_DIR / "model_visualization_summary.txt"
    create_summary(generated_files, summary_path)

    print("Model visualization completed.")
    print(f"Figures directory: {FIGURE_DIR}")
    print(f"Summary:           {summary_path}")


if __name__ == "__main__":
    run()
