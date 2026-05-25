import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Project configuration
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/tables/diagnostics")
ANNUALIZATION_FACTOR = 252

CRISIS_WINDOWS = {
    "pre_crisis": ("2005-01-01", "2007-06-30"),
    "crisis": ("2007-07-01", "2009-06-30"),
    "post_crisis": ("2009-07-01", "2012-12-31"),
}

def calculate_core_diagnostics(series: pd.Series, prefix="") -> dict:
    """Calculates only the critical diagnostic tests for a single time series."""
    # Clean and mean-center the data
    x = series.dropna().astype(float)
    x_centered = x - x.mean()
    x_squared = x_centered.pow(2)

    # Return empty dictionary if insufficient data
    if len(x) < 60:  
        return {}

    # 1. Descriptive Statistics
    results = {
        f"{prefix}n_obs": len(x),
        f"{prefix}annualized_vol": x.std(ddof=1) * np.sqrt(ANNUALIZATION_FACTOR),
        f"{prefix}skewness": skew(x, bias=False),
        f"{prefix}kurtosis_excess": kurtosis(x, fisher=True, bias=False),
    }

    # 2. Stationarity (ADF Test)
    adf_res = adfuller(x, autolag="AIC")
    results[f"{prefix}adf_pvalue"] = adf_res[1]

    # 3. Normality (Jarque-Bera)
    jb_res = jarque_bera(x)
    results[f"{prefix}jb_pvalue"] = jb_res.pvalue

    # 4. Volatility Clustering (Squared Ljung-Box Test only)
    lb_res = acorr_ljungbox(x_squared, lags=[10], return_df=True)
    results[f"{prefix}lb_squared_pvalue_lag10"] = float(lb_res["lb_pvalue"].iloc[0])

    return results

def run_diagnostics():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    returns_df = pd.read_csv(PROCESSED_DIR / "log_returns.csv", index_col="Date", parse_dates=True)
    panel_df = pd.read_csv(PROCESSED_DIR / "panel_dataset.csv", parse_dates=["Date"])
    
    # --- 1. FULL SAMPLE ANALYSIS ---
    full_sample_results = []
    for asset in returns_df.columns:
        stats = calculate_core_diagnostics(returns_df[asset])
        stats["Asset"] = asset
        
        # Add the clustering flag based on Squared Ljung-Box
        stats["Volatility_Clustering_Present"] = "Yes" if stats["lb_squared_pvalue_lag10"] < 0.05 else "No"
        full_sample_results.append(stats)
        
    full_df = pd.DataFrame(full_sample_results)
    
    # Reorder columns for better readability
    cols = ["Asset", "n_obs", "annualized_vol", "skewness", "kurtosis_excess", 
            "adf_pvalue", "jb_pvalue", "lb_squared_pvalue_lag10", "Volatility_Clustering_Present"]
    full_df = full_df[cols]
    full_df.to_csv(OUTPUT_DIR / "core_diagnostics_full.csv", index=False)


    # --- 2. PERIOD-BASED (CRISIS) ANALYSIS ---
    period_results = []
    for asset in sorted(panel_df["Asset"].unique()):
        for period in ["pre_crisis", "crisis", "post_crisis"]:
            subset = panel_df[(panel_df["Asset"] == asset) & (panel_df["Period"] == period)]["Return"]
            
            stats = calculate_core_diagnostics(subset)
            if stats:
                stats["Asset"] = asset
                stats["Period"] = period
                period_results.append(stats)

    period_df = pd.DataFrame(period_results)
    
    # Keep only the most critical metrics for the period comparison
    period_cols = ["Asset", "Period", "n_obs", "annualized_vol", "skewness", "kurtosis_excess"]
    period_df = period_df[period_cols]
    period_df.to_csv(OUTPUT_DIR / "core_diagnostics_by_period.csv", index=False)

    print("Cleaned diagnostic reports generated successfully:")
    print("- core_diagnostics_full.csv (For GARCH motivation via Squared Ljung-Box)")
    print("- core_diagnostics_by_period.csv (For crisis period volatility comparison)")

if __name__ == "__main__":
    run_diagnostics()