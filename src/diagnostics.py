from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/diagnostics")
ANNUALIZATION_FACTOR = 252

CRISIS_WINDOWS = {
    "pre_crisis": ("2005-01-01", "2007-06-30"),
    "crisis": ("2007-07-01", "2009-06-30"),
    "post_crisis": ("2009-07-01", "2012-12-31"),
}

def calculate_core_diagnostics(series: pd.Series, prefix="") -> dict:
    """Calculates critical diagnostic tests including ARCH-LM for Heteroskedasticity."""
    x = series.dropna().astype(float)
    x_centered = x - x.mean()
    x_squared = x_centered.pow(2)

    if len(x) < 60:  
        return {}

    # 1. Descriptive Statistics
    results = {
        f"{prefix}Ann. Vol (%)": x.std(ddof=1) * np.sqrt(ANNUALIZATION_FACTOR),
        f"{prefix}Skewness": skew(x, bias=False),
        f"{prefix}Excess Kurtosis": kurtosis(x, fisher=True, bias=False),
    }

    # 2. Stationarity (ADF Test)
    adf_res = adfuller(x, autolag="AIC")
    results[f"{prefix}ADF p-val"] = adf_res[1]

    # 3. Normality (Jarque-Bera)
    jb_res = jarque_bera(x)
    results[f"{prefix}JB p-val"] = jb_res.pvalue

    # 4. Autocorrelation in Returns
    lb_res = acorr_ljungbox(x, lags=[10], return_df=True)
    results[f"{prefix}LB p-val (Lag 10)"] = float(lb_res["lb_pvalue"].iloc[0])

    # 5. Autocovariance/Clustering in Variance
    lb_sq_res = acorr_ljungbox(x_squared, lags=[10], return_df=True)
    results[f"{prefix}LB^2 p-val (Lag 10)"] = float(lb_sq_res["lb_pvalue"].iloc[0])

    # 6. Conditional Heteroskedasticity (ARCH-LM Test)
    arch_test = het_arch(x_centered, nlags=10)
    results[f"{prefix}ARCH-LM p-val"] = float(arch_test[1])

    return results

def save_dataframe_as_image(df: pd.DataFrame, filename: str, title: str):
    df_display = df.copy()

    for col in df_display.select_dtypes(include=[float]).columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#1f497d")
        elif row % 2 == 0:
            cell.set_facecolor("#f2f2f2")
        else:
            cell.set_facecolor("white")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close()

def run_diagnostics():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    returns_df = pd.read_csv(PROCESSED_DIR / "log_returns.csv", index_col="Date", parse_dates=True)
    panel_df = pd.read_csv(PROCESSED_DIR / "panel_dataset.csv", parse_dates=["Date"])
    
    # --- 1. FULL SAMPLE ANALYSIS ---
    full_sample_results = []
    for asset in returns_df.columns:
        stats = calculate_core_diagnostics(returns_df[asset])
        stats["Asset"] = asset
        
        full_sample_results.append(stats)
        
    full_df = pd.DataFrame(full_sample_results)
    
    cols = ["Asset", "Ann. Vol (%)", "Skewness", "Excess Kurtosis", 
            "ADF p-val", "JB p-val", "LB p-val (Lag 10)", "LB^2 p-val (Lag 10)", 
            "ARCH-LM p-val"]
    full_df = full_df[cols]
    
    full_df.to_csv(OUTPUT_DIR / "core_diagnostics_full.csv", index=False)
    save_dataframe_as_image(full_df, "core_diagnostics_full.png", "Full Sample Core Diagnostics")

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
    # Filter columns for the period breakdown
    period_cols = ["Asset", "Period", "Ann. Vol (%)", "Skewness", "Excess Kurtosis"]
    period_df = period_df[period_cols]
    
    period_df.to_csv(OUTPUT_DIR / "core_diagnostics_by_period.csv", index=False)
    save_dataframe_as_image(period_df, "core_diagnostics_by_period.png", "Descriptive Statistics by Period")

    print(f"Diagnostics successfully generated in '{OUTPUT_DIR}':")
    print(" - CSV files saved.")
    print(" - High-res PNG tables rendered with fixed column widths and clean labels.")

if __name__ == "__main__":
    run_diagnostics()