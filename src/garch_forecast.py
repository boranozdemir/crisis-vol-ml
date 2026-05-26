import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import scipy.stats as stats

warnings.filterwarnings("ignore")

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "log_returns.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasts"
PLOT_DIR = OUTPUT_DIR / "plots"

# --- Configuration ---
ASSETS = ["SPY", "XLF", "KBE"]

MODEL_SPECS = {
    "GARCH_11": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
    "GJR_GARCH_11": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "EGARCH_11": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
}

# We will test the models specifically during the Crisis period
CRISIS_START = "2007-07-01"
CRISIS_END = "2009-06-30"
ROLLING_WINDOW = 250  # 1 trading year of historical data for each step

def run_var_forecasts():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    sns.set_theme(style="whitegrid", palette="muted")

    all_results = []
    backtest_summary = []

    print("Starting Rolling Window VaR Forecasting...")

    for asset in ASSETS:
        series = returns[asset].dropna()
        crisis_series = series.loc[CRISIS_START:CRISIS_END]
        
        if crisis_series.empty:
            continue
            
        for model_name, spec in MODEL_SPECS.items():
            print(f"Processing Out-of-Sample Forecast for {asset} -> {model_name}")
            
            forecast_dates = []
            actual_returns = []
            var_95 = []
            var_99 = []
            
            # Step-by-step Rolling Window Loop
            for date in crisis_series.index:
                loc_idx = series.index.get_loc(date)
                
                # Get the trailing 250 days strictly BEFORE the current date
                train_data = series.iloc[loc_idx - ROLLING_WINDOW : loc_idx]
                
                if len(train_data) < ROLLING_WINDOW:
                    continue
                    
                try:
                    # 1. Fit the model on the rolling window
                    am = arch_model(
                        train_data, mean="Constant", vol=spec["vol"], 
                        p=spec["p"], o=spec["o"], q=spec["q"], 
                        dist=spec["dist"], rescale=False
                    )
                    res = am.fit(disp="off", show_warning=False)
                    
                    # 2. Forecast exactly 1 day ahead
                    fc = res.forecast(horizon=1, align='origin')
                    pred_var = fc.variance.iloc[-1, 0]
                    pred_mu = fc.mean.iloc[-1, 0]
                    nu = res.params.get("nu", 5.0)  # Degrees of freedom for Student-t
                    
                    # 3. Calculate VaR (Value at Risk) using Student-t PPF
                    # arch uses a standardized Student-t, so we scale by sqrt((nu-2)/nu)
                    scale = np.sqrt((nu - 2) / nu) if nu > 2 else 1.0
                    q_95 = stats.t.ppf(0.05, nu) * scale
                    q_99 = stats.t.ppf(0.01, nu) * scale
                    
                    v95 = pred_mu + np.sqrt(pred_var) * q_95
                    v99 = pred_mu + np.sqrt(pred_var) * q_99
                    
                    forecast_dates.append(date)
                    actual_returns.append(series.loc[date])
                    var_95.append(v95)
                    var_99.append(v99)
                    
                except Exception:
                    continue  # Skip day if optimization fails to converge
            
            # --- Save Daily Results ---
            results_df = pd.DataFrame({
                "Date": forecast_dates,
                "Actual_Return": actual_returns,
                "VaR_95": var_95,
                "VaR_99": var_99
            }).set_index("Date")
            
            results_df["Asset"] = asset
            results_df["Model"] = model_name
            
            # Identify Violations (when losses exceed VaR)
            results_df["Violation_95"] = results_df["Actual_Return"] < results_df["VaR_95"]
            results_df["Violation_99"] = results_df["Actual_Return"] < results_df["VaR_99"]
            all_results.append(results_df.reset_index())
            
            # --- Build Backtest Summary ---
            total_days = len(results_df)
            backtest_summary.append({
                "Asset": asset,
                "Model": model_name,
                "Total_Days": total_days,
                "Expected_99_Breaches": round(total_days * 0.01, 1),
                "Actual_99_Breaches": results_df["Violation_99"].sum(),
                "Expected_95_Breaches": round(total_days * 0.05, 1),
                "Actual_95_Breaches": results_df["Violation_95"].sum()
            })

            # --- Generate Beautiful Forecast Plots ---
            plt.figure(figsize=(12, 6))
            
            # Plot returns
            plt.plot(results_df.index, results_df["Actual_Return"], label="Actual Returns", color="gray", alpha=0.6, linewidth=1)
            # Plot VaR limit
            plt.plot(results_df.index, results_df["VaR_99"], label="99% VaR Forecast", color="navy", linewidth=1.5)
            
            # Highlight breaches with red dots
            breaches = results_df[results_df["Violation_99"]]
            plt.scatter(breaches.index, breaches["Actual_Return"], color="red", label="VaR 99% Violation (Crash)", zorder=5)
            
            plt.title(f"Out-of-Sample VaR (99%) Forecast during Crisis - {asset} ({model_name})", fontweight="bold", pad=15)
            plt.ylabel("Daily Return (%)")
            plt.legend(loc="lower left")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"var_forecast_{asset}_{model_name}.png", dpi=300)
            plt.close()

    # --- Save Final CSVs ---
    pd.concat(all_results, ignore_index=True).to_csv(OUTPUT_DIR / "daily_var_forecasts.csv", index=False)
    pd.DataFrame(backtest_summary).to_csv(OUTPUT_DIR / "var_backtest_summary.csv", index=False)
    
    print("\nForecasting completed successfully!")
    print(f"Summary table saved to: {OUTPUT_DIR / 'var_backtest_summary.csv'}")
    print(f"Forecast plots saved to:  {PLOT_DIR}")

if __name__ == "__main__":
    run_var_forecasts()