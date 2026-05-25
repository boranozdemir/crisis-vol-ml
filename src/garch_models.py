from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "log_returns.csv"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "garch"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "garch"

ASSETS = ["SPY", "XLF", "KBE"]

PERIODS = {
    "full_sample": (None, None),
    "pre_crisis": ("2005-01-01", "2007-06-30"),
    "crisis": ("2007-07-01", "2009-06-30"),
    "post_crisis": ("2009-07-01", "2012-12-31"),
}

MODEL_SPECS = {
    "GARCH_11": {"vol": "GARCH", "p": 1, "o": 0, "q": 1, "dist": "t"},
    "GJR_GARCH_11": {"vol": "GARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
    "EGARCH_11": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1, "dist": "t"},
}

MIN_OBS = 250

def ensure_dirs():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def estimate_garch_models():
    ensure_dirs()
    
    # Load and prepare returns
    returns = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True).sort_index()
    
    parameters_list = []
    cond_vol_list = []
    std_resid_list = []

    print("Starting GARCH estimations...")

    for asset in ASSETS:
        for period_name, (start, end) in PERIODS.items():
            
            # Slice data for the period
            if start and end:
                series = returns.loc[start:end, asset].dropna()
            else:
                series = returns[asset].dropna()

            if len(series) < MIN_OBS:
                continue

            for model_name, spec in MODEL_SPECS.items():
                try:
                    # Fit the model
                    am = arch_model(
                        series, 
                        mean="Constant", 
                        vol=spec["vol"], 
                        p=spec["p"], 
                        o=spec["o"], 
                        q=spec["q"], 
                        dist=spec["dist"], 
                        rescale=False
                    )
                    res = am.fit(disp="off", show_warning=False)

                    # Extract Key Parameters & P-Values
                    params = res.params
                    pvals = res.pvalues  
                    
                    alpha = params.get("alpha[1]", np.nan)
                    beta = params.get("beta[1]", np.nan)
                    gamma = params.get("gamma[1]", 0.0) # 0 if not GJR/EGARCH

                    if model_name == "GARCH_11":
                        persistence = alpha + beta
                    elif model_name == "GJR_GARCH_11":
                        persistence = alpha + beta + 0.5 * gamma
                    elif model_name == "EGARCH_11":
                        persistence = beta  # EGARCH için kalıcılık sadece beta'dır
                    else:
                        persistence = np.nan
                    
                    # Store Results (Now with p-values!)
                    parameters_list.append({
                        "Asset": asset,
                        "Period": period_name,
                        "Model": model_name,
                        "n_obs": res.nobs,
                        "AIC": res.aic,
                        "BIC": res.bic,
                        "alpha_1": alpha,
                        "alpha_1_pval": pvals.get("alpha[1]", np.nan),
                        "gamma_1": params.get("gamma[1]", np.nan),
                        "gamma_1_pval": pvals.get("gamma[1]", np.nan),  # <-- Araştırma sorusunun kalbi
                        "beta_1": beta,
                        "beta_1_pval": pvals.get("beta[1]", np.nan),
                        "persistence": persistence
                    })

                    # Extract Conditional Volatility & Standardized Residuals
                    cond_vol = pd.DataFrame({
                        "Date": series.index,
                        "Asset": asset,
                        "Period": period_name,
                        "Model": model_name,
                        "Conditional_Volatility": res.conditional_volatility
                    })
                    cond_vol_list.append(cond_vol)

                    std_resid = pd.DataFrame({
                        "Date": series.index,
                        "Asset": asset,
                        "Period": period_name,
                        "Model": model_name,
                        "Standardized_Residuals": res.std_resid
                    })
                    std_resid_list.append(std_resid)

                except Exception as e:
                    print(f"Failed to fit {model_name} for {asset} in {period_name}: {e}")

    # --- Compile and Save DataFrames ---
    
    if parameters_list:
        params_df = pd.DataFrame(parameters_list)
        params_df.to_csv(TABLE_DIR / "garch_parameters_and_fit.csv", index=False)
        print(f"Saved parameters to: {TABLE_DIR / 'garch_parameters_and_fit.csv'}")

    if cond_vol_list:
        cond_vol_df = pd.concat(cond_vol_list, ignore_index=True)
        cond_vol_df.to_csv(MODEL_OUTPUT_DIR / "conditional_volatility.csv", index=False)
        print(f"Saved conditional volatility to: {MODEL_OUTPUT_DIR / 'conditional_volatility.csv'}")

    if std_resid_list:
        std_resid_df = pd.concat(std_resid_list, ignore_index=True)
        std_resid_df.to_csv(MODEL_OUTPUT_DIR / "standardized_residuals.csv", index=False)
        print(f"Saved standardized residuals to: {MODEL_OUTPUT_DIR / 'standardized_residuals.csv'}")

    print("GARCH model estimations completed.")

if __name__ == "__main__":
    estimate_garch_models()