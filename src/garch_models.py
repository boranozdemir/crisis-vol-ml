from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/models")

def save_dataframe_as_image(df: pd.DataFrame, filename: str, title: str):
    """Renders a pandas DataFrame as a high-quality, readable PNG table."""
    df_display = df.copy()
    
    # Format floats to 4 decimal places
    for col in df_display.select_dtypes(include=[float]).columns:
        df_display[col] = df_display[col].apply(lambda x: f"{x:.4f}")

    # Set up fixed, robust figure size parameters
    num_cols = len(df.columns)
    num_rows = len(df)
    
    # Base dimensions: 1.5 inches per column width, 0.5 inches per row height
    fig_width = max(10, num_cols * 1.5) 
    fig_height = max(3, num_rows * 0.6 + 1.5) 
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.axis('off')

    # Create table
    table = ax.table(cellText=df_display.values, 
                     colLabels=df_display.columns, 
                     loc='center', 
                     cellLoc='center')
    
    # Enforce clear font sizing and vertical scaling
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Increased font size for readability
    table.scale(1, 2.5)     # Significantly increase row height

    # Set uniform column widths
    col_width = 1.0 / num_cols
    for (row, col), cell in table.get_celld().items():
        cell.set_width(col_width)
        
        # Style Header
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=13)
            cell.set_facecolor('#1f497d')
        # Style Data Rows
        else:
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('white')

    plt.title(title, fontweight="bold", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_conditional_variance(asset: str, realized_var: pd.Series, cond_var: pd.Series, model_name: str):
    """Overlays the model's conditional variance over the realized volatility proxy."""
    plt.figure(figsize=(10, 5))
    
    plt.plot(realized_var.index, realized_var, color='gray', alpha=0.3, linewidth=1, label="Realized Proxy ($r_t^2$)")
    plt.plot(cond_var.index, cond_var, color='darkred', linewidth=1.5, label=f"Cond. Variance ({model_name})")
    
    plt.axvspan(pd.Timestamp("2007-07-01"), pd.Timestamp("2009-06-30"), color='gray', alpha=0.15, label='Crisis Period')
    
    plt.title(f"Volatility Fit: {asset} ({model_name})", fontsize=12, fontweight='bold')
    plt.ylabel("Variance")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{asset}_conditional_variance_fit.png", dpi=300)
    plt.close()

def run_model_selection():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    returns_df = pd.read_csv(PROCESSED_DIR / "log_returns.csv", index_col="Date", parse_dates=True)
    
    model_grid = {
        "GARCH(1,1)": {"vol": "Garch", "p": 1, "o": 0, "q": 1},
        "GARCH(1,2)": {"vol": "Garch", "p": 1, "o": 0, "q": 2},
        "GARCH(2,1)": {"vol": "Garch", "p": 2, "o": 0, "q": 1},
        "GJR-GARCH(1,1)": {"vol": "Garch", "p": 1, "o": 1, "q": 1},
        "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}
    }

    evaluation_records = []

    for asset in returns_df.columns:
        print(f"--- Processing {asset} ---")
        y = returns_df[asset].dropna()
        y_squared = y ** 2
        
        best_bic = np.inf
        best_model_name = ""
        best_cond_var = None
        
        for name, specs in model_grid.items():
            try:
                am = arch_model(y, mean="Constant", vol=specs["vol"], p=specs["p"], o=specs["o"], q=specs["q"], dist="t", rescale=True)
                res = am.fit(disp="off")
                
                # Diagnostics calculations
                std_resid = res.resid / res.conditional_volatility
                lb_res = acorr_ljungbox(std_resid.dropna(), lags=[10], return_df=True)
                lb_sq_res = acorr_ljungbox(std_resid.dropna()**2, lags=[10], return_df=True)
                
                # Max p-value of variance equation parameters (to check significance)
                var_pvalues = res.pvalues[res.pvalues.index.str.contains('omega|alpha|beta|gamma')]
                max_pval = var_pvalues.max()
                
                # Persistence Calculation
                persistence = np.nan
                if specs["vol"] == "Garch" and specs["o"] == 0:
                    persistence = sum(res.params[res.params.index.str.contains('alpha|beta')])
                elif specs["vol"] == "Garch" and specs["o"] > 0: # GJR
                    alpha = sum(res.params[res.params.index.str.contains('alpha')])
                    beta = sum(res.params[res.params.index.str.contains('beta')])
                    gamma = sum(res.params[res.params.index.str.contains('gamma')])
                    persistence = alpha + beta + (0.5 * gamma)
                elif specs["vol"] == "EGARCH":
                    persistence = sum(res.params[res.params.index.str.contains('beta')])

                bic = res.bic
                
                evaluation_records.append({
                    "Asset": asset,
                    "Model": name,
                    "BIC": bic,
                    "Persist.": persistence,
                    "Max p-val": max_pval,
                    "Res.LB p-val": lb_res['lb_pvalue'].iloc[0],
                    "Res.LB^2 p-val": lb_sq_res['lb_pvalue'].iloc[0]
                })
                
                if bic < best_bic:
                    best_bic = bic
                    best_model_name = name
                    scale = res.scale
                    best_cond_var = (res.conditional_volatility / scale) ** 2
                    
            except Exception as e:
                print(f"Failed to fit {name} for {asset}: {e}")

        plot_conditional_variance(asset, y_squared, best_cond_var, "GJR-GARCH(1,1)")

    eval_df = pd.DataFrame(evaluation_records)
    eval_df.to_csv(OUTPUT_DIR / "model_evaluation_matrix.csv", index=False)
    
    for asset in eval_df['Asset'].unique():
        asset_df = eval_df[eval_df['Asset'] == asset].drop(columns=['Asset'])
        save_dataframe_as_image(asset_df, f"{asset}_model_evaluation.png", f"{asset} Model Evaluation Matrix")

    print(f"\nModeling complete. Clean and robust results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_model_selection()