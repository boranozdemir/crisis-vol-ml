import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error

FORECAST_DIR = Path("outputs/ml_forecast")
INPUT_FILE = FORECAST_DIR / "daily_oos_predictions.csv"

def calculate_qlike(actual: pd.Series, forecast: pd.Series) -> float:
    """Calculates the QLIKE loss function, strictly penalizing variance under-prediction."""
    eps = 1e-6
    forecast_safe = np.maximum(forecast, eps)
    actual_safe = np.maximum(actual, eps)
    qlike_values = np.log(forecast_safe) + (actual_safe / forecast_safe)
    return np.mean(qlike_values)

def save_dataframe_as_image(df: pd.DataFrame, filename: str, title: str):

    df_display = df.copy()
    
    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]):
            def format_metric(x):
                if pd.isna(x) or np.isinf(x) or x > 1e6:
                    return "∞ (Diverged)"
                return f"{x:.4f}"
            df_display[col] = df_display[col].apply(format_metric)

    num_cols = len(df.columns)
    num_rows = len(df)
    
    fig_width = max(10, num_cols * 1.5) 
    fig_height = max(3, num_rows * 0.6 + 1.5) 
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)
    ax.axis('off')

    table = ax.table(cellText=df_display.values, colLabels=df_display.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12) 
    table.scale(1, 2.5)     

    col_width = 1.0 / num_cols
    for (row, col), cell in table.get_celld().items():
        cell.set_width(col_width)
        if row == 0:
            cell.set_text_props(weight='bold', color='white', fontsize=13)
            cell.set_facecolor('#1f497d')
        else:
            cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')

    plt.title(title, fontweight="bold", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(FORECAST_DIR / filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_forecast_comparison(asset: str, results_df: pd.DataFrame, best_model: str):
    """Plots the actual realized volatility proxy against GJR-GARCH and the Best Model."""
    plt.figure(figsize=(12, 6))
    
    # Ensure Date is index for plotting
    plot_df = results_df.set_index('Date')
    plot_df.index = pd.to_datetime(plot_df.index)
    
    # Plot Actual Proxy
    plt.plot(plot_df.index, plot_df['Actual_Proxy'], color='gray', alpha=0.3, linewidth=1, label="Actual Proxy ($r_t^2$)")
    
    # Plot Baseline Ekonometri (GJR-GARCH)
    plt.plot(plot_df.index, plot_df['GJR-GARCH(1,1)'], color='navy', linestyle='--', alpha=0.7, linewidth=1.2, label="TS Baseline (GJR-GARCH)")
    
    # Plot The Ultimate Winner (if it's different from GJR-GARCH)
    if best_model != 'GJR-GARCH(1,1)':
        plt.plot(plot_df.index, plot_df[best_model], color='darkred', linewidth=1.5, label=f"OOS Best Model ({best_model})")
    
    # Crisis Window Highlighting
    plt.axvline(plot_df.index[0], color='black', linestyle='-.', linewidth=2, label="Rolling OOS Start")
    plt.axvspan(plot_df.index[0], plot_df.index[-1], color='darkorange', alpha=0.1, label='OOS Crisis Window')
    
    plt.title(f"Forecast Comparison for ML: {asset}", fontsize=14, fontweight='bold')
    plt.ylabel("Variance")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(FORECAST_DIR / f"{asset}_hybrid_forecast_comparison.png", dpi=300)
    plt.close()

def run_visualizer():
    if not INPUT_FILE.exists():
        print(f"Error: Could not find {INPUT_FILE}. Run forecast_engine.py first.")
        return

    print("Loading forecast data...")
    df = pd.read_csv(INPUT_FILE)
    
    models = ["GARCH(1,1)", "GJR-GARCH(1,1)", "RF_Pure", "XGB_Pure", "RF_Hybrid", "XGB_Hybrid"]
    
    for asset in df['Asset'].unique():
        print(f"\nEvaluating {asset}...")
        asset_df = df[df['Asset'] == asset].copy()
        actual = asset_df['Actual_Proxy']
        
        records = []
        for model in models:
            forecast = asset_df[model]
            
            # Check for diverged models (NaNs or Infs)
            if forecast.isna().all() or np.isinf(forecast).any() or forecast.max() > 1e6:
                rmse, mae, qlike = np.inf, np.inf, np.inf
            else:
                rmse = np.sqrt(mean_squared_error(actual, forecast))
                mae = mean_absolute_error(actual, forecast)
                qlike = calculate_qlike(actual, forecast)
                
            records.append({
                "Model": model,
                "RMSE": rmse,
                "MAE": mae,
                "QLIKE": qlike
            })
            
        metrics_df = pd.DataFrame(records)
        metrics_df = metrics_df.sort_values(by='QLIKE', ascending=True)
        
        # Save Evaluation Table
        save_dataframe_as_image(metrics_df, f"{asset}_hybrid_evaluation_matrix.png", f"{asset} Out-of-Sample Hybrid Evaluation")
        
        # Identify the winner
        best_model = metrics_df.iloc[0]['Model']
        print(f"  -> Best Model (Lowest QLIKE): {best_model}")
        
        # Plot Time Series Comparison
        plot_forecast_comparison(asset, asset_df, best_model)
        
    print(f"\n{'='*50}")
    print(f"Visualization complete! Check the {FORECAST_DIR} folder for PNG files.")

if __name__ == "__main__":
    run_visualizer()