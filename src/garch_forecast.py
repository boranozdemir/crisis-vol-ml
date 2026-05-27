from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Project configuration
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/forecasts")

# --- EXPANDING WINDOW OOS DATES (VERİ KISITLAMASI OLMADAN) ---
EVAL_START_DATE = "2007-07-01" # Subprime krizinin ilk dalgaları
EVAL_END_DATE = "2009-12-31"   # Krizin ve toparlanmanın sonu

def save_dataframe_as_image(df: pd.DataFrame, filename: str, title: str):
    df_display = df.copy()
    
    # Kırılmaz Formatta "Diverged" (Patlama) Filtresi
    for col in df_display.columns:
        if pd.api.types.is_numeric_dtype(df_display[col]):
            def format_metric(x):
                # Eğer değer NaN, Sonsuz veya 1 Milyondan büyük (Patlamış) ise:
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
    plt.savefig(OUTPUT_DIR / filename, bbox_inches='tight', dpi=300)
    plt.close()

def plot_forecast_comparison(asset: str, actual_var: pd.Series, best_forecast: pd.Series, base_forecast: pd.Series, best_name: str):
    plt.figure(figsize=(12, 6))
    
    plt.plot(actual_var.index, actual_var, color='gray', alpha=0.3, linewidth=1, label="Actual Proxy ($r_t^2$)")
    plt.plot(base_forecast.index, base_forecast, color='navy', linestyle='--', alpha=0.7, linewidth=1.2, label="OOS Baseline GARCH(1,1)")
    plt.plot(best_forecast.index, best_forecast, color='darkred', linewidth=1.5, label=f"OOS Best Model ({best_name})")
    
    plt.axvline(pd.Timestamp(EVAL_START_DATE), color='black', linestyle='-.', linewidth=2, label="Rolling OOS Start")
    plt.axvspan(pd.Timestamp(EVAL_START_DATE), pd.Timestamp(EVAL_END_DATE), color='darkorange', alpha=0.1, label='OOS Evaluation Window')
    
    # Grafiğin görsel odağını tüm kriz periyodunu kapsayacak şekilde genişlettik
    plt.xlim(pd.Timestamp("2006-07-01"), pd.Timestamp("2010-12-31"))
    
    plt.title(f"1-Step Ahead Expanding Window Forecast: {asset}", fontsize=14, fontweight='bold')
    plt.ylabel("Variance")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{asset}_rolling_forecast_comparison.png", dpi=300)
    plt.close()

def calculate_qlike(actual: pd.Series, forecast: pd.Series) -> float:
    eps = 1e-6
    forecast_safe = np.maximum(forecast, eps)
    actual_safe = np.maximum(actual, eps)
    qlike_values = np.log(forecast_safe) + (actual_safe / forecast_safe)
    return np.mean(qlike_values)

def run_forecast_evaluation():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    returns_df = pd.read_csv(PROCESSED_DIR / "log_returns.csv", index_col="Date", parse_dates=True)
    
    model_grid = {
        "GARCH(1,1)": {"vol": "Garch", "p": 1, "o": 0, "q": 1},
        "GARCH(1,2)": {"vol": "Garch", "p": 1, "o": 0, "q": 2},
        "GARCH(2,1)": {"vol": "Garch", "p": 2, "o": 0, "q": 1},
        "GJR-GARCH(1,1)": {"vol": "Garch", "p": 1, "o": 1, "q": 1},
        "EGARCH(1,1)": {"vol": "EGARCH", "p": 1, "o": 1, "q": 1}
    }

    forecast_records = []

    for asset in returns_df.columns:
        print(f"\n{'='*40}")
        print(f"--- Running 1-Step Ahead Expanding Window OOS Forecast for {asset} ---")
        print(f"This requires fitting models day-by-day. Please wait...")
        
        y = returns_df[asset].dropna()
        actual_proxy = y ** 2
        
        oos_mask = (y.index >= EVAL_START_DATE) & (y.index <= EVAL_END_DATE)
        oos_dates = y.index[oos_mask]
        
        asset_forecasts = {}
        
        for name, specs in model_grid.items():
            print(f"  -> Processing {name}...")
            
            daily_forecasts = pd.Series(index=oos_dates, dtype=float)
            
            for date in oos_dates:
                train_y = y.loc[y.index < date]
                
                try:
                    am = arch_model(train_y, mean="Constant", vol=specs["vol"], p=specs["p"], o=specs["o"], q=specs["q"], dist="t", rescale=True)
                    res = am.fit(disp="off")
                    
                    fcast = res.forecast(horizon=1, reindex=False)
                    pred_var = (np.sqrt(fcast.variance.iloc[-1, 0]) / res.scale) ** 2
                    daily_forecasts[date] = pred_var
                    
                except Exception:
                    daily_forecasts[date] = np.nan
            
            daily_forecasts = daily_forecasts.ffill().bfill()
            asset_forecasts[name] = daily_forecasts
            
            actual_eval = actual_proxy.loc[oos_dates]
            
            if daily_forecasts.isna().all() or np.isinf(daily_forecasts).any() or daily_forecasts.max() > 1e6:
                 rmse = np.inf
                 mae = np.inf
                 qlike = np.inf
            else:
                 rmse = np.sqrt(mean_squared_error(actual_eval, daily_forecasts))
                 mae = mean_absolute_error(actual_eval, daily_forecasts)
                 qlike = calculate_qlike(actual_eval, daily_forecasts)
            
            forecast_records.append({
                "Asset": asset,
                "Model": name,
                "OOS RMSE": rmse,
                "OOS MAE": mae,
                "OOS QLIKE": qlike
            })

        asset_df = pd.DataFrame([r for r in forecast_records if r["Asset"] == asset])
        valid_models = asset_df[asset_df['OOS QLIKE'] != np.inf]
        
        if not valid_models.empty:
            best_model_name = valid_models.loc[valid_models['OOS QLIKE'].idxmin()]['Model']
        else:
            best_model_name = "GARCH(1,1)" 

        plot_forecast_comparison(
            asset=asset, 
            actual_var=actual_proxy.loc[EVAL_START_DATE:EVAL_END_DATE], 
            best_forecast=asset_forecasts[best_model_name], 
            base_forecast=asset_forecasts["GARCH(1,1)"], 
            best_name=best_model_name
        )

    forecast_df = pd.DataFrame(forecast_records)
    forecast_df.to_csv(OUTPUT_DIR / "expanding_window_forecast_evaluation.csv", index=False)
    
    for asset in forecast_df['Asset'].unique():
        df_subset = forecast_df[forecast_df['Asset'] == asset].drop(columns=['Asset'])
        df_subset = df_subset.sort_values(by='OOS QLIKE', ascending=True)
        save_dataframe_as_image(df_subset, f"{asset}_expanding_forecast_evaluation.png", f"{asset} 1-Step Ahead Expanding Forecast")

    print(f"\n{'='*40}")
    print(f"Expanding Window Forecasting complete! Results and plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_forecast_evaluation()