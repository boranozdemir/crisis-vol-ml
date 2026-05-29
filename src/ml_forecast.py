from pathlib import Path
import numpy as np
import pandas as pd
from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/ml_forecast")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_START_DATE = "2007-07-01" 
EVAL_END_DATE = "2009-12-31"   

def create_ml_features(series: pd.Series) -> pd.DataFrame:
    """Generates lagged and rolling features for Machine Learning models."""
    df = pd.DataFrame({"Return": series})
    df["Target_Proxy"] = df["Return"] ** 2  # The y value we want to predict
    
    # 1. Lagged Returns and Squared Returns (Momentum & Volatility Base)
    df["Lag_1"] = df["Return"].shift(1)
    df["Lag_2"] = df["Return"].shift(2)
    df["Lag_1_Sq"] = df["Lag_1"] ** 2
    df["Lag_4_Sq"] = df["Return"].shift(4) ** 2
    
    # 2. Rolling Volatility (Structural Regimes)
    df["Roll_Std_5"] = df["Return"].rolling(window=5).std()
    df["Roll_Std_22"] = df["Return"].rolling(window=22).std()
    
    return df

def run_forecast_engine():

    returns_df = pd.read_csv(PROCESSED_DIR / "log_returns.csv", index_col="Date", parse_dates=True)
    
    garch_grid = {
        "GJR-GARCH(1,1)": {"vol": "Garch", "p": 1, "o": 1, "q": 1}
    }

    all_daily_forecasts = []

    for asset in returns_df.columns:
        print(f"\n{'='*50}")
        print(f"--- Starting Forecast Engine for: {asset} ---")
        
        y = returns_df[asset].dropna()
        actual_proxy = y ** 2
        
        # Build base ML features for the entire timeline
        features_df = create_ml_features(y)
        
        oos_mask = (y.index >= EVAL_START_DATE) & (y.index <= EVAL_END_DATE)
        oos_dates = y.index[oos_mask]
        
        total_days = len(oos_dates)
        
        for i, date in enumerate(oos_dates):
            if i % 50 == 0:
                print(f"  -> Processing date {date.strftime('%Y-%m-%d')} ({i}/{total_days})...")
                
            # 1. SPLIT DATA: Strictly isolate data prior to the current prediction date
            train_y = y.loc[y.index < date]
            train_features = features_df.loc[features_df.index < date].copy()
            test_features_base = features_df.loc[[date]].copy() # Just the row for 'date'
            
            # Dictionary to store all models' predictions for this specific day
            daily_results = {"Date": date, "Asset": asset, "Actual_Proxy": actual_proxy.loc[date]}
            
            gjr_in_sample_var = None
            gjr_forecast_var = None
            
            # 2. GARCH ENGINE
            for name, specs in garch_grid.items():
                try:
                    am = arch_model(train_y, mean="Constant", vol=specs["vol"], p=specs["p"], o=specs["o"], q=specs["q"], dist="t", rescale=True)
                    res = am.fit(disp="off")
                    
                    # 1-step ahead OOS prediction
                    fcast = res.forecast(horizon=1, reindex=False)
                    pred_var = (np.sqrt(fcast.variance.iloc[-1, 0]) / res.scale) ** 2
                    daily_results[name] = pred_var
                    
                    # UNIFORM ML FEED: Always capture GJR-GARCH details for the ML Hybrid model
                    if name == "GJR-GARCH(1,1)":
                        # rescale the conditional volatility back to the original return space
                        cond_var = (res.conditional_volatility / res.scale) ** 2
                        # Align with the training features index
                        gjr_in_sample_var = cond_var.reindex(train_features.index).bfill()
                        gjr_forecast_var = pred_var
                        
                except Exception:
                    daily_results[name] = np.nan
                    if name == "GJR-GARCH(1,1)":
                        gjr_in_sample_var = pd.Series(np.nan, index=train_features.index)
                        gjr_forecast_var = np.nan
            
            # 3. MACHINE LEARNING ENGINE
            y_train = train_features["Target_Proxy"]
            X_train_pure = train_features.drop(columns=["Target_Proxy", "Return"])
            X_test_pure = test_features_base.drop(columns=["Target_Proxy", "Return"])
            
            # Construct Hybrid Features (use GJR-GARCH features into ML)
            X_train_hybrid = X_train_pure.copy()
            X_train_hybrid["GJR_Var"] = gjr_in_sample_var
            
            X_test_hybrid = X_test_pure.copy()
            X_test_hybrid["GJR_Var"] = gjr_forecast_var
            
            # Clean NaNs
            valid_idx = X_train_hybrid.dropna().index
            
            if len(valid_idx) > 100 : 
                # Model 1: Pure Random Forest
                rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                rf.fit(X_train_pure.loc[valid_idx], y_train.loc[valid_idx])
                daily_results["RF_Pure"] = rf.predict(X_test_pure)[0]
                
                # Model 2: Pure XGBoost
                xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
                xgb_model.fit(X_train_pure.loc[valid_idx], y_train.loc[valid_idx])
                daily_results["XGB_Pure"] = xgb_model.predict(X_test_pure)[0]
                
                # Model 3: Hybrid Random Forest (GARCH + RF)
                rf_hybrid = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                rf_hybrid.fit(X_train_hybrid.loc[valid_idx], y_train.loc[valid_idx])
                daily_results["RF_Hybrid"] = rf_hybrid.predict(X_test_hybrid)[0]
                
                # Model 4: Hybrid XGBoost (GARCH + XGB)
                xgb_hybrid = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=-1)
                xgb_hybrid.fit(X_train_hybrid.loc[valid_idx], y_train.loc[valid_idx])
                daily_results["XGB_Hybrid"] = xgb_hybrid.predict(X_test_hybrid)[0]
                
            all_daily_forecasts.append(daily_results)
            
    # 4. CONSOLIDATE AND EXPORT
    results_df = pd.DataFrame(all_daily_forecasts)
    
    output_filepath = OUTPUT_DIR / "daily_oos_predictions.csv"
    results_df.to_csv(output_filepath, index=False)
    
    print(f"\n{'='*50}")
    print(f"Forecast Engine successfully completed!")
    print(f"Raw daily predictions saved to: {output_filepath}")

if __name__ == "__main__":
    run_forecast_engine()