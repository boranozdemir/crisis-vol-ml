"""
Rolling machine-learning volatility forecasts for the 2008 crisis project.

This script is intended to be placed directly under:

    src/ml_rolling_models.py

It builds adaptive one-step-ahead volatility forecasts using rolling training
windows. It complements the strict pre-crisis train/test experiment in
src/ml_models.py.

Input
-----
data/processed/ml_features_panel.csv

Output
------
outputs/models/ml_rolling/ml_rolling_predictions.csv
outputs/tables/ml_rolling/ml_rolling_performance_overall.csv
outputs/tables/ml_rolling/ml_rolling_performance_by_period.csv
outputs/tables/ml_rolling/ml_rolling_feature_importance.csv
outputs/tables/ml_rolling/ml_rolling_summary.txt
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------

def find_project_root() -> Path:
    """
    Find the project root.

    Expected structure:

    crisis-vol-ml/
    ├── data/
    ├── outputs/
    └── src/
        ├── ml_features.py
        ├── ml_models.py
        └── ml_rolling_models.py

    The function is robust to being executed from project root.
    """

    current = Path(__file__).resolve()

    for parent in [current.parent, *current.parents]:
        if (parent / "data" / "processed").exists() and (parent / "src").exists():
            return parent

    # Fallback for unusual execution environments
    cwd = Path.cwd()
    if (cwd / "data" / "processed").exists():
        return cwd

    raise FileNotFoundError(
        "Could not locate project root. Expected to find data/processed/ "
        "above the current script or in the current working directory."
    )


PROJECT_ROOT = find_project_root()

INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features_panel.csv"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "ml_rolling"
TABLE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables" / "ml_rolling"

TARGET_COL = "target_squared_return_t_plus_1"

TEST_START_DATE = "2007-07-01"
ROLLING_WINDOW = 500
MIN_TRAIN_SIZE = 250

# Refit every N observations to keep runtime manageable.
# Set to 1 for full daily refitting.
REFIT_EVERY = 5

RANDOM_STATE = 42
EPS = 1e-8


# ---------------------------------------------------------------------
# Optional model imports
# ---------------------------------------------------------------------

HAS_XGBOOST = False
HAS_LIGHTGBM = False

try:
    from xgboost import XGBRegressor  # type: ignore

    HAS_XGBOOST = True
except Exception:
    XGBRegressor = None  # type: ignore

try:
    from lightgbm import LGBMRegressor  # type: ignore

    HAS_LIGHTGBM = True
except Exception:
    LGBMRegressor = None  # type: ignore


@dataclass
class RollingConfig:
    rolling_window: int = ROLLING_WINDOW
    min_train_size: int = MIN_TRAIN_SIZE
    refit_every: int = REFIT_EVERY
    test_start_date: str = TEST_START_DATE


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def ensure_output_dirs() -> None:
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column-name variants.

    The previous scripts may create columns with names such as:
    asset, ticker, symbol, date, period, target, etc.

    This function maps them into canonical names:
    Date, Asset, Period, target_squared_return_t_plus_1.
    """

    df = df.copy()

    # Remove accidental index columns from CSV exports
    unnamed_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    rename_map = {}

    for col in df.columns:
        clean = str(col).strip()
        lower = clean.lower()

        if lower in {"date", "datetime", "time", "timestamp"}:
            rename_map[col] = "Date"

        elif lower in {
            "asset",
            "assets",
            "ticker",
            "tickers",
            "symbol",
            "symbols",
            "asset_name",
            "asset_id",
        }:
            rename_map[col] = "Asset"

        elif lower in {
            "period",
            "regime",
            "sample_period",
            "market_period",
            "crisis_period",
        }:
            rename_map[col] = "Period"

        elif lower in {
            "target_squared_return_t_plus_1",
            "target",
            "y",
            "rv_t_plus_1",
            "squared_return_t_plus_1",
            "target_rv",
            "target_volatility",
        }:
            rename_map[col] = TARGET_COL

    df = df.rename(columns=rename_map)

    return df


def load_feature_panel(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature panel not found: {path}\n"
            "Run src/ml_features.py before this script."
        )

    df = pd.read_csv(path)
    df = normalize_column_names(df)

    required_cols = ["Date", "Asset", TARGET_COL]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            "ml_features_panel.csv is missing required columns: "
            f"{missing}\n\n"
            f"Available columns are:\n{list(df.columns)}\n\n"
            "This script expects panel-format ML data with one row per "
            "Date-Asset observation. If the file is wide-format, rerun "
            "src/ml_features.py and use data/processed/ml_features_panel.csv."
        )

    if "Period" not in df.columns:
        df["Period"] = "unknown"

    df["Date"] = pd.to_datetime(df["Date"])
    df["Asset"] = df["Asset"].astype(str)

    df = df.sort_values(["Asset", "Date"]).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    metadata_cols = {
        "Date",
        "Asset",
        "Period",
        TARGET_COL,
    }

    leakage_keywords = [
        "target",
        "t_plus_1",
        "future",
        "lead",
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    feature_cols: List[str] = []

    for col in numeric_cols:
        if col in metadata_cols:
            continue

        lower = col.lower()

        if any(keyword in lower for keyword in leakage_keywords):
            continue

        if df[col].notna().sum() == 0:
            continue

        feature_cols.append(col)

    if not feature_cols:
        raise ValueError(
            "No usable numeric feature columns found in ml_features_panel.csv."
        )

    return feature_cols


def find_naive_feature(feature_cols: List[str]) -> Optional[str]:
    """
    Try to detect the 22-day rolling volatility/squared-return feature.

    If not found, the benchmark falls back to the average of the last 22
    available target observations from the training window.
    """

    candidates = []

    for col in feature_cols:
        c = col.lower()
        if "22" in c and ("squared" in c or "sq" in c or "rv" in c or "vol" in c):
            candidates.append(col)

    preferred = [
        col for col in candidates
        if "mean" in col.lower() or "avg" in col.lower() or "rolling" in col.lower()
    ]

    if preferred:
        return preferred[0]

    if candidates:
        return candidates[0]

    return None


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------

def build_model_registry() -> Dict[str, object]:
    models: Dict[str, object] = {
        "Ridge": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=120,
                        max_depth=4,
                        min_samples_leaf=10,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=120,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=10,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=150,
                        learning_rate=0.03,
                        max_depth=2,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if HAS_LIGHTGBM:
        models["LightGBM"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=150,
                        learning_rate=0.03,
                        max_depth=3,
                        min_child_samples=20,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    return models


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.maximum(np.asarray(y_true, dtype=float), EPS)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), EPS)

    ratio = y_true / y_pred
    loss = ratio - np.log(ratio) - 1.0

    return float(np.mean(loss))


def evaluate_predictions(pred_df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []

    for keys, group in pred_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        y_true = group["actual"].to_numpy(dtype=float)
        y_pred = group["prediction"].to_numpy(dtype=float)

        record = {col: value for col, value in zip(group_cols, keys)}
        record.update(
            {
                "n_obs": len(group),
                "rmse": rmse(y_true, y_pred),
                "mae": mae(y_true, y_pred),
                "qlike": qlike(y_true, y_pred),
                "actual_mean": float(np.mean(y_true)),
                "prediction_mean": float(np.mean(y_pred)),
            }
        )

        rows.append(record)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


# ---------------------------------------------------------------------
# Rolling forecast
# ---------------------------------------------------------------------

def make_naive_prediction(
    train_df: pd.DataFrame,
    row: pd.Series,
    naive_feature: Optional[str],
) -> float:
    if naive_feature is not None and pd.notna(row.get(naive_feature, np.nan)):
        pred = float(row[naive_feature])
    else:
        pred = float(train_df[TARGET_COL].tail(22).mean())

    if not np.isfinite(pred):
        pred = float(train_df[TARGET_COL].mean())

    return max(pred, EPS)


def extract_feature_importance(
    model: object,
    feature_cols: List[str],
) -> List[Tuple[str, float]]:
    if isinstance(model, Pipeline):
        final_estimator = model.named_steps.get("model")
    else:
        final_estimator = model

    values: Optional[np.ndarray] = None

    if hasattr(final_estimator, "feature_importances_"):
        values = np.asarray(final_estimator.feature_importances_, dtype=float)
    elif hasattr(final_estimator, "coef_"):
        values = np.abs(np.asarray(final_estimator.coef_, dtype=float).ravel())

    if values is None or len(values) != len(feature_cols):
        return []

    if values.sum() > 0:
        values = values / values.sum()

    pairs = list(zip(feature_cols, values))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    return [(feature, float(value)) for feature, value in pairs]


def rolling_forecast_asset(
    df_asset: pd.DataFrame,
    feature_cols: List[str],
    model_registry: Dict[str, object],
    config: RollingConfig,
    naive_feature: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    asset = str(df_asset["Asset"].iloc[0])
    df_asset = df_asset.sort_values("Date").reset_index(drop=True)

    test_start = pd.Timestamp(config.test_start_date)
    test_positions = df_asset.index[df_asset["Date"] >= test_start].tolist()

    prediction_records = []
    importance_records = []

    fitted_models: Dict[str, Optional[object]] = {
        model_name: None for model_name in model_registry
    }

    last_fit_position: Dict[str, Optional[int]] = {
        model_name: None for model_name in model_registry
    }

    for pos in test_positions:
        row = df_asset.iloc[pos]

        if pd.isna(row[TARGET_COL]):
            continue

        train_start = max(0, pos - config.rolling_window)
        train_df = df_asset.iloc[train_start:pos].dropna(subset=[TARGET_COL])

        if len(train_df) < config.min_train_size:
            continue

        actual = float(row[TARGET_COL])
        period = row.get("Period", "unknown")
        date = row["Date"]

        naive_pred = make_naive_prediction(train_df, row, naive_feature)

        prediction_records.append(
            {
                "Date": date,
                "Asset": asset,
                "Period": period,
                "scope": "asset_specific_rolling",
                "model": "Naive_Rolling22",
                "actual": actual,
                "prediction": naive_pred,
                "train_start": train_df["Date"].min(),
                "train_end": train_df["Date"].max(),
                "n_train": len(train_df),
                "refit_every": 1,
            }
        )

        X_test = row[feature_cols].to_frame().T

        for model_name, model_template in model_registry.items():
            should_refit = (
                fitted_models[model_name] is None
                or last_fit_position[model_name] is None
                or (pos - int(last_fit_position[model_name])) >= config.refit_every
            )

            if should_refit:
                train_model_df = train_df.dropna(subset=feature_cols + [TARGET_COL])

                if len(train_model_df) < config.min_train_size:
                    continue

                X_train = train_model_df[feature_cols]
                y_train = train_model_df[TARGET_COL].astype(float)

                model = clone(model_template)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train, y_train)

                fitted_models[model_name] = model
                last_fit_position[model_name] = pos

            model = fitted_models[model_name]

            if model is None:
                continue

            try:
                pred = float(model.predict(X_test)[0])
            except Exception:
                continue

            prediction_records.append(
                {
                    "Date": date,
                    "Asset": asset,
                    "Period": period,
                    "scope": "asset_specific_rolling",
                    "model": model_name,
                    "actual": actual,
                    "prediction": max(pred, EPS),
                    "train_start": train_df["Date"].min(),
                    "train_end": train_df["Date"].max(),
                    "n_train": len(train_df),
                    "refit_every": config.refit_every,
                }
            )

    # Final-window feature importance
    if prediction_records and test_positions:
        final_pos = test_positions[-1]
        train_start = max(0, final_pos - config.rolling_window)

        final_train = df_asset.iloc[train_start:final_pos].dropna(
            subset=feature_cols + [TARGET_COL]
        )

        if len(final_train) >= config.min_train_size:
            X_final = final_train[feature_cols]
            y_final = final_train[TARGET_COL].astype(float)

            for model_name, model_template in model_registry.items():
                try:
                    model = clone(model_template)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model.fit(X_final, y_final)

                    importances = extract_feature_importance(model, feature_cols)

                    for feature, importance in importances:
                        importance_records.append(
                            {
                                "Asset": asset,
                                "scope": "asset_specific_rolling",
                                "model": model_name,
                                "feature": feature,
                                "importance": importance,
                                "importance_window_end": df_asset.iloc[final_pos]["Date"],
                            }
                        )

                except Exception:
                    continue

    return pd.DataFrame(prediction_records), pd.DataFrame(importance_records)


def run_rolling_forecasts(
    df: pd.DataFrame,
    feature_cols: List[str],
    config: RollingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model_registry = build_model_registry()
    naive_feature = find_naive_feature(feature_cols)

    all_predictions = []
    all_importances = []

    for asset, df_asset in df.groupby("Asset"):
        print(f"Running rolling forecasts for {asset}...")

        pred_asset, imp_asset = rolling_forecast_asset(
            df_asset=df_asset,
            feature_cols=feature_cols,
            model_registry=model_registry,
            config=config,
            naive_feature=naive_feature,
        )

        if not pred_asset.empty:
            all_predictions.append(pred_asset)

        if not imp_asset.empty:
            all_importances.append(imp_asset)

    if not all_predictions:
        raise RuntimeError("No rolling predictions were produced.")

    pred_df = pd.concat(all_predictions, ignore_index=True)
    pred_df["Date"] = pd.to_datetime(pred_df["Date"])

    if all_importances:
        imp_df = pd.concat(all_importances, ignore_index=True)
    else:
        imp_df = pd.DataFrame(
            columns=[
                "Asset",
                "scope",
                "model",
                "feature",
                "importance",
                "importance_window_end",
            ]
        )

    return pred_df, imp_df


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def build_summary(
    pred_df: pd.DataFrame,
    overall_perf: pd.DataFrame,
    period_perf: pd.DataFrame,
    feature_cols: List[str],
    config: RollingConfig,
) -> str:
    lines = []

    lines.append("Rolling machine-learning volatility forecast summary")
    lines.append("=" * 55)
    lines.append("")
    lines.append("Purpose:")
    lines.append(
        "This file summarizes adaptive rolling-window machine-learning benchmarks "
        "for next-day squared-return volatility forecasting."
    )
    lines.append("")
    lines.append("Forecast target:")
    lines.append(f"- {TARGET_COL}")
    lines.append("")
    lines.append("Design:")
    lines.append(f"- Test starts at: {config.test_start_date}")
    lines.append(f"- Rolling train window: {config.rolling_window} observations")
    lines.append(f"- Minimum train size: {config.min_train_size} observations")
    lines.append(f"- Refit frequency: every {config.refit_every} test observations")
    lines.append("- Split is chronological; no random shuffling is used.")
    lines.append("")
    lines.append("Feature count:")
    lines.append(f"- {len(feature_cols)} numeric predictive features")
    lines.append("")
    lines.append("Successful prediction rows:")
    lines.append(f"- {len(pred_df)}")
    lines.append("")

    skipped = []

    if not HAS_XGBOOST:
        skipped.append("XGBoost not installed. Install with: pip install xgboost")
    if not HAS_LIGHTGBM:
        skipped.append("LightGBM not installed. Install with: pip install lightgbm")

    if skipped:
        lines.append("Skipped optional models:")
        for item in skipped:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("Main reading rules:")
    lines.append("- Lower RMSE, MAE, and QLIKE indicate better forecast performance.")
    lines.append("- QLIKE is the preferred volatility forecast loss.")
    lines.append("- Naive_Rolling22 is a rolling-volatility benchmark.")
    lines.append("- Rolling ML results test adaptive forecasting, not structural interpretation.")
    lines.append("")

    lines.append("Best models by QLIKE, overall:")

    best_overall = overall_perf.loc[
        overall_perf.groupby(["scope", "Asset"])["qlike"].idxmin()
    ].sort_values(["scope", "Asset"])

    for _, row in best_overall.iterrows():
        lines.append(
            f"- {row['scope']} | asset={row['Asset']}: "
            f"{row['model']} "
            f"(QLIKE={row['qlike']:.6f}, RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f})"
        )

    lines.append("")
    lines.append("Best models by QLIKE, by test period:")

    best_period = period_perf.loc[
        period_perf.groupby(["scope", "Asset", "Period"])["qlike"].idxmin()
    ].sort_values(["scope", "Asset", "Period"])

    for _, row in best_period.iterrows():
        lines.append(
            f"- {row['scope']} | asset={row['Asset']} | period={row['Period']}: "
            f"{row['model']} (QLIKE={row['qlike']:.6f})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    ensure_output_dirs()

    config = RollingConfig()

    df = load_feature_panel()
    feature_cols = get_feature_columns(df)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Loaded feature panel: {df.shape}")
    print(f"Selected {len(feature_cols)} numeric features.")
    print(f"Rolling window: {config.rolling_window}")
    print(f"Refit every {config.refit_every} observations.")

    pred_df, imp_df = run_rolling_forecasts(
        df=df,
        feature_cols=feature_cols,
        config=config,
    )

    overall_perf = evaluate_predictions(
        pred_df,
        group_cols=["scope", "Asset", "model"],
    )

    period_perf = evaluate_predictions(
        pred_df,
        group_cols=["scope", "Asset", "Period", "model"],
    )

    pred_path = MODEL_OUTPUT_DIR / "ml_rolling_predictions.csv"
    overall_path = TABLE_OUTPUT_DIR / "ml_rolling_performance_overall.csv"
    period_path = TABLE_OUTPUT_DIR / "ml_rolling_performance_by_period.csv"
    importance_path = TABLE_OUTPUT_DIR / "ml_rolling_feature_importance.csv"
    summary_path = TABLE_OUTPUT_DIR / "ml_rolling_summary.txt"

    pred_df.to_csv(pred_path, index=False)
    overall_perf.to_csv(overall_path, index=False)
    period_perf.to_csv(period_path, index=False)
    imp_df.to_csv(importance_path, index=False)

    summary = build_summary(
        pred_df=pred_df,
        overall_perf=overall_perf,
        period_perf=period_perf,
        feature_cols=feature_cols,
        config=config,
    )

    summary_path.write_text(summary, encoding="utf-8")

    print("")
    print("Rolling ML forecasting completed.")
    print(f"Predictions saved to: {pred_path}")
    print(f"Overall performance saved to: {overall_path}")
    print(f"Period performance saved to: {period_path}")
    print(f"Feature importance saved to: {importance_path}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
