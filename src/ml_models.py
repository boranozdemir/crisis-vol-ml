"""
ml_models.py

Machine-learning benchmarks for next-day volatility forecasting in the
2008 mortgage crisis project.

This script reads the supervised ML feature datasets produced by
src/models/ml_features.py, trains several machine-learning models, and evaluates
one-step-ahead volatility proxy forecasts.

Forecast target:
    target_squared_return_t_plus_1 = r_{t+1}^2

Main inputs:
    data/processed/ml_features_panel.csv
    data/processed/ml_features_wide.csv

Main outputs:
    outputs/tables/ml/ml_model_performance_overall.csv
    outputs/tables/ml/ml_model_performance_by_period.csv
    outputs/tables/ml/ml_feature_importance.csv
    outputs/tables/ml/ml_model_summary.txt
    outputs/models/ml/ml_predictions.csv

Run:
    python src/models/ml_models.py

Required packages:
    pip install pandas numpy scikit-learn

Optional packages:
    pip install xgboost lightgbm
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Project paths
# -----------------------------------------------------------------------------


def find_project_root() -> Path:
    """Find project root by walking upward from this file."""
    current = Path(__file__).resolve()
    for parent in [current.parent, *current.parents]:
        if (parent / "data").exists() and (parent / "src").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()

PANEL_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features_panel.csv"
WIDE_FEATURE_PATH = PROJECT_ROOT / "data" / "processed" / "ml_features_wide.csv"

TABLE_DIR = PROJECT_ROOT / "outputs" / "tables" / "ml"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "models" / "ml"


# -----------------------------------------------------------------------------
# Empirical setup
# -----------------------------------------------------------------------------

ASSETS = ["SPY", "XLF", "KBE"]
TARGET_COL = "target_squared_return_t_plus_1"
DATE_COL = "Date"
ASSET_COL = "asset"
PERIOD_COL = "period"

TRAIN_END = "2007-06-30"
TEST_START = "2007-07-01"

RANDOM_STATE = 42
EPS = 1e-8

# Columns that should never be used as predictors.
NON_FEATURE_COLUMNS = {
    DATE_COL,
    ASSET_COL,
    PERIOD_COL,
    "target_squared_return_t_plus_1",
    "target_abs_return_t_plus_1",
    "target_return_t_plus_1",
}

# Naive volatility benchmark using information available at time t.
NAIVE_FEATURE_COL = "rolling_mean_squared_return_22d"


@dataclass(frozen=True)
class ModelRunResult:
    """Container for one fitted ML model result."""

    scope: str
    asset: str
    model_name: str
    predictions: pd.DataFrame
    estimator: Optional[object]
    feature_columns: List[str]
    error: Optional[str] = None


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------


def ensure_output_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_feature_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run src/models/ml_features.py before ml_models.py."
        )

    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.sort_values([ASSET_COL, DATE_COL]).reset_index(drop=True)
    return df


def infer_feature_columns(df: pd.DataFrame) -> List[str]:
    """Infer numeric feature columns while excluding metadata and targets."""
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in NON_FEATURE_COLUMNS]

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    if not feature_cols:
        raise ValueError("No numeric feature columns were found.")

    return feature_cols


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------


def build_model_registry() -> Tuple[Dict[str, object], List[str]]:
    """Build ML model registry and report unavailable optional models."""
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
                        n_estimators=400,
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
                        n_estimators=250,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=10,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    skipped: List[str] = []

    try:
        from xgboost import XGBRegressor  # type: ignore

        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=2,
                        min_child_weight=5,
                        subsample=0.90,
                        colsample_bytree=0.90,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        skipped.append("XGBoost not installed. Install with: pip install xgboost")

    try:
        from lightgbm import LGBMRegressor  # type: ignore

        models["LightGBM"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        max_depth=3,
                        num_leaves=7,
                        min_child_samples=20,
                        objective="regression",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    except ImportError:
        skipped.append("LightGBM not installed. Install with: pip install lightgbm")

    return models, skipped


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def clip_variance_forecast(y_pred: np.ndarray | pd.Series) -> np.ndarray:
    """Variance forecasts must be strictly positive for QLIKE."""
    arr = np.asarray(y_pred, dtype=float)
    return np.clip(arr, EPS, None)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss for volatility forecasts.

    Uses the common form log(h_t) + rv_t / h_t, where h_t is the variance
    forecast and rv_t is the realized volatility proxy. This avoids undefined
    log(rv_t / h_t) terms when the realized proxy is zero.
    """
    h = clip_variance_forecast(y_pred)
    rv = np.asarray(y_true, dtype=float)
    rv = np.clip(rv, 0.0, None)
    return float(np.mean(np.log(h) + rv / h))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred_pos = clip_variance_forecast(y_pred)
    return {
        "rmse": rmse(y_true, y_pred_pos),
        "mae": mae(y_true, y_pred_pos),
        "qlike": qlike(y_true, y_pred_pos),
        "mean_prediction": float(np.mean(y_pred_pos)),
        "mean_actual": float(np.mean(y_true)),
    }


# -----------------------------------------------------------------------------
# Training and prediction
# -----------------------------------------------------------------------------


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split: train on pre-crisis, test on crisis/post-crisis."""
    train = df[df[DATE_COL] <= pd.Timestamp(TRAIN_END)].copy()
    test = df[df[DATE_COL] >= pd.Timestamp(TEST_START)].copy()
    return train, test


def fit_predict_model(
    model_name: str,
    estimator: object,
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: List[str],
    scope: str,
    asset_label: str,
) -> ModelRunResult:
    """Fit one ML model and produce test-set predictions."""
    try:
        X_train = train[feature_columns]
        y_train = train[TARGET_COL].astype(float)
        X_test = test[feature_columns]

        est = clone(estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X_train, y_train)
            raw_pred = est.predict(X_test)

        pred = clip_variance_forecast(raw_pred)
        predictions = test[[DATE_COL, ASSET_COL, PERIOD_COL, TARGET_COL]].copy()
        predictions["scope"] = scope
        predictions["trained_asset"] = asset_label
        predictions["model"] = model_name
        predictions["prediction_raw"] = raw_pred
        predictions["prediction"] = pred

        return ModelRunResult(
            scope=scope,
            asset=asset_label,
            model_name=model_name,
            predictions=predictions,
            estimator=est,
            feature_columns=feature_columns,
            error=None,
        )
    except Exception as exc:  # pragma: no cover - model failures depend on environment
        empty = pd.DataFrame()
        return ModelRunResult(
            scope=scope,
            asset=asset_label,
            model_name=model_name,
            predictions=empty,
            estimator=None,
            feature_columns=feature_columns,
            error=str(exc),
        )


def build_naive_predictions(test: pd.DataFrame, scope: str, asset_label: str) -> pd.DataFrame:
    """Build a no-training benchmark based on rolling 22-day variance."""
    if NAIVE_FEATURE_COL not in test.columns:
        raise ValueError(f"Naive feature column is missing: {NAIVE_FEATURE_COL}")

    pred = test[[DATE_COL, ASSET_COL, PERIOD_COL, TARGET_COL]].copy()
    pred["scope"] = scope
    pred["trained_asset"] = asset_label
    pred["model"] = "Naive_Rolling22"
    pred["prediction_raw"] = test[NAIVE_FEATURE_COL].astype(float).values
    pred["prediction"] = clip_variance_forecast(pred["prediction_raw"])
    return pred


def run_asset_specific_models(panel: pd.DataFrame, models: Dict[str, object]) -> List[ModelRunResult]:
    """Train separate ML models for each asset."""
    results: List[ModelRunResult] = []

    for asset in sorted(panel[ASSET_COL].unique()):
        asset_df = panel[panel[ASSET_COL] == asset].copy()
        feature_cols = infer_feature_columns(asset_df)
        train, test = split_train_test(asset_df)

        if train.empty or test.empty:
            results.append(
                ModelRunResult(
                    scope="asset_specific",
                    asset=asset,
                    model_name="ALL",
                    predictions=pd.DataFrame(),
                    estimator=None,
                    feature_columns=feature_cols,
                    error=f"Empty train/test split for {asset}",
                )
            )
            continue

        # Naive benchmark.
        naive_pred = build_naive_predictions(test, scope="asset_specific", asset_label=asset)
        results.append(
            ModelRunResult(
                scope="asset_specific",
                asset=asset,
                model_name="Naive_Rolling22",
                predictions=naive_pred,
                estimator=None,
                feature_columns=[NAIVE_FEATURE_COL],
            )
        )

        for model_name, estimator in models.items():
            results.append(
                fit_predict_model(
                    model_name=model_name,
                    estimator=estimator,
                    train=train,
                    test=test,
                    feature_columns=feature_cols,
                    scope="asset_specific",
                    asset_label=asset,
                )
            )

    return results


def run_pooled_models(wide: pd.DataFrame, models: Dict[str, object]) -> List[ModelRunResult]:
    """Train pooled ML models across assets using asset/period dummies."""
    results: List[ModelRunResult] = []
    feature_cols = infer_feature_columns(wide)
    train, test = split_train_test(wide)

    if train.empty or test.empty:
        return [
            ModelRunResult(
                scope="pooled",
                asset="ALL",
                model_name="ALL",
                predictions=pd.DataFrame(),
                estimator=None,
                feature_columns=feature_cols,
                error="Empty pooled train/test split",
            )
        ]

    # Naive benchmark.
    naive_pred = build_naive_predictions(test, scope="pooled", asset_label="ALL")
    results.append(
        ModelRunResult(
            scope="pooled",
            asset="ALL",
            model_name="Naive_Rolling22",
            predictions=naive_pred,
            estimator=None,
            feature_columns=[NAIVE_FEATURE_COL],
        )
    )

    for model_name, estimator in models.items():
        results.append(
            fit_predict_model(
                model_name=model_name,
                estimator=estimator,
                train=train,
                test=test,
                feature_columns=feature_cols,
                scope="pooled",
                asset_label="ALL",
            )
        )

    return results


# -----------------------------------------------------------------------------
# Result aggregation
# -----------------------------------------------------------------------------


def collect_predictions(results: Iterable[ModelRunResult]) -> pd.DataFrame:
    frames = [r.predictions for r in results if r.predictions is not None and not r.predictions.empty]
    if not frames:
        raise RuntimeError("No predictions were produced by ML models.")
    predictions = pd.concat(frames, axis=0, ignore_index=True)
    predictions = predictions.sort_values(["scope", "model", ASSET_COL, DATE_COL]).reset_index(drop=True)
    return predictions


def performance_table(predictions: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []

    for keys, group in predictions.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y_true = group[TARGET_COL].astype(float).values
        y_pred = group["prediction"].astype(float).values
        metrics = evaluate_predictions(y_true, y_pred)
        row = dict(zip(group_cols, keys))
        row.update(metrics)
        row["n_obs"] = int(len(group))
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [col for col in group_cols if col in out.columns]
    out = out.sort_values(sort_cols + ["qlike", "rmse"]).reset_index(drop=True)
    return out


def extract_feature_importance(results: Iterable[ModelRunResult]) -> pd.DataFrame:
    """Extract feature importances/coefs when available."""
    rows = []

    for result in results:
        if result.estimator is None or result.error is not None:
            continue

        est = result.estimator
        if hasattr(est, "named_steps") and "model" in est.named_steps:
            model = est.named_steps["model"]
        else:
            model = est

        values = None
        importance_type = None

        if hasattr(model, "feature_importances_"):
            values = np.asarray(model.feature_importances_, dtype=float)
            importance_type = "feature_importance"
        elif hasattr(model, "coef_"):
            values = np.asarray(model.coef_, dtype=float).ravel()
            importance_type = "coefficient"

        if values is None:
            continue

        # Align length defensively.
        n = min(len(values), len(result.feature_columns))
        for feature, value in zip(result.feature_columns[:n], values[:n]):
            rows.append(
                {
                    "scope": result.scope,
                    "trained_asset": result.asset,
                    "model": result.model_name,
                    "importance_type": importance_type,
                    "feature": feature,
                    "value": float(value),
                    "abs_value": float(abs(value)),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["scope", "trained_asset", "model", "importance_type", "feature", "value", "abs_value"]
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["scope", "trained_asset", "model", "abs_value"], ascending=[True, True, True, False])
    return out.reset_index(drop=True)


def summarize_runs(results: Iterable[ModelRunResult], skipped_optional_models: List[str]) -> str:
    results = list(results)
    failures = [r for r in results if r.error is not None]
    successes = [r for r in results if r.error is None]

    lines = [
        "Machine-learning volatility forecast summary",
        "============================================",
        "",
        "Purpose:",
        "This file summarizes machine-learning benchmarks for next-day squared-return volatility forecasting.",
        "",
        "Forecast target:",
        f"- {TARGET_COL}",
        "",
        "Train/test design:",
        f"- Train: observations with Date <= {TRAIN_END}",
        f"- Test: observations with Date >= {TEST_START}",
        "- This is a chronological split, not a random split.",
        "",
        "Modeling scopes:",
        "- asset_specific: separate model trained for each asset.",
        "- pooled: one model trained across all assets using asset/period dummies.",
        "",
        f"Successful model runs: {len(successes)}",
        f"Failed model runs: {len(failures)}",
        "",
    ]

    if skipped_optional_models:
        lines.append("Skipped optional models:")
        for item in skipped_optional_models:
            lines.append(f"- {item}")
        lines.append("")

    if failures:
        lines.append("Failures:")
        for item in failures:
            lines.append(f"- {item.scope} | {item.asset} | {item.model_name}: {item.error}")
        lines.append("")

    lines.extend(
        [
            "Main reading rules:",
            "- Lower RMSE, MAE, and QLIKE indicate better forecast performance.",
            "- QLIKE is the preferred volatility forecast loss because it penalizes proportional variance forecast errors.",
            "- The Naive_Rolling22 model is a simple rolling-volatility benchmark.",
            "- These ML results are forecast benchmarks; they do not replace GJR-GARCH parameter interpretation.",
        ]
    )

    return "\n".join(lines)


def add_best_model_flags(perf: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Add flags indicating the best model by each metric within a group."""
    out = perf.copy()
    ranking_group = [col for col in group_cols if col != "model"]

    for metric in ["rmse", "mae", "qlike"]:
        flag_col = f"best_by_{metric}"
        out[flag_col] = False
        idx = out.groupby(ranking_group)[metric].idxmin()
        out.loc[idx, flag_col] = True

    return out


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    ensure_output_dirs()

    panel = read_feature_data(PANEL_FEATURE_PATH)
    wide = read_feature_data(WIDE_FEATURE_PATH)

    models, skipped_optional_models = build_model_registry()

    all_results: List[ModelRunResult] = []
    all_results.extend(run_asset_specific_models(panel, models))
    all_results.extend(run_pooled_models(wide, models))

    predictions = collect_predictions(all_results)
    predictions.to_csv(MODEL_OUTPUT_DIR / "ml_predictions.csv", index=False)

    overall_perf = performance_table(
        predictions,
        group_cols=["scope", "trained_asset", ASSET_COL, "model"],
    )
    overall_perf = add_best_model_flags(overall_perf, group_cols=["scope", "trained_asset", ASSET_COL, "model"])
    overall_perf.to_csv(TABLE_DIR / "ml_model_performance_overall.csv", index=False)

    period_perf = performance_table(
        predictions,
        group_cols=["scope", "trained_asset", ASSET_COL, PERIOD_COL, "model"],
    )
    period_perf = add_best_model_flags(
        period_perf,
        group_cols=["scope", "trained_asset", ASSET_COL, PERIOD_COL, "model"],
    )
    period_perf.to_csv(TABLE_DIR / "ml_model_performance_by_period.csv", index=False)

    feature_importance = extract_feature_importance(all_results)
    feature_importance.to_csv(TABLE_DIR / "ml_feature_importance.csv", index=False)

    summary = summarize_runs(all_results, skipped_optional_models)

    # Add compact best-model highlights.
    lines = [summary, "", "Best models by QLIKE, overall:"]
    if not overall_perf.empty:
        best = overall_perf.loc[overall_perf.groupby(["scope", "trained_asset", ASSET_COL])["qlike"].idxmin()]
        for _, row in best.sort_values(["scope", "trained_asset", ASSET_COL]).iterrows():
            lines.append(
                f"- {row['scope']} | trained={row['trained_asset']} | asset={row[ASSET_COL]}: "
                f"{row['model']} (QLIKE={row['qlike']:.6f}, RMSE={row['rmse']:.6f}, MAE={row['mae']:.6f})"
            )

    lines.extend(["", "Best models by QLIKE, by test period:"])
    if not period_perf.empty:
        best_period = period_perf.loc[
            period_perf.groupby(["scope", "trained_asset", ASSET_COL, PERIOD_COL])["qlike"].idxmin()
        ]
        for _, row in best_period.sort_values(["scope", "trained_asset", ASSET_COL, PERIOD_COL]).iterrows():
            lines.append(
                f"- {row['scope']} | trained={row['trained_asset']} | asset={row[ASSET_COL]} | {row[PERIOD_COL]}: "
                f"{row['model']} (QLIKE={row['qlike']:.6f})"
            )

    (TABLE_DIR / "ml_model_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("ML volatility forecasting completed.")
    print(f"Predictions saved to: {MODEL_OUTPUT_DIR / 'ml_predictions.csv'}")
    print(f"Performance tables saved to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
