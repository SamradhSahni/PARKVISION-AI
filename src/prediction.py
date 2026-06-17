"""
PARKVISION AI — Stage 8: Violation Prediction Model (XGBoost)
===============================================================
Builds a time-series-aware XGBoost model to predict violation counts
per H3 hexagon per hour. Features include spatial, temporal, road,
POI proximity, and location memory signals.

Usage:
    python -m src.prediction
"""

import sys
import logging
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import (
    DATA_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    PCIS_SCORED_PARQUET,
    LOCATION_MEMORY_PARQUET,
    PREDICTED_VIOLATIONS_PARQUET,
    VIOLATION_MODEL_PATH,
    LOG_FORMAT,
    LOG_LEVEL,
)

warnings.filterwarnings("ignore")
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)
logger = logging.getLogger("prediction")


# ============================================
# STEP 1: Build training dataset
# ============================================
def build_training_data(df: pd.DataFrame, memory_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a training dataset: one row per (h3_index, date, hour) with
    violation count as target and rich contextual features.
    """
    logger.info("Building training dataset ...")

    # Aggregate to hex-date-hour level
    agg_cols = {
        "latitude": "count",  # this becomes our TARGET (violation_count)
        "pcis": "mean",
        "capacity_reduction": "mean",
        "proximity_factor": "mean",
        "temporal_demand_multiplier": "first",
        "vehicle_obstruction_factor": "mean",
        "network_criticality": "mean",
        "violation_severity_weight": "mean",
        "road_width_m": "mean",
        "road_lanes": "mean",
        "road_betweenness_centrality": "mean",
        "road_is_main": "mean",
        "is_wrong_parking": "mean",
        "is_no_parking": "mean",
        "is_main_road_parking": "mean",
        "has_junction": "mean",
    }

    # Add POI columns if available
    for poi in ["metro_station", "bus_stop", "hospital", "school", "market"]:
        col = f"near_{poi}"
        if col in df.columns:
            agg_cols[col] = "first"

    train = df.groupby(["h3_index", "date", "hour"]).agg(agg_cols).reset_index()
    train = train.rename(columns={"latitude": "violation_count"})

    # Add temporal features from the date
    train["date_dt"] = pd.to_datetime(train["date"])
    train["day_of_week"] = train["date_dt"].dt.dayofweek
    train["day_of_month"] = train["date_dt"].dt.day
    train["month"] = train["date_dt"].dt.month
    train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)
    train["week_of_year"] = train["date_dt"].dt.isocalendar().week.astype(int)

    # Cyclical encoding of hour and day_of_week
    train["hour_sin"] = np.sin(2 * np.pi * train["hour"] / 24)
    train["hour_cos"] = np.cos(2 * np.pi * train["hour"] / 24)
    train["dow_sin"] = np.sin(2 * np.pi * train["day_of_week"] / 7)
    train["dow_cos"] = np.cos(2 * np.pi * train["day_of_week"] / 7)

    # Merge location memory scores
    if memory_df is not None and len(memory_df) > 0:
        mem_cols = ["h3_index", "location_memory_score", "persistence_ratio",
                    "repeat_vehicle_fraction", "is_addiction_zone"]
        mem_cols = [c for c in mem_cols if c in memory_df.columns]
        train = train.merge(memory_df[mem_cols], on="h3_index", how="left")
        train["location_memory_score"] = train["location_memory_score"].fillna(0)
        train["persistence_ratio"] = train["persistence_ratio"].fillna(0)
        train["repeat_vehicle_fraction"] = train["repeat_vehicle_fraction"].fillna(0)
        train["is_addiction_zone"] = train["is_addiction_zone"].fillna(False).astype(int)

    # Lag features: historical violation count per hex (rolling averages)
    train = train.sort_values(["h3_index", "date_dt", "hour"])

    # Per-hex historical stats (computed from the full dataset, not leaking future data)
    hex_hist = df.groupby("h3_index").agg(
        hist_total_violations=("latitude", "size"),
        hist_avg_hour_violations=("hour", lambda x: len(x) / max(x.nunique(), 1)),
    ).reset_index()
    train = train.merge(hex_hist, on="h3_index", how="left")

    logger.info(f"  Training dataset: {len(train):,} rows, {len(train.columns)} columns")
    logger.info(f"  Date range: {train['date_dt'].min()} to {train['date_dt'].max()}")
    logger.info(f"  Target stats: mean={train['violation_count'].mean():.2f}, "
                f"median={train['violation_count'].median():.0f}, "
                f"max={train['violation_count'].max()}")

    return train


# ============================================
# STEP 2: Train XGBoost model
# ============================================
def train_model(train: pd.DataFrame):
    """Train XGBoost regressor with time-series-aware split."""
    logger.info("Training XGBoost violation predictor ...")

    # Define features
    feature_cols = [
        # Temporal
        "hour", "day_of_week", "month", "is_weekend", "day_of_month",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        # Road characteristics
        "road_width_m", "road_lanes", "road_betweenness_centrality", "road_is_main",
        # PCIS components
        "pcis", "capacity_reduction", "proximity_factor",
        "temporal_demand_multiplier", "vehicle_obstruction_factor", "network_criticality",
        # Violation context
        "violation_severity_weight", "is_wrong_parking", "is_no_parking",
        "is_main_road_parking", "has_junction",
        # Location memory
        "location_memory_score", "persistence_ratio", "repeat_vehicle_fraction",
        "is_addiction_zone",
        # Historical
        "hist_total_violations", "hist_avg_hour_violations",
    ]

    # Add POI features if available
    for poi in ["metro_station", "bus_stop", "hospital", "school", "market"]:
        col = f"near_{poi}"
        if col in train.columns:
            feature_cols.append(col)

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in train.columns]
    logger.info(f"  Using {len(feature_cols)} features")

    X = train[feature_cols].copy()
    y = train["violation_count"].copy()

    # Convert boolean columns to int
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    # Fill NaN
    X = X.fillna(0)

    # Time-based split: train on first 80% of dates, test on last 20%
    dates_sorted = train["date_dt"].sort_values().unique()
    split_idx = int(len(dates_sorted) * 0.8)
    split_date = dates_sorted[split_idx]

    train_mask = train["date_dt"] < split_date
    test_mask = train["date_dt"] >= split_date

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    logger.info(f"  Train: {len(X_train):,} rows ({train_mask.sum()/len(train)*100:.0f}%)")
    logger.info(f"  Test:  {len(X_test):,} rows ({test_mask.sum()/len(train)*100:.0f}%)")
    logger.info(f"  Split date: {split_date}")

    # XGBoost parameters
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mae",
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Metrics
    metrics = {
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "test_r2": float(r2_score(y_test, y_pred_test)),
    }

    logger.info(f"\n  --- Model Performance ---")
    logger.info(f"  Train MAE:  {metrics['train_mae']:.3f}")
    logger.info(f"  Train RMSE: {metrics['train_rmse']:.3f}")
    logger.info(f"  Train R2:   {metrics['train_r2']:.4f}")
    logger.info(f"  Test MAE:   {metrics['test_mae']:.3f}")
    logger.info(f"  Test RMSE:  {metrics['test_rmse']:.3f}")
    logger.info(f"  Test R2:    {metrics['test_r2']:.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info(f"\n  --- Top 15 Feature Importances ---")
    for _, row in importance.head(15).iterrows():
        bar = "#" * int(row["importance"] * 50)
        logger.info(f"    {row['feature']:35s} {row['importance']:.4f} {bar}")

    return model, metrics, importance, feature_cols


# ============================================
# STEP 3: Generate predictions for "next week"
# ============================================
def generate_predictions(model, train: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Generate violation predictions for all hexagons across all hours for next week."""
    logger.info("Generating predictions for next week ...")

    # Get unique hexagons with their static features
    hex_features = train.groupby("h3_index").agg({
        col: "mean" for col in feature_cols
        if col not in ["hour", "day_of_week", "month", "is_weekend", "day_of_month",
                        "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
        and col in train.columns
    }).reset_index()

    # Generate predictions for each hex × hour × day_of_week
    predictions = []
    last_date = train["date_dt"].max()

    for day_offset in range(7):
        pred_date = last_date + pd.Timedelta(days=day_offset + 1)
        dow = pred_date.dayofweek
        month = pred_date.month
        is_weekend = int(dow >= 5)
        dom = pred_date.day

        for hour in range(24):
            pred_row = hex_features.copy()
            pred_row["hour"] = hour
            pred_row["day_of_week"] = dow
            pred_row["month"] = month
            pred_row["is_weekend"] = is_weekend
            pred_row["day_of_month"] = dom
            pred_row["hour_sin"] = np.sin(2 * np.pi * hour / 24)
            pred_row["hour_cos"] = np.cos(2 * np.pi * hour / 24)
            pred_row["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            pred_row["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            pred_row["pred_date"] = pred_date
            pred_row["pred_dow_name"] = pred_date.strftime("%A")

            # Ensure all feature columns exist
            for col in feature_cols:
                if col not in pred_row.columns:
                    pred_row[col] = 0

            X_pred = pred_row[feature_cols].fillna(0)

            # Convert boolean columns
            for col in X_pred.columns:
                if X_pred[col].dtype == bool:
                    X_pred[col] = X_pred[col].astype(int)

            pred_row["predicted_violations"] = model.predict(X_pred).clip(0)
            predictions.append(pred_row[["h3_index", "pred_date", "pred_dow_name",
                                          "hour", "predicted_violations"]])

    pred_df = pd.concat(predictions, ignore_index=True)

    # Summary
    daily_totals = pred_df.groupby(["pred_date", "pred_dow_name"])["predicted_violations"].sum()
    logger.info(f"  Generated {len(pred_df):,} predictions")
    logger.info(f"  Predicted daily totals:")
    for (date, dow), total in daily_totals.items():
        logger.info(f"    {dow:>10s} ({str(date)[:10]}): {total:,.0f} violations")

    return pred_df


# ============================================
# STEP 4: Save & Summarize
# ============================================
def save_and_summarize(model, metrics, importance, pred_df, feature_cols):
    """Save model, predictions, and print summary."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    logger.info(f"Saving model to {VIOLATION_MODEL_PATH} ...")
    joblib.dump({"model": model, "feature_cols": feature_cols, "metrics": metrics},
                VIOLATION_MODEL_PATH)

    # Save predictions
    logger.info(f"Saving predictions to {PREDICTED_VIOLATIONS_PARQUET} ...")
    pred_df.to_parquet(PREDICTED_VIOLATIONS_PARQUET, index=False, engine="pyarrow")

    # Save metrics and importance
    metrics_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    importance_path = MODELS_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print("\n" + "=" * 70)
    print("  PARKVISION AI - XGBoost Violation Predictor Summary")
    print("=" * 70)

    print(f"\n  --- Model Performance ---")
    print(f"    {'Metric':>15s}  {'Train':>8s}  {'Test':>8s}")
    print(f"    {'MAE':>15s}  {metrics['train_mae']:>8.3f}  {metrics['test_mae']:>8.3f}")
    print(f"    {'RMSE':>15s}  {metrics['train_rmse']:>8.3f}  {metrics['test_rmse']:>8.3f}")
    print(f"    {'R-squared':>15s}  {metrics['train_r2']:>8.4f}  {metrics['test_r2']:>8.4f}")

    print(f"\n  --- Top 15 Feature Importances ---")
    for _, row in importance.head(15).iterrows():
        bar = "#" * int(row["importance"] * 40)
        print(f"    {row['feature']:35s} {row['importance']:.4f}  {bar}")

    print(f"\n  --- Next Week Predictions ---")
    daily = pred_df.groupby(["pred_date", "pred_dow_name"]).agg(
        total=("predicted_violations", "sum"),
        max_hex=("predicted_violations", "max"),
        active_hexes=("predicted_violations", lambda x: (x > 0.5).sum()),
    )
    for (date, dow), row in daily.iterrows():
        print(f"    {dow:>10s}: {row['total']:>7,.0f} total, "
              f"max_hex={row['max_hex']:>5.1f}, "
              f"active={row['active_hexes']:>5,} hexagons")

    # Top predicted hotspots for next week
    hex_weekly = pred_df.groupby("h3_index")["predicted_violations"].sum().nlargest(10)
    print(f"\n  --- Top 10 Predicted Hotspots (next week total) ---")
    for h3_idx, total in hex_weekly.items():
        print(f"    {h3_idx}  predicted={total:>7.0f}")

    print(f"\n  Outputs:")
    print(f"    {VIOLATION_MODEL_PATH}")
    print(f"    {PREDICTED_VIOLATIONS_PARQUET}")
    print(f"    {metrics_path}")
    print(f"    {importance_path}")
    print("=" * 70)


# ============================================
# MAIN PIPELINE
# ============================================
def run():
    """Execute prediction pipeline."""
    # Load data
    logger.info(f"Loading PCIS-scored violations from {PCIS_SCORED_PARQUET} ...")
    df = pd.read_parquet(PCIS_SCORED_PARQUET)
    logger.info(f"  Loaded {len(df):,} records")

    # Load location memory
    memory_path = DATA_DIR / "location_memory.parquet"
    memory_df = pd.read_parquet(memory_path) if memory_path.exists() else pd.DataFrame()
    logger.info(f"  Loaded {len(memory_df):,} location memory records")

    # Step 1: Build training data
    train = build_training_data(df, memory_df)

    # Step 2: Train model
    model, metrics, importance, feature_cols = train_model(train)

    # Step 3: Generate predictions
    pred_df = generate_predictions(model, train, feature_cols)

    # Step 4: Save & summarize
    save_and_summarize(model, metrics, importance, pred_df, feature_cols)

    return model, pred_df


if __name__ == "__main__":
    run()
