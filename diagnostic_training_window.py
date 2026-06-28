#!/usr/bin/env python3
"""
Diagnostic script: Compare training window strategies
- Option A: All data (2014-2024)
- Option B: Recent 2 years (2022-2024)

This helps decide if old data helps or hurts the forecast.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "features_model_all_sites.csv"

sys.path.append(str(PROJECT_ROOT))
from src.models import (
    elastic_net,
    random_forest,
    gradient_boosting,
    quantile_regression_forest,
)


def evaluate_forecast(y_true, y_pred) -> dict:
    """Calculate metrics"""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def clip_non_negative(values):
    """Ensure non-negative predictions"""
    return np.clip(np.asarray(values, dtype=float), 0, None)


print("\n" + "=" * 80)
print("DIAGNOSTIC: Training Window Strategy")
print("=" * 80)

# Load data
print("\n1. Loading features...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"   Total data: {len(df)} rows, {df['timestamp'].min()} to {df['timestamp'].max()}")

# Prepare features (same as NB03)
print("\n2. Engineering features...")
df_model = df.copy()
df_model["energy_mwh"] = df_model["power"].clip(lower=0) / 1000
df_model["hour"] = df_model["timestamp"].dt.hour
df_model["dayofweek"] = df_model["timestamp"].dt.dayofweek
df_model["month"] = df_model["timestamp"].dt.month

df_model["hour_sin"] = np.sin(2 * np.pi * df_model["hour"] / 24)
df_model["hour_cos"] = np.cos(2 * np.pi * df_model["hour"] / 24)
df_model["dow_sin"] = np.sin(2 * np.pi * df_model["dayofweek"] / 7)
df_model["dow_cos"] = np.cos(2 * np.pi * df_model["dayofweek"] / 7)
df_model["month_sin"] = np.sin(2 * np.pi * df_model["month"] / 12)
df_model["month_cos"] = np.cos(2 * np.pi * df_model["month"] / 12)

# Gate-close features
df_model["gate_close"] = (
    df_model["timestamp"].dt.normalize() - pd.Timedelta(hours=12)
)

full_hourly_idx = pd.date_range(
    df["timestamp"].min().normalize() - pd.Timedelta(days=8),
    df["timestamp"].max(),
    freq="h",
)

energy_hourly = (
    df_model.set_index("timestamp")["energy_mwh"]
    .groupby(level=0).mean()
    .reindex(full_hourly_idx)
)
energy_roll24 = energy_hourly.rolling(24, min_periods=12).mean()
energy_roll168 = energy_hourly.rolling(168, min_periods=72).mean()

for offset_h in [0, 12, 24, 48, 168]:
    col = "energy_at_gate" if offset_h == 0 else f"energy_gate_m{offset_h}h"
    df_model[col] = (
        df_model["gate_close"] - pd.Timedelta(hours=offset_h)
    ).map(energy_hourly).values

df_model["energy_roll24_at_gate"] = df_model["gate_close"].map(energy_roll24).values
df_model["energy_roll168_at_gate"] = df_model["gate_close"].map(energy_roll168).values

BASE_FEATURES = [
    "energy_at_gate",
    "energy_gate_m12h",
    "energy_gate_m24h",
    "energy_gate_m48h",
    "energy_gate_m168h",
    "energy_roll24_at_gate",
    "energy_roll168_at_gate",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
]

NWP_FEATURES = [f for f in [
    "nwp_ws100", "nwp_ws10", "nwp_wd100", "nwp_wd10",
    "nwp_temp2m", "nwp_surface_pressure", "nwp_total_cloud_cover",
    "nwp_wind_shear_diff", "nwp_wind_shear_ratio",
    "nwp_sin_wd100", "nwp_cos_wd100", "nwp_sin_wd10", "nwp_cos_wd10",
    "nwp_ws100_lag3h", "nwp_ws100_lag6h", "nwp_ws100_lag12h", "nwp_ws100_lag24h",
    "nwp_ws100_rollmean3h", "nwp_ws100_rollmean6h", "nwp_ws100_rollmean24h",
    "nwp_temp2m_lag3h", "nwp_temp2m_lag6h", "nwp_temp2m_lag12h",
] if f in df_model.columns]

CAMS_FEATURES = [f for f in [
    "cams_ghi", "cams_clearsky_index",
    "cams_ghi_lag3h", "cams_ghi_lag6h", "cams_ghi_lag12h", "cams_ghi_lag24h",
    "cams_clearsky_index_lag3h", "cams_clearsky_index_lag6h",
    "cams_clearsky_index_lag12h", "cams_clearsky_index_lag24h",
    "cams_ghi_rollmean3h", "cams_ghi_rollmean6h", "cams_ghi_rollmean24h",
    "cams_clearsky_index_rollmean3h", "cams_clearsky_index_rollmean6h",
    "cams_clearsky_index_rollmean24h",
] if f in df_model.columns]

FEATURES_ALL = BASE_FEATURES + NWP_FEATURES + CAMS_FEATURES
TARGET = "energy_mwh"

df_model = df_model.dropna(subset=FEATURES_ALL + [TARGET]).reset_index(drop=True)
X = df_model[FEATURES_ALL]
y = df_model[TARGET]

print(f"   Features: {len(FEATURES_ALL)} total")

# Train-Test Split (same as NB03)
TEST_SIZE = 0.2
split_idx = int(len(df_model) * (1 - TEST_SIZE))

X_test = X.iloc[split_idx:].copy()
y_test = y.iloc[split_idx:].copy()
test_start_date = df_model.loc[split_idx, "timestamp"]

print(f"\n3. Train-Test split:")
print(f"   Test set: {test_start_date} to {df_model['timestamp'].iloc[-1]}")
print(f"   Test size: {len(X_test)} samples")

# Define training windows
WINDOW_ALL = 0  # All available data
WINDOW_2Y = 365 * 2  # Last 2 years
WINDOW_1Y = 365  # Last 1 year

windows = {
    "Option A (All data)": WINDOW_ALL,
    "Option B (2 years)": WINDOW_2Y,
    "Option C (1 year)": WINDOW_1Y,
}

results = {}

for option_name, window_days in windows.items():
    print(f"\n{'=' * 80}")
    print(f"{option_name}")
    print(f"{'=' * 80}")

    if window_days == 0:
        # All data
        X_train = X.iloc[:split_idx].copy()
        y_train = y.iloc[:split_idx].copy()
        train_start = df_model.iloc[0]["timestamp"]
        train_end = df_model.iloc[split_idx - 1]["timestamp"]
    else:
        # Recent window
        cutoff_idx = split_idx - window_days
        if cutoff_idx < 0:
            print(f"   WARNING: Window {window_days} days exceeds training data!")
            cutoff_idx = 0

        X_train = X.iloc[cutoff_idx:split_idx].copy()
        y_train = y.iloc[cutoff_idx:split_idx].copy()
        train_start = df_model.iloc[cutoff_idx]["timestamp"]
        train_end = df_model.iloc[split_idx - 1]["timestamp"]

    print(f"   Training data: {train_start} to {train_end}")
    print(f"   Training samples: {len(X_train)}")

    # Train multiple models
    models_to_test = {
        "Elastic Net": None,
        "Random Forest": None,
        "Gradient Boosting": None,
        "QRF (q50)": None,
    }

    metrics_list = []

    print(f"\n   Training models...")

    # Elastic Net
    try:
        en_model = elastic_net.train(X_train, y_train, alpha=0.1, l1_ratio=0.5, random_state=42)
        y_pred = clip_non_negative(elastic_net.predict(en_model, X_test))
        metrics = evaluate_forecast(y_test, y_pred)
        metrics["model"] = "Elastic Net"
        metrics_list.append(metrics)
        print(f"     Elastic Net:      RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
    except Exception as e:
        print(f"     Elastic Net:      FAILED ({str(e)[:50]})")

    # Random Forest
    try:
        rf_model = random_forest.train(X_train, y_train, n_estimators=300, min_samples_leaf=2, random_state=42)
        y_pred = clip_non_negative(random_forest.predict(rf_model, X_test))
        metrics = evaluate_forecast(y_test, y_pred)
        metrics["model"] = "Random Forest"
        metrics_list.append(metrics)
        print(f"     Random Forest:    RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
    except Exception as e:
        print(f"     Random Forest:    FAILED ({str(e)[:50]})")

    # Gradient Boosting
    try:
        gb_model = gradient_boosting.train(X_train, y_train, n_estimators=100, learning_rate=0.03, max_depth=2, min_samples_leaf=20)
        y_pred = clip_non_negative(gradient_boosting.predict(gb_model, X_test))
        metrics = evaluate_forecast(y_test, y_pred)
        metrics["model"] = "Gradient Boosting"
        metrics_list.append(metrics)
        print(f"     Gradient Boosting: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
    except Exception as e:
        print(f"     Gradient Boosting: FAILED ({str(e)[:50]})")

    # QRF
    try:
        qrf_model = quantile_regression_forest.train(X_train, y_train, n_estimators=300, min_samples_leaf=5, random_state=42)
        qrf_preds = quantile_regression_forest.predict_all(qrf_model, X_test, quantiles=(0.5,))
        y_pred = clip_non_negative(qrf_preds["q50"].to_numpy())
        metrics = evaluate_forecast(y_test, y_pred)
        metrics["model"] = "QRF (q50)"
        metrics_list.append(metrics)
        print(f"     QRF (q50):        RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}")
    except Exception as e:
        print(f"     QRF (q50):        FAILED ({str(e)[:50]})")

    # Average metrics
    df_metrics = pd.DataFrame(metrics_list)
    avg_rmse = df_metrics["RMSE"].mean()
    avg_mae = df_metrics["MAE"].mean()
    avg_r2 = df_metrics["R2"].mean()

    print(f"\n   Average across models:")
    print(f"     RMSE: {avg_rmse:.4f}")
    print(f"     MAE:  {avg_mae:.4f}")
    print(f"     R2:   {avg_r2:.4f}")

    results[option_name] = {
        "avg_rmse": avg_rmse,
        "avg_mae": avg_mae,
        "avg_r2": avg_r2,
        "metrics": df_metrics,
    }

# Summary and recommendation
print(f"\n{'=' * 80}")
print("SUMMARY & RECOMMENDATION")
print(f"{'=' * 80}")

summary_df = pd.DataFrame({
    option: {
        "Avg RMSE": results[option]["avg_rmse"],
        "Avg MAE": results[option]["avg_mae"],
        "Avg R2": results[option]["avg_r2"],
    }
    for option in results.keys()
}).T

print(f"\n{summary_df.to_string()}\n")

# Find best option
best_option = min(results.keys(), key=lambda x: results[x]["avg_rmse"])
best_rmse = results[best_option]["avg_rmse"]
all_rmse = results["Option A (All data)"]["avg_rmse"]
improvement = ((all_rmse - best_rmse) / all_rmse) * 100

print(f"Best option: {best_option}")
print(f"  → RMSE improvement: {improvement:.2f}% vs Option A")

if best_option == "Option A (All data)":
    print(f"\nRECOMMENDATION: Keep current strategy (Option A)")
    print(f"  Reason: All data performs best. Old data is helpful!")
elif improvement > 5:
    print(f"\nRECOMMENDATION: Switch to {best_option}")
    print(f"  Reason: {improvement:.1f}% RMSE improvement is significant")
    print(f"  Action: Update NB03 to use recent window only")
else:
    print(f"\nRECOMMENDATION: Either option is acceptable")
    print(f"  Improvement is marginal ({improvement:.1f}%)")
    print(f"  Keep current (Option A) for simplicity")

print(f"\n{'=' * 80}\n")
