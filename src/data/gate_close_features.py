"""
Gate-Close Feature Engineering for Day-Ahead Forecasting.

Extracted from 03_forecasting_models.ipynb to be reused in single-day notebooks.
All features are computed relative to gate_close time (noon of day D-1 for day D+1 bid).
"""

import pandas as pd
import numpy as np


def engineer_gate_close_features(df: pd.DataFrame, lookback_days: int = 8) -> pd.DataFrame:
    """
    Build gate-close-referenced features for day-ahead forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: timestamp (datetime), power (float), wind_speed (float),
                      hour, dayofweek, month, hour_sin, hour_cos, dow_sin, dow_cos
    lookback_days : int
        Days to look back for rolling windows (default 8)

    Returns
    -------
    pd.DataFrame
        Original df plus gate-close features
    """
    df = df.copy()

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Gate-close: all 24-hour bids for day D+1 are placed at 12:00 noon on day D
    # gate_close = midnight of target date - 12h = noon of previous day
    df["gate_close"] = (
        df["timestamp"].dt.normalize() - pd.Timedelta(hours=12)
    )

    # Energy features: create hourly data with full grid to handle gaps
    df["energy_mwh"] = df["power"].clip(lower=0) / 1000

    full_hourly_idx = pd.date_range(
        df["timestamp"].min().normalize() - pd.Timedelta(days=lookback_days),
        df["timestamp"].max(),
        freq="h",
    )

    energy_hourly = (
        df.set_index("timestamp")["energy_mwh"]
        .groupby(level=0).mean()
        .reindex(full_hourly_idx)
    )

    energy_roll24 = energy_hourly.rolling(24, min_periods=12).mean()
    energy_std24 = energy_hourly.rolling(24, min_periods=12).std()
    energy_roll168 = energy_hourly.rolling(168, min_periods=72).mean()
    energy_std168 = energy_hourly.rolling(168, min_periods=72).std()

    # Point features: production value at gate-close and hours before
    for offset_h in [0, 12, 24, 48, 168]:
        col = "energy_at_gate" if offset_h == 0 else f"energy_gate_m{offset_h}h"
        df[col] = (
            df["gate_close"] - pd.Timedelta(hours=offset_h)
        ).map(energy_hourly).values

    # Rolling features at gate-close time
    df["energy_roll24_at_gate"] = df["gate_close"].map(energy_roll24).values
    df["energy_roll_std24_at_gate"] = df["gate_close"].map(energy_std24).values
    df["energy_roll168_at_gate"] = df["gate_close"].map(energy_roll168).values
    df["energy_roll_std168_at_gate"] = df["gate_close"].map(energy_std168).values

    # Wind features: same pattern
    if "wind_speed" in df.columns:
        wind_hourly = (
            df.set_index("timestamp")["wind_speed"]
            .groupby(level=0).mean()
            .reindex(full_hourly_idx)
        )

        wind_roll24 = wind_hourly.rolling(24, min_periods=12).mean()
        wind_std24 = wind_hourly.rolling(24, min_periods=12).std()
        wind_roll168 = wind_hourly.rolling(168, min_periods=72).mean()
        wind_std168 = wind_hourly.rolling(168, min_periods=72).std()

        for offset_h in [0, 12, 24, 168]:
            col = "wind_at_gate" if offset_h == 0 else f"wind_gate_m{offset_h}h"
            df[col] = (
                df["gate_close"] - pd.Timedelta(hours=offset_h)
            ).map(wind_hourly).values

        df["wind_roll24_at_gate"] = df["gate_close"].map(wind_roll24).values
        df["wind_roll_std24_at_gate"] = df["gate_close"].map(wind_std24).values
        df["wind_roll168_at_gate"] = df["gate_close"].map(wind_roll168).values
        df["wind_roll_std168_at_gate"] = df["gate_close"].map(wind_std168).values

    return df


def get_gate_close_features() -> list[str]:
    """Return list of all gate-close features"""
    return [
        "energy_at_gate",
        "energy_gate_m12h",
        "energy_gate_m24h",
        "energy_gate_m48h",
        "energy_gate_m168h",
        "energy_roll24_at_gate",
        "energy_roll_std24_at_gate",
        "energy_roll168_at_gate",
        "energy_roll_std168_at_gate",
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "wind_at_gate",
        "wind_gate_m12h",
        "wind_gate_m24h",
        "wind_gate_m168h",
        "wind_roll24_at_gate",
        "wind_roll_std24_at_gate",
        "wind_roll168_at_gate",
        "wind_roll_std168_at_gate",
    ]
