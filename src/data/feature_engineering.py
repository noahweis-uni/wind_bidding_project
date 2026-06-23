# feature_engineering.py
# -------------------------------------------------------
# Verarbeitet kombinierte Rohdaten (alle Standorte, alle Jahre)
# zu einem ML-fertigen stündlichen Datensatz.
#
# WICHTIG: Die Standort-Spalte ('site') bleibt jetzt erhalten,
# damit einzelne Windkraftwerke separat analysiert werden können
# (z.B. für Single-Plant Single-Day Forecasting in NB03).
#
# Aggregation erfolgt PRO STANDORT auf Stundenwerte, nicht mehr
# über alle Standorte summiert/gemittelt.
#
# Output: data/processed/final_dataset.csv
# -------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert 10-Minuten-Daten auf Stundenwerte, PRO STANDORT getrennt.

    Wichtig: Die Aggregation gruppiert nach (site, Stunde), damit
    Daten verschiedener Standorte nicht vermischt werden.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "power"])

    if "site" not in df.columns:
        raise ValueError(
            "Spalte 'site' fehlt im Rohdaten-DataFrame. "
            "Bitte load_data.load_production() verwenden, das liefert 'site' mit."
        )

    hourly = (
        df.groupby(["site", pd.Grouper(key="timestamp", freq="h")])[["power", "wind_speed"]]
        .mean()
        .reset_index()
        .sort_values(["site", "timestamp"])
        .reset_index(drop=True)
    )

    return hourly


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fügt zyklische Zeitfeatures hinzu."""
    df = df.copy()
    df["hour"]      = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"]     = df["timestamp"].dt.month

    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def build_final_dataset(df_prod: pd.DataFrame) -> pd.DataFrame:
    """
    Hauptfunktion: Rohdaten → ML-fertiger stündlicher DataFrame.

    Parameters
    ----------
    df_prod : pd.DataFrame
        Kombinierte Rohdaten aus load_data.load_production().
        Erwartet Spalten: timestamp, power, wind_speed, site.

    Returns
    -------
    pd.DataFrame
        Stündlicher DataFrame mit Spalten:
        site, timestamp, power, wind_speed, hour, dayofweek, month,
        hour_sin, hour_cos, dow_sin, dow_cos.

        Enthält weiterhin ALLE Standorte (nicht gefiltert) — die
        Auswahl eines einzelnen Standorts erfolgt in Notebook 03
        über z.B. df[df["site"] == SELECTED_PLANT].
    """
    df = aggregate_to_hourly(df_prod)
    df = add_time_features(df)
    df = df.dropna().reset_index(drop=True)

    # Spaltenreihenfolge: site und timestamp zuerst
    cols = ["site", "timestamp", "power", "wind_speed",
            "hour", "dayofweek", "month", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    df = df[cols]

    return df