# feature_engineering.py
# -------------------------------------------------------
# Verarbeitet kombinierte Rohdaten (alle Standorte, alle Jahre)
# zu einem ML-fertigen stündlichen Datensatz.
#
# Strategie bei mehreren Standorten:
#   Pro Zeitstempel wird über alle vorhandenen Standorte gemittelt.
#   Das ergibt eine "Portfolio-Sicht" auf die Gesamtproduktion.
#
# Output: data/processed/final_dataset.csv
# -------------------------------------------------------

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert 10-Minuten-Daten auf Stundenwerte.

    Bei mehreren Standorten: erst pro Standort auf Stunden aggregieren,
    dann über alle Standorte summieren (Gesamtportfolio).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "power"])

    if "site" in df.columns:
        # Pro Standort stündlich aggregieren
        hourly_per_site = (
            df.groupby(["site", pd.Grouper(key="timestamp", freq="h")])[["power", "wind_speed"]]
            .mean()
            .reset_index()
        )

        # Über alle Standorte summieren (Gesamtleistung) bzw. mitteln (Windgeschwindigkeit)
        hourly = (
            hourly_per_site.groupby("timestamp")
            .agg(
                power=("power", "sum"),          # Summe aller Standorte
                wind_speed=("wind_speed", "mean"), # Mittlere Windgeschwindigkeit
                n_sites=("site", "count"),         # Wie viele Standorte je Stunde
            )
            .reset_index()
        )

        # Nur Stunden behalten wo alle Standorte Daten liefern
        n_total_sites = df["site"].nunique()
        n_before = len(hourly)
        hourly = hourly[hourly["n_sites"] == n_total_sites].drop(columns="n_sites")
        n_dropped = n_before - len(hourly)
        if n_dropped > 0:
            warnings.warn(
                f"{n_dropped} Stunden entfernt, weil nicht alle {n_total_sites} "
                f"Standorte Daten lieferten."
            )
    else:
        # Einzelner Standort
        hourly = (
            df.set_index("timestamp")
            .resample("h")[["power", "wind_speed"]]
            .mean()
            .reset_index()
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
        Erwartet Spalten: timestamp, power, wind_speed, (site).

    Returns
    -------
    pd.DataFrame
        Stündlicher DataFrame mit allen Features.
    """
    df = aggregate_to_hourly(df_prod)
    df = add_time_features(df)
    df = df.dropna().reset_index(drop=True)
    return df