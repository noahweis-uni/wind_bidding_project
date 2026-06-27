# load_nwp.py
# -------------------------------------------------------
# Lädt NWP-Wettervorhersagedaten (stündlich, 2017–2025)
# für alle 4 Windkraft-Standorte.
#
# METHODISCHER HINWEIS:
#   NWP-Daten (Numerical Weather Prediction) sind echte
#   Wettervorhersagen für die Zielstunde — im Gegensatz zu
#   ERA5-Reanalyse, die nur gemessene Vergangenheit abbildet.
#   NWP-Features werden als lag_0 (kein Shift) verwendet, da
#   sie zum Gate-Close-Zeitpunkt (12:00 Uhr Tag D-1) als
#   Vorhersage für Tag D bereits verfügbar wären.
#
# Verfügbare Spalten:
#   nwp_ws100          – Windgeschwindigkeit 100m [m/s]  ← Hauptfeature
#   nwp_wd100          – Windrichtung 100m [°]
#   nwp_ws10           – Windgeschwindigkeit 10m [m/s]
#   nwp_wd10           – Windrichtung 10m [°]
#   nwp_temp2m         – Temperatur 2m [°C]
#   nwp_surface_pressure – Luftdruck [hPa]
#   nwp_cloud_cover    – Bewölkungsgrad [%]
#
# Dateinamenskonvention: nwp_{standort}.csv
# Ablageort:            data/raw/nwp/
# -------------------------------------------------------

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NWP_DIR      = PROJECT_ROOT / "data" / "raw" / "nwp"

# Mapping Standortname → Dateiname
NWP_FILES = {
    "Schonungen":  "nwp_schonungen.csv",
    "Schwanfeld":  "nwp_schwanfeld.csv",
    "Trabelsdorf": "nwp_trabelsdorf.csv",
    "Obbach":      "nwp_obbach.csv",
}

# NWP-Spalten die direkt als lag_0 Features genutzt werden
NWP_COLS = [
    "nwp_ws100",
    "nwp_wd100",
    "nwp_ws10",
    "nwp_temp2m",
    "nwp_surface_pressure",
    "nwp_cloud_cover",
]

# Feature-Sets fuer die zwei Szenarien
# Szenario A: Nur historische SCADA-Lags (kein NWP)
FEATURES_HISTORICAL = [
    "energy_lag_24",
    "energy_lag_48",
    "wind_speed_lag_24",
    "energy_roll_mean_72",
    "wind_roll_mean_72",
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
]

# Szenario B: NWP + SCADA-Lags
FEATURES_NWP = [
    # NWP Zielstunden-Vorhersage (lag_0, methodisch korrekt da Vorhersage)
    "nwp_ws100",
    "nwp_wd100",
    "nwp_temp2m",
    "nwp_surface_pressure",
    "nwp_cloud_cover",
    # SCADA-Produktionshistorie (Day-Ahead-konform, lag_24)
    "energy_lag_24",
    "energy_lag_48",
    "energy_roll_mean_72",
    # Zeitfeatures
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
]


def load_nwp(
    site: str = "Schonungen",
    nwp_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lädt NWP-Vorhersagedaten für einen Standort.

    Parameters
    ----------
    site : str
        Standortname. Muss in NWP_FILES vorhanden sein.
    nwp_dir : Path | None
        Ordner mit NWP-CSV-Dateien. None → NWP_DIR aus Modulkonstante.

    Returns
    -------
    pd.DataFrame
        Spalten: timestamp + NWP_COLS
    """
    if site not in NWP_FILES:
        raise ValueError(
            f"Unbekannter Standort '{site}'. "
            f"Verfügbar: {list(NWP_FILES.keys())}"
        )

    base = Path(nwp_dir) if nwp_dir else NWP_DIR

    if not base.exists():
        raise FileNotFoundError(
            f"NWP-Ordner nicht gefunden: {base}\n"
            "Bitte NWP_DIR in load_nwp.py anpassen."
        )

    path = base / NWP_FILES[site]
    if not path.exists():
        raise FileNotFoundError(f"NWP-Datei nicht gefunden: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])

    # Sicherstellen dass alle erwarteten Spalten vorhanden sind
    missing_cols = [c for c in NWP_COLS if c not in df.columns]
    if missing_cols:
        warnings.warn(f"Fehlende NWP-Spalten: {missing_cols}")

    df = df[["timestamp"] + [c for c in NWP_COLS if c in df.columns]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    if verbose:
        print(f"NWP geladen ({site}): {len(df):,} Stunden | "
              f"{df['timestamp'].min().date()} bis {df['timestamp'].max().date()}")

    return df


def merge_nwp_into_dataset(
    df_scada: pd.DataFrame,
    df_nwp: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merged NWP-Features (lag_0) in den SCADA-Datensatz über timestamp.

    NWP-Werte werden OHNE Shift gejoined — der Vorhersagewert für
    Stunde t wird direkt als Feature für Stunde t verwendet. Das ist
    methodisch korrekt, weil NWP-Daten echte Vorhersagen sind die
    zum Gate-Close-Zeitpunkt bereits verfügbar wären.

    Parameters
    ----------
    df_scada : pd.DataFrame
        SCADA-Datensatz mit 'timestamp'-Spalte.
    df_nwp : pd.DataFrame
        NWP-Datensatz aus load_nwp().

    Returns
    -------
    pd.DataFrame
        df_scada mit zusätzlichen NWP-Spalten.
    """
    df = df_scada.merge(
        df_nwp[["timestamp"] + NWP_COLS],
        on="timestamp",
        how="left",
    )

    missing = df["nwp_ws100"].isna().sum()
    if missing > 0:
        warnings.warn(
            f"{missing} Stunden ohne NWP-Daten "
            f"({missing / len(df) * 100:.1f}%)."
        )

    return df


def load_and_merge_nwp(
    df_scada: pd.DataFrame,
    site: str = "Schonungen",
    nwp_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Shortcut: NWP laden und direkt in df_scada einmergen.

    Beispiel
    --------
    df = load_and_merge_nwp(df, site="Schonungen")
    FEATURES = FEATURES_NWP
    """
    df_nwp = load_nwp(site=site, nwp_dir=nwp_dir, verbose=verbose)
    return merge_nwp_into_dataset(df_scada, df_nwp)