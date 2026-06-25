# load_era5.py
# -------------------------------------------------------
# Lädt ERA5 NetCDF-Dateien (stündlich, ECMWF Reanalyse)
# und extrahiert meteorologische Features für den
# nächstgelegenen Gitterpunkt zu einem Standort.
#
# METHODISCHER HINWEIS (ERA5 als lag_0):
#   ERA5-Reanalysedaten werden hier als Proxy für eine
#   perfekte Wettervorhersage verwendet (lag_0 = Zielstunden-
#   wert direkt, kein Shift). Das entspricht dem Idealfall
#   einer fehlerfreien NWP-Vorhersage und stellt eine obere
#   Schranke der erreichbaren Prognosegenauigkeit dar.
#   In der Praxis wäre dieser Wert nicht zum Gate-Close-
#   Zeitpunkt bekannt — er simuliert aber genau das, was
#   eine gute Wettervorhersage liefern würde.
#   → In der Arbeit explizit als "Oracle-Wetter-Szenario"
#     bzw. "Upper Bound" kennzeichnen.
#
# Variablen in den Dateien:
#   u100, v100  – Windkomponenten auf 100m (Nabenhöhe)
#   u10,  v10   – Windkomponenten auf 10m
#   t2m         – Temperatur 2m [K]
#   sp          – Oberflächenluftdruck [Pa]
#   tcc         – Gesamtbedeckungsgrad [0–1]
#
# Dateinamenskonvention: era5_YYYY_MM.nc
# Ablageort:             data/raw/era5/
#
# Aufgerufen von: notebooks/02_preprocessing.ipynb
#                notebooks/03_single_day_forecasting.ipynb
# -------------------------------------------------------

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ERA5_DIR     = PROJECT_ROOT / "data" / "raw" / "era5"

# Standort-Koordinaten (für nearest-neighbour Auswahl)
SITE_COORDS = {
    "Schonungen":  {"lat": 50.04, "lon": 10.23},
    "Schwanfeld":  {"lat": 49.97, "lon": 10.17},
    "Trabelsdorf": {"lat": 49.89, "lon": 10.73},
    "Obbach":      {"lat": 50.05, "lon": 10.15},
}

# ERA5-Features die direkt (lag_0) als NWP-Proxy genutzt werden
ERA5_DIRECT_COLS = [
    "era5_wind_100m",      # Windgeschwindigkeit 100m (Hauptfeature)
    "era5_wind_dir_100m",  # Windrichtung 100m
    "era5_wind_10m",       # Windgeschwindigkeit 10m (Zusatz)
    "era5_t2m_c",          # Temperatur [°C]
    "era5_sp_hpa",         # Luftdruck [hPa]
    "era5_tcc",            # Bewölkungsgrad [0–1]
]

# Empfohlenes Feature-Set (ERA5 lag_0 + SCADA-Lags)
ERA5_FEATURES = [
    # ERA5 Zielstunden-Werte (NWP-Proxy, lag_0)
    "era5_wind_100m",
    "era5_wind_dir_100m",
    "era5_t2m_c",
    "era5_sp_hpa",
    # SCADA-basierte Produktionshistorie (Day-Ahead-konform, lag_24)
    "energy_lag_24",
    "energy_lag_48",
    "energy_roll_mean_72",
    # Zeitfeatures (zyklisch)
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
]


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def _wind_speed(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Berechnet Windgeschwindigkeit aus u/v-Komponenten."""
    return np.sqrt(u**2 + v**2)


def _wind_direction(u: xr.DataArray, v: xr.DataArray) -> xr.DataArray:
    """Berechnet meteorologische Windrichtung in Grad (0=Nord, 90=Ost)."""
    return (180 + np.degrees(np.arctan2(u, v))) % 360


# ---------------------------------------------------------------------------
# Einzelne Datei laden
# ---------------------------------------------------------------------------

def load_era5_file(path: Path, site: str = "Schonungen") -> pd.DataFrame:
    """
    Lädt eine einzelne ERA5 NetCDF-Datei und gibt einen stündlichen
    DataFrame mit meteorologischen Features zurück.

    Parameters
    ----------
    path : Path
        Pfad zur .nc-Datei.
    site : str
        Standortname für nearest-neighbour Koordinatenauswahl.

    Returns
    -------
    pd.DataFrame
        Spalten: timestamp + ERA5_DIRECT_COLS
    """
    if site not in SITE_COORDS:
        raise ValueError(
            f"Unbekannter Standort '{site}'. "
            f"Verfügbar: {list(SITE_COORDS.keys())}"
        )

    coords = SITE_COORDS[site]
    lat, lon = coords["lat"], coords["lon"]

    ds = xr.open_dataset(path)

    # Nächsten Gitterpunkt wählen
    ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")

    # Windgeschwindigkeit und -richtung auf 100m und 10m
    ws100 = _wind_speed(ds_point["u100"], ds_point["v100"])
    wd100 = _wind_direction(ds_point["u100"], ds_point["v100"])
    ws10  = _wind_speed(ds_point["u10"],  ds_point["v10"])

    df = pd.DataFrame({
        "timestamp":          pd.to_datetime(ds_point["valid_time"].values),
        "era5_wind_100m":     ws100.values.astype(float),
        "era5_wind_dir_100m": wd100.values.astype(float),
        "era5_wind_10m":      ws10.values.astype(float),
        "era5_t2m_c":         (ds_point["t2m"].values - 273.15).astype(float),
        "era5_sp_hpa":        (ds_point["sp"].values / 100).astype(float),
        "era5_tcc":           ds_point["tcc"].values.astype(float),
    })

    ds.close()
    return df


# ---------------------------------------------------------------------------
# Alle Dateien laden
# ---------------------------------------------------------------------------

def load_era5_all(
    era5_dir: Path | None = None,
    site: str = "Schonungen",
    pattern: str = "era5_*.nc",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lädt alle ERA5 NetCDF-Dateien aus einem Ordner und kombiniert sie
    zu einem kontinuierlichen stündlichen DataFrame.

    Parameters
    ----------
    era5_dir : Path | None
        Ordner mit .nc-Dateien. None → ERA5_DIR aus Modulkonstante.
    site : str
        Standort für nearest-neighbour Auswahl.
    pattern : str
        Glob-Pattern für die Dateinamen (default: 'era5_*.nc').

    Returns
    -------
    pd.DataFrame
        Sortierter, deduplizierter stündlicher DataFrame mit
        timestamp + ERA5_DIRECT_COLS.
    """
    base = Path(era5_dir) if era5_dir else ERA5_DIR

    if not base.exists():
        raise FileNotFoundError(
            f"ERA5-Ordner nicht gefunden: {base}\n"
            "Bitte ERA5_DIR in load_era5.py anpassen."
        )

    files = sorted(base.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"Keine ERA5-Dateien gefunden in: {base} (Pattern: {pattern})"
        )

    if verbose:
        print(f"Lade {len(files)} ERA5-Dateien für Standort '{site}'...")

    dfs = []
    for f in files:
        if verbose:
            print(f"  {f.name}")
        try:
            dfs.append(load_era5_file(f, site=site))
        except Exception as e:
            warnings.warn(f"Fehler beim Laden von {f.name}: {e}")

    if not dfs:
        raise ValueError("Keine ERA5-Daten geladen.")

    combined = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if verbose:
        print(
            f"\nERA5 geladen: {len(combined):,} Stunden | "
            f"{combined['timestamp'].min().date()} bis "
            f"{combined['timestamp'].max().date()}"
        )

    return combined


# ---------------------------------------------------------------------------
# In SCADA-Datensatz einmergen (lag_0 — kein Shift!)
# ---------------------------------------------------------------------------

def merge_era5_into_dataset(
    df_scada: pd.DataFrame,
    df_era5: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merged ERA5-Features (lag_0) in den SCADA-Datensatz über timestamp.

    ERA5-Werte werden OHNE Shift gejoined — der Wert der Zielstunde t
    wird direkt als Feature für Stunde t verwendet. Das simuliert eine
    perfekte NWP-Vorhersage (Oracle-Wetter-Szenario / Upper Bound).

    Parameters
    ----------
    df_scada : pd.DataFrame
        SCADA-Datensatz (final_dataset.csv), muss 'timestamp'-Spalte haben.
    df_era5 : pd.DataFrame
        ERA5-Datensatz aus load_era5_all().

    Returns
    -------
    pd.DataFrame
        df_scada mit zusätzlichen ERA5-Spalten (ERA5_DIRECT_COLS).
    """
    df = df_scada.merge(df_era5[["timestamp"] + ERA5_DIRECT_COLS],
                        on="timestamp", how="left")

    missing = df["era5_wind_100m"].isna().sum()
    total   = len(df)
    if missing > 0:
        warnings.warn(
            f"{missing} von {total} Stunden ({missing/total*100:.1f}%) "
            "ohne ERA5-Daten — fehlende .nc-Dateien für diese Monate?\n"
            "Stunden mit NaN in ERA5-Features werden beim dropna() entfernt."
        )

    return df


# ---------------------------------------------------------------------------
# Convenience: alles in einem Schritt
# ---------------------------------------------------------------------------

def load_and_merge_era5(
    df_scada: pd.DataFrame,
    era5_dir: Path | None = None,
    site: str = "Schonungen",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lädt alle ERA5-Dateien und merged sie direkt in df_scada.
    Shortcut für den häufigsten Anwendungsfall.

    Beispiel
    --------
    df = load_and_merge_era5(df_scada, site="Schonungen")
    """
    df_era5 = load_era5_all(era5_dir=era5_dir, site=site, verbose=verbose)
    return merge_era5_into_dataset(df_scada, df_era5)