# load_data.py
# -------------------------------------------------------
# Lädt alle Windkraft-Rohdaten aus 4 Standorten:
#   - Schonungen, Schwanfeld, Trabelsdorf: deutsches Format
#   - Obbach: englisches Format (Header ab Zeile 10)
#
# Ordnerstruktur:
#   data/raw/Daten zur Windkennlinie_2025-09-17/
#     Obbach/       2021/ 2022/ 2023/ 2024/ 2025/  ← Jahres-Unterordner
#     Schonungen/   *.xlsx  (direkt)
#     Schwanfeld/   *.xlsx  (direkt)
#     Trabelsdorf/  *.xlsx  (direkt)
# -------------------------------------------------------

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WIND_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "Daten zur Windkennlinie_2025-09-17"
REBAP_DIR     = PROJECT_ROOT / "data" / "raw" / "daten zu rebap preisen"


def get_production_path() -> Path:
    """Gibt den Ordner mit den Wind-Produktionsdaten zurück."""
    return WIND_DATA_DIR


# ---------------------------------------------------------------------------
# Format-Erkennung und Einlesen einzelner Dateien
# ---------------------------------------------------------------------------

def _is_obbach_format(path: Path) -> bool:
    """Prüft ob die Datei das Obbach-Format hat (englischer Header ab Zeile 10)."""
    try:
        df = pd.read_excel(path, header=None, nrows=12)
        # Obbach hat "Time Stamp" in Zeile 10
        row10 = [str(v) for v in df.iloc[10].values if pd.notna(v)]
        return any("Time Stamp" in v for v in row10)
    except Exception:
        return False


def _load_standard_format(path: Path, site: str) -> pd.DataFrame | None:
    """
    Liest Schonungen / Schwanfeld / Trabelsdorf Format.
    Spalten: Datum, Leistung (Ø) [kW], Windgeschwindigkeit (Ø) [m/s]
    """
    try:
        df = pd.read_excel(path)
        df = df.rename(columns={
            "Datum":                        "timestamp",
            "Leistung (Ø) [kW]":            "power",
            "Windgeschwindigkeit (Ø) [m/s]": "wind_speed",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")
        df = df[["timestamp", "power", "wind_speed"]].dropna(subset=["timestamp"])
        df["site"] = site
        return df
    except Exception as e:
        warnings.warn(f"Fehler (Standard-Format) {path.name}: {e}")
        return None


def _load_obbach_format(path: Path) -> pd.DataFrame | None:
    """
    Liest Obbach-Format.
    Header in Zeile 10 (0-basiert), relevante Spalten:
    Time Stamp, Power(kW), Wind speed(m/s)
    """
    try:
        df = pd.read_excel(path, header=10)
        df = df.rename(columns={
            "Time Stamp":      "timestamp",
            "Power(kW)":       "power",
            "Wind speed(m/s)": "wind_speed",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df[["timestamp", "power", "wind_speed"]].dropna(subset=["timestamp"])
        df["site"] = "Obbach"
        return df
    except Exception as e:
        warnings.warn(f"Fehler (Obbach-Format) {path.name}: {e}")
        return None


def _load_single_file(path: Path, site: str) -> pd.DataFrame | None:
    """Erkennt Format automatisch und liest die Datei ein."""
    if site == "Obbach" or _is_obbach_format(path):
        return _load_obbach_format(path)
    return _load_standard_format(path, site)


# ---------------------------------------------------------------------------
# Alle Dateien laden
# ---------------------------------------------------------------------------

def load_production(
    sites: list[str] | None = None,
    years: list[int] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Lädt alle Produktionsdaten aus allen Standorten und Jahren.

    Parameters
    ----------
    sites : list[str] | None
        Nur bestimmte Standorte, z.B. ["Schonungen", "Schwanfeld"].
        None = alle 4 Standorte.
    years : list[int] | None
        Nur bestimmte Jahre, z.B. [2019, 2020, 2021].
        None = alle verfügbaren Jahre.

    Returns
    -------
    pd.DataFrame
        Kombinierter DataFrame mit Spalten:
        timestamp, power, wind_speed, site
    """
    if not WIND_DATA_DIR.exists():
        raise FileNotFoundError(
            f"Wind-Datenordner nicht gefunden:\n  {WIND_DATA_DIR}\n"
            "Bitte WIND_DATA_DIR in load_data.py anpassen."
        )

    # Alle Standort-Unterordner
    site_dirs = [d for d in WIND_DATA_DIR.iterdir() if d.is_dir()]

    # Nach Standort filtern
    if sites is not None:
        sites_lower = [s.lower() for s in sites]
        site_dirs = [d for d in site_dirs if d.name.lower() in sites_lower]

    if not site_dirs:
        raise ValueError(f"Keine Standort-Ordner gefunden. sites={sites}")

    all_dfs = []

    for site_dir in sorted(site_dirs):
        site_name = site_dir.name
        files = sorted(site_dir.rglob("*.xlsx"))

        # Auto-filter Obbach to 2023 if not specified
        site_years = years
        if site_name.lower() == "obbach" and years is None:
            site_years = [2023]
            if verbose:
                print(f"[INFO] {site_name}: auto-filtering to 2023 (use years param to override)")

        # Nach Jahr filtern (Dateiname oder Ordnername muss Jahr enthalten)
        if site_years is not None:
            years_str = [str(y) for y in site_years]
            files = [
                f for f in files
                if any(y in f.name or y in f.parent.name for y in years_str)
            ]

        if not files:
            if verbose:
                print(f"  {site_name}: keine Dateien (nach Filter)")
            continue

        if verbose:
            print(f"\n{site_name}: {len(files)} Datei(en)")

        # Pro Standort sammeln (für Obbach-Aggregation)
        site_dfs = []
        for f in files:
            if verbose:
                print(f"  Lade: {f.name}")
            df = _load_single_file(f, site_name)
            if df is not None and len(df) > 0:
                site_dfs.append(df)
            elif verbose:
                print(f"    [SKIP] uebersprungen (leer oder Fehler)")

        # Obbach: aggregiere alle Turbinen (WEA 1-5) pro timestamp
        if site_name.lower() == "obbach" and site_dfs:
            # Aggregiere in chunks pro Datei bevor concat um RAM zu sparen
            agg_dfs = []
            for df in site_dfs:
                agg = df.groupby("timestamp")[["power", "wind_speed", "site"]].agg(
                    {"power": "mean", "wind_speed": "mean", "site": "first"}
                ).reset_index()
                agg_dfs.append(agg)
            site_df_agg = pd.concat(agg_dfs, ignore_index=True)
            # Finale aggregation
            site_df_agg = (
                site_df_agg
                .groupby("timestamp")[["power", "wind_speed", "site"]]
                .agg({"power": "mean", "wind_speed": "mean", "site": "first"})
                .reset_index()
            )
            all_dfs.append(site_df_agg)
            if verbose:
                print(f"  -> Aggregiert zu {len(site_df_agg)} Stunden")
        else:
            # Andere Standorte: nur concatenieren
            all_dfs.extend(site_dfs)

    if not all_dfs:
        raise ValueError("Keine Daten geladen.")

    combined = (
        pd.concat(all_dfs, ignore_index=True)
        .sort_values(["site", "timestamp"])
        .reset_index(drop=True)
    )

    if verbose:
        print(f"\n{'='*50}")
        print(f"Gesamt: {len(combined):,} Zeilen")
        print(f"Standorte: {combined['site'].unique().tolist()}")
        print(f"Zeitraum:  {combined['timestamp'].min()} bis {combined['timestamp'].max()}")

    return combined


# ---------------------------------------------------------------------------
# reBAP-Preise
# ---------------------------------------------------------------------------

def load_rebap(rebap_dir: Path | None = None) -> pd.DataFrame:
    """
    Lädt alle reBAP-CSV-Dateien und gibt einen stündlich aggregierten
    DataFrame zurück.
    """
    base = Path(rebap_dir) if rebap_dir else REBAP_DIR

    if not base.exists():
        raise FileNotFoundError(f"reBAP-Ordner nicht gefunden: {base}")

    files = sorted(base.glob("reBAP *.csv"))
    if not files:
        raise FileNotFoundError(f"Keine 'reBAP *.csv' Dateien in: {base}")

    print(f"Lade {len(files)} reBAP-Dateien...")

    dfs = []
    for f in files:
        df = pd.read_csv(
            f, sep=";", decimal=",", encoding="utf-8-sig",
            usecols=["Datum", "von", "reBAP unterdeckt", "reBAP ueberdeckt"],
        )
        df["timestamp"] = pd.to_datetime(
            df["Datum"] + " " + df["von"],
            format="%d.%m.%Y %H:%M", errors="coerce",
        )
        df = df.dropna(subset=["timestamp"]).rename(columns={
            "reBAP unterdeckt": "rebap_under",
            "reBAP ueberdeckt": "rebap_over",
        })
        df["rebap"] = (df["rebap_under"] + df["rebap_over"]) / 2
        dfs.append(df[["timestamp", "rebap", "rebap_under", "rebap_over"]])

    rebap_raw = pd.concat(dfs, ignore_index=True).sort_values("timestamp")

    rebap_hourly = (
        rebap_raw.set_index("timestamp")
        .resample("h")[["rebap", "rebap_under", "rebap_over"]]
        .mean()
        .reset_index()
    )

    print(f"reBAP: {rebap_hourly.shape[0]:,} Stunden | "
          f"{rebap_hourly['timestamp'].min().date()} bis {rebap_hourly['timestamp'].max().date()}")

    return rebap_hourly