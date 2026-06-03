# load_data.py
# -------------------------------------------------------
# Zweck: Excel-Rohdaten einlesen und als DataFrame zurueckgeben.
#
# TODO:
#   - Pfade zu euren Excel-Dateien anpassen (production, prices, rebap)
#   - Spaltennamen pruefen - muessen exakt mit euren Excel-Headern uebereinstimmen
#   - Bei mehreren Sheets: sheet_name=... Parameter setzen
#
# Aufgerufen von: notebooks/01_data_understanding.ipynb
#                notebooks/02_preprocessing.ipynb
# -------------------------------------------------------

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_CONFIG_PATH = PROJECT_ROOT / "data" / "raw" / "data_sources.json"

PRODUCTION_PATH = "data/raw/production.xlsx"
PRICES_DA_PATH  = "data/raw/day_ahead_prices.xlsx"
REBAP_PATH      = "data/raw/Daten zu reBAP Preisen/reBAP unterdeckt 2016-2025.csv"


def _resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _load_data_config() -> dict:
    if not DATA_CONFIG_PATH.exists():
        return {}
    with DATA_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _save_data_config(config: dict) -> None:
    DATA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)


def set_production_path(path: str) -> Path:
    """
    Speichert den Produktionspfad zentral fuer alle Notebooks.
    Pfad wird relativ zum Projektroot gespeichert, damit die config
    auf jedem Rechner funktioniert.
    """
    resolved_path = _resolve_path(path)
    try:
        store_path = str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        store_path = str(resolved_path)
    config = _load_data_config()
    config["production_path"] = store_path
    _save_data_config(config)
    return resolved_path


def get_production_path() -> Path:
    """
    Liefert den zentral konfigurierten Produktionspfad.
    """
    config = _load_data_config()
    configured_path = config.get("production_path", PRODUCTION_PATH)
    return _resolve_path(configured_path)


def set_rebap_path(path: str) -> Path:
    """
    Speichert den reBAP-Pfad zentral fuer alle Notebooks.
    """
    resolved_path = _resolve_path(path)
    try:
        store_path = str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        store_path = str(resolved_path)
    config = _load_data_config()
    config["rebap_path"] = store_path
    _save_data_config(config)
    return resolved_path


def get_rebap_path() -> Path:
    """
    Liefert den zentral konfigurierten reBAP-Pfad.
    """
    config = _load_data_config()
    configured_path = config.get("rebap_path", REBAP_PATH)
    return _resolve_path(configured_path)


def load_production(path: str = PRODUCTION_PATH) -> pd.DataFrame:
    """
    Laedt die 10-Minuten-Produktionsdaten.
    Erwartet Spalten: Datum, Leistung (O) [kW], Windgeschwindigkeit (O) [m/s], ...
    """
    if path == PRODUCTION_PATH:
        path = get_production_path()
    else:
        path = _resolve_path(path)

    # TODO: sheet_name anpassen falls noetig
    df = pd.read_excel(path)
    return df


def load_day_ahead_prices(path: str = PRICES_DA_PATH) -> pd.DataFrame:
    """
    Laedt Day-Ahead Strompreise (stuendlich).
    Erwartet Spalten: timestamp, price_da
    """
    # TODO: Spaltennamen anpassen
    df = pd.read_excel(_resolve_path(path))
    return df


def load_rebap(path: str = None) -> pd.DataFrame:
    """
    Laedt reBAP-Preise (15-Minuten-Aufloesung, 2016-2025).
    Gibt DataFrame mit DatetimeIndex und Spalten:
      price_rebap_under  [EUR/MWh] – reBAP Unterdeckung
      price_rebap_over   [EUR/MWh] – reBAP Ueberdeckung
    """
    resolved = _resolve_path(path) if path else get_rebap_path()
    df = pd.read_csv(
        resolved,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    df["timestamp"] = pd.to_datetime(
        df["Datum"] + " " + df["von"],
        format="%d.%m.%Y %H:%M",
    )
    df = df.rename(columns={
        "reBAP unterdeckt": "price_rebap_under",
        "reBAP ueberdeckt": "price_rebap_over",
    })
    df = df[["timestamp", "price_rebap_under", "price_rebap_over"]].set_index("timestamp")
    return df
