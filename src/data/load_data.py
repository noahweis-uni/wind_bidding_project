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
PRICES_DA_PATH = "data/raw/day_ahead_prices.xlsx"
REBAP_PATH = "data/raw/rebap.xlsx"


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
    """
    resolved_path = _resolve_path(path)
    config = _load_data_config()
    config["production_path"] = str(resolved_path)
    _save_data_config(config)
    return resolved_path


def get_production_path() -> Path:
    """
    Liefert den zentral konfigurierten Produktionspfad.
    """
    config = _load_data_config()
    configured_path = config.get("production_path", PRODUCTION_PATH)
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


def load_rebap(path: str = REBAP_PATH) -> pd.DataFrame:
    """
    Laedt reBAP (Regelenergie-Bilanzkreisabrechnung) Preise (stuendlich).
    Erwartet Spalten: timestamp, price_rebap
    """
    # TODO: Spaltennamen anpassen
    df = pd.read_excel(_resolve_path(path))
    return df
