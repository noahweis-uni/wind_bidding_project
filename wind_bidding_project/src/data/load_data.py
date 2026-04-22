# load_data.py
# -------------------------------------------------------
# Zweck: Excel-Rohdaten einlesen und als DataFrame zurückgeben.
#
# TODO:
#   - Pfade zu euren Excel-Dateien anpassen (production, prices, rebap)
#   - Spaltennamen prüfen – müssen exakt mit euren Excel-Headern übereinstimmen
#   - Bei mehreren Sheets: sheet_name=... Parameter setzen
#
# Aufgerufen von: notebooks/02_preprocessing.ipynb
# -------------------------------------------------------

import pandas as pd


PRODUCTION_PATH = "data/raw/production.xlsx"
PRICES_DA_PATH  = "data/raw/day_ahead_prices.xlsx"
REBAP_PATH      = "data/raw/rebap.xlsx"


def load_production(path: str = PRODUCTION_PATH) -> pd.DataFrame:
    """
    Lädt die 10-Minuten-Produktionsdaten.
    Erwartet Spalten: Datum, Leistung (Ø) [kW], Windgeschwindigkeit (Ø) [m/s], ...
    """
    # TODO: sheet_name anpassen falls nötig
    df = pd.read_excel(path)
    return df


def load_day_ahead_prices(path: str = PRICES_DA_PATH) -> pd.DataFrame:
    """
    Lädt Day-Ahead Strompreise (stündlich).
    Erwartet Spalten: timestamp, price_da
    """
    # TODO: Spaltennamen anpassen
    df = pd.read_excel(path)
    return df


def load_rebap(path: str = REBAP_PATH) -> pd.DataFrame:
    """
    Lädt reBAP (Regelenergie-Bilanzkreisabrechnung) Preise (stündlich).
    Erwartet Spalten: timestamp, price_rebap
    """
    # TODO: Spaltennamen anpassen
    df = pd.read_excel(path)
    return df
