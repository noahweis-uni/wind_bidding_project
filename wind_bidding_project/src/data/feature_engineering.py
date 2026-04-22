# feature_engineering.py
# -------------------------------------------------------
# Zweck: Rohdaten bereinigen, auf Stundenwerte aggregieren,
#        Features bauen und alles zu einem Datensatz mergen.
#
# TODO:
#   - Spaltennamen (PRODUCTION_COL, WIND_COL, ...) an eure Excel anpassen
#   - Timestamp-Format prüfen (pd.to_datetime klappt meist automatisch)
#   - Fehlende Werte-Strategie entscheiden: droppen oder interpolieren?
#   - Preisdaten mergen sobald vorhanden
#
# Output: data/processed/final_dataset.csv
# Aufgerufen von: notebooks/02_preprocessing.ipynb
# -------------------------------------------------------

import pandas as pd


# --- Spaltennamen anpassen ---
TIMESTAMP_COL  = "Datum"
PRODUCTION_COL = "Leistung (Ø) [kW]"
WIND_COL       = "Windgeschwindigkeit (Ø) [m/s]"
ROTOR_COL      = "Rotordrehzahl (Ø) [U/min]"   # optional
GONDEL_COL     = "Gondelposition [°]"            # optional


def aggregate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert 10-Minuten-Daten auf Stundenwerte (Mittelwert).
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.set_index("timestamp")

    # TODO: Spalten auswählen die ihr aggregieren wollt
    cols = [PRODUCTION_COL, WIND_COL]
    df_hourly = df[cols].resample("h").mean()
    df_hourly = df_hourly.rename(columns={
        PRODUCTION_COL: "power",
        WIND_COL:       "wind_speed",
    })
    return df_hourly.reset_index()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt zyklische Zeitfeatures hinzu: hour, dayofweek, month.
    sin/cos-Encoding damit Modelle die Zyklizität verstehen.
    """
    import numpy as np
    df = df.copy()
    df["hour"]       = df["timestamp"].dt.hour
    df["dayofweek"]  = df["timestamp"].dt.dayofweek
    df["month"]      = df["timestamp"].dt.month

    # Zyklisches Encoding – besser als plain integers
    df["hour_sin"]   = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]    = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]    = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def merge_prices(df_prod: pd.DataFrame,
                 df_da: pd.DataFrame,
                 df_rebap: pd.DataFrame) -> pd.DataFrame:
    """
    Merged Produktionsdaten mit Preisdaten über timestamp.
    TODO: Sicherstellen dass alle drei DataFrames eine 'timestamp'-Spalte haben.
    """
    df = df_prod.merge(df_da,    on="timestamp", how="left")
    df = df.merge(df_rebap,      on="timestamp", how="left")
    return df


def build_final_dataset(df_prod: pd.DataFrame,
                        df_da: pd.DataFrame   = None,
                        df_rebap: pd.DataFrame = None) -> pd.DataFrame:
    """
    Hauptfunktion: alles zusammen.
    Gibt ML-fertigen DataFrame zurück.
    """
    df = aggregate_to_hourly(df_prod)
    df = add_time_features(df)

    # TODO: Auskommentieren sobald Preisdaten vorhanden
    # if df_da is not None and df_rebap is not None:
    #     df = merge_prices(df, df_da, df_rebap)

    df = df.dropna()
    return df
