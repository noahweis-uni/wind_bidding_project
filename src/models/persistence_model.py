# persistence_model.py
# -------------------------------------------------------
# Zweck: Persistence Model als Baseline-Benchmark.
#        lag=1:  ŷ(t) = y(t-1)  – letzter Zeitschritt
#        lag=24: ŷ(t) = y(t-24) – gleiche Stunde Vortag
# -------------------------------------------------------

import numpy as np


FEATURES = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET   = "power"


def build_model(last_obs: float, lag: int = 1) -> dict:
    return {"last_obs": float(last_obs), "lag": int(lag)}


def train(X_train: np.ndarray, y_train: np.ndarray, lag: int = 1) -> dict:
    """Speichert die letzten `lag` Trainingswerte als Übergangspuffer."""
    lag = int(lag)
    return {
        "tail":     np.asarray(y_train[-lag:], dtype=float),
        "lag":      lag,
        "last_obs": float(y_train[-1]),
    }


def predict(model: dict, X_test: np.ndarray,
            y_test: np.ndarray = None) -> np.ndarray:
    """
    Verschiebt y_test intern um `lag` Schritte.
    Die ersten `lag` Werte werden aus dem Trainingspuffer gefüllt.
    y_test=None: konstante Prognose = letzter bekannter Wert.
    """
    lag  = model.get("lag", 1)
    tail = model.get("tail", np.array([model["last_obs"]]))
    n    = len(X_test)

    if y_test is None:
        return np.full(n, tail[-1])

    combined = np.concatenate([tail, np.asarray(y_test, dtype=float)])
    return combined[:n]
