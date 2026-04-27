# persistence_model.py
# -------------------------------------------------------
# Zweck: Persistence Model als Baseline-Benchmark.
#        ŷ(t+1) = y(t) – simpelste mögliche Prognose.
#        Schlägt dieses Modell nicht, ist etwas falsch.
#
# TODO: nichts – keine Parameter zu setzen.
# -------------------------------------------------------

import numpy as np


FEATURES = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET   = "power"


def build_model(last_obs: float) -> dict:
    return {"last_obs": float(last_obs)}


def train(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Speichert nur den letzten Trainingswert als Übergangspunkt."""
    return build_model(last_obs=y_train[-1])


def predict(model: dict, X_test: np.ndarray,
            y_prev: np.ndarray = None) -> np.ndarray:
    """
    y_prev: tatsächliche Werte um 1 Schritt verschoben (echter Persistence-Forecast).
            [y_train[-1], y_test[0], y_test[1], ..., y_test[-2]]
            Wird nicht übergeben: konstante Vorhersage = letzter Trainingswert.
    """
    if y_prev is not None:
        return np.asarray(y_prev, dtype=float)
    return np.full(len(X_test), model["last_obs"])
