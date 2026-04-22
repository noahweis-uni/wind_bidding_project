# quantile_regression.py
# -------------------------------------------------------
# Zweck: Quantile Regression – das Hauptmodell für Bidding.
#        Gibt direkt q_alpha zurück → optimal für Newsvendor.
#
# Kernidee: y* = q_alpha, wobei alpha aus der Newsvendor-Logik kommt:
#           alpha = c_under / (c_under + c_over)
#
# TODO:
#   - FEATURES anpassen
#   - QUANTILES anpassen je nach Bedarf
#   - alpha (= tau im Paper) aus Kostenstruktur ableiten
# -------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor


FEATURES  = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET    = "power"

# Zu schätzende Quantile
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


def build_model(quantile: float, alpha: float = 0.0) -> QuantileRegressor:
    """
    quantile: das Quantil das vorhergesagt werden soll (z.B. 0.5 = Median)
    alpha:    L1-Regularisierung (0 = keine Regularisierung)
    """
    return QuantileRegressor(quantile=quantile, alpha=alpha, solver="highs")


def train_all_quantiles(X_train: np.ndarray,
                        y_train: np.ndarray,
                        quantiles: list = QUANTILES) -> dict:
    """
    Trainiert ein Modell pro Quantil.
    Gibt dict zurück: {0.1: model, 0.5: model, ...}
    """
    models = {}
    for q in quantiles:
        m = build_model(q)
        m.fit(X_train, y_train)
        models[q] = m
        print(f"  Quantile {q:.2f} trainiert")
    return models


def predict_quantile(models: dict,
                     X_test: np.ndarray,
                     quantile: float) -> np.ndarray:
    """
    Vorhersage für ein einzelnes Quantil.
    """
    return models[quantile].predict(X_test)


def predict_all_quantiles(models: dict, X_test: np.ndarray) -> pd.DataFrame:
    """
    Vorhersage für alle Quantile → DataFrame mit einer Spalte pro Quantil.
    """
    preds = {}
    for q, m in models.items():
        preds[f"q{int(q*100):02d}"] = m.predict(X_test)
    return pd.DataFrame(preds)


def optimal_bid(models: dict, X_test: np.ndarray,
                c_under: float = 1.0, c_over: float = 1.0) -> np.ndarray:
    """
    Berechnet das optimale Gebot direkt aus der Newsvendor-Logik.
    tau = c_under / (c_under + c_over)
    y* = q_tau
    TODO: c_under und c_over aus echten Preisdaten ableiten.
    """
    tau = c_under / (c_under + c_over)
    # Nächstes verfügbares Quantil finden
    available = sorted(models.keys())
    closest   = min(available, key=lambda q: abs(q - tau))
    print(f"  Newsvendor tau={tau:.3f} → nächstes Quantil: {closest}")
    return predict_quantile(models, X_test, closest)
