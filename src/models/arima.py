# arima.py
# -------------------------------------------------------
# Zweck: ARIMA als univariater Zeitreihen-Baseline.
#        Modelliert Windleistung nur auf Basis vergangener
#        Zielwerte (keine Exogenous-Features).
#
# TODO:
#   - order=(p,d,q) ggf. per auto_arima optimieren
#   - Für Produktiveinsatz rolling forecast durch
#     apply() mit update() ersetzen (deutlich schneller)
# -------------------------------------------------------

import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA as _ARIMA


FEATURES = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET   = "power"

_ORDER = (2, 1, 2)


def build_model(y_train: np.ndarray, order: tuple = _ORDER):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ARIMA(np.asarray(y_train, dtype=float), order=order).fit()


def train(y_train: np.ndarray, order: tuple = _ORDER):
    """Trainiert ARIMA auf der Zielreihe y_train."""
    return build_model(y_train, order=order)


def predict(model, steps: int) -> np.ndarray:
    """
    Gibt einen Punkt-Forecast für die nächsten `steps` Schritte zurück.
    Kein Rolling – schnell, aber akkumuliert Fehler über längere Horizonte.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.forecast(steps=steps)
