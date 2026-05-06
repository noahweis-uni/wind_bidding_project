# arima.py
# -------------------------------------------------------
# Zweck: ARIMA als univariater Zeitreihen-Baseline.
#        Modelliert Windleistung nur auf Basis vergangener
#        Zielwerte (keine Exogenous-Features).
#
# TODO:
#   - Für Produktiveinsatz rolling forecast durch
#     apply() mit update() ersetzen (deutlich schneller)
# -------------------------------------------------------

import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA as _ARIMA


FEATURES = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET   = "power"

_ORDER = (2, 1, 2)


def find_order(y_train: np.ndarray,
               max_p: int = 5, max_q: int = 5,
               information_criterion: str = "aic") -> tuple:
    """
    Bestimmt (p, d, q) automatisch via pmdarima.auto_arima.
    Fällt auf _ORDER zurück falls pmdarima nicht installiert ist.
    """
    try:
        import pmdarima as pm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = pm.auto_arima(
                np.asarray(y_train, dtype=float),
                max_p=max_p, max_q=max_q,
                information_criterion=information_criterion,
                stepwise=True, seasonal=False,
                error_action="ignore", suppress_warnings=True,
            )
        return result.order
    except ImportError:
        return _ORDER


def build_model(y_train: np.ndarray, order: tuple = _ORDER):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _ARIMA(np.asarray(y_train, dtype=float), order=order).fit()


def train(y_train: np.ndarray, order: tuple = None) -> object:
    """
    Trainiert ARIMA auf der Zielreihe y_train.
    order=None → auto_arima wählt (p,d,q) automatisch (empfohlen).
    order=(p,d,q) → fixer Wert, schneller aber ggf. suboptimal.
    """
    if order is None:
        order = find_order(y_train)
        print(f"  auto_arima gewählt: order={order}")
    return build_model(y_train, order=order)


def predict(model, steps: int) -> np.ndarray:
    """
    Gibt einen Punkt-Forecast für die nächsten `steps` Schritte zurück.
    Kein Rolling – schnell, aber akkumuliert Fehler über längere Horizonte.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.forecast(steps=steps)
