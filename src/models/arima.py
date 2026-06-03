# arima.py

import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA as _ARIMA


FEATURES = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET   = "power"

_ORDER = (2, 1, 2)


def _as_1d_float(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr.reshape(1) if arr.ndim == 0 else arr.ravel()


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


def update(model, y_new: np.ndarray):
    y_new = _as_1d_float(y_new)
    if y_new.size == 0:
        return model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model.extend(y_new)


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


def predict(model, steps: int, y_observed: np.ndarray = None) -> np.ndarray:
    """
    Gibt einen Punkt-Forecast für die nächsten `steps` Schritte zurück.
    Mit y_observed wird der Zustand schnell ohne Refit fortgeschrieben.
    """
    steps = int(steps)
    if steps <= 0:
        return np.empty(0, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if y_observed is None:
            return _as_1d_float(model.forecast(steps=steps))

        y_observed = _as_1d_float(y_observed)
        n_update = min(steps, y_observed.size)
        if n_update == 0:
            return _as_1d_float(model.forecast(steps=steps))

        updated = model.extend(y_observed[:n_update])
        y_pred = _as_1d_float(updated.fittedvalues)[:n_update]

        if n_update < steps:
            tail = _as_1d_float(updated.forecast(steps=steps - n_update))
            y_pred = np.concatenate([y_pred, tail])
        return y_pred


def rolling_predict(model, y_observed: np.ndarray) -> np.ndarray:
    y_observed = _as_1d_float(y_observed)
    return predict(model, steps=y_observed.size, y_observed=y_observed)
