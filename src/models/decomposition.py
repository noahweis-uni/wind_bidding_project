# decomposition.py

import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


FEATURES = []
TARGET   = "power"
PERIOD   = 24  # stündliche Daten → tägliche Saisonalität


def decompose(y: np.ndarray, period: int = PERIOD, model: str = "additive"):
    """Dekomponiert die Zeitreihe in Trend, Saisonalität und Residuum."""
    return seasonal_decompose(
        np.asarray(y, dtype=float),
        model=model,
        period=period,
        extrapolate_trend="freq",
    )


def train(y_train: np.ndarray, period: int = PERIOD) -> dict:
    """
    Dekomponiert y_train und fittet einen linearen Trend für den Forecast.
    Gibt ein dict zurück, das für predict() benötigt wird.
    """
    result = decompose(y_train, period=period)

    t  = np.arange(len(y_train)).reshape(-1, 1)
    lr = LinearRegression().fit(t, result.trend)

    return {
        "decomposition": result,
        "trend_model":   lr,
        "seasonal":      result.seasonal,
        "period":        period,
        "n_train":       len(y_train),
    }


def predict(model: dict, steps: int) -> np.ndarray:
    """Forecast via extrapoliertem Trend + wiederholtem Saisonmuster."""
    steps    = int(steps)
    period   = model["period"]
    n_train  = model["n_train"]
    lr       = model["trend_model"]
    seasonal = model["seasonal"]

    t_future     = np.arange(n_train, n_train + steps).reshape(-1, 1)
    trend_fc     = lr.predict(t_future)

    idx_seasonal = np.arange(n_train, n_train + steps) % period
    seasonal_fc  = seasonal[idx_seasonal]

    return trend_fc + seasonal_fc
