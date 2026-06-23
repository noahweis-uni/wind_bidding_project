"""
ARIMA model for wind power point forecasting.

This module trains a (S)ARIMA model on the production target series and
produces rolling/step-ahead day-ahead point forecasts. It serves as a
classical statistical baseline alongside the persistence baseline.

Note:
    ARIMA here is fitted purely on the univariate target series (energy
    production), not on exogenous features. This keeps it comparable to
    Persistence as a "no-features" baseline, and highlights how much
    value the feature-based ML models add.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _looks_like_order(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (tuple, list)):
        return len(value) == 3
    if isinstance(value, np.ndarray):
        return value.ndim == 1 and value.size == 3
    return False


def _normalize_order(order) -> tuple[int, int, int]:
    if order is None:
        return (2, 0, 2)
    if isinstance(order, np.ndarray):
        order = order.tolist()
    if isinstance(order, (tuple, list)):
        if len(order) != 3:
            raise ValueError("order must be a tuple/list of length 3")
        return tuple(int(v) for v in order)
    raise ValueError("order must be a tuple/list of length 3")


def _normalize_seasonal_order(seasonal_order) -> tuple[int, int, int, int]:
    if seasonal_order is None:
        return (0, 0, 0, 0)
    if isinstance(seasonal_order, np.ndarray):
        seasonal_order = seasonal_order.tolist()
    if isinstance(seasonal_order, (tuple, list)):
        if len(seasonal_order) != 4:
            raise ValueError("seasonal_order must be a tuple/list of length 4")
        return tuple(int(v) for v in seasonal_order)
    raise ValueError("seasonal_order must be a tuple/list of length 4")


def train(*args, order=(2, 0, 2), seasonal_order=None):
    """
    Fit an ARIMA (or SARIMA) model on the training target series.

    The function supports both calling styles:
    - train(y_train, order=(p, d, q))
    - train(X_train, y_train)

    The first style is used by the rest of the project; the second one
    matches the notebook usage in this repository.
    """
    if len(args) == 1:
        y_train = args[0]
    elif len(args) == 2:
        first, second = args
        if _looks_like_order(second):
            y_train = first
            order = second
        else:
            y_train = second
    else:
        raise TypeError("train expects either one or two positional arguments")

    y_train = np.asarray(y_train, dtype=float).ravel()
    order = _normalize_order(order)
    seasonal_order = _normalize_seasonal_order(seasonal_order)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = ARIMA(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit()

    return fitted


def predict_day_ahead(
    fitted_model,
    y_history: np.ndarray | pd.Series,
    n_test: int,
    horizon: int = 24,
    clip_negative: bool = True,
):
    """
    Generate rolling day-ahead forecasts over the test period.

    For each day-ahead step, the model forecasts `horizon` hours into
    the future using the model fitted on training data, refit with the
    extended history via `apply` / `append` (no re-estimation of
    parameters, only state update). This mimics a realistic day-ahead
    setting where parameters are estimated once and the forecast origin
    rolls forward.

    Parameters
    ----------
    fitted_model:
        ARIMA model fitted on the training series (output of `train`).
    y_history:
        Full target series (train + test), in chronological order.
        Used to update the model state as we roll through the test set.
    n_test:
        Number of test observations (length of test set).
    horizon:
        Forecast horizon in steps. For hourly data with day-ahead
        forecasts produced once per day, horizon=24.
    clip_negative:
        If True, negative predictions are clipped to zero.

    Returns
    -------
    np.ndarray
        Array of length n_test with one-step-ahead-style forecasts,
        each value taken from the appropriate position within its
        24h-ahead forecast block.
    """
    y_history = np.asarray(y_history, dtype=float)
    n_train = len(y_history) - n_test

    preds = np.zeros(n_test)

    # Roll forward in blocks of `horizon`
    current_model = fitted_model

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for block_start in range(0, n_test, horizon):
            block_end = min(block_start + horizon, n_test)
            block_len = block_end - block_start

            # Forecast `horizon` steps ahead from current state
            forecast = current_model.forecast(steps=block_len)
            preds[block_start:block_end] = forecast

            # Update model state with the true observed values
            # (no re-estimation, just state update)
            new_obs = y_history[n_train + block_start: n_train + block_end]
            current_model = current_model.append(new_obs, refit=False)

    if clip_negative:
        preds = np.clip(preds, 0, None)

    return preds


def predict(
    fitted_model,
    steps_or_frame=None,
    steps: int = None,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Simple multi-step-ahead forecast from the end of the fitted series.

    Supports multiple calling styles:
    - predict(model, 24)
    - predict(model, X_test)
    - predict(model, steps=24)

    Parameters
    ----------
    fitted_model:
        Fitted ARIMA model.
    steps_or_frame:
        Either an integer number of steps to forecast or a DataFrame/Series/
        array-like object whose length defines the forecast horizon.
    steps:
        Alternative keyword argument for number of forecast steps.
    clip_negative:
        If True, negative predictions are clipped to zero.

    Returns
    -------
    np.ndarray
        Forecast values.
    """
    if steps is not None:
        forecast_steps = int(steps)
    elif steps_or_frame is not None:
        if isinstance(steps_or_frame, (int, np.integer)):
            forecast_steps = int(steps_or_frame)
        else:
            forecast_steps = len(steps_or_frame)
    else:
        raise ValueError("Either steps_or_frame or steps must be provided")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = fitted_model.forecast(steps=forecast_steps)

    y_pred = np.asarray(y_pred)

    if clip_negative:
        y_pred = np.clip(y_pred, 0, None)

    return y_pred


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Evaluate predictions using standard forecast metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def get_model_info(fitted_model) -> dict[str, object]:
    """
    Return basic information about the fitted ARIMA model.
    """
    return {
        "order": fitted_model.model.order,
        "seasonal_order": fitted_model.model.seasonal_order,
        "aic": float(fitted_model.aic),
        "bic": float(fitted_model.bic),
        "params": fitted_model.params.to_dict() if hasattr(fitted_model.params, "to_dict") else None,
    }