"""
Linear Regression model for wind power forecasting.

This module contains a simple sklearn-based Linear Regression model
with a consistent train/predict interface for the project notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    fit_intercept: bool = True,
) -> LinearRegression:
    """
    Train a Linear Regression model.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    fit_intercept:
        Whether to calculate the intercept for this model.

    Returns
    -------
    LinearRegression
        Fitted sklearn Linear Regression model.
    """
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    return model


def predict(
    model: LinearRegression,
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Generate predictions with a trained Linear Regression model.

    Parameters
    ----------
    model:
        Trained Linear Regression model.
    X_test:
        Test features.
    clip_negative:
        If True, negative predictions are clipped to zero.
        This is useful for wind power forecasting because production
        cannot be negative.

    Returns
    -------
    np.ndarray
        Predicted values.
    """
    y_pred = model.predict(X_test)

    if clip_negative:
        y_pred = np.clip(y_pred, 0, None)

    return y_pred


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Evaluate predictions using standard forecast metrics.

    Parameters
    ----------
    y_true:
        True target values.
    y_pred:
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary containing MAE, RMSE and R2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def get_coefficients(
    model: LinearRegression,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return model coefficients for interpretability.

    Parameters
    ----------
    model:
        Trained Linear Regression model.
    feature_names:
        Names of the input features. If None, generic names are used.

    Returns
    -------
    pd.DataFrame
        Coefficients sorted by absolute importance.
    """
    coefficients = model.coef_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    })

    return coef_df.sort_values("abs_coefficient", ascending=False)