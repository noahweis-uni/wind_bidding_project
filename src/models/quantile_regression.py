"""
Quantile Regression models for probabilistic wind power forecasting.

This module trains one QuantileRegressor per requested quantile and provides
a consistent train/predict/evaluate interface for the project notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    quantiles: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
    alpha: float = 0.0,
    solver: str = "highs",
) -> dict[float, Pipeline]:
    """
    Train one Quantile Regression model for each requested quantile.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    quantiles:
        Quantiles to train, e.g. (0.25, 0.5, 0.75).
    alpha:
        L1 regularization strength used by sklearn's QuantileRegressor.
        alpha=0.0 means no regularization.
    solver:
        Solver used by QuantileRegressor, usually 'highs'.

    Returns
    -------
    dict[float, Pipeline]
        Dictionary mapping each quantile to its fitted sklearn pipeline.
    """
    models: dict[float, Pipeline] = {}

    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {q}.")

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "quantile_regressor",
                    QuantileRegressor(
                        quantile=q,
                        alpha=alpha,
                        solver=solver,
                    ),
                ),
            ]
        )

        model.fit(X_train, y_train)
        models[float(q)] = model

    return models


def predict_quantile(
    models: dict[float, Pipeline],
    X_test: np.ndarray | pd.DataFrame,
    quantile: float,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Predict one specific quantile.

    Parameters
    ----------
    models:
        Dictionary of trained quantile models.
    X_test:
        Test features.
    quantile:
        Quantile to predict.
    clip_negative:
        If True, negative predictions are clipped to zero.

    Returns
    -------
    np.ndarray
        Predicted quantile values.
    """
    if quantile not in models:
        available = sorted(models.keys())
        raise ValueError(
            f"Quantile {quantile} not available. "
            f"Available quantiles: {available}"
        )

    y_pred = models[quantile].predict(X_test)

    if clip_negative:
        y_pred = np.clip(y_pred, 0, None)

    return y_pred


def predict_all(
    models: dict[float, Pipeline],
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> pd.DataFrame:
    """
    Predict all available quantiles.

    Parameters
    ----------
    models:
        Dictionary of trained quantile models.
    X_test:
        Test features.
    clip_negative:
        If True, negative predictions are clipped to zero.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per quantile, e.g. q25, q50, q75.
    """
    predictions = {}

    for q in sorted(models.keys()):
        column_name = f"q{int(q * 100):02d}"
        predictions[column_name] = predict_quantile(
            models=models,
            X_test=X_test,
            quantile=q,
            clip_negative=clip_negative,
        )

    return pd.DataFrame(predictions)


def predict_median(
    models: dict[float, Pipeline],
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Predict the median forecast, i.e. q=0.5.

    Parameters
    ----------
    models:
        Dictionary of trained quantile models.
    X_test:
        Test features.
    clip_negative:
        If True, negative predictions are clipped to zero.

    Returns
    -------
    np.ndarray
        Median predictions.
    """
    return predict_quantile(
        models=models,
        X_test=X_test,
        quantile=0.5,
        clip_negative=clip_negative,
    )


def pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    quantile: float,
) -> float:
    """
    Compute the pinball loss for one quantile.

    Parameters
    ----------
    y_true:
        True target values.
    y_pred:
        Predicted quantile values.
    quantile:
        Quantile level.

    Returns
    -------
    float
        Mean pinball loss.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)

    return float(np.mean(loss))


def evaluate_median(
    y_true: np.ndarray | pd.Series,
    y_pred_median: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Evaluate the median forecast using standard point forecast metrics.

    Parameters
    ----------
    y_true:
        True target values.
    y_pred_median:
        Predicted median values.

    Returns
    -------
    dict[str, float]
        Dictionary containing MAE, RMSE and R2.
    """
    mae = mean_absolute_error(y_true, y_pred_median)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_median))
    r2 = r2_score(y_true, y_pred_median)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def evaluate_quantiles(
    y_true: np.ndarray | pd.Series,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate all quantile predictions using pinball loss.

    Parameters
    ----------
    y_true:
        True target values.
    predictions:
        DataFrame returned by predict_all(), with columns like q25, q50, q75.

    Returns
    -------
    pd.DataFrame
        Evaluation table with pinball loss per quantile.
    """
    rows = []

    for column in predictions.columns:
        if not column.startswith("q"):
            continue

        q = int(column.replace("q", "")) / 100
        loss = pinball_loss(y_true, predictions[column].values, q)

        rows.append({
            "quantile": q,
            "pinball_loss": loss,
        })

    return pd.DataFrame(rows)


def get_coefficients(
    models: dict[float, Pipeline],
    quantile: float = 0.5,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return coefficients of one trained Quantile Regression model.

    Parameters
    ----------
    models:
        Dictionary of trained quantile models.
    quantile:
        Quantile whose model coefficients should be returned.
    feature_names:
        Names of the input features. If None, generic names are used.

    Returns
    -------
    pd.DataFrame
        Coefficients sorted by absolute importance.
    """
    if quantile not in models:
        available = sorted(models.keys())
        raise ValueError(
            f"Quantile {quantile} not available. "
            f"Available quantiles: {available}"
        )

    regressor = models[quantile].named_steps["quantile_regressor"]
    coefficients = regressor.coef_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    })

    return coef_df.sort_values("abs_coefficient", ascending=False)