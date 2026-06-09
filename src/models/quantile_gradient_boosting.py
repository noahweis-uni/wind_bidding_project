"""
Quantile Gradient Boosting for probabilistic wind power forecasting.

This module trains one GradientBoostingRegressor with quantile loss for each
requested quantile. It is a tree-based probabilistic forecasting model and is
well suited for Newsvendor-based bidding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    quantiles: list[float] | tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9),
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> dict[float, GradientBoostingRegressor]:
    """
    Train one Quantile Gradient Boosting model per requested quantile.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    quantiles:
        Quantiles to train, e.g. (0.25, 0.5, 0.75).
    n_estimators:
        Number of boosting stages.
    learning_rate:
        Shrinks the contribution of each tree.
    max_depth:
        Maximum depth of individual regression trees.
    min_samples_leaf:
        Minimum number of samples required at a leaf node.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    dict[float, GradientBoostingRegressor]
        Dictionary mapping quantiles to fitted models.
    """
    models: dict[float, GradientBoostingRegressor] = {}

    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {q}.")

        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=float(q),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )

        model.fit(X_train, y_train)
        models[float(q)] = model

    return models


def predict_quantile(
    models: dict[float, GradientBoostingRegressor],
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
    models: dict[float, GradientBoostingRegressor],
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> pd.DataFrame:
    """
    Predict all available quantiles.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns like q10, q25, q50, q75, q90.
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
    models: dict[float, GradientBoostingRegressor],
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Predict the median forecast, i.e. q=0.5.
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
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)

    return float(np.mean(loss))


def evaluate_quantiles(
    y_true: np.ndarray | pd.Series,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate all quantile predictions using pinball loss.
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


def evaluate_median(
    y_true: np.ndarray | pd.Series,
    y_pred_median: np.ndarray | pd.Series,
) -> dict[str, float]:
    """
    Evaluate the median forecast using point forecast metrics.
    """
    mae = mean_absolute_error(y_true, y_pred_median)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_median))
    r2 = r2_score(y_true, y_pred_median)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def get_feature_importance(
    models: dict[float, GradientBoostingRegressor],
    quantile: float = 0.5,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return feature importances of one trained quantile model.
    """
    if quantile not in models:
        available = sorted(models.keys())
        raise ValueError(
            f"Quantile {quantile} not available. "
            f"Available quantiles: {available}"
        )

    model = models[quantile]
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    return importance_df.sort_values("importance", ascending=False)


def get_model_info(
    models: dict[float, GradientBoostingRegressor],
) -> pd.DataFrame:
    """
    Return basic information about all trained quantile models.
    """
    rows = []

    for q, model in sorted(models.items()):
        rows.append({
            "quantile": q,
            "loss": model.loss,
            "alpha": model.alpha,
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "min_samples_leaf": model.min_samples_leaf,
            "random_state": model.random_state,
        })

    return pd.DataFrame(rows)