"""
Random Forest model for wind power forecasting.

This module provides a standard RandomForestRegressor for point forecasts.
It includes a consistent train/predict/evaluate interface and feature
importance extraction for the project notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """
    Train a Random Forest model for point forecasting.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum depth of the trees.
    min_samples_leaf:
        Minimum number of samples required at a leaf node.
    random_state:
        Random seed for reproducibility.
    n_jobs:
        Number of parallel jobs.

    Returns
    -------
    RandomForestRegressor
        Fitted Random Forest model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(X_train, y_train)
    return model


def predict(
    model: RandomForestRegressor,
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Generate predictions with a trained Random Forest model.

    Parameters
    ----------
    model:
        Trained RandomForestRegressor.
    X_test:
        Test features.
    clip_negative:
        If True, negative predictions are clipped to zero.

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


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return feature importances of the trained Random Forest.

    Parameters
    ----------
    model:
        Trained RandomForestRegressor.
    feature_names:
        Names of the input features. If None, generic names are used.

    Returns
    -------
    pd.DataFrame
        Feature importances sorted descending.
    """
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    return importance_df.sort_values("importance", ascending=False)


def get_model_info(model: RandomForestRegressor) -> dict[str, object]:
    """
    Return basic information about the trained Random Forest.

    Parameters
    ----------
    model:
        Trained RandomForestRegressor.

    Returns
    -------
    dict[str, object]
        Basic model configuration.
    """
    return {
        "n_estimators": model.n_estimators,
        "max_depth": model.max_depth,
        "min_samples_leaf": model.min_samples_leaf,
        "random_state": model.random_state,
        "n_features_in": model.n_features_in_,
    }