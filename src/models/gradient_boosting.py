"""
Gradient Boosting model for wind power point forecasting.

This module provides a standard GradientBoostingRegressor for point forecasts.
It is a tree-based black-box model and can later be explained using feature
importance or SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    """
    Train a Gradient Boosting model for point forecasting.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
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
    GradientBoostingRegressor
        Fitted Gradient Boosting model.
    """
    model = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    return model


def predict(
    model: GradientBoostingRegressor,
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Generate predictions with a trained Gradient Boosting model.

    Parameters
    ----------
    model:
        Trained GradientBoostingRegressor.
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
    model: GradientBoostingRegressor,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return feature importances of the trained Gradient Boosting model.
    """
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    return importance_df.sort_values("importance", ascending=False)


def get_model_info(model: GradientBoostingRegressor) -> dict[str, object]:
    """
    Return basic information about the trained Gradient Boosting model.
    """
    return {
        "loss": model.loss,
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "min_samples_leaf": model.min_samples_leaf,
        "random_state": model.random_state,
        "n_features_in": model.n_features_in_,
    }