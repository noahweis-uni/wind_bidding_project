"""
XGBoost model for wind power point forecasting.

This module provides a standard XGBRegressor for point forecasts.
It is a tree-based black-box model, often the strongest benchmark
for tabular data, and can be explained using feature importance or SHAP.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    min_child_weight: int = 5,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
) -> XGBRegressor:
    """
    Train an XGBoost model for point forecasting.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    n_estimators:
        Number of boosting rounds.
    learning_rate:
        Shrinks the contribution of each tree.
    max_depth:
        Maximum depth of individual regression trees.
    min_child_weight:
        Minimum sum of instance weight needed in a child (regularization,
        analogous to min_samples_leaf).
    subsample:
        Fraction of rows used per boosting round (regularization).
    colsample_bytree:
        Fraction of features used per tree (regularization).
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    XGBRegressor
        Fitted XGBoost model.
    """
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def predict(
    model: XGBRegressor,
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Generate predictions with a trained XGBoost model.

    Parameters
    ----------
    model:
        Trained XGBRegressor.
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
    model: XGBRegressor,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return feature importances of the trained XGBoost model.
    """
    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    })

    return importance_df.sort_values("importance", ascending=False)


def get_model_info(model: XGBRegressor) -> dict[str, object]:
    """
    Return basic information about the trained XGBoost model.
    """
    return {
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "max_depth": model.max_depth,
        "min_child_weight": model.min_child_weight,
        "subsample": model.subsample,
        "colsample_bytree": model.colsample_bytree,
        "random_state": model.random_state,
        "n_features_in": model.n_features_in_,
    }