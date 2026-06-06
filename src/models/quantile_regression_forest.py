"""
Quantile Regression Forest for probabilistic wind power forecasting.

This module approximates conditional quantiles using a RandomForestRegressor.
For each test observation, predictions from all trees are collected and
empirical quantiles are computed.

The module provides a consistent train/predict/evaluate interface for the
project notebooks.
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
    min_samples_leaf: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """
    Train a Random Forest model for quantile prediction.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    n_estimators:
        Number of trees in the forest.
    max_depth:
        Maximum tree depth. If None, nodes are expanded until leaves are pure
        or until all leaves contain fewer than min_samples_split samples.
    min_samples_leaf:
        Minimum number of samples required to be at a leaf node.
        Larger values make quantile estimates smoother.
    random_state:
        Random seed for reproducibility.
    n_jobs:
        Number of jobs to run in parallel.

    Returns
    -------
    RandomForestRegressor
        Fitted random forest model.
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


def _tree_predictions(
    model: RandomForestRegressor,
    X: np.ndarray | pd.DataFrame,
) -> np.ndarray:
    """
    Collect predictions from all trees.

    Parameters
    ----------
    model:
        Trained RandomForestRegressor.
    X:
        Input features.

    Returns
    -------
    np.ndarray
        Array with shape (n_samples, n_trees).
    """
    tree_preds = np.column_stack([
        tree.predict(X) for tree in model.estimators_
    ])

    return tree_preds


def predict_quantile(
    model: RandomForestRegressor,
    X_test: np.ndarray | pd.DataFrame,
    quantile: float,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Predict one empirical quantile from tree predictions.

    Parameters
    ----------
    model:
        Trained RandomForestRegressor.
    X_test:
        Test features.
    quantile:
        Quantile to predict, e.g. 0.25, 0.5 or 0.75.
    clip_negative:
        If True, negative predictions are clipped to zero.

    Returns
    -------
    np.ndarray
        Predicted quantile values.
    """
    if not 0 < quantile < 1:
        raise ValueError(f"Quantile must be between 0 and 1, got {quantile}.")

    tree_preds = _tree_predictions(model, X_test)
    y_pred = np.quantile(tree_preds, quantile, axis=1)

    if clip_negative:
        y_pred = np.clip(y_pred, 0, None)

    return y_pred


def predict_all(
    model: RandomForestRegressor,
    X_test: np.ndarray | pd.DataFrame,
    quantiles: list[float] | tuple[float, ...] = (0.25, 0.5, 0.75),
    clip_negative: bool = True,
) -> pd.DataFrame:
    """
    Predict multiple quantiles.

    Parameters
    ----------
    model:
        Trained RandomForestRegressor.
    X_test:
        Test features.
    quantiles:
        Quantiles to predict.
    clip_negative:
        If True, negative predictions are clipped to zero.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column per quantile, e.g. q25, q50, q75.
    """
    predictions = {}

    for q in quantiles:
        column_name = f"q{int(q * 100):02d}"
        predictions[column_name] = predict_quantile(
            model=model,
            X_test=X_test,
            quantile=q,
            clip_negative=clip_negative,
        )

    return pd.DataFrame(predictions)


def predict_median(
    model: RandomForestRegressor,
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Predict the median forecast, i.e. q=0.5.

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
        Median predictions.
    """
    return predict_quantile(
        model=model,
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


def get_feature_importance(
    model: RandomForestRegressor,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return feature importances of the underlying Random Forest.

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