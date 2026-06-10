"""
Elastic Net model for wind power forecasting.

Elastic Net is a regularized linear regression model that combines
L1 regularization (Lasso) and L2 regularization (Ridge). It is useful
as an interpretable point forecast model with feature selection and
coefficient shrinkage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    alpha: float = 0.1,
    l1_ratio: float = 0.5,
    fit_intercept: bool = True,
    max_iter: int = 10_000,
    random_state: int = 42,
) -> Pipeline:
    """
    Train an Elastic Net regression model.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    alpha:
        Overall regularization strength. Higher values increase regularization.
    l1_ratio:
        Mixing parameter between L1 and L2 regularization.
        l1_ratio=1.0 corresponds to Lasso.
        l1_ratio=0.0 corresponds to Ridge-like behavior.
    fit_intercept:
        Whether to estimate the intercept.
    max_iter:
        Maximum number of optimization iterations.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline with StandardScaler and ElasticNet.
    """
    if not 0 <= l1_ratio <= 1:
        raise ValueError(f"l1_ratio must be between 0 and 1, got {l1_ratio}.")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "elastic_net",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    fit_intercept=fit_intercept,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    return model


def predict(
    model: Pipeline,
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Generate predictions with a trained Elastic Net model.

    Parameters
    ----------
    model:
        Trained sklearn pipeline.
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
    model: Pipeline,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return Elastic Net coefficients for interpretability.

    Parameters
    ----------
    model:
        Trained sklearn pipeline.
    feature_names:
        Names of the input features. If None, generic names are used.

    Returns
    -------
    pd.DataFrame
        Coefficients sorted by absolute importance.
    """
    elastic_net = model.named_steps["elastic_net"]
    coefficients = elastic_net.coef_

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(coefficients))]

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    })

    return coef_df.sort_values("abs_coefficient", ascending=False)


def get_model_info(model: Pipeline) -> dict[str, object]:
    """
    Return basic information about the trained Elastic Net model.

    Parameters
    ----------
    model:
        Trained sklearn pipeline.

    Returns
    -------
    dict[str, object]
        Basic model configuration.
    """
    elastic_net = model.named_steps["elastic_net"]

    return {
        "alpha": elastic_net.alpha,
        "l1_ratio": elastic_net.l1_ratio,
        "fit_intercept": elastic_net.fit_intercept,
        "max_iter": elastic_net.max_iter,
        "n_iter": int(elastic_net.n_iter_),
    }