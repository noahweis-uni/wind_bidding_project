"""
Neural Network model for wind power forecasting.

This module uses sklearn's MLPRegressor as a feed-forward neural network.
It provides a consistent train/predict/evaluate interface for the project notebooks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    activation: str = "relu",
    alpha: float = 0.0001,
    learning_rate_init: float = 0.001,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """
    Train a feed-forward neural network for regression.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    hidden_layer_sizes:
        Number of neurons per hidden layer.
    activation:
        Activation function, e.g. 'relu' or 'tanh'.
    alpha:
        L2 regularization strength.
    learning_rate_init:
        Initial learning rate.
    max_iter:
        Maximum number of training iterations.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    Pipeline
        Fitted sklearn pipeline with StandardScaler and MLPRegressor.
    """
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPRegressor(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
                    max_iter=max_iter,
                    random_state=random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
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
    Generate predictions with a trained neural network.

    Parameters
    ----------
    model:
        Trained sklearn pipeline.
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


def get_model_info(model: Pipeline) -> dict[str, object]:
    """
    Return basic information about the trained neural network.

    Parameters
    ----------
    model:
        Trained sklearn pipeline.

    Returns
    -------
    dict[str, object]
        Basic model information such as hidden layers and number of iterations.
    """
    mlp = model.named_steps["mlp"]

    return {
        "hidden_layer_sizes": mlp.hidden_layer_sizes,
        "activation": mlp.activation,
        "alpha": mlp.alpha,
        "learning_rate_init": mlp.learning_rate_init,
        "n_iter": mlp.n_iter_,
        "loss": float(mlp.loss_),
    }