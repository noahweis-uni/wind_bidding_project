"""
Quantile Neural Network for probabilistic wind power forecasting.

This module implements a simple feed-forward neural network that predicts
multiple quantiles at once. It is trained with the pinball loss and can be used
as a probabilistic black-box forecasting model.

The module contains forecasting logic only.
Bidding and Newsvendor logic should remain in src/bidding/.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class QuantileMLP(nn.Module):
    """
    Simple feed-forward neural network for multi-quantile regression.
    """

    def __init__(
        self,
        n_features: int,
        n_quantiles: int,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        input_dim = n_features

        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, n_quantiles))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class QuantileNeuralNetworkModel:
    """
    Container for the trained neural network, scaler and quantile configuration.
    """

    model: QuantileMLP
    scaler: StandardScaler
    quantiles: tuple[float, ...]
    device: str
    feature_names: list[str] | None = None
    train_losses: list[float] | None = None
    val_losses: list[float] | None = None


def _validate_quantiles(quantiles: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    quantiles = tuple(float(q) for q in quantiles)

    for q in quantiles:
        if not 0 < q < 1:
            raise ValueError(f"Quantile must be between 0 and 1, got {q}.")

    if list(quantiles) != sorted(quantiles):
        raise ValueError("Quantiles must be sorted in ascending order.")

    return quantiles


def pinball_loss_torch(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantiles: tuple[float, ...],
) -> torch.Tensor:
    """
    Compute multi-quantile pinball loss.

    Parameters
    ----------
    y_true:
        Tensor with shape (n_samples,).
    y_pred:
        Tensor with shape (n_samples, n_quantiles).
    quantiles:
        Quantile levels.

    Returns
    -------
    torch.Tensor
        Mean pinball loss over all samples and quantiles.
    """
    y_true = y_true.view(-1, 1)
    q = torch.tensor(
        quantiles,
        dtype=y_pred.dtype,
        device=y_pred.device,
    ).view(1, -1)

    error = y_true - y_pred
    loss = torch.maximum(q * error, (q - 1) * error)

    return loss.mean()


def train(
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    quantiles: tuple[float, ...] | list[float] = (0.1, 0.25, 0.5, 0.75, 0.9),
    hidden_layer_sizes: tuple[int, ...] = (64, 32),
    dropout: float = 0.0,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    batch_size: int = 64,
    max_epochs: int = 500,
    validation_fraction: float = 0.1,
    patience: int = 30,
    random_state: int = 42,
    device: str | None = None,
) -> QuantileNeuralNetworkModel:
    """
    Train a Quantile Neural Network.

    Parameters
    ----------
    X_train:
        Training features.
    y_train:
        Training target values.
    quantiles:
        Quantiles to predict, e.g. (0.1, 0.25, 0.5, 0.75, 0.9).
    hidden_layer_sizes:
        Hidden layer sizes of the MLP.
    dropout:
        Dropout rate.
    learning_rate:
        Learning rate for Adam optimizer.
    weight_decay:
        L2 regularization in Adam optimizer.
    batch_size:
        Mini-batch size.
    max_epochs:
        Maximum number of epochs.
    validation_fraction:
        Fraction of training data used for validation.
    patience:
        Early stopping patience.
    random_state:
        Random seed.
    device:
        "cpu" or "cuda". If None, automatically chooses cuda if available.

    Returns
    -------
    QuantileNeuralNetworkModel
        Trained model container.
    """
    quantiles = _validate_quantiles(quantiles)

    if isinstance(X_train, pd.DataFrame):
        feature_names = list(X_train.columns)
    else:
        feature_names = None

    X_arr = np.asarray(X_train, dtype=np.float32)
    y_arr = np.asarray(y_train, dtype=np.float32)

    if X_arr.ndim != 2:
        raise ValueError("X_train must be a 2D array or DataFrame.")

    if y_arr.ndim != 1:
        y_arr = y_arr.reshape(-1)

    if len(X_arr) != len(y_arr):
        raise ValueError("X_train and y_train must have the same length.")

    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    n_samples = len(X_arr)
    val_size = int(n_samples * validation_fraction)

    if val_size <= 0:
        raise ValueError("validation_fraction too small; validation set would be empty.")

    train_size = n_samples - val_size

    # Time-series-safe validation split: last part of training data is validation.
    X_fit = X_arr[:train_size]
    y_fit = y_arr[:train_size]
    X_val = X_arr[train_size:]
    y_val = y_arr[train_size:]

    scaler = StandardScaler()
    X_fit_scaled = scaler.fit_transform(X_fit).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)

    train_dataset = TensorDataset(
        torch.tensor(X_fit_scaled, dtype=torch.float32),
        torch.tensor(y_fit, dtype=torch.float32),
    )

    val_x = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    val_y = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = QuantileMLP(
        n_features=X_arr.shape[1],
        n_quantiles=len(quantiles),
        hidden_layer_sizes=hidden_layer_sizes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_loss = np.inf
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            batch_pred = model(batch_x)
            loss = pinball_loss_torch(batch_y, batch_pred, quantiles)

            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.detach().cpu().item()))

        mean_train_loss = float(np.mean(batch_losses))
        train_losses.append(mean_train_loss)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = pinball_loss_torch(val_y, val_pred, quantiles)
            val_loss_value = float(val_loss.detach().cpu().item())

        val_losses.append(val_loss_value)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)

    return QuantileNeuralNetworkModel(
        model=model,
        scaler=scaler,
        quantiles=quantiles,
        device=device,
        feature_names=feature_names,
        train_losses=train_losses,
        val_losses=val_losses,
    )


def predict_all(
    fitted_model: QuantileNeuralNetworkModel,
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
    enforce_monotonicity: bool = True,
) -> pd.DataFrame:
    """
    Predict all trained quantiles.

    Parameters
    ----------
    fitted_model:
        Trained QuantileNeuralNetworkModel.
    X_test:
        Test features.
    clip_negative:
        If True, negative predictions are clipped to zero.
    enforce_monotonicity:
        If True, predicted quantiles are sorted row-wise to avoid quantile crossing.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns like q10, q25, q50, q75, q90.
    """
    X_arr = np.asarray(X_test, dtype=np.float32)
    X_scaled = fitted_model.scaler.transform(X_arr).astype(np.float32)

    fitted_model.model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(fitted_model.device)
        preds = fitted_model.model(x_tensor).detach().cpu().numpy()

    if enforce_monotonicity:
        preds = np.sort(preds, axis=1)

    if clip_negative:
        preds = np.clip(preds, 0, None)

    columns = [f"q{int(q * 100):02d}" for q in fitted_model.quantiles]

    return pd.DataFrame(preds, columns=columns)


def predict_quantile(
    fitted_model: QuantileNeuralNetworkModel,
    X_test: np.ndarray | pd.DataFrame,
    quantile: float,
    clip_negative: bool = True,
    enforce_monotonicity: bool = True,
) -> np.ndarray:
    """
    Predict one specific quantile.
    """
    quantile = float(quantile)

    if quantile not in fitted_model.quantiles:
        raise ValueError(
            f"Quantile {quantile} not available. "
            f"Available quantiles: {fitted_model.quantiles}"
        )

    preds = predict_all(
        fitted_model=fitted_model,
        X_test=X_test,
        clip_negative=clip_negative,
        enforce_monotonicity=enforce_monotonicity,
    )

    col = f"q{int(quantile * 100):02d}"
    return preds[col].to_numpy()


def predict_median(
    fitted_model: QuantileNeuralNetworkModel,
    X_test: np.ndarray | pd.DataFrame,
    clip_negative: bool = True,
    enforce_monotonicity: bool = True,
) -> np.ndarray:
    """
    Predict the median forecast, i.e. q=0.5.
    """
    return predict_quantile(
        fitted_model=fitted_model,
        X_test=X_test,
        quantile=0.5,
        clip_negative=clip_negative,
        enforce_monotonicity=enforce_monotonicity,
    )


def pinball_loss(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    quantile: float,
) -> float:
    """
    Compute pinball loss for one quantile forecast.
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

    This is only descriptive. The main comparison with probabilistic forecasts
    should be based on Pinball Loss and economic bidding performance.
    """
    mae = mean_absolute_error(y_true, y_pred_median)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_median))
    r2 = r2_score(y_true, y_pred_median)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
    }


def get_training_history(
    fitted_model: QuantileNeuralNetworkModel,
) -> pd.DataFrame:
    """
    Return training and validation loss history.
    """
    return pd.DataFrame({
        "epoch": np.arange(1, len(fitted_model.train_losses) + 1),
        "train_loss": fitted_model.train_losses,
        "val_loss": fitted_model.val_losses,
    })