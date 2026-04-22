# neural_net.py
# -------------------------------------------------------
# Zweck: Kleines MLP (Multi-Layer Perceptron) als Black-Box Modell.
#        Dient dem Vergleich: Black Box vs. Interpretierbar.
#
# TODO:
#   - FEATURES anpassen
#   - Architektur (hidden_sizes) nach Bedarf anpassen
#   - epochs und batch_size sind gute Defaults für kleine Datensätze
#   - Normalisierung: X muss vor Training skaliert werden! (StandardScaler)
# -------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


FEATURES = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET   = "power"


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list = [64, 32]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train(X_train: np.ndarray, y_train: np.ndarray,
          epochs: int = 100, batch_size: int = 64,
          lr: float = 1e-3) -> MLP:
    """
    Trainiert das MLP mit MSE-Loss.
    TODO: X_train sollte vorher mit StandardScaler normiert werden.
    """
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model     = MLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        # TODO: Logging einbauen wenn gewünscht
        # if (epoch+1) % 20 == 0: print(f"Epoch {epoch+1}: loss={loss.item():.4f}")

    return model


def predict(model: MLP, X_test: np.ndarray) -> np.ndarray:
    """
    TODO: X_test mit demselben Scaler wie X_train transformieren!
    """
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X_test)).numpy()
