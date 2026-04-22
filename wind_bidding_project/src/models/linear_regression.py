# linear_regression.py
# -------------------------------------------------------
# Zweck: Lineare Regression als interpretierbare Baseline.
#        Gibt Punkt-Prognose zurück (kein Quantil).
#
# TODO:
#   - FEATURES anpassen: welche Spalten als Input?
#   - Polynomial-Features aktivieren wenn gewünscht (Kommentar unten)
# -------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# Welche Spalten werden als Features genutzt?
FEATURES = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET   = "power"


def build_model(polynomial_degree: int = 1) -> Pipeline:
    """
    Baut lineare Regression, optional mit Polynomial Features.
    degree=1 → plain linear
    degree=2 → quadratische Terms (besser für Windkurve)
    """
    steps = []
    if polynomial_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=polynomial_degree, include_bias=False)))
    steps.append(("lr", LinearRegression()))
    return Pipeline(steps)


def train(X_train: np.ndarray, y_train: np.ndarray, polynomial_degree: int = 2):
    model = build_model(polynomial_degree)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)


def get_coefficients(model, feature_names: list) -> pd.Series:
    """
    Gibt Koeffizienten zurück – wichtig für Interpretierbarkeit.
    TODO: Funktioniert nur bei polynomial_degree=1.
    """
    lr = model.named_steps["lr"]
    return pd.Series(lr.coef_, index=feature_names).sort_values(ascending=False)
