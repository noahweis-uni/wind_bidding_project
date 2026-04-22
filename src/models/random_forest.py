# random_forest.py
# -------------------------------------------------------
# Zweck: Random Forest als robustes ML-Modell (tree-based).
#        Gibt Punkt-Prognose + Feature Importance zurück.
#
# TODO:
#   - FEATURES anpassen
#   - Hyperparameter (n_estimators, max_depth) sind Defaults –
#     kein aufwendiges Tuning nötig laut Protokoll
# -------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURES = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET   = "power"


def build_model(n_estimators: int = 200, max_depth: int = None, random_state: int = 42):
    """
    n_estimators=200 ist ein guter Default.
    max_depth=None → Bäume wachsen bis zu den Blättern.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )


def train(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
    model = build_model()
    model.fit(X_train, y_train)
    return model


def predict(model: RandomForestRegressor, X_test: np.ndarray) -> np.ndarray:
    return model.predict(X_test)


def feature_importance(model: RandomForestRegressor, feature_names: list) -> pd.Series:
    """
    Gibt Feature Importances zurück – wichtig für XAI-Teil.
    Kann direkt als Bar-Plot visualisiert werden.
    """
    return pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)
