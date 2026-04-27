# quantile_regression_forest.py
# -------------------------------------------------------
# Zweck: Quantile Regression Forest (QRF) – probabilistisches Modell.
#        Erweitert Random Forest: Quantile werden aus den Blattknoten
#        der Trainingsbeobachtungen geschätzt (Meinshausen, 2006).
#        Liefert schärfere Intervalle als lineares QR.
#
# Implementierung: pure scikit-learn, keine Extra-Abhängigkeit.
#
# TODO:
#   - n_estimators erhöhen für schärfere Intervalle (kostet Rechenzeit)
#   - QUANTILES anpassen je nach Bidding-Bedarf
# -------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


FEATURES  = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET    = "power"
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]


class QuantileRegressionForest:
    """
    QRF nach Meinshausen (2006): gewichtete Quantil-Schätzung aus Blattknoten.
    Interface angelehnt an sklearn-Regressoren (fit / predict).
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = None,
                 random_state: int = 42):
        self.rf_ = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.y_train_: np.ndarray = None
        self.train_leaves_: np.ndarray = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileRegressionForest":
        self.rf_.fit(X, y)
        self.y_train_     = np.asarray(y, dtype=np.float32)
        self.train_leaves_ = self.rf_.apply(X)   # (n_train, n_trees)
        return self

    def predict(self, X: np.ndarray, quantile: float = 0.5) -> np.ndarray:
        """
        Gibt das quantile-Quantil der bedingten Verteilung zurück.
        Laufzeit: O(n_test * n_trees) – einige Sekunden für typische Datensätze.
        """
        test_leaves = self.rf_.apply(X)       # (n_test, n_trees)
        n_test      = len(X)

        # Gewichtsmatrix aufbauen: Anteil Bäume, in denen Test- und Trainpunkt
        # im selben Blatt landen → (n_test, n_train)
        weights = np.zeros((n_test, len(self.y_train_)), dtype=np.float32)
        for t in range(self.rf_.n_estimators):
            weights += (test_leaves[:, t:t+1] == self.train_leaves_[:, t])

        weights /= weights.sum(axis=1, keepdims=True)

        # Gewichtetes Quantil je Testpunkt
        sorter   = np.argsort(self.y_train_)
        y_sorted = self.y_train_[sorter]
        preds    = np.empty(n_test)
        for i in range(n_test):
            cumw    = np.cumsum(weights[i, sorter])
            idx     = np.searchsorted(cumw, quantile)
            preds[i] = y_sorted[min(idx, len(y_sorted) - 1)]
        return preds


def build_model(n_estimators: int = 200, max_depth: int = None,
                random_state: int = 42) -> QuantileRegressionForest:
    return QuantileRegressionForest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )


def train(X_train: np.ndarray, y_train: np.ndarray) -> QuantileRegressionForest:
    model = build_model()
    model.fit(X_train, y_train)
    return model


def predict(model: QuantileRegressionForest,
            X_test: np.ndarray,
            quantile: float = 0.5) -> np.ndarray:
    """Vorhersage für ein einzelnes Quantil (Default: Median)."""
    return model.predict(X_test, quantile=quantile)


def predict_all_quantiles(model: QuantileRegressionForest,
                           X_test: np.ndarray,
                           quantiles: list = QUANTILES) -> pd.DataFrame:
    """Vorhersage für alle Quantile → DataFrame mit einer Spalte pro Quantil."""
    preds = {}
    for q in quantiles:
        print(f"  Quantile {q:.2f} wird berechnet...")
        preds[f"q{int(q*100):02d}"] = model.predict(X_test, quantile=q)
    return pd.DataFrame(preds)


def optimal_bid(model: QuantileRegressionForest,
                X_test: np.ndarray,
                c_under: float = 1.0, c_over: float = 1.0) -> np.ndarray:
    """
    Optimales Gebot aus Newsvendor-Logik: tau = c_under / (c_under + c_over).
    """
    tau = c_under / (c_under + c_over)
    print(f"  Newsvendor tau={tau:.3f}")
    return predict(model, X_test, quantile=tau)
