# decision_rules.py
# -------------------------------------------------------
# Zweck: Aus Forecast → Gebot y berechnen.
#        Drei Strategien: Naive, Quantile-based, Oracle.
#
# Das ist der Kern des Projekts – hier wird Forecast zu Entscheidung.
#
# TODO:
#   - c_under und c_over aus echten Preisdaten ableiten
#     (z.B. Differenz Day-Ahead zu reBAP)
#   - Oracle-Strategie braucht y_true → nur für Evaluation verwenden
# -------------------------------------------------------

import numpy as np
from src.bidding import newsvendor


def naive_bid(y_pred: np.ndarray) -> np.ndarray:
    """
    Benchmark 1 – Naiv: Gebot = Punkt-Prognose.
    Keine Berücksichtigung von Unsicherheit.
    y* = y_hat
    """
    return y_pred.copy()


def quantile_bid(qr_models: dict,
                 X_test: np.ndarray,
                 c_under: float = 1.0,
                 c_over: float = 1.0) -> np.ndarray:
    """
    Newsvendor-optimales Gebot:
    τ* = c_under / (c_under + c_over)  →  y* = q_{τ*}
    Interpoliert zwischen verfügbaren Quantilen.
    """
    return newsvendor.optimal_bid(qr_models, X_test, c_under, c_over)


def oracle_bid(y_true: np.ndarray) -> np.ndarray:
    """
    Benchmark 3 – Oracle (What-If):
    Bietet exakt die wahre Produktion → maximaler Profit.
    NUR für Evaluation, nicht in echter Anwendung verfügbar!
    """
    return y_true.copy()


def persistence_bid(y_true: np.ndarray, lag: int = 24) -> np.ndarray:
    """
    Extra-Baseline: Persistence – Gebot = Produktion von vor 24h.
    TODO: Randbehandlung prüfen (erste 24 Werte fehlen).
    """
    bid = np.roll(y_true, lag)
    bid[:lag] = y_true[:lag].mean()  # Auffüllen mit Mittelwert
    return bid
