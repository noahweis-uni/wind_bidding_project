# metrics.py
# -------------------------------------------------------
# Zweck: Alle Evaluationsmetriken an einem Ort.
#        Forecast-Metriken UND Economic-Metriken.
#
# TODO: nichts – diese Datei ist direkt verwendbar.
#       Bei Bedarf weitere Metriken ergänzen.
# -------------------------------------------------------

import numpy as np
import pandas as pd


# ── Forecast-Metriken ──────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_pred - y_true)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ── Economic-Metriken (Newsvendor) ────────────────────

def newsvendor_loss(y_true: np.ndarray,
                   y_bid: np.ndarray,
                   c_under: float = 1.0,
                   c_over: float = 1.0) -> float:
    """
    Newsvendor-Loss (Erwartungswert über alle Stunden).
    c_under: Kosten bei Unterdeckung (bid > actual), E[max(reBAP - DA, 0)]
    c_over:  Kosten bei Überdeckung  (actual > bid), E[max(DA - reBAP, 0)]
    L = c_under * max(bid - actual, 0) + c_over * max(actual - bid, 0)
    """
    return float(np.mean(
        c_under * np.maximum(y_bid - y_true, 0) +
        c_over  * np.maximum(y_true - y_bid, 0)
    ))


def profit(y_true: np.ndarray,
           y_bid: np.ndarray,
           price_da: np.ndarray,
           price_rebap: np.ndarray) -> float:
    """
    Profit-Berechnung mit Day-Ahead und reBAP.
    revenue = price_da * y_bid + price_rebap * (y_true - y_bid)
    TODO: Vorzeichen-Konvention für reBAP prüfen.
    """
    revenue = price_da * y_bid + price_rebap * (y_true - y_bid)
    return np.mean(revenue)


def regret(y_true: np.ndarray,
           y_bid: np.ndarray,
           price_da: np.ndarray,
           price_rebap: np.ndarray) -> float:
    """
    Regret = Oracle-Profit minus Modell-Profit.
    Oracle bietet exakt die wahre Produktion.
    """
    oracle_profit = np.mean(price_da * y_true)
    model_profit  = profit(y_true, y_bid, price_da, price_rebap)
    return oracle_profit - model_profit


def summary_table(results: dict) -> pd.DataFrame:
    """
    Erstellt Vergleichstabelle für alle Modelle.
    results = {'LinearRegression': {'rmse': ..., 'profit': ..., 'regret': ...}, ...}
    """
    return pd.DataFrame(results).T.round(4)
