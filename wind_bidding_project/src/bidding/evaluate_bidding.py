# evaluate_bidding.py
# -------------------------------------------------------
# Zweck: Economic-Performance aller Bidding-Strategien berechnen.
#        Das ist die wichtigste Evaluation im Projekt.
#
# TODO:
#   - price_da und price_rebap aus echten Daten übergeben
#   - c_under / c_over aus Preisdifferenzen ableiten
#   - Wenn keine Preisdaten vorhanden: Platzhalter-Werte nutzen
# -------------------------------------------------------

import numpy as np
import pandas as pd
from src.utils.metrics import newsvendor_loss, profit, regret, summary_table
from src.bidding.decision_rules import naive_bid, quantile_bid, oracle_bid, persistence_bid


def evaluate_strategy(y_true: np.ndarray,
                      y_bid: np.ndarray,
                      strategy_name: str,
                      price_da: np.ndarray,
                      price_rebap: np.ndarray,
                      c_under: float = 1.0,
                      c_over: float = 1.0) -> dict:
    """Berechnet alle Economic-Metriken für eine Strategie."""
    return {
        "newsvendor_loss": newsvendor_loss(y_true, y_bid, c_over, c_under),
        "profit":          profit(y_true, y_bid, price_da, price_rebap),
        "regret":          regret(y_true, y_bid, price_da, price_rebap),
    }


def evaluate_all_strategies(y_true: np.ndarray,
                             y_pred_lr: np.ndarray,
                             y_pred_rf: np.ndarray,
                             y_pred_nn: np.ndarray,
                             qr_models: dict,
                             X_test: np.ndarray,
                             price_da: np.ndarray,
                             price_rebap: np.ndarray,
                             c_under: float = 1.0,
                             c_over: float = 1.0) -> pd.DataFrame:
    """
    Vergleicht alle Strategien auf einmal.
    Gibt Tabelle zurück: Modell × Metrik.
    """
    strategies = {
        "Naive (LR)":     naive_bid(y_pred_lr),
        "Naive (RF)":     naive_bid(y_pred_rf),
        "Naive (NN)":     naive_bid(y_pred_nn),
        "Quantile Bid":   quantile_bid(qr_models, X_test, c_under, c_over),
        "Persistence":    persistence_bid(y_true),
        "Oracle":         oracle_bid(y_true),
    }

    results = {}
    for name, y_bid in strategies.items():
        results[name] = evaluate_strategy(
            y_true, y_bid, name, price_da, price_rebap, c_under, c_over
        )

    df = summary_table(results)
    print("\n── Bidding Evaluation ───────────────────")
    print(df.to_string())
    return df
