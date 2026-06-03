# evaluate_bidding.py
# -------------------------------------------------------
# Zweck: Economic-Performance aller Bidding-Strategien berechnen.
#        Das ist die wichtigste Evaluation im Projekt.
# -------------------------------------------------------

import numpy as np
import pandas as pd
from src.utils.metrics import newsvendor_loss, profit, regret, summary_table
from src.bidding import newsvendor
from src.bidding.decision_rules import naive_bid, quantile_bid, oracle_bid, persistence_bid


def evaluate_strategy(y_true: np.ndarray,
                      y_bid: np.ndarray,
                      price_da: np.ndarray,
                      price_rebap: np.ndarray,
                      c_under: float,
                      c_over: float) -> dict:
    """Berechnet alle Economic-Metriken für eine Strategie."""
    return {
        "nv_loss": newsvendor_loss(y_true, y_bid, c_under, c_over),
        "profit":  float(np.mean(profit(y_true, y_bid, price_da, price_rebap))),
        "regret":  regret(y_true, y_bid, price_da, price_rebap),
    }


def evaluate_all_strategies(y_true: np.ndarray,
                             y_pred_lr: np.ndarray,
                             y_pred_rf: np.ndarray,
                             y_pred_nn: np.ndarray,
                             qr_models: dict,
                             X_test: np.ndarray,
                             price_da: np.ndarray,
                             price_rebap: np.ndarray) -> pd.DataFrame:
    """
    Vergleicht alle Strategien auf einmal.
    c_under / c_over werden automatisch aus den Preisdaten abgeleitet.
    Gibt Tabelle zurück: Strategie × Metrik.
    """
    c_under, c_over = newsvendor.compute_costs(price_da, price_rebap)
    tau = newsvendor.optimal_quantile(c_under, c_over)
    print(f"  c_under={c_under:.4f}, c_over={c_over:.4f} → τ*={tau:.3f}")

    strategies = {
        "Naive (LR)":      naive_bid(y_pred_lr),
        "Naive (RF)":      naive_bid(y_pred_rf),
        "Naive (NN)":      naive_bid(y_pred_nn),
        "Newsvendor (QR)": quantile_bid(qr_models, X_test, c_under, c_over),
        "Persistence":     persistence_bid(y_true),
        "Oracle":          oracle_bid(y_true),
    }

    results = {}
    for name, y_bid in strategies.items():
        results[name] = evaluate_strategy(
            y_true, y_bid, price_da, price_rebap, c_under, c_over
        )

    df = summary_table(results)
    print("\n── Bidding Evaluation ───────────────────")
    print(df.to_string())
    return df
