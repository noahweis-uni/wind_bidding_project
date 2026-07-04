# newsvendor.py
# -------------------------------------------------------
# Schicht 1 – Gewinnfunktion:
#   π_t = p_DA,t · b_t + p_reBAP,t · (y_t - b_t)
#
# Schicht 2 – Newsvendor-Verlustfunktion:
#   L_t = c_under · max(b_t - y_t, 0) + c_over · max(y_t - b_t, 0)
#   c_under = E[max(p_reBAP - p_DA, 0)]  → Unterdeckung teuer wenn reBAP > DA
#   c_over  = E[max(p_DA - p_reBAP, 0)]  → Überdeckung teuer wenn DA > reBAP
#
# Schicht 3 – Optimales Quantil:
#   Minimierung von E[L] (Schicht 2) ergibt F(b*) = c_over / (c_under + c_over).
#   Herleitung: dE[L]/db = c_under·F(b) − c_over·(1−F(b)) = 0.
#   τ* = c_over / (c_under + c_over)
# -------------------------------------------------------

import numpy as np
from src.models import quantile_regression


def compute_costs(p_da: np.ndarray, p_rebap: np.ndarray) -> tuple[float, float]:
    """
    Leitet c_under und c_over direkt aus den Preiszeitreihen ab.

    Returns:
        c_under: mittlere Kosten bei Unterdeckung (bid > actual)
        c_over:  mittlere Kosten bei Überdeckung  (actual > bid)
    """
    c_under = float(np.mean(np.maximum(p_rebap - p_da, 0)))
    c_over  = float(np.mean(np.maximum(p_da - p_rebap, 0)))
    return c_under, c_over


def optimal_quantile(c_under: float, c_over: float) -> float:
    """
    Kostenoptimales Gebotsquantil des Newsvendor-Problems.

    Minimierung von E[L] mit L = c_under·(b-y)+ + c_over·(y-b)+ liefert
    F(b*) = c_over / (c_under + c_over). Ist Unterdeckung teurer
    (c_under > c_over), so ist tau* < 0.5: konservativ *unter* dem Median
    bieten, um die teure Untereinspeisung selten zu machen.
    """
    return c_over / (c_under + c_over)


def loss(b: np.ndarray, y: np.ndarray,
         c_under: float, c_over: float) -> np.ndarray:
    """
    Newsvendor-Verlust pro Stunde (Schicht 2).
    L_t = c_under · max(b_t - y_t, 0) + c_over · max(y_t - b_t, 0)
    """
    return (c_under * np.maximum(b - y, 0) +
            c_over  * np.maximum(y - b, 0))


def profit(b: np.ndarray, y: np.ndarray,
           p_da: np.ndarray, p_rebap: np.ndarray) -> np.ndarray:
    """
    Gewinn pro Stunde (Schicht 1).
    π_t = p_DA,t · b_t + p_reBAP,t · (y_t - b_t)
    """
    return p_da * b + p_rebap * (y - b)


def optimal_bid(qr_models: dict, X_test: np.ndarray,
                c_under: float, c_over: float) -> np.ndarray:
    """
    Optimales Gebot via Newsvendor-Lösung (Schicht 3).
    Interpoliert linear zwischen den nächsten verfügbaren Quantilen.
    """
    tau       = optimal_quantile(c_under, c_over)
    available = sorted(qr_models.keys())

    if tau in qr_models:
        print(f"  Newsvendor τ*={tau:.3f} → exaktes Quantil vorhanden")
        return quantile_regression.predict_quantile(qr_models, X_test, tau)

    lower = max((q for q in available if q <= tau), default=available[0])
    upper = min((q for q in available if q >= tau), default=available[-1])

    if lower == upper:
        return quantile_regression.predict_quantile(qr_models, X_test, lower)

    w      = (tau - lower) / (upper - lower)
    q_low  = quantile_regression.predict_quantile(qr_models, X_test, lower)
    q_high = quantile_regression.predict_quantile(qr_models, X_test, upper)
    print(f"  Newsvendor τ*={tau:.3f} → interpoliert zwischen q{lower:.2f} und q{upper:.2f}")
    return (1 - w) * q_low + w * q_high
