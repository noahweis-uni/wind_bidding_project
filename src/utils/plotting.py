# plotting.py
# -------------------------------------------------------
<<<<<<< HEAD
# Zweck: Wiederverwendbare Plot-Funktionen fuer alle Notebooks.
# -------------------------------------------------------

from pathlib import Path

=======
# Zweck: Wiederverwendbare Plot-Funktionen für alle Notebooks.
#
# TODO:
#   - SAVE_PATH anpassen wenn Plots automatisch gespeichert werden sollen
#   - Farbschema nach Geschmack anpassen
# -------------------------------------------------------

>>>>>>> origin/main
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

<<<<<<< HEAD
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVE_PATH = PROJECT_ROOT / "results" / "figures"
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def plot_forecast_vs_actual(y_true, y_pred, model_name: str, n: int = 200):
    """Zeitreihen-Plot: Prognose vs. tatsaechliche Produktion."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true[:n], label="Actual", alpha=0.8)
    ax.plot(y_pred[:n], label=model_name, alpha=0.8, linestyle="--")
    ax.set_title(f"Forecast vs. Actual - {model_name}")
=======
SAVE_PATH = "results/figures/"


def plot_forecast_vs_actual(y_true, y_pred, model_name: str, n: int = 200):
    """Zeitreihen-Plot: Prognose vs. tatsächliche Produktion."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true[:n],  label="Actual",        alpha=0.8)
    ax.plot(y_pred[:n],  label=model_name,       alpha=0.8, linestyle="--")
    ax.set_title(f"Forecast vs. Actual – {model_name}")
>>>>>>> origin/main
    ax.set_xlabel("Stunde")
    ax.set_ylabel("Leistung [kW]")
    ax.legend()
    plt.tight_layout()
<<<<<<< HEAD
    plt.savefig(SAVE_PATH / f"forecast_{model_name.lower().replace(' ', '_')}.png", dpi=150)
=======
    plt.savefig(f"{SAVE_PATH}forecast_{model_name.lower().replace(' ', '_')}.png", dpi=150)
>>>>>>> origin/main
    plt.show()


def plot_quantile_forecast(y_true, q10, q50, q90, n: int = 200):
<<<<<<< HEAD
    """Zeigt Median + Konfidenzband (10-90%) vs. Actual."""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(n)
    ax.fill_between(x, q10[:n], q90[:n], alpha=0.2, label="10-90% Band")
=======
    """Zeigt Median + Konfidenzband (10–90%) vs. Actual."""
    fig, ax = plt.subplots(figsize=(12, 4))
    x = range(n)
    ax.fill_between(x, q10[:n], q90[:n], alpha=0.2, label="10–90% Band")
>>>>>>> origin/main
    ax.plot(list(x), q50[:n], label="Median (q50)", linewidth=1.5)
    ax.plot(list(x), y_true[:n], label="Actual", alpha=0.7, linestyle="--")
    ax.set_title("Quantile Forecast")
    ax.set_xlabel("Stunde")
    ax.set_ylabel("Leistung [kW]")
    ax.legend()
    plt.tight_layout()
<<<<<<< HEAD
    plt.savefig(SAVE_PATH / "quantile_forecast.png", dpi=150)
=======
    plt.savefig(f"{SAVE_PATH}quantile_forecast.png", dpi=150)
>>>>>>> origin/main
    plt.show()


def plot_metric_comparison(results: dict, metric: str = "rmse"):
<<<<<<< HEAD
    """Balkendiagramm: Metrik-Vergleich ueber alle Modelle."""
=======
    """Balkendiagramm: Metrik-Vergleich über alle Modelle."""
>>>>>>> origin/main
    models = list(results.keys())
    values = [results[m][metric] for m in models]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(models, values, color="#4C72B0", edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
<<<<<<< HEAD
    ax.set_title(f"Modellvergleich - {metric.upper()}")
    ax.set_ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(SAVE_PATH / f"comparison_{metric}.png", dpi=150)
=======
    ax.set_title(f"Modellvergleich – {metric.upper()}")
    ax.set_ylabel(metric.upper())
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}comparison_{metric}.png", dpi=150)
>>>>>>> origin/main
    plt.show()


def plot_profit_regret(results: dict):
    """Doppel-Balken: Profit und Regret nebeneinander."""
    models = list(results.keys())
<<<<<<< HEAD
    profits = [results[m].get("profit", 0) for m in models]
    regrets = [results[m].get("regret", 0) for m in models]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, profits, width, label="Profit", color="#55A868")
    ax.bar(x + width / 2, regrets, width, label="Regret", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Profit & Regret - Modellvergleich")
    ax.legend()
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "profit_regret.png", dpi=150)
=======
    profits  = [results[m].get("profit", 0)  for m in models]
    regrets  = [results[m].get("regret", 0)  for m in models]
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width/2, profits, width, label="Profit",  color="#55A868")
    ax.bar(x + width/2, regrets, width, label="Regret",  color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_title("Profit & Regret – Modellvergleich")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}profit_regret.png", dpi=150)
>>>>>>> origin/main
    plt.show()
