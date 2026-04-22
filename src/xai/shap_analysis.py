# shap_analysis.py
# -------------------------------------------------------
# Zweck: SHAP-basierte Erklärung für Random Forest und Neural Net.
#        Beantwortet: Welche Features treiben das Gebot?
#
# TODO:
#   - shap installieren: pip install shap
#   - X_test und feature_names anpassen
#   - Bei Neural Net: DeepExplainer oder KernelExplainer verwenden
#     (KernelExplainer ist langsamer aber universell)
# -------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

SAVE_PATH = "results/figures/"


def shap_random_forest(rf_model, X_test: np.ndarray, feature_names: list):
    """
    TreeExplainer ist schnell und exakt für baumbasierte Modelle.
    Gibt SHAP-Values zurück und zeigt Summary Plot.
    """
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    # Summary Plot: Feature Importance + Richtung
    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      show=False)
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}shap_rf_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    return shap_values


def shap_neural_net(nn_model, X_test: np.ndarray,
                    X_background: np.ndarray, feature_names: list):
    """
    KernelExplainer für Neural Net (modellunabhängig, aber langsam).
    X_background: kleines Sample aus Trainingsdaten als Referenz (50-100 Punkte reichen).
    TODO: Für schnellere Ergebnisse X_background klein halten.
    """
    import torch

    def predict_fn(X):
        nn_model.eval()
        with torch.no_grad():
            return nn_model(torch.FloatTensor(X)).numpy()

    explainer   = shap.KernelExplainer(predict_fn, X_background)
    # TODO: nsamples erhöhen für genauere Schätzung (aber langsamer)
    shap_values = explainer.shap_values(X_test[:100], nsamples=200)

    shap.summary_plot(shap_values, X_test[:100],
                      feature_names=feature_names,
                      show=False)
    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}shap_nn_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    return shap_values


def plot_feature_importance_comparison(rf_importance: pd.Series,
                                       lr_coefs: pd.Series):
    """
    Vergleicht Feature Importance von RF vs. Koeffizienten von LR.
    Zeigt: Einigen sich beide Modelle auf die wichtigsten Features?
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    lr_coefs.abs().sort_values().plot(kind="barh", ax=axes[0], color="#4C72B0")
    axes[0].set_title("Linear Regression – |Koeffizienten|")

    rf_importance.sort_values().plot(kind="barh", ax=axes[1], color="#55A868")
    axes[1].set_title("Random Forest – Feature Importance")

    plt.tight_layout()
    plt.savefig(f"{SAVE_PATH}feature_importance_comparison.png", dpi=150)
    plt.show()
