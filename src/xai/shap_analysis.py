# shap_analysis.py
# -------------------------------------------------------
# Zweck: SHAP-basierte Erklärung der Gebotsentscheidung auf Feature-Ebene.
#        Welche Features treiben das optimale Gebot τ* (Newsvendor-Modell)?
#
# Methodik: Lundberg & Lee (2017), arXiv:1705.07874v2.
# -------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

SAVE_PATH = Path(__file__).resolve().parents[2] / "results" / "figures"
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def shap_random_forest(rf_model, X_test: np.ndarray, feature_names: list,
                       max_samples: int = 150):
    """
    TreeExplainer für RF. max_samples begrenzt die Laufzeit bei tiefen Wäldern
    (300 Bäume × N Punkte — jenseits von ~150 Punkten kaum Gewinn für Feature-Ranking).
    """
    X = X_test[:max_samples]
    explainer   = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X, check_additivity=False)

    shap.summary_plot(shap_values, X,
                      feature_names=feature_names,
                      show=False)
    plt.tight_layout()
    plt.savefig(SAVE_PATH / "shap_rf_summary.png", dpi=150, bbox_inches="tight")
    plt.show()

    return shap_values


def shap_qgb(
    qgb_models: dict,
    tau_star: float,
    X_test: np.ndarray,
    feature_names: list,
) -> np.ndarray:
    """
    TreeExplainer für das τ*-Quantilmodell des Quantile Gradient Boosting.

    Erklärt direkt das Modell, das für das optimale Gebot im Newsvendor-Sinn
    trainiert wurde (GradientBoostingRegressor mit loss='quantile', alpha=τ*).
    SHAP-Werte messen den additiven Beitrag jedes Features zur Vorhersage
    des τ*-Quantils (Lundberg & Lee 2017, Theorem 1).

    Parameters
    ----------
    qgb_models:
        Dict {quantile: GradientBoostingRegressor} aus quantile_gradient_boosting.train().
    tau_star:
        Optimales Gebot-Quantil (Newsvendor-τ*), muss als Key in qgb_models vorhanden sein.
    X_test:
        Testdaten, Shape (n_samples, n_features).
    feature_names:
        Feature-Namen, z.B. ['wind_speed', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'].

    Returns
    -------
    np.ndarray
        SHAP-Values, Shape (n_samples, n_features).
    """
    if tau_star not in qgb_models:
        available = sorted(qgb_models.keys())
        raise ValueError(
            f"tau_star={tau_star} nicht in qgb_models. Verfügbare Quantile: {available}"
        )

    model     = qgb_models[tau_star]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      show=False)
    plt.title(f"QGB SHAP – Quantil τ*={tau_star:.2f} (Gebotsentscheidung)")
    plt.tight_layout()
    plt.savefig(
        SAVE_PATH / f"shap_qgb_tau{int(tau_star * 100):02d}_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    return shap_values


def shap_qrf(
    qrf_model,
    X_test: np.ndarray,
    feature_names: list,
) -> np.ndarray:
    """
    TreeExplainer für den Quantile Regression Forest.

    Hinweis: Der zugrundeliegende RandomForestRegressor minimiert den MSE und
    approximiert E[y | X]. Die SHAP-Werte erklären daher den bedingten
    Erwartungswert als Proxy für die Quantilattribution – nicht direkt das
    τ*-Quantil. Für eine direkte Quantilerklärung ist shap_qgb() vorzuziehen.

    Parameters
    ----------
    qrf_model:
        Trainierter RandomForestRegressor aus quantile_regression_forest.train().
    X_test:
        Testdaten, Shape (n_samples, n_features).
    feature_names:
        Feature-Namen.

    Returns
    -------
    np.ndarray
        SHAP-Values, Shape (n_samples, n_features).
    """
    explainer   = shap.TreeExplainer(qrf_model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test,
                      feature_names=feature_names,
                      show=False)
    plt.title("QRF SHAP – Proxy via bedingtem Erwartungswert E[y|X]")
    plt.tight_layout()
    plt.savefig(
        SAVE_PATH / "shap_qrf_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.show()

    return shap_values


def shap_neural_net(
    nn_model,
    X_test: np.ndarray,
    X_train: np.ndarray,
    feature_names: list,
    n_background: int = 50,
    nsamples: int = 300,
) -> np.ndarray:
    """
    KernelExplainer für das sklearn-Pipeline-Neural-Net (MLPRegressor).

    KernelExplainer ist modellunabhängig und approximiert SHAP-Werte via
    gewichtete lineare Regression über Koalitionen von Features
    (Lundberg & Lee 2017, Abschn. 3.3). Bei 5 Features ist der kombinatorische
    Raum (2^5=32 Koalitionen) überschaubar; nsamples=300 liefert eine stabile
    Schätzung ohne übermäßige Laufzeit.

    Background-Sample: n_background Cluster-Zentren aus X_train via shap.kmeans()
    repräsentieren die marginale Feature-Verteilung als Referenz für E[f(X)].

    Parameters
    ----------
    nn_model:
        Trainierte sklearn Pipeline (StandardScaler + MLPRegressor).
    X_test:
        Testdaten, Shape (n_samples, n_features). Wird auf [:100] begrenzt.
    X_train:
        Trainingsdaten zur Berechnung des Background-Samples via kmeans.
    feature_names:
        Feature-Namen.
    n_background:
        Anzahl kmeans-Cluster für den Background (50–100 empfohlen).
    nsamples:
        Anzahl Koalitions-Samples pro Datenpunkt für KernelExplainer.

    Returns
    -------
    np.ndarray
        SHAP-Values, Shape (min(n_samples, 100), n_features).
    """
    # Lambda-Wrapper verhindert, dass shap.convert_to_model versucht
    # feature_names_in_ auf dem sklearn-Pipeline-Objekt zu setzen (read-only property).
    predict_fn  = lambda X: nn_model.predict(np.asarray(X))
    background  = shap.kmeans(X_train, n_background)
    explainer   = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test[:100], nsamples=nsamples)

    shap.summary_plot(shap_values, X_test[:100],
                      feature_names=feature_names,
                      show=False)
    plt.title("Neural Net SHAP – KernelExplainer (Gebotsentscheidung)")
    plt.tight_layout()
    plt.savefig(
        SAVE_PATH / "shap_nn_summary.png",
        dpi=150,
        bbox_inches="tight",
    )
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
    plt.savefig(SAVE_PATH / "feature_importance_comparison.png", dpi=150)
    plt.show()
