# evaluate_forecast.py
# -------------------------------------------------------
# Zweck: Forecast-Metriken für alle Modelle berechnen und vergleichen.
#
# TODO: nichts – direkt verwendbar sobald Modelle trainiert sind.
# -------------------------------------------------------

import numpy as np
import pandas as pd
from src.utils.metrics import rmse, mae, bias
from src.models import (linear_regression, random_forest, neural_net,
                        quantile_regression, persistence_model,
                        quantile_regression_forest)


def evaluate_all(models: dict,
                 X_test: np.ndarray,
                 X_sc_test: np.ndarray,
                 y_test: np.ndarray) -> pd.DataFrame:
    """
    Berechnet RMSE, MAE und Bias für alle Modelle.
    X_sc_test: skalierte Features für Neural Net.
    """
    results = {}

    # Persistence Model – y_prev = [y_train[-1], y_test[0], ..., y_test[-2]]
    pm_model = models["Persistence"]
    y_prev   = np.concatenate([[pm_model["last_obs"]], y_test[:-1]])
    y_pred_pm = persistence_model.predict(pm_model, X_test, y_prev)
    results["Persistence"] = {
        "rmse": rmse(y_test, y_pred_pm),
        "mae": mae(y_test, y_pred_pm),
        "bias": bias(y_test, y_pred_pm),
    }

    # Linear Regression
    y_pred_lr = linear_regression.predict(models["LinearRegression"], X_test)
    results["LinearRegression"] = {
        "rmse": rmse(y_test, y_pred_lr),
        "mae": mae(y_test, y_pred_lr),
        "bias": bias(y_test, y_pred_lr),
    }

    # Random Forest
    y_pred_rf = random_forest.predict(models["RandomForest"], X_test)
    results["RandomForest"] = {
        "rmse": rmse(y_test, y_pred_rf),
        "mae": mae(y_test, y_pred_rf),
        "bias": bias(y_test, y_pred_rf),
    }

    # Neural Network (braucht skalierte Features)
    y_pred_nn = neural_net.predict(models["NeuralNet"], X_sc_test)
    results["NeuralNet"] = {
        "rmse": rmse(y_test, y_pred_nn),
        "mae": mae(y_test, y_pred_nn),
        "bias": bias(y_test, y_pred_nn),
    }

    # Quantile Regression – Median als Punkt-Prognose
    y_pred_qr = quantile_regression.predict_quantile(models["QuantileRegression"], X_test, 0.5)
    results["QuantileRegression (q50)"] = {
        "rmse": rmse(y_test, y_pred_qr),
        "mae": mae(y_test, y_pred_qr),
        "bias": bias(y_test, y_pred_qr),
    }

    # Quantile Regression Forest – Median als Punkt-Prognose
    y_pred_qrf = quantile_regression_forest.predict(models["QRF"], X_test, quantile=0.5)
    results["QRF (q50)"] = {
        "rmse": rmse(y_test, y_pred_qrf),
        "mae": mae(y_test, y_pred_qrf),
        "bias": bias(y_test, y_pred_qrf),
    }

    df_results = pd.DataFrame(results).T.round(4)
    print("\n── Forecast Evaluation ──────────────────")
    print(df_results.to_string())
    return df_results
