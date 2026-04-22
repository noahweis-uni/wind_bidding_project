# train.py
# -------------------------------------------------------
# Zweck: Einheitlicher Trainings-Wrapper für alle Modelle.
#        Daten splitten, Features vorbereiten, Modelle trainieren.
#
# TODO:
#   - FEATURES und TARGET anpassen wenn nötig
#   - TEST_SIZE: Anteil Testdaten (default 20%)
#   - Scaler ist wichtig für Neural Net – bei anderen Modellen optional
# -------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from src.models import linear_regression, random_forest, neural_net, quantile_regression


FEATURES  = ["wind_speed", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
TARGET    = "power"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def prepare_data(df: pd.DataFrame):
    """
    Splittet DataFrame in Train/Test und gibt numpy arrays zurück.
    TODO: Sicherstellen dass df keine NaN-Werte enthält.
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )
    # shuffle=False wichtig bei Zeitreihen!
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    StandardScaler – MUSS für Neural Net verwendet werden.
    Optional für RF/LinReg aber schadet nicht.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


def train_all_models(df: pd.DataFrame) -> dict:
    """
    Trainiert alle 4 Modelle und gibt sie als dict zurück.
    Speichert Modelle in results/forecasts/.

    TODO: Pfade anpassen wenn gewünscht.
    """
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_sc_train, X_sc_test, scaler    = scale_features(X_train, X_test)

    print("Training Linear Regression...")
    lr_model = linear_regression.train(X_train, y_train, polynomial_degree=2)

    print("Training Random Forest...")
    rf_model = random_forest.train(X_train, y_train)

    print("Training Neural Network...")
    nn_model = neural_net.train(X_sc_train, y_train)

    print("Training Quantile Regression...")
    qr_models = quantile_regression.train_all_quantiles(X_train, y_train)

    models = {
        "LinearRegression": lr_model,
        "RandomForest":     rf_model,
        "NeuralNet":        nn_model,
        "QuantileRegression": qr_models,
    }

    # TODO: Modelle speichern
    # joblib.dump(lr_model, "results/forecasts/linear_regression.pkl")
    # joblib.dump(rf_model, "results/forecasts/random_forest.pkl")
    # torch.save(nn_model.state_dict(), "results/forecasts/neural_net.pt")

    return models, X_train, X_test, X_sc_train, X_sc_test, y_train, y_test, scaler
