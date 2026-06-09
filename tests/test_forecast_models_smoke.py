"""
Smoke tests for all forecasting models.

Run from project root with:
python tests/test_forecast_models_smoke.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from src.models import (
    linear_regression,
    elastic_net,
    random_forest,
    gradient_boosting,
    quantile_regression,
    quantile_regression_forest,
    quantile_gradient_boosting,
    quantile_neural_network,
)


def make_dummy_data(n_samples: int = 300, n_features: int = 8):
    """Create small artificial dataset for testing model interfaces."""
    rng = np.random.default_rng(42)

    X = pd.DataFrame(
        rng.normal(size=(n_samples, n_features)),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    y = (
        0.5 * X["feature_0"]
        - 0.2 * X["feature_1"]
        + 0.1 * rng.normal(size=n_samples)
    )

    # Wind production should be non-negative
    y = np.clip(y, 0, None)

    split = int(n_samples * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = pd.Series(y[:split])
    y_test = pd.Series(y[split:])

    return X_train, X_test, y_train, y_test


def check_prediction(name: str, y_pred, expected_len: int):
    """Check prediction output."""
    y_pred = np.asarray(y_pred)

    assert len(y_pred) == expected_len, f"{name}: wrong prediction length"
    assert np.all(np.isfinite(y_pred)), f"{name}: prediction contains NaN or inf"
    assert y_pred.ndim == 1, f"{name}: prediction should be 1D"

    print(f"OK: {name}")


def main():
    X_train, X_test, y_train, y_test = make_dummy_data()
    n_test = len(y_test)

    print("Testing point forecast models...")

    # Linear Regression
    model = linear_regression.train(X_train, y_train)
    pred = linear_regression.predict(model, X_test)
    check_prediction("linear_regression", pred, n_test)

    # Elastic Net
    model = elastic_net.train(X_train, y_train)
    pred = elastic_net.predict(model, X_test)
    check_prediction("elastic_net", pred, n_test)

    # Random Forest
    model = random_forest.train(
        X_train,
        y_train,
        n_estimators=20,
        min_samples_leaf=2,
    )
    pred = random_forest.predict(model, X_test)
    check_prediction("random_forest", pred, n_test)

    # Gradient Boosting
    model = gradient_boosting.train(
        X_train,
        y_train,
        n_estimators=20,
    )
    pred = gradient_boosting.predict(model, X_test)
    check_prediction("gradient_boosting", pred, n_test)

    print("\nTesting probabilistic forecast models...")

    quantiles = (0.25, 0.5, 0.75)

    # Quantile Regression
    models = quantile_regression.train(
        X_train,
        y_train,
        quantiles=quantiles,
    )
    preds = quantile_regression.predict_all(models, X_test)

    assert list(preds.columns) == ["q25", "q50", "q75"], "quantile_regression: wrong columns"
    assert len(preds) == n_test, "quantile_regression: wrong number of rows"
    assert np.all(np.isfinite(preds.values)), "quantile_regression: NaN or inf"

    print("OK: quantile_regression")

    # Quantile Regression Forest
    model = quantile_regression_forest.train(
        X_train,
        y_train,
        n_estimators=20,
        min_samples_leaf=2,
    )
    preds = quantile_regression_forest.predict_all(
        model,
        X_test,
        quantiles=quantiles,
    )

    assert list(preds.columns) == ["q25", "q50", "q75"], "qrf: wrong columns"
    assert len(preds) == n_test, "qrf: wrong number of rows"
    assert np.all(np.isfinite(preds.values)), "qrf: NaN or inf"

    print("OK: quantile_regression_forest")

    # Quantile Gradient Boosting
    models = quantile_gradient_boosting.train(
        X_train,
        y_train,
        quantiles=quantiles,
        n_estimators=20,
    )
    preds = quantile_gradient_boosting.predict_all(models, X_test)

    assert list(preds.columns) == ["q25", "q50", "q75"], "quantile_gradient_boosting: wrong columns"
    assert len(preds) == n_test, "quantile_gradient_boosting: wrong number of rows"
    assert np.all(np.isfinite(preds.values)), "quantile_gradient_boosting: NaN or inf"

    print("OK: quantile_gradient_boosting")

    # Quantile Neural Network
    model = quantile_neural_network.train(
        X_train,
        y_train,
        quantiles=quantiles,
        hidden_layer_sizes=(16, 8),
        max_epochs=20,
        patience=5,
        batch_size=32,
    )
    preds = quantile_neural_network.predict_all(model, X_test)

    assert list(preds.columns) == ["q25", "q50", "q75"], "quantile_neural_network: wrong columns"
    assert len(preds) == n_test, "quantile_neural_network: wrong number of rows"
    assert np.all(np.isfinite(preds.values)), "quantile_neural_network: NaN or inf"

    print("OK: quantile_neural_network")

    print("\nAll forecast model smoke tests passed.")


if __name__ == "__main__":
    main()