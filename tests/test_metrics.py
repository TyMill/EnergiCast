import numpy as np
import pandas as pd

from energicast.metrics.metrics import (
    empirical_crps,
    energy_weighted_rmse,
    imbalance_cost,
    multi_quantile_pinball,
    pinball_loss,
)


def test_pinball_and_multi_quantile_metrics():
    index = pd.date_range("2024-01-01", periods=5, freq="H")
    y_true = pd.Series(np.linspace(0, 4, 5), index=index)
    forecasts = {
        "q10": y_true - 0.5,
        "q50": y_true,
        "q90": y_true + 0.5,
    }
    single = pinball_loss(y_true, forecasts["q10"], 0.1)
    results = multi_quantile_pinball(y_true, forecasts, [0.1, 0.5, 0.9])
    assert single == results["q10"]
    assert results["pinball_mean"] <= max(results.values())
    crps = empirical_crps(y_true, forecasts, [0.1, 0.5, 0.9])
    assert crps >= 0


def test_energy_weighted_rmse_and_imbalance_cost():
    index = pd.date_range("2024-01-01", periods=4, freq="H")
    y_true = pd.Series([10, 12, 11, 13], index=index)
    y_pred = y_true + pd.Series([0.5, -0.2, 0.3, -0.4], index=index)
    price = pd.Series([100, 80, 60, 40], index=index)
    rmse = energy_weighted_rmse(y_true, y_pred, price=price)
    assert rmse > 0
    cost = imbalance_cost(y_true, y_pred)
    assert cost > 0
