"""Weather scenario utilities for uncertainty experiments."""

import numpy as np
import pandas as pd


def naive_weather_ensemble(
    X_base: pd.DataFrame, n: int = 10, scale: float = 0.1
) -> list[pd.DataFrame]:
    """Perturb baseline covariates with Gaussian noise to form an ensemble."""

    ens = []
    for _ in range(n):
        noise = np.random.normal(0, scale, size=X_base.shape)
        Xi = X_base.copy()
        Xi.loc[:, :] = Xi.values + noise
        ens.append(Xi)
    return ens
