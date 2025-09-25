"""Weather feature helpers used for deterministic baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simple_weather_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate deterministic weather proxies with lagged covariates."""

    hours = index.hour + index.minute / 60.0
    dayofyear = index.dayofyear
    df = pd.DataFrame(index=index)
    df["temp_2m"] = (
        12 + 8 * np.sin(2 * np.pi * hours / 24.0) + 2 * np.cos(2 * np.pi * dayofyear / 365.0)
    )
    df["wind_speed"] = 5 + 1.5 * np.cos(2 * np.pi * hours / 12.0)
    df["ghi_clear_sky"] = np.clip(900 * np.sin(np.pi * hours / 24.0), a_min=0, a_max=None)
    for lag in (1, 6, 24):
        df[f"temp_lag_{lag}"] = df["temp_2m"].shift(lag).bfill()
        df[f"wind_lag_{lag}"] = df["wind_speed"].shift(lag).bfill()
    return df


__all__ = ["simple_weather_features"]
