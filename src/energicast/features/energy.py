"""Energy specific feature generation."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from ..data.pv import solar_position_features
from ..data.weather import simple_weather_features


HOLIDAY_DAYS = {(1, 1), (5, 1), (12, 25), (12, 26)}


def _is_holiday(
    index: pd.DatetimeIndex, extra: Iterable[tuple[int, int]] | None = None
) -> pd.Series:
    holidays = set(HOLIDAY_DAYS)
    if extra is not None:
        holidays.update(extra)
    return index.to_series().apply(lambda ts: (ts.month, ts.day) in holidays)


def make_energy_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate calendar, solar and weather derived features."""

    df = pd.DataFrame(index=index)
    df["hour"] = index.hour
    df["dow"] = index.dayofweek
    df["month"] = index.month
    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)
    df["is_holiday"] = _is_holiday(index).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7.0)

    solar = solar_position_features(index)
    solar["solar_ramp_1h"] = solar["solar_elevation"].diff().fillna(0.0)
    solar["solar_ramp_3h"] = solar["solar_elevation"].diff(3).fillna(0.0)

    weather = simple_weather_features(index)
    weather["temp_ramp_1h"] = weather["temp_2m"].diff().fillna(0.0)
    weather["wind_ramp_1h"] = weather["wind_speed"].diff().fillna(0.0)

    df = df.join(solar)
    df = df.join(weather)
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df


__all__ = ["make_energy_features"]
