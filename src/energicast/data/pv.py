"""Lightweight solar position helper used in feature generation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

try:  # pragma: no cover - exercised when pvlib is installed
    from pvlib.solarposition import get_solarposition as _pvlib_get_solarposition
except Exception:  # pragma: no cover - fallback branch used during tests

    def _pvlib_get_solarposition(times, latitude, longitude):
        raise ImportError


def _fallback_solarposition(
    index: pd.DatetimeIndex, latitude: float, longitude: float
) -> pd.DataFrame:
    """Approximate solar position using a simple declination model."""

    if index.tz is None:
        index = index.tz_localize("UTC")
    # Convert to fractional hours from local solar time approximation.
    day_of_year = index.dayofyear.to_numpy()
    fractional_hour = index.hour + index.minute / 60.0
    gamma = 2.0 * math.pi / 365.0 * (day_of_year - 1)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    lat_rad = np.deg2rad(latitude)
    hour_angle = np.deg2rad((fractional_hour - 12.0) * 15.0) + np.deg2rad(longitude / 15.0)
    cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(
        hour_angle
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    zenith = np.arccos(cos_zenith)
    elevation = np.pi / 2.0 - zenith
    azimuth = np.arctan2(
        -np.sin(hour_angle),
        np.tan(decl) * np.cos(lat_rad) - np.sin(lat_rad) * np.cos(hour_angle),
    )
    return pd.DataFrame(
        {
            "solar_zenith": np.rad2deg(zenith),
            "solar_elevation": np.rad2deg(elevation),
            "solar_azimuth": np.rad2deg(azimuth),
        },
        index=index,
    )


def solar_position_features(
    index: pd.DatetimeIndex, latitude: float = 52.0, longitude: float = 21.0
) -> pd.DataFrame:
    """Return solar geometry features for the provided timestamps."""

    try:
        data = _pvlib_get_solarposition(index, latitude=latitude, longitude=longitude)
        columns = {
            "apparent_zenith": "solar_zenith",
            "apparent_elevation": "solar_elevation",
            "azimuth": "solar_azimuth",
        }
        df = data.loc[:, columns.keys()].rename(columns=columns)
    except Exception:
        df = _fallback_solarposition(index, latitude, longitude)
    df["solar_cos_zenith"] = np.cos(np.deg2rad(df["solar_zenith"]))
    df["solar_day_fraction"] = (index.hour + index.minute / 60.0) / 24.0
    return df


__all__ = ["solar_position_features"]
