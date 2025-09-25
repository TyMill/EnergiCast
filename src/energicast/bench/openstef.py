"""Benchmark helpers inspired by the OpenSTEF dataset."""

import pandas as pd


def load_openstef_demo() -> pd.Series:
    """Return a synthetic hourly load profile for quick experiments."""

    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="H", tz="UTC")
    return pd.Series(range(len(idx)), index=idx, name="load")
