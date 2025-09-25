"""Validation utilities shared between AutoML search and pipelines."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class RollingOriginConfig:
    """Configuration container for rolling-origin evaluation."""

    horizon: int
    test_size: int
    n_windows: int = 1
    step_size: int = 24


@dataclass
class RollingWindow:
    """Indices describing a single rolling-origin split."""

    train: pd.DatetimeIndex
    test: pd.DatetimeIndex


def rolling_origin_windows(
    index: pd.DatetimeIndex, config: RollingOriginConfig
) -> List[RollingWindow]:
    """Return rolling-origin splits for the provided :class:`~pandas.DatetimeIndex`."""

    if config.test_size < config.horizon:
        raise ValueError("test_size must be at least as large as the forecast horizon")
    n = len(index)
    if n <= config.test_size:
        raise ValueError("Not enough observations for rolling validation")
    last_start = n - config.test_size
    first_start = last_start - (config.n_windows - 1) * config.step_size
    if first_start <= 0:
        raise ValueError("Rolling windows exceed available history")

    windows: List[RollingWindow] = []
    for offset in range(config.n_windows):
        start = first_start + offset * config.step_size
        train_idx = index[:start]
        test_idx = index[start : start + config.horizon]
        if len(test_idx) == 0:
            continue
        windows.append(RollingWindow(train=train_idx, test=test_idx))
    return windows


def validation_records_frame(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Convert rolling validation records into a flat :class:`~pandas.DataFrame`."""

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame.from_records(records)

    def _is_non_string_iterable(value: object) -> bool:
        return isinstance(value, Iterable) and not isinstance(value, (str, bytes))

    list_columns = [
        column for column in frame.columns if frame[column].map(_is_non_string_iterable).all()
    ]
    for column in list_columns:
        frame[column] = frame[column].apply(list)
    return frame


__all__ = [
    "RollingOriginConfig",
    "RollingWindow",
    "rolling_origin_windows",
    "validation_records_frame",
]
