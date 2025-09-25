"""Synthetic gap filling utilities for energy time series."""

from typing import Optional

import pandas as pd


class GapFiller:
    """Hybrid gap filler.

    v0: forward/backward fill + seasonal interpolation.
    v1+: diffusion/cGAN with ramp-rate and non-negativity constraints.
    """

    def __init__(self, max_gap_hours: int = 6, enforce_non_negative: bool = True):
        self.max_gap_hours = max_gap_hours
        self.enforce_non_negative = enforce_non_negative

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None):
        """Estimate seasonal statistics used during :meth:`transform`."""

        self.freq = pd.infer_freq(y.index) or "H"
        return self

    def transform(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> pd.Series:
        """Fill gaps via directional fills and hour-of-day means."""

        s = y.copy()
        df = s.to_frame("y")
        df["hod"] = df.index.hour
        means = df.groupby("hod")["y"].transform("mean")
        s = s.fillna(method="ffill", limit=self.max_gap_hours)
        s = s.fillna(method="bfill", limit=self.max_gap_hours)
        s = s.fillna(means)
        if self.enforce_non_negative:
            s = s.clip(lower=0)
        return s

    def fit_transform(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> pd.Series:
        return self.fit(y, X).transform(y, X)
