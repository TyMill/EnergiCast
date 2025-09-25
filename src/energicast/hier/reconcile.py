"""Hierarchy reconciliation helpers."""

from __future__ import annotations

import pandas as pd


def mint_reconcile(bottom_forecasts: pd.DataFrame, S: pd.DataFrame) -> pd.DataFrame:
    """Placeholder MinT reconciliation using simple aggregation."""

    return bottom_forecasts @ S.T


__all__ = ["mint_reconcile"]
