"""Utility metrics for probabilistic and energy-weighted evaluation."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd


def _format_quantile(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


def pinball_loss(y_true: pd.Series, y_pred: pd.Series, q: float) -> float:
    """Compute the pinball loss for a single quantile forecast."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be aligned")
    errors = y_true - y_pred
    losses = np.maximum(q * errors, (q - 1) * errors)
    return float(np.mean(losses))


def multi_quantile_pinball(
    y_true: pd.Series,
    forecasts: Mapping[str, pd.Series] | pd.DataFrame,
    quantiles: Sequence[float],
) -> Dict[str, float]:
    """Return pinball loss for each quantile and the average across them."""

    if isinstance(forecasts, pd.DataFrame):
        frame = forecasts
    else:
        frame = pd.DataFrame({k: v for k, v in forecasts.items()})
    results: Dict[str, float] = {}
    for q in quantiles:
        column = _format_quantile(q)
        if column not in frame:
            raise KeyError(f"Forecasts missing column '{column}'")
        results[column] = pinball_loss(y_true, frame[column], q)
    results["pinball_mean"] = float(np.mean(list(results.values())))
    return results


def empirical_crps(
    y_true: pd.Series,
    forecasts: Mapping[str, pd.Series] | pd.DataFrame,
    quantiles: Sequence[float],
) -> float:
    """Approximate the CRPS empirically from quantile forecasts."""

    losses = []
    for q, loss in multi_quantile_pinball(y_true, forecasts, quantiles).items():
        if q == "pinball_mean":
            continue
        quantile = int(q[1:]) / 100.0
        losses.append((quantile, loss))
    if not losses:
        raise ValueError("No quantile forecasts provided")
    losses = sorted(losses, key=lambda x: x[0])
    quantile_levels = np.array([x[0] for x in losses])
    pinballs = np.array([x[1] for x in losses])
    return float(np.trapezoid(pinballs, quantile_levels))


def energy_weighted_rmse(
    y_true: pd.Series,
    y_pred: pd.Series,
    price: pd.Series | None = None,
) -> float:
    """Root mean squared error weighted by price or absolute load."""

    weights = price if price is not None else y_true.abs()
    weights = weights.replace(0, np.nan).fillna(1.0)
    errors = (y_true - y_pred) ** 2
    return float(np.sqrt((errors * weights).sum() / weights.sum()))


def imbalance_cost(
    y_true: pd.Series,
    y_pred: pd.Series,
    under_penalty: float = 1.5,
    over_penalty: float = 1.0,
) -> float:
    """Estimate imbalance cost with asymmetric penalties."""

    diff = y_pred - y_true
    under = diff[diff < 0].abs().sum() * under_penalty
    over = diff[diff >= 0].sum() * over_penalty
    return float((under + over) / len(y_true))


__all__ = [
    "pinball_loss",
    "multi_quantile_pinball",
    "empirical_crps",
    "energy_weighted_rmse",
    "imbalance_cost",
]
