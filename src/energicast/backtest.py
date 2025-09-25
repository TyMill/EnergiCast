"""Utilities for rolling backtests and reporting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import pandas as pd

from .metrics.metrics import empirical_crps, energy_weighted_rmse, multi_quantile_pinball
from .pipeline import ForecastPipeline


def _format_quantile(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


@dataclass
class BacktestResult:
    metrics: pd.DataFrame
    summary: Dict[str, float]
    output_dir: Optional[Path] = None


def run_backtest(
    pipeline: ForecastPipeline,
    df: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> BacktestResult:
    """Fit the pipeline, compute metrics and optionally persist artefacts."""

    pipeline.fit(df)
    if pipeline.validation_results_ is None or pipeline.validation_results_.empty:
        raise ValueError("Pipeline configuration must enable rolling validation")

    records = []
    quantiles = pipeline.config.quantiles
    validation = pipeline.validation_results_.copy()
    for row in validation.itertuples(index=False):
        index = pd.to_datetime(row.index)
        y_true = pd.Series(row.y_true, index=index)
        forecast_series = {
            column: pd.Series(getattr(row, column), index=index)
            for column in validation.columns
            if column.startswith("q") and hasattr(row, column)
        }
        pinballs = multi_quantile_pinball(y_true, forecast_series, quantiles)
        crps = empirical_crps(y_true, forecast_series, quantiles)
        median_col = _format_quantile(0.5)
        if median_col not in forecast_series:
            median_col = _format_quantile(quantiles[len(quantiles) // 2])
        rmse = energy_weighted_rmse(y_true, forecast_series[median_col])
        record = {
            "series_id": row.series_id,
            "window": row.window,
            "crps": crps,
            "energy_rmse": rmse,
        }
        record.update(pinballs)
        records.append(record)

    metrics_df = pd.DataFrame.from_records(records)
    summary = metrics_df.drop(columns=["series_id", "window"], errors="ignore").mean().to_dict()

    output_path = None
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        reports_dir = output_path / "reports"
        reports_dir.mkdir(exist_ok=True)
        metrics_df.to_csv(output_path / "metrics.csv", index=False)
        with open(output_path / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        for series_id, group in validation.groupby("series_id"):
            last = group.sort_values("window").iloc[-1]
            idx = pd.to_datetime(last["index"])
            y_true = pd.Series(last["y_true"], index=idx)
            median_col = _format_quantile(0.5)
            if median_col not in last.index or not hasattr(last, median_col):
                median_col = _format_quantile(quantiles[len(quantiles) // 2])
            median_values = pd.Series(getattr(last, median_col), index=idx)
            fig, ax = plt.subplots(figsize=(6, 3))
            y_true.plot(ax=ax, label="actual")
            median_values.plot(ax=ax, label="median forecast")
            ax.set_title(f"Series {series_id} - window {last['window']}")
            ax.legend()
            fig.tight_layout()
            fig.savefig(reports_dir / f"{series_id}_window{last['window']}.png")
            plt.close(fig)

    return BacktestResult(metrics=metrics_df, summary=summary, output_dir=output_path)


__all__ = ["BacktestResult", "run_backtest"]
