import numpy as np
import pandas as pd

from energicast.backtest import run_backtest
from energicast.config import DatasetConfig, TrainingConfig
from energicast.pipeline import ForecastPipeline


def _make_dataset(n_periods: int = 5 * 24) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=n_periods, freq="H")
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        {
            "timestamp": index,
            "load": np.sin(np.arange(n_periods) / 24.0) + rng.normal(scale=0.05, size=n_periods),
            "temp": 10 + np.cos(np.arange(n_periods) / 24.0),
            "series_id": "A",
        }
    )
    return data


def test_run_backtest_generates_artifacts(tmp_path):
    df = _make_dataset()
    config = TrainingConfig(
        freq="H",
        horizon=6,
        history_window=36,
        model="xgb",
        model_params={"base_model_params": {"n_estimators": 20, "max_depth": 3}},
        dataset=DatasetConfig(
            time_column="timestamp",
            target_column="load",
            series_id_column="series_id",
            known_covariates=["temp"],
        ),
        validation={"method": "rolling", "test_size": 24, "step_size": 6, "n_windows": 2},
    )
    pipeline = ForecastPipeline(config=config)
    result = run_backtest(pipeline, df, output_dir=tmp_path)
    assert not result.metrics.empty
    assert "crps" in result.metrics.columns
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "summary.json").exists()
    reports = list((tmp_path / "reports").glob("*.png"))
    assert reports
