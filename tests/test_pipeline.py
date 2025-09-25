import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from zoneinfo import ZoneInfo

from energicast.config import DatasetConfig, TrainingConfig
from energicast.pipeline import ForecastPipeline


def _make_dataset(n_periods: int = 7 * 24, n_series: int = 2) -> pd.DataFrame:
    base_index = pd.date_range("2024-01-01", periods=n_periods, freq="H")
    frames = []
    rng = np.random.default_rng(7)
    for sid in range(n_series):
        values = np.sin(np.arange(n_periods) / 24.0) + sid * 0.2
        noise = rng.normal(scale=0.05, size=n_periods)
        frame = pd.DataFrame(
            {
                "timestamp": base_index,
                "load": values + noise,
                "temp": 10 + 5 * np.cos(np.arange(n_periods) / 24.0),
                "series_id": f"S{sid}",
            }
        )
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df.loc[df.index[5:8], "load"] = np.nan
    return df


def test_pipeline_fit_and_predict_multi_series():
    df = _make_dataset()
    config = TrainingConfig(
        freq="H",
        horizon=6,
        history_window=48,
        quantiles=[0.1, 0.5, 0.9],
        model="xgb",
        dataset=DatasetConfig(
            time_column="timestamp",
            target_column="load",
            series_id_column="series_id",
            known_covariates=["temp"],
            expected_series_ids=["S0", "S1"],
        ),
    )

    pipeline = ForecastPipeline(config=config)
    pipeline.fit(df)
    predictions = pipeline.predict()

    assert isinstance(predictions, pd.DataFrame)
    assert predictions.index.names == ["series_id", "timestamp"]
    assert predictions.shape[0] == config.horizon * df["series_id"].nunique()
    assert predictions.filter(like="q").notnull().all().all()

    # Quantile monotonicity
    pivot = predictions.reset_index().pivot(index="timestamp", columns="series_id")
    assert (pivot["q10"] <= pivot["q50"]).all().all()
    assert (pivot["q50"] <= pivot["q90"]).all().all()


def test_training_config_validation():
    with pytest.raises(ValidationError):
        TrainingConfig(
            freq="invalid",
            dataset=DatasetConfig(time_column="t", target_column="y"),
        )


def test_training_config_timezone_handling():
    cfg = TrainingConfig(timezone="Europe/Warsaw")
    assert cfg.timezone_info is not None
    assert isinstance(cfg.timezone_info, ZoneInfo)
    assert cfg.timezone_info.key == "Europe/Warsaw"

    with pytest.raises(ValidationError):
        TrainingConfig(timezone="Not/AZone")


def test_pipeline_converts_timezones():
    df = _make_dataset(n_periods=48, n_series=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")

    config = TrainingConfig(
        freq="H",
        horizon=4,
        history_window=24,
        model="xgb",
        timezone="Europe/Berlin",
        dataset=DatasetConfig(
            time_column="timestamp",
            target_column="load",
            series_id_column="series_id",
            known_covariates=["temp"],
            expected_series_ids=["S0"],
        ),
    )

    pipeline = ForecastPipeline(config=config)
    pipeline.fit(df)
    preds = pipeline.predict()

    idx = preds.index.get_level_values("timestamp")
    assert idx.tz is not None
    assert idx.tz.key == "Europe/Berlin"


def test_pipeline_persistence(tmp_path):
    df = _make_dataset(n_periods=72, n_series=1)
    config = TrainingConfig(
        freq="H",
        horizon=6,
        history_window=24,
        model="xgb",
        dataset=DatasetConfig(
            time_column="timestamp",
            target_column="load",
            series_id_column="series_id",
            known_covariates=["temp"],
        ),
    )
    pipeline = ForecastPipeline(config=config)
    pipeline.fit(df)
    save_path = tmp_path / "model"
    pipeline.save(save_path)

    restored = ForecastPipeline.load(save_path)
    preds = restored.predict()
    assert not preds.empty
    assert preds.filter(like="q").notnull().all().all()


def test_pipeline_rolling_validation_records_metrics():
    df = _make_dataset(n_periods=96, n_series=1)
    config = TrainingConfig(
        freq="H",
        horizon=6,
        history_window=24,
        model="xgb",
        dataset=DatasetConfig(
            time_column="timestamp",
            target_column="load",
            series_id_column="series_id",
            known_covariates=["temp"],
        ),
        validation={
            "method": "rolling",
            "test_size": 24,
            "step_size": 6,
            "n_windows": 2,
        },
    )
    pipeline = ForecastPipeline(config=config)
    pipeline.fit(df)
    assert pipeline.validation_results_ is not None
    assert {"series_id", "window", "y_true"}.issubset(pipeline.validation_results_.columns)
