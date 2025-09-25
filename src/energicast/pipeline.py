"""End-to-end orchestration utilities for training forecasting models.

The goal is to provide a deterministic set of steps that transform raw data
frames into inputs suitable for the underlying forecaster.  The pipeline keeps
track of the configuration, handles multi-series datasets and wires feature
engineering, imputers and models together.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pandas as pd

from . import models as _loaded_models  # noqa: F401  # ensure model registry is populated
from .config import TrainingConfig
from .features.energy import make_energy_features
from .impute.synth import GapFiller
from .models.registry import ForecastModel, MODEL_REGISTRY
from zoneinfo import ZoneInfo


FeatureGenerator = Callable[[pd.DatetimeIndex], pd.DataFrame]


def _default_feature_generators() -> List[FeatureGenerator]:
    return [make_energy_features]


@dataclass
class PipelineArtifacts:
    model: ForecastModel
    target: Optional[pd.Series] = None
    features: Optional[pd.DataFrame] = None


class ForecastPipeline:
    """Coordinate preprocessing steps and delegate to the selected model."""

    def __init__(
        self,
        config: TrainingConfig,
        model_factory: Optional[Callable[[], ForecastModel]] = None,
        imputer: Optional[GapFiller] = None,
        feature_generators: Optional[List[FeatureGenerator]] = None,
    ) -> None:
        self.config = config
        self.dataset = config.dataset
        self.model_name = config.model
        self.model_params = dict(config.model_params)
        self._model_factory = model_factory or self._default_model_factory
        self._imputer = imputer or GapFiller()
        self._feature_generators = feature_generators or _default_feature_generators()
        self._artifacts: Dict[str, PipelineArtifacts] = {}
        self._expected_series_ids = (
            {str(v) for v in self.dataset.expected_series_ids}
            if self.dataset.expected_series_ids is not None
            else None
        )
        self.validation_results_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------ helpers
    def _default_model_factory(self) -> ForecastModel:
        params = {
            "horizon": self.config.horizon,
            "quantiles": self.config.quantiles,
            "max_lag": self.config.history_window,
        }
        params.update(self.model_params)
        return MODEL_REGISTRY.create(self.model_name, **params)

    def _iter_series(self, df: pd.DataFrame) -> Iterator[Tuple[str, pd.DataFrame]]:
        if self.dataset.series_id_column:
            for key, group in df.groupby(self.dataset.series_id_column):
                yield str(key), group.drop(columns=self.dataset.series_id_column)
        else:
            yield "__default__", df

    def _ensure_timezone(self, timestamps: pd.Series) -> pd.Series:
        tzinfo: Optional[ZoneInfo] = self.config.timezone_info
        if tzinfo is None:
            return timestamps
        if timestamps.dt.tz is None:
            return timestamps.dt.tz_localize(tzinfo)
        return timestamps.dt.tz_convert(tzinfo)

    def _prepare_single_series(self, df: pd.DataFrame) -> pd.DataFrame:
        ts_col = self.dataset.time_column
        out = df.copy()
        timestamps = pd.to_datetime(out[ts_col], utc=False)
        timestamps = self._ensure_timezone(timestamps)
        out[ts_col] = timestamps
        out = out.sort_values(ts_col)
        out = out.set_index(ts_col)
        if self.dataset.require_complete_history:
            freq = pd.tseries.frequencies.to_offset(self.config.freq)
            idx = pd.date_range(out.index.min(), out.index.max(), freq=freq, tz=out.index.tz)
            out = out.reindex(idx)
        return out

    def _build_known_covariates(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not self.dataset.known_covariates:
            return None
        cov = df[self.dataset.known_covariates].copy()
        cov = cov.sort_index()
        cov = cov.fillna(method="ffill").fillna(method="bfill")
        return cov

    def _make_feature_matrix(
        self, index: pd.DatetimeIndex, base: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        frames = []
        for generator in self._feature_generators:
            feats = generator(index)
            if not isinstance(feats, pd.DataFrame):
                raise TypeError("Feature generators must return pandas.DataFrame objects")
            frames.append(feats)
        if base is not None:
            frames.append(base)
        if not frames:
            return pd.DataFrame(index=index)
        df = pd.concat(frames, axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        df = df.loc[index]
        return df

    def _fit_single_series(
        self, series_id: str, df: pd.DataFrame, prepared: Optional[pd.DataFrame] = None
    ) -> None:
        prepared_df = prepared if prepared is not None else self._prepare_single_series(df)
        target = prepared_df[self.dataset.target_column]
        target_imputed = self._imputer.fit_transform(target)

        covariates = self._build_known_covariates(prepared_df)
        features = self._make_feature_matrix(prepared_df.index, covariates)

        model = self._model_factory()
        model.fit(target_imputed, features)

        self._artifacts[series_id] = PipelineArtifacts(
            model=model, target=target_imputed, features=features
        )

    def fit(self, df: pd.DataFrame) -> "ForecastPipeline":
        self.validation_results_ = None
        seen_ids = set()
        for series_id, subset in self._iter_series(df):
            prepared = self._prepare_single_series(subset)
            self._maybe_validate_series(series_id, prepared)
            self._fit_single_series(series_id, subset, prepared=prepared)
            seen_ids.add(series_id)
        if self._expected_series_ids is not None:
            missing = self._expected_series_ids.difference(seen_ids)
            if missing:
                raise ValueError("Missing series in training data: " + ", ".join(sorted(missing)))
        return self

    # ----------------------------------------------------------------- inference
    def _future_index(self, model: ForecastModel) -> pd.DatetimeIndex:
        last = getattr(model, "last_timestamp_", None)
        if last is None:
            raise AttributeError("Model does not expose 'last_timestamp_' attribute")
        freq = pd.tseries.frequencies.to_offset(self.config.freq)
        return pd.date_range(last + freq, periods=self.config.horizon, freq=freq, tz=last.tz)

    def _prepare_future_covariates(
        self,
        series_id: str,
        future_known_covariates: Optional[pd.DataFrame],
        index: pd.DatetimeIndex,
    ) -> Optional[pd.DataFrame]:
        if future_known_covariates is None or not self.dataset.known_covariates:
            return None
        df = future_known_covariates.copy()
        if self.dataset.series_id_column and self.dataset.series_id_column in df.columns:
            df = df[df[self.dataset.series_id_column] == series_id]
        if self.dataset.time_column in df.columns:
            df = df.set_index(pd.to_datetime(df[self.dataset.time_column]))
            df = df.drop(columns=[self.dataset.time_column])
        df = df.reindex(index)
        return df[self.dataset.known_covariates]

    def predict(
        self,
        future_known_covariates: Optional[pd.DataFrame] = None,
        horizon: Optional[int] = None,
    ) -> pd.DataFrame:
        outputs = []
        steps = horizon or self.config.horizon
        for series_id, artifacts in self._artifacts.items():
            model = artifacts.model
            index = self._future_index(model)[:steps]
            cov = self._prepare_future_covariates(series_id, future_known_covariates, index)
            features = self._make_feature_matrix(index, cov)
            preds = model.predict(steps=len(index), X_future=features)
            df = pd.DataFrame({k: v for k, v in preds.items()})
            df.index = index
            df.index.name = self.dataset.time_column
            if self.dataset.series_id_column:
                df[self.dataset.series_id_column] = series_id
                df = df.set_index(self.dataset.series_id_column, append=True).reorder_levels(
                    [self.dataset.series_id_column, self.dataset.time_column]
                )
            outputs.append(df)
        if not outputs:
            return pd.DataFrame()
        result = pd.concat(outputs).sort_index()
        return result

    # ------------------------------------------------------------ validation
    def _rolling_origin_windows(
        self, index: pd.DatetimeIndex
    ) -> List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
        cfg = self.config.validation
        if cfg.test_size < self.config.horizon:
            raise ValueError("test_size must be at least as large as the forecast horizon")
        n = len(index)
        if n <= cfg.test_size:
            raise ValueError("Not enough observations for rolling validation")
        last_start = n - cfg.test_size
        first_start = last_start - (cfg.n_windows - 1) * cfg.step_size
        if first_start <= 0:
            raise ValueError("Rolling windows exceed available history")
        windows: List[Tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
        for offset in range(cfg.n_windows):
            start = first_start + offset * cfg.step_size
            train_idx = index[:start]
            test_idx = index[start : start + self.config.horizon]
            if len(test_idx) == 0:
                continue
            windows.append((train_idx, test_idx))
        return windows

    def _maybe_validate_series(self, series_id: str, prepared: pd.DataFrame) -> None:
        if self.config.validation.method != "rolling":
            return
        target = prepared[self.dataset.target_column]
        covariates = self._build_known_covariates(prepared)
        features = self._make_feature_matrix(prepared.index, covariates)
        windows = self._rolling_origin_windows(prepared.index)
        records: List[Dict[str, object]] = []
        for window_id, (train_idx, test_idx) in enumerate(windows):
            train_target = target.loc[train_idx]
            train_features = features.loc[train_idx]
            imputer = GapFiller(max_gap_hours=self._imputer.max_gap_hours)
            train_target_imputed = imputer.fit_transform(train_target)
            model = self._model_factory()
            model.fit(train_target_imputed, train_features)
            future_features = features.loc[test_idx]
            preds = model.predict(len(test_idx), X_future=future_features)
            record: Dict[str, object] = {
                "series_id": series_id,
                "window": window_id,
                "index": [ts.isoformat() for ts in test_idx],
                "y_true": target.loc[test_idx].to_list(),
            }
            for q, values in preds.items():
                record[q] = values.loc[test_idx].to_list()
            records.append(record)
        if records:
            frame = pd.DataFrame.from_records(records)
            if self.validation_results_ is None:
                self.validation_results_ = frame
            else:
                self.validation_results_ = pd.concat(
                    [self.validation_results_, frame], ignore_index=True
                )

    # ------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / "config.json"
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump({"training_config": self.config.model_dump(mode="json")}, handle)
        models_dir = path / "models"
        models_dir.mkdir(exist_ok=True)
        metadata: Dict[str, Dict[str, str]] = {}
        for series_id, artifacts in self._artifacts.items():
            series_dir = models_dir / series_id
            artifacts.model.save(series_dir)
            metadata[series_id] = {"model": self.model_name}
        with open(path / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle)

    @classmethod
    def load(cls, path: Path) -> "ForecastPipeline":
        path = Path(path)
        with open(path / "config.json", "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        config = TrainingConfig(**payload["training_config"])
        pipeline = cls(config=config)
        with open(path / "metadata.json", "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        models_dir = path / "models"
        for series_id, meta in metadata.items():
            model_name = meta.get("model", config.model)
            model_cls = MODEL_REGISTRY.get(model_name)
            model = model_cls.load(models_dir / series_id)  # type: ignore[attr-defined]
            pipeline._artifacts[series_id] = PipelineArtifacts(model=model)
        return pipeline


__all__ = ["ForecastPipeline"]
