"""Configuration objects shared across the code base.

The first iteration of the library only exposed a couple of loosely defined
attributes (frequency, horizon, quantiles).  As the project grows we need a
central place where assumptions about the input data and the training regime
are validated.  Pydantic is already a dependency, so we leverage it to provide
runtime validation with helpful error messages.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Dict, Iterable, List, Optional

from pandas.tseries.frequencies import to_offset
from pydantic import BaseModel, Field, field_validator, model_validator
from zoneinfo import ZoneInfo


def _ensure_unique(items: Iterable[str]) -> None:
    seen = set()
    for name in items:
        if name in seen:
            raise ValueError(f"Duplicate column '{name}' declared in dataset configuration")
        seen.add(name)


class DatasetConfig(BaseModel):
    """Describe how raw data frames should be interpreted by the pipeline."""

    time_column: str = Field(default="timestamp", description="Column with timestamps")
    target_column: str = Field(default="target", description="Column with the target series")
    series_id_column: Optional[str] = Field(
        default=None,
        description="Optional column identifying multiple series in the same dataframe",
    )
    known_covariates: List[str] = Field(
        default_factory=list,
        description="Columns with future-known dynamic features (e.g., weather forecasts)",
    )
    static_covariates: List[str] = Field(
        default_factory=list,
        description="Columns constant for a series (e.g., capacity, latitude)",
    )
    expected_series_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional whitelist of supported series identifiers",
    )
    require_complete_history: bool = Field(
        default=True,
        description="If True, ensure every series is reindexed to the requested frequency",
    )

    @model_validator(mode="after")
    def _validate_columns(self) -> "DatasetConfig":  # pragma: no cover - validated via tests
        candidates = [self.time_column, self.target_column]
        if self.series_id_column:
            candidates.append(self.series_id_column)
        candidates.extend(self.known_covariates)
        candidates.extend(self.static_covariates)
        _ensure_unique(candidates)
        if self.expected_series_ids is not None:
            _ensure_unique(self.expected_series_ids)
            if not self.expected_series_ids:
                raise ValueError("expected_series_ids cannot be an empty list")
            if self.series_id_column is None:
                raise ValueError("Series identifiers require 'series_id_column' to be configured")
        return self


class ValidationConfig(BaseModel):
    method: str = Field(default="holdout", description="holdout or rolling validation")
    test_size: int = Field(default=24, description="Number of time steps reserved for validation")
    step_size: int = Field(default=24, description="Stride between rolling windows")
    n_windows: int = Field(default=1, description="Number of validation windows for rolling")

    @field_validator("method")
    @classmethod
    def _check_method(cls, value: str) -> str:
        valid = {"holdout", "rolling"}
        if value not in valid:
            raise ValueError(f"Validation method must be one of {valid}, received '{value}'")
        return value

    @model_validator(mode="after")
    def _check_positive(self) -> "ValidationConfig":
        for attr in ("test_size", "step_size", "n_windows"):
            if getattr(self, attr) <= 0:
                raise ValueError(f"{attr} must be strictly positive")
        return self


class TrainingConfig(BaseModel):
    freq: str = Field(default="H", description="Series frequency, e.g., 'H' for hourly")
    timezone: Optional[str] = Field(
        default=None,
        description="IANA timezone name applied to timestamps during preprocessing",
    )
    model: str = Field(default="xgb", description="Registered model name used by the pipeline")
    model_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the model constructor",
    )
    horizon: int = Field(default=24, description="Forecast horizon in time steps")
    history_window: int = Field(
        default=7 * 24,
        description="How many past steps should be retained for models requiring lags",
    )
    quantiles: List[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    seed: int = Field(default=42, description="Random seed for deterministic components")
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @field_validator("freq")
    @classmethod
    def _validate_freq(cls, value: str) -> str:
        try:
            to_offset(value)
        except ValueError as exc:  # pragma: no cover - exercised via tests
            raise ValueError(f"Invalid pandas frequency alias '{value}'") from exc
        return value

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        try:
            ZoneInfo(value)
        except Exception as exc:  # pragma: no cover - depends on system tz database
            raise ValueError(f"Unknown timezone identifier '{value}'") from exc
        return value

    @field_validator("horizon", "history_window")
    @classmethod
    def _validate_positive(cls, value: int, field) -> int:
        if value <= 0:
            raise ValueError(f"{field.alias} must be strictly positive")
        return value

    @field_validator("quantiles")
    @classmethod
    def _validate_quantiles(cls, values: List[float]) -> List[float]:
        if not values:
            raise ValueError("At least one quantile must be provided")
        if any(not (0.0 < q < 1.0) for q in values):
            raise ValueError("Quantiles must lie strictly between 0 and 1")
        sorted_values = sorted(values)
        if sorted_values != list(values):
            raise ValueError("Quantiles must be sorted increasingly")
        return sorted_values

    @model_validator(mode="after")
    def _check_history_vs_horizon(self) -> "TrainingConfig":
        if self.history_window <= self.horizon:
            raise ValueError("history_window must exceed the forecast horizon to provide context")
        return self

    @cached_property
    def timezone_info(self) -> Optional[ZoneInfo]:
        """Return the configured timezone as a ZoneInfo object."""

        if self.timezone is None:
            return None
        return ZoneInfo(self.timezone)


class CostConfig(BaseModel):
    price_series: Optional[str] = Field(
        default=None,
        description="Column name with price for energy-weighted metrics",
    )
    imbalance_penalty: float = 1.0


class AutoMLConfig(BaseModel):
    models: List[str] = Field(default_factory=lambda: ["xgb", "arima"])  # "tft", "prophet" optional
    n_trials: int = 50
