"""Model registry and persistence interface for forecasting models."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypeVar

import pandas as pd


class ForecastModel(Protocol):
    """Protocol describing the minimum interface expected by the pipeline."""

    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> ForecastModel:
        """Fit the model using the provided target and optional covariates."""

    def predict(self, steps: int, X_future: pd.DataFrame | None = None) -> dict[str, pd.Series]:
        """Generate probabilistic forecasts for the requested number of steps."""

    def save(self, path: Path) -> None:
        """Persist the trained model and its state to ``path``."""

    @classmethod
    def load(cls, path: Path) -> ForecastModel:
        """Load a previously persisted model instance from ``path``."""

    last_timestamp_: pd.Timestamp  # pylint: disable=pointless-statement


T = TypeVar("T", bound=ForecastModel)


@dataclass
class ModelRegistry:
    """Simple in-memory registry mapping model names to implementations."""

    _registry: dict[str, type[ForecastModel]]

    def __init__(self) -> None:
        self._registry = {}

    def register(self, name: str, cls: type[T]) -> type[T]:
        key = name.lower()
        if key in self._registry:
            raise ValueError(f"Model '{name}' already registered")
        self._registry[key] = cls
        return cls

    def get(self, name: str) -> type[ForecastModel]:
        try:
            return self._registry[name.lower()]
        except KeyError as exc:  # pragma: no cover - error propagated to caller
            raise KeyError(f"Unknown model '{name}'. Available: {sorted(self._registry)}") from exc

    def create(self, name: str, **kwargs) -> ForecastModel:
        cls = self.get(name)
        return cls(**kwargs)  # type: ignore[call-arg]

    def names(self) -> Iterable[str]:
        return tuple(sorted(self._registry))


MODEL_REGISTRY = ModelRegistry()


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator registering a model implementation under the provided name."""

    def decorator(cls: type[T]) -> type[T]:
        return MODEL_REGISTRY.register(name, cls)

    return decorator


__all__ = ["ForecastModel", "MODEL_REGISTRY", "register_model"]
