"""Lightweight Optuna-based search utilities for forecasting models."""

from typing import Any, Dict

import optuna


class AutoML:
    """Coordinate model selection over the registry using Optuna trials."""

    def __init__(self, registry: Dict[str, Any], metric_fn, n_trials: int = 50):
        """Initialise the search object with a registry and scoring function."""

        self.registry = registry
        self.metric_fn = metric_fn
        self.n_trials = n_trials
        self.best_ = None

    def fit(self, y, X=None, horizon: int = 24):
        """Run the optimisation loop and store the best trial parameters."""

        def objective(trial: optuna.Trial):
            name = trial.suggest_categorical("model", list(self.registry.keys()))
            cls = self.registry[name]
            model = cls(horizon=horizon)
            model.fit(y, X)
            f = model.predict(steps=horizon, X_future=X.tail(horizon) if X is not None else None)
            yhat = f.get("q50") or f.get("q0.5")
            score = self.metric_fn(y.tail(horizon), yhat)
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        self.best_ = study.best_params
        return self
