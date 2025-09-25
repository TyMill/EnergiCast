"""Gradient boosted tree forecaster with autoregressive features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .registry import register_model


def _format_quantile(q: float) -> str:
    return f"q{int(round(q * 100)):02d}"


@register_model("xgb")
class XGBForecaster:
    """Direct autoregressive forecaster with empirical quantile adjustments."""

    def __init__(
        self,
        horizon: int = 24,
        quantiles: Optional[List[float]] = None,
        max_lag: int = 48,
        base_model_params: Optional[Dict[str, float]] = None,
    ) -> None:
        self.h = horizon
        self.quantiles = sorted(set(quantiles or [0.5]))
        if 0.5 not in self.quantiles:
            self.quantiles.append(0.5)
            self.quantiles.sort()
        self.max_lag = max_lag
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": 1,
        }
        if base_model_params:
            params.update(base_model_params)
        self._base_model_params = params

        self.base_model_: Optional[XGBRegressor] = None
        self.feature_columns_: List[str] = []
        self.lag_columns_: List[str] = []
        self.history_: Optional[pd.Series] = None
        self.last_timestamp_: Optional[pd.Timestamp] = None
        self.quantile_offsets_: Dict[float, float] = {}

    # ------------------------------------------------------------------ training
    def _make_training_matrix(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        window = min(self.max_lag, len(y))
        if window <= 1:
            raise ValueError("Not enough observations to build lagged features")
        lagged = {f"lag_{i}": y.shift(i) for i in range(1, window + 1)}
        frame = pd.concat(lagged.values(), axis=1)
        frame.columns = list(lagged.keys())
        if X is not None:
            frame = pd.concat([frame, X], axis=1)
        frame["target"] = y
        frame = frame.dropna()
        if frame.empty:
            raise ValueError("Lag construction resulted in an empty training matrix")
        return frame

    def fit(self, y: pd.Series, X: Optional[pd.DataFrame] = None) -> "XGBForecaster":
        data = self._make_training_matrix(y, X)
        X_train = data.drop(columns=["target"])
        y_train = data["target"]

        model = XGBRegressor(**self._base_model_params)
        model.fit(X_train, y_train)

        self.base_model_ = model
        self.feature_columns_ = list(X_train.columns)
        self.lag_columns_ = [c for c in self.feature_columns_ if c.startswith("lag_")]
        self.history_ = y.loc[data.index].tail(len(self.lag_columns_))
        self.last_timestamp_ = y.loc[data.index].index.max()

        residuals = y_train - pd.Series(model.predict(X_train), index=y_train.index)
        self.quantile_offsets_ = {
            q: float(np.quantile(residuals, q)) for q in self.quantiles if not np.isclose(q, 0.5)
        }
        return self

    # ---------------------------------------------------------------- inference
    def _lag_features_from_history(self, history: pd.Series) -> Dict[str, float]:
        values = history.to_numpy()
        feats: Dict[str, float] = {}
        for i, col in enumerate(self.lag_columns_, start=1):
            if len(values) >= i:
                feats[col] = float(values[-i])
            else:
                feats[col] = float(values[0])
        return feats

    def _ensure_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_columns_ if col not in df.columns]
        for column in missing:
            df[column] = 0.0
        return df[self.feature_columns_]

    def predict(self, steps: int, X_future: Optional[pd.DataFrame] = None) -> Dict[str, pd.Series]:
        if self.base_model_ is None or self.history_ is None or self.last_timestamp_ is None:
            raise RuntimeError("Model must be fitted before calling predict")
        freq = self.history_.index.freq or pd.infer_freq(self.history_.index)
        if freq is None:
            freq = "H"
        index = pd.date_range(
            start=self.last_timestamp_ + pd.tseries.frequencies.to_offset(freq),
            periods=steps,
            freq=freq,
            tz=self.last_timestamp_.tz,
        )
        if X_future is not None:
            future = X_future.copy()
        else:
            future = pd.DataFrame(index=index)
        future = future.reindex(index)
        preds = {q: [] for q in self.quantiles}
        history = self.history_.copy()

        for ts in index:
            lag_feats = self._lag_features_from_history(history)
            row = future.loc[[ts]] if ts in future.index else pd.DataFrame(index=[ts])
            row = row.copy()
            for col, value in lag_feats.items():
                row[col] = value
            row = self._ensure_feature_frame(row)
            median_pred = float(self.base_model_.predict(row)[0])
            for q in self.quantiles:
                if np.isclose(q, 0.5):
                    preds[q].append(median_pred)
                else:
                    preds[q].append(median_pred + self.quantile_offsets_.get(q, 0.0))
            history = pd.concat([history, pd.Series([median_pred], index=[ts])])

        return {_format_quantile(q): pd.Series(values, index=index) for q, values in preds.items()}

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path) -> None:
        """Persist the fitted estimator and its auxiliary state."""

        if self.base_model_ is None or self.history_ is None or self.last_timestamp_ is None:
            raise RuntimeError("Model must be fitted before saving")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        model_path = path / "model.json"
        self.base_model_.save_model(str(model_path))
        state = {
            "h": self.h,
            "quantiles": self.quantiles,
            "max_lag": self.max_lag,
            "base_model_params": self._base_model_params,
            "feature_columns": self.feature_columns_,
            "lag_columns": self.lag_columns_,
            "history_index": [ts.isoformat() for ts in self.history_.index],
            "history_values": self.history_.to_list(),
            "last_timestamp": self.last_timestamp_.isoformat(),
            "quantile_offsets": self.quantile_offsets_,
        }
        with open(path / "state.json", "w", encoding="utf-8") as handle:
            json.dump(state, handle)

    @classmethod
    def load(cls, path: Path) -> "XGBForecaster":
        """Reconstruct a fitted estimator from ``path``."""

        path = Path(path)
        with open(path / "state.json", "r", encoding="utf-8") as handle:
            state = json.load(handle)
        obj = cls(
            horizon=state["h"],
            quantiles=state["quantiles"],
            max_lag=state["max_lag"],
            base_model_params=state["base_model_params"],
        )
        model = XGBRegressor(**obj._base_model_params)
        model.load_model(str(path / "model.json"))
        obj.base_model_ = model
        obj.feature_columns_ = list(state["feature_columns"])
        obj.lag_columns_ = list(state["lag_columns"])
        history_index = pd.DatetimeIndex(pd.to_datetime(state["history_index"]))
        obj.history_ = pd.Series(state["history_values"], index=history_index)
        obj.last_timestamp_ = pd.Timestamp(state["last_timestamp"])
        obj.quantile_offsets_ = {float(k): float(v) for k, v in state["quantile_offsets"].items()}
        return obj


__all__ = ["XGBForecaster"]
