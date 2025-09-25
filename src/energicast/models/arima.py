"""Classical ETS forecaster powered by StatsForecast."""

from typing import Dict, List

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ETS


class ARIMAForecaster:
    """ETS baseline via StatsForecast (ARIMA/ETS family)."""

    def __init__(self, horizon: int = 24, quantiles: List[float] = [0.5]):
        self.h = horizon
        self.q = quantiles
        self.sf = None

    def fit(self, y: pd.Series, X: pd.DataFrame = None):
        """Estimate the ETS model on the provided target series."""

        df = y.rename("y").to_frame()
        df["unique_id"] = "series"
        df = df.reset_index().rename(columns={df.index.name or "index": "ds"})
        self.sf = StatsForecast(models=[ETS(season_length=24)], freq="H")
        self.sf.fit(df)
        return self

    def predict(self, steps: int, X_future: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """Return deterministic forecasts repeated for each requested quantile."""

        fc = self.sf.predict(h=steps)
        fc = fc.set_index("ds")["ETS"]
        return {f"q{int(q*100):.0f}": fc for q in self.q}
