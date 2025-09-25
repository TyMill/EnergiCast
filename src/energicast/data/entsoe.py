import pandas as pd
import numpy as np


def load_entsoe_sample(n_days: int = 60) -> pd.Series:
    """Synthetic series resembling a daily load profile (hourly)."""
    rng = pd.date_range(
        end=pd.Timestamp.utcnow().floor("H"), periods=n_days * 24, freq="H", tz="UTC"
    )
    base = 1000 + 200 * np.sin(2 * np.pi * (rng.hour) / 24)  # diurnal cycle
    noise = np.random.normal(0, 30, size=len(rng))
    s = pd.Series(base + noise, index=rng, name="load")
    return s
