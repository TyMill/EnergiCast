import pandas as pd


def ensure_freq(s: pd.Series, freq: str = "H") -> pd.Series:
    return s.asfreq(freq)
