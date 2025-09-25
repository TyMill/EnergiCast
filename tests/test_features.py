import pandas as pd

from energicast.features.energy import make_energy_features


def test_energy_features_include_calendar_weather_and_ramps():
    index = pd.date_range("2024-12-24", periods=48, freq="H", tz="UTC")
    feats = make_energy_features(index)
    expected_columns = {
        "is_weekend",
        "is_holiday",
        "solar_elevation",
        "solar_ramp_1h",
        "temp_lag_6",
        "wind_ramp_1h",
    }
    assert expected_columns.issubset(feats.columns)
    assert feats.loc["2024-12-25 12:00:00+00:00", "is_holiday"] == 1
    assert feats["is_weekend"].isin({0, 1}).all()
    assert feats["solar_cos_zenith"].between(-1, 1).all()
    assert feats["temp_lag_6"].isna().sum() == 0
