# Szybki start

Ten przewodnik prowadzi przez instalację EnergiCast oraz wykonanie minimalnej prognozy.

## Instalacja

Zalecana jest instalacja w trybie deweloperskim (editable), aby mieć dostęp do przykładowych
skryptów i dokumentacji:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[docs]
```

Instalacja `.[docs]` zapewnia wszystkie zależności wymagane do budowania dokumentacji oraz
opcji CLI.

## Minimalny przykład pipeline

Poniższy fragment korzysta z pliku `examples/pv_sample.csv`, aby wytrenować pipeline i
wykonać prognozę jednodniową.

```python
import pandas as pd
from energicast.config import DatasetConfig, TrainingConfig, ValidationConfig
from energicast.pipeline import ForecastPipeline

# 1. Wczytaj dane
frame = pd.read_csv("examples/pv_sample.csv", parse_dates=["timestamp"])

# 2. Przygotuj konfigurację
cfg = TrainingConfig(
    model="xgb",
    horizon=24,
    dataset=DatasetConfig(
        time_column="timestamp",
        target_column="power",
        known_covariates=["temp"],
        series_id_column="site",
    ),
    validation=ValidationConfig(method="rolling", test_size=48, n_windows=2, step_size=24),
)

# 3. Zbuduj i dopasuj pipeline
pipeline = ForecastPipeline(config=cfg)
pipeline.fit(frame)

# 4. Wygeneruj prognozę
forecast = pipeline.predict()
print(forecast.tail())
```

## CLI

Te same kroki można wykonać z poziomu wiersza poleceń, korzystając z pliku
`examples/pv_config.yaml`:

```bash
python -m energicast.cli train --config examples/pv_config.yaml --out runs/demo_model
python -m energicast.cli backtest --config examples/pv_config.yaml --out runs/demo_backtest
python -m energicast.cli report --backtest-dir runs/demo_backtest
```

Polecenie `export` opisano szczegółowo w przewodniku dotyczącym wdrażania.
