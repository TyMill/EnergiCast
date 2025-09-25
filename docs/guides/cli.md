# CLI

EnergiCast wykorzystuje [Typer](https://typer.tiangolo.com/) do ekspozycji funkcji przez CLI.
Główne polecenia znajdują się w `energicast.cli`.

## Dostępne komendy

```bash
python -m energicast.cli --help
```

### train

```bash
python -m energicast.cli train --config examples/pv_config.yaml --out runs/model_xgb
```

- Ładuje plik YAML z konfiguracją (`TrainingConfig` + sekcja danych).
- Trenuje pipeline (`ForecastPipeline.fit`).
- Zapisuje artefakty w katalogu `--out` (konfiguracja, modele, metadata).

### backtest

```bash
python -m energicast.cli backtest --config examples/pv_config.yaml --out runs/backtest_xgb
```

- Uruchamia `ForecastPipeline.fit` z włączoną walidacją rolling.
- Przekazuje pipeline i dane do `run_backtest`.
- Zapisuje raporty (`metrics.csv`, `summary.json`, `reports/*.png`).

### export

```bash
python -m energicast.cli export --model-dir runs/model_xgb --fmt pickle --out runs/export_xgb
```

- Wczytuje pipeline z katalogu (`ForecastPipeline.load`).
- Wywołuje `deploy.export.export_pipeline`, aby przygotować artefakty do wdrożenia.
- Dołącza metadane (`metadata.json`) z informacją o wersji pakietu.

### report

```bash
python -m energicast.cli report --backtest-dir runs/backtest_xgb
```

- Wczytuje `metrics.csv` i `summary.json`.
- Wyświetla w konsoli agregaty oraz statystyki per serię.

## Najlepsze praktyki

- Ścieżki w konfiguracji YAML mogą być względne wobec położenia pliku konfiguracyjnego.
- CLI zgłasza błędy, jeśli brakuje sekcji `training` lub plików w katalogach wynikowych.
- Polecenia można łączyć w pipeline CI (np. `train` → `backtest` → `export`).
