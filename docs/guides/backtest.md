# Backtest

`energicast.backtest.run_backtest` umożliwia przeprowadzenie spójnego backtestu dla pipeline.

## Przepływ działania

1. `ForecastPipeline.fit` wykonuje trening oraz walidację kroczącą i zapisuje wyniki w
   `pipeline.validation_results_`.
2. `run_backtest` odczytuje każdą próbkę walidacyjną, liczy metryki (`multi_quantile_pinball`,
   `empirical_crps`, `energy_weighted_rmse`) i agreguje wyniki do ramki.
3. Podsumowanie średnich metryk zapisywane jest w `summary.json`, a pełna tabela w
   `metrics.csv`.
4. Dla ostatniego okna każdej serii generowany jest wykres PNG (z wykorzystaniem Matplotlib).

## Raporty

- `BacktestResult.metrics` – ramka z metrykami per okno i serię.
- `BacktestResult.summary` – średnie wartości metryk.
- `BacktestResult.output_dir` – katalog z plikami (`metrics.csv`, `summary.json`, `reports/`).

## Tryb CLI

Polecenie `python -m energicast.cli backtest` oczekuje tych samych parametrów co `train`:

```bash
python -m energicast.cli backtest --config examples/pv_config.yaml --out runs/demo_backtest
```

Po zakończeniu `runs/demo_backtest` zawiera CSV, JSON oraz wykresy.

## Wskazówki

- Upewnij się, że `TrainingConfig.validation.method` ustawiono na `"rolling"`, aby
  backtest miał dostęp do danych.
- `run_backtest` wymaga co najmniej jednego okna walidacyjnego; w przeciwnym wypadku
  zgłosi wyjątek `ValueError`.
