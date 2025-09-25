# EnergiCast

EnergiCast to biblioteka AutoML do prognozowania energii, łącząca klasyczne modele
czasowe z funkcjami fizycznymi oraz narzędziami do walidacji i wdrażania. W repozytorium
znajdziesz komplet narzędzi obejmujących przygotowanie danych, inżynierię cech,
syntetyczną imputację, metryki kosztowe oraz eksport modeli do produkcji.

## Najważniejsze możliwości

- **Pipeline** – `ForecastPipeline` scala konfigurację (`TrainingConfig`), imputer
  `GapFiller`, generatory cech (`make_energy_features`) i modele z rejestru.
- **Modele** – gradient boosting (`XGBForecaster`), ETS/ARIMA (`ARIMAForecaster`) oraz
  planowana implementacja Temporal Fusion Transformer (`tft`).
- **AutoML** – moduły `energicast.automl` dostarczają pętlę Optuna i pomocniki do
  walidacji kroczącej.
- **Funkcje energetyczne** – kalendarz, cechy solarne (`solar_position_features`) i
  pogodowe (`simple_weather_features`).
- **Imputacja** – `GapFiller` łączy kierunkowe wypełnianie z sezonową średnią godzinową.
- **Metryki** – pinball, CRPS i energy-weighted RMSE (`energicast.metrics.metrics`).
- **Backtest** – `run_backtest` generuje raporty walidacyjne wraz z wykresami.
- **CLI** – polecenia `train`, `backtest`, `export`, `report` zbudowane na Typerze.

## Architektura w skrócie

```text
+-----------------+      +------------------+      +------------------+
| Dane surowe     | ---> | ForecastPipeline | ---> | Modele (XGB/ETS) |
+-----------------+      +------------------+      +------------------+
        |                         |                           |
        v                         v                           v
   Imputacja (GapFiller)   Cechy (make_energy_features)   Walidacja krocząca
        |                         |                           |
        +-------------------------+---------------------------+
                                  v
                           Backtest & Metryki
```

Repozytorium zawiera również szereg modułów pomocniczych: `energicast.deploy` (eksport
modeli), `energicast.hier` (rekonsyliacja hierarchiczna), `energicast.scenarios`
(scenariusze pogody) oraz `energicast.bench` (dane demonstracyjne OpenSTEF).

W kolejnych rozdziałach znajdziesz przewodniki krok po kroku oraz referencję API.
