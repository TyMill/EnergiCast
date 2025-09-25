# EnergiCast

**EnergiCast** to biblioteka AutoML do prognozowania energii, łącząca klasyczne modele
szeregów czasowych z cechami fizycznymi oraz narzędziami do walidacji, backtestu i wdrażania.
Repozytorium zawiera komplet komponentów: przygotowanie danych, inżynierię cech, imputację,
metryki kosztowe, pipeline treningowy i eksport modeli.

👉 **Dokumentacja on-line:** <https://tymill.github.io/EnergiCast/>

---

## Szybki start

- Zainstaluj i uruchom przykład: zobacz **[Getting started](getting-started.md)**.
- Uruchom CLI: komendy `train`, `backtest`, `export`, `report` – patrz **[CLI](guides/cli.md)**.

---

## Przewodniki (Guides)

- **Pipeline** – jak działa `ForecastPipeline` (ładowanie → imputacja → cechy → trening → predykcja):  
  [guides/pipeline.md](guides/pipeline.md)

- **Modele** – XGB/ETS-ARIMA oraz plan TFT, interfejsy i konfiguracja:  
  [guides/models.md](guides/models.md)

- **AutoML i walidacja** – pętle Optuna, rolling-origin, konfiguracja CV:  
  [guides/automl.md](guides/automl.md)

- **Cechy i dane** – `make_energy_features`, święta/weekendy, ramp-rate, lagi pogody; `solar_position_features`:  
  [guides/features.md](guides/features.md)

- **Imputacja** – `GapFiller`, warianty sezonowe i testy „dziur”:  
  [guides/impute.md](guides/impute.md)

- **Metryki** – pinball (multi-quantile), CRPS (empiryczny), energy-weighted RMSE, koszt niezbilansowania:  
  [guides/metrics.md](guides/metrics.md)

- **Backtest** – `run_backtest`, logi metryk i wykresy raportowe:  
  [guides/backtest.md](guides/backtest.md)

- **CLI** – `train`, `backtest`, `export`, `report` z przykładami:  
  [guides/cli.md](guides/cli.md)

- **Deploy/Export** – metadane, wersjonowanie artefaktów i eksport:  
  [guides/deploy.md](guides/deploy.md)

- **Hierarchie i rekonsyliacja** – struktury przestrzenne/temporalne, MinT (w przygotowaniu):  
  [guides/hierarchy.md](guides/hierarchy.md)

- **Scenariusze pogody** – generacja scenariuszy, niepewność i korelacje (w przygotowaniu):  
  [guides/scenarios.md](guides/scenarios.md)

- **Benchmarki** – OpenSTEF i procedury weryfikacji (w przygotowaniu):  
  [guides/benchmarks.md](guides/benchmarks.md)

---

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


---

## Wkład i standardy

- Zasady współpracy i stylu kodu: **[Contributing](contributing.md)**
- Zgłaszanie błędów/feature’ów: Issues/PR w repozytorium GitHub.

!!! note
    **API Reference** pojawi się wkrótce (sekcja „API Reference” w nawigacji).  
    Obecnie skupiamy się na przewodnikach z folderu `guides/`.

