# AutoML i walidacja

Moduł `energicast.automl` zapewnia proste narzędzia do automatycznego strojenia modeli.

## Pętla Optuna (`AutoML`)

`AutoML` pobiera rejestr modeli (np. `MODEL_REGISTRY.registry`) oraz funkcję oceny.
Kluczowe elementy:

- **Przestrzeń wyszukiwania** – nazwy modeli deklarowane w rejestrze.
- **Obliczanie metryk** – obiekt `metric_fn` powinien zwracać wartość do minimalizacji
  (np. średnią stratę pinball na ostatnim oknie walidacyjnym).
- **Krok treningu** – w każdej próbie tworzony jest nowy model (`cls(horizon=horizon)`),
  dopasowywany do danych `y` i opcjonalnych cech `X`.
- **Wynik** – po `n_trials` najlepsze hiperparametry są dostępne pod `AutoML.best_`.

## Walidacja krocząca

`energicast.automl.validation` udostępnia strukturę `RollingOriginConfig` oraz funkcję
`rolling_origin_windows`, która odwzorowuje logikę pipeline. Typowy workflow:

```python
from energicast.automl.validation import RollingOriginConfig, rolling_origin_windows

index = frame.index
cfg = RollingOriginConfig(horizon=24, test_size=96, n_windows=3, step_size=24)
windows = rolling_origin_windows(index, cfg)
```

Każdy element listy to `RollingWindow(train=..., test=...)`, zawierający indeksy
wykorzystywane do trenowania i ewaluacji. Funkcja `validation_records_frame` przekształca
wyniki walidacji (np. listę słowników) do ramki danych kompatybilnej z `ForecastPipeline`.

## Rolling-origin w praktyce

- **test_size** powinno być co najmniej tak duże jak `horizon`, ponieważ walidacja
  korzysta z pełnych okien prognozy.
- **step_size** steruje odległością pomiędzy kolejnymi oknami; wartość 24 oznacza
  przesunięcie o jeden dzień przy danych godzinowych.
- **n_windows** definiuje liczbę walidacji – w pipeline wyniki trafiają do
  `validation_results_` i są wykorzystywane w `backtest.run_backtest`.

## Integracja z Optuna

Walidację można połączyć z AutoML poprzez agregację metryk z każdego okna i przekazanie ich
jako `metric_fn`. Nic nie stoi na przeszkodzie, aby w `metric_fn` korzystać z funkcji
z `energicast.metrics.metrics`, np. `multi_quantile_pinball`.
