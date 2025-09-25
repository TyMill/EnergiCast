# Metryki

Moduł `energicast.metrics.metrics` zawiera funkcje używane podczas backtestu i AutoML.

## Pinball loss i warianty

- `pinball_loss(y_true, y_pred, q)` – strata dla pojedynczego kwantyla.
- `multi_quantile_pinball(y_true, forecasts, quantiles)` – zwraca słownik z wynikami
  dla każdego kwantyla oraz średnią (`pinball_mean`).

## CRPS

`empirical_crps` aproksymuje Continuous Ranked Probability Score poprzez numeryczne
całkowanie strat pinball dla dostarczonych kwantyli. Funkcja oczekuje, że prognozy są
podane jako `DataFrame` lub mapowanie `{kolumna_kwantylowa: Series}`.

## Energy-weighted RMSE

`energy_weighted_rmse` oblicza błąd średniokwadratowy ważony ceną lub absolutną wartością
obciążenia. Przy braku kolumny cenowej pipeline korzysta z samej serii obciążenia.

## Koszt niezbilansowania

`imbalance_cost` pozwala przydzielić asymetryczne kary za niedoszacowanie (`under_penalty`)
 i przeszacowanie (`over_penalty`). Może być używana jako funkcja celu dla AutoML.

## Integracja z backtestem

`backtest.run_backtest` używa `multi_quantile_pinball`, `empirical_crps` oraz
`energy_weighted_rmse` do generowania raportów CSV i podsumowań JSON.
