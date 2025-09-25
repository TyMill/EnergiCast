# Modele

EnergiCast udostępnia wspólny interfejs modeli poprzez rejestr `MODEL_REGISTRY`.
Każdy model implementuje metody `fit`, `predict` oraz (opcjonalnie) `save`/`load`.

## XGBForecaster (`energicast.models.xgb`)

- **Charakterystyka**: bezpośredni model autoregresyjny oparty na `xgboost.XGBRegressor`.
- **Cechy wejściowe**: pipeline tworzy opóźnienia (`lag_1`, `lag_2`, ...), a także
  scala cechy energetyczne i kowariaty pogodowe.
- **Kwantyle**: model dopasowuje medianę i estymuje przesunięcia kwantylowe na podstawie
  reszt treningowych (`quantile_offsets_`).
- **Persistencja**: `save(path)` zapisuje model oraz stan (`state.json`), a `load(path)`
  umożliwia odtworzenie estymatora.

## ARIMAForecaster (`energicast.models.arima`)

- **Silnik**: `StatsForecast` z modelem ETS (`statsforecast.models.ETS`).
- **Zastosowanie**: szybki baseline bez dodatkowych cech; `predict` zwraca tę samą serię
  dla każdego żądanego kwantyla.
- **Wymagania**: dane wejściowe muszą być konwertowane do formatu `unique_id`/`ds`
  oczekiwanego przez StatsForecast.

## TFTForecaster (`energicast.models.tft`)

- Moduł zawiera obecnie opis planowanej integracji z Temporal Fusion Transformer.
- Docelowo będzie oparty na PyTorch Lightning z obsługą cech statycznych i dynamicznych.
- W przewodniku pozostawiamy go jako *roadmapę* – w rejestrze można go traktować jako
  placeholder do przyszłej implementacji.

## Rejestr modeli

Rejestr (`energicast.models.registry.MODEL_REGISTRY`) umożliwia rejestrowanie modeli
przez dekorator `@register_model("alias")`. Pipeline korzysta z rejestru przy
instancjonowaniu modeli na podstawie pola `TrainingConfig.model`.
